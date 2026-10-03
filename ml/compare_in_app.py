#!/usr/bin/env python
"""
Phase 3, step 6: does the fine-tuned model help INSIDE TalentSync?
Uses your real role_matcher.py and validation_test_suite.py. Nothing in them is edited.

    py -3.12 ml\\compare_in_app.py          (run from the repo root, after train_final.py)

2 x 2 design, fixed in advance
    model     A/B = pretrained (all-MiniLM-L6-v2)   vs   C/D = fine-tuned (models\\minilm-resume-jd-ft)
    template  "current"  = the app's templated sentences (_build_candidate_text / _build_role_text)
              "compact"  = the format the fine-tuned model was trained on:
                           candidate "Experience: <roles>. Skills: <skills>."
                           role      "Role: <name>. Key criteria: <core skills>. Requirements: <optional skills>."
    A pretrained + current (what ships today)     B fine-tuned + current
    D pretrained + compact (template control)     C fine-tuned + compact

Reported for each configuration
    1. as shipped      the app's own thresholds and weights, run through the real match_roles()
    2. best-of-grid    the best cell of one shared weight/threshold grid (optimistic: tuned on the suite)
    3. LOO-tuned       leave-one-case-out tuning on that grid (honest out-of-sample estimate of "tuned")
Two scoring modes
    suite      exactly validation_test_suite.py's logic
    corrected  same, except that for cases with expected_rank = None, the role being ABSENT counts as a
               pass. The suite scores absence as FAILED, which penalises the correct rejection and rewards
               models whose higher cosines let more roles clear the 0.25 threshold.

Caveat: the suite has 21 hand-written cases and was probably used to tune the current weights, so it
favours the configuration that was tuned on it (A). One case = ~4.8 points of effectiveness.

Outputs ml/results/compare_in_app.md / .json
"""

import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

import role_matcher as rm  # noqa: E402
from validation_test_suite import TEST_CASES  # noqa: E402
from eval import paired_bootstrap  # noqa: E402
from inspect_netsol import Tee, hr  # noqa: E402

DEF_W, DEF_ST, DEF_OT = (0.55, 0.35, 0.10), 0.25, 0.15
W_SEM = [0.35, 0.45, 0.55, 0.65, 0.75]
W_BONUS = [0.0, 0.10]
SEM_THR = [round(x, 2) for x in np.arange(0.0, 0.701, 0.05)]
OV_THR = [0.0, 0.15, 0.30]


# --------------------------------------------------------------------------- #
# Templates
# --------------------------------------------------------------------------- #
def compact_role_text(role_name, role_profile):
    parts = [f"Role: {role_name}."]
    core = ", ".join(role_profile.get("core_skills", []))
    opt = ", ".join(role_profile.get("optional_skills", []))
    if core:
        parts.append(f"Key criteria: {core}.")
    if opt:
        parts.append(f"Requirements: {opt}.")
    return " ".join(parts)


def compact_candidate_text(skills, roles, tools):
    parts = []
    if roles:
        parts.append("Experience: " + "; ".join(roles) + ".")
    if skills:
        parts.append("Skills: " + ", ".join(skills) + ".")
    return " ".join(parts) or "Skills: none."


# --------------------------------------------------------------------------- #
# Scoring (mirrors validation_test_suite.py; 'corrected' fixes negative cases)
# --------------------------------------------------------------------------- #
def score_case(ranked_names, case, mode):
    exp, er = case["expected_role"].lower(), case["expected_rank"]
    if not ranked_names:
        return 1.0 if (mode == "corrected" and er is None) else 0.0
    matches = [i + 1 for i, n in enumerate(ranked_names) if exp in n.lower()]
    if not matches:
        return 1.0 if (mode == "corrected" and er is None) else 0.0
    rank = matches[0]
    if er is None:
        return 0.5 if rank <= 5 else 1.0
    return 1.0 if rank <= er else 0.5


def expected_rank_of(ranked_names, case):
    exp = case["expected_role"].lower()
    for i, n in enumerate(ranked_names):
        if exp in n.lower():
            return i + 1
    return None


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--finetuned", default=os.path.join(ROOT, "models", "minilm-resume-jd-ft"))
    ap.add_argument("--base", default="all-MiniLM-L6-v2")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--out", default=os.path.join(ROOT, "ml", "results", "compare_in_app"))
    args = ap.parse_args()
    sys.stdout = Tee(args.out + ".txt")

    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity

    if not os.path.isdir(args.finetuned):
        print(f"ERROR: fine-tuned model not found at {args.finetuned}. Run ml\\train_final.py first.")
        sys.exit(1)
    models = {"pretrained": SentenceTransformer(args.base), "fine-tuned": SentenceTransformer(args.finetuned)}

    orig = {"cand": rm._build_candidate_text, "role": rm._build_role_text}
    templates = {"current": (orig["cand"], orig["role"]), "compact": (compact_candidate_text, compact_role_text)}
    configs = {"A pretrained + current": ("pretrained", "current"),
               "B fine-tuned + current": ("fine-tuned", "current"),
               "C fine-tuned + compact": ("fine-tuned", "compact"),
               "D pretrained + compact": ("pretrained", "compact")}

    cases = TEST_CASES
    n_cases = len(cases)
    role_names = list(rm.ROLE_PROFILES)
    role_lower = [r.lower() for r in role_names]
    ov = np.array([[rm.compute_skill_overlap(c["skills"], rm.ROLE_PROFILES[r]) for r in role_names] for c in cases])
    bo = np.array([[rm.compute_role_match_bonus(c["skills"], c["roles"], r) for r in role_names] for c in cases])
    neg = [i for i, c in enumerate(cases) if c["expected_rank"] is None]

    hr(f"SETUP: {n_cases} test cases, {len(role_names)} roles, negative cases (expected_rank=None): "
       f"{[cases[i]['name'] for i in neg]}")

    # ---- semantic matrices (same recipe as the app: encode, cosine, round 4) ----
    sem = {}
    for cname, (mname, tname) in configs.items():
        bc, br = templates[tname]
        m = models[mname]
        ce = m.encode([bc(c["skills"], c["roles"], c["tools"]) for c in cases])
        re_ = m.encode([br(r, rm.ROLE_PROFILES[r]) for r in role_names])
        sem[cname] = np.round(cosine_similarity(ce, re_), 4)

    hr("SEMANTIC SCALE (mean cosine over all case x role pairs; shows why thresholds need re-tuning)")
    for cname in configs:
        s = sem[cname]
        print(f"  {cname:26s} mean={s.mean():.3f}  p10={np.percentile(s, 10):.3f}  p90={np.percentile(s, 90):.3f}  "
              f"share >= 0.25: {(s >= 0.25).mean():.0%}")

    # ---- ranking helper replicating match_roles ----
    def ranked(c_idx, S, w, st, ot):
        sem_r, ov_r, bo_r = S[c_idx], ov[c_idx], bo[c_idx]
        keep = np.where((ov_r >= ot) & (sem_r >= st))[0]
        if keep.size == 0:
            return [], []
        final = np.round(np.clip(w[0] * sem_r[keep] + w[1] * ov_r[keep] + w[2] * bo_r[keep], 0, 1), 4)
        order = sorted(range(keep.size), key=lambda k: (-final[k], -ov_r[keep[k]], -sem_r[keep[k]], role_lower[keep[k]]))
        return [role_names[keep[k]] for k in order], [float(final[k]) for k in order]

    # ---- as shipped: the real match_roles() ----
    rm.generate_job_links = lambda *a, **k: {}  # no link building during evaluation
    shipped = {}
    for cname, (mname, tname) in configs.items():
        rm._model, rm._role_embeddings = models[mname], None
        rm._build_candidate_text, rm._build_role_text = templates[tname]
        shipped[cname] = []
        for c in cases:
            recs = rm.match_roles(c["skills"], c["roles"], c["tools"])
            shipped[cname].append(([r["role"] for r in recs], [r["final_score"] for r in recs]))
    rm._build_candidate_text, rm._build_role_text = orig["cand"], orig["role"]

    hr("REPLICATION CHECK (my grid code vs the real match_roles at the app's own settings)")
    for cname in configs:
        same = sum(ranked(i, sem[cname], DEF_W, DEF_ST, DEF_OT)[0] == shipped[cname][i][0] for i in range(n_cases))
        print(f"  {cname:26s} identical rankings in {same}/{n_cases} cases"
              + ("" if same == n_cases else "   (differences come from float rounding at 4 decimals; small)"))

    # ---- grid ----
    cells = [((ws, 1.0 - ws - wb, wb), st, ot) for ws in W_SEM for wb in W_BONUS for st in SEM_THR for ot in OV_THR
             if 1.0 - ws - wb >= -1e-9]
    dist = np.array([abs(c[0][0] - DEF_W[0]) + abs(c[0][2] - DEF_W[2]) + abs(c[1] - DEF_ST) + abs(c[2] - DEF_OT)
                     for c in cells])
    grid = {}
    for cname in configs:
        grid[cname] = {}
        for mode in ("suite", "corrected"):
            M = np.zeros((len(cells), n_cases))
            for ci, (w, st, ot) in enumerate(cells):
                for i in range(n_cases):
                    M[ci, i] = score_case(ranked(i, sem[cname], w, st, ot)[0], cases[i], mode)
            grid[cname][mode] = M

    def loo(M):
        tot, out = M.sum(1), np.zeros(M.shape[1])
        for i in range(M.shape[1]):
            means = (tot - M[:, i]) / (M.shape[1] - 1)
            cand = np.where(means >= means.max() - 1e-12)[0]
            out[i] = M[cand[np.argmin(dist[cand])], i]
        return out

    # ---- tables ----
    results, per_case = {}, {}
    for cname in configs:
        results[cname] = {}
        for mode in ("suite", "corrected"):
            ship = np.array([score_case(shipped[cname][i][0], cases[i], mode) for i in range(n_cases)])
            M = grid[cname][mode]
            lo = loo(M)
            results[cname][mode] = {"shipped": float(ship.mean()), "best_of_grid": float(M.mean(1).max()),
                                    "loo_tuned": float(lo.mean())}
            per_case[(cname, mode)] = {"shipped": ship, "loo": lo}

    md = "# In-app comparison (validation suite, 21 cases)\n"
    md += ("\nEffectiveness = mean case score (pass 1, partial 0.5, fail 0). 'suite' = the suite's own logic; "
           "'corrected' counts correct absence as a pass for expected_rank=None cases. One case = 4.8 points. "
           "best-of-grid is optimistic (tuned on the suite); LOO-tuned is the honest tuned estimate.\n")
    for mode in ("suite", "corrected"):
        hr(f"EFFECTIVENESS, scoring = {mode}")
        head = "| config | as shipped | best-of-grid (optimistic) | LOO-tuned |\n|---|---|---|---|\n"
        rows = "".join(f"| {c} | {results[c][mode]['shipped']:.1%} | {results[c][mode]['best_of_grid']:.1%} | "
                       f"{results[c][mode]['loo_tuned']:.1%} |\n" for c in configs)
        print("  " + (head + rows).replace("\n", "\n  "))
        md += f"\n## Effectiveness ({mode} scoring)\n\n{head}{rows}"

    hr("BEHAVIOUR AS SHIPPED (what a user would see)")
    head = "| config | median # recs per case | top final score on negative cases (max) |\n|---|---|---|\n"
    rows = ""
    behav = {}
    for c in configs:
        n_recs = [len(shipped[c][i][0]) for i in range(n_cases)]
        tops = [shipped[c][i][1][0] if shipped[c][i][1] else 0.0 for i in neg]
        behav[c] = {"median_recs": float(np.median(n_recs)), "max_top_score_negative": float(max(tops)) if tops else None}
        rows += f"| {c} | {np.median(n_recs):.0f} | {max(tops):.0%} |\n"
    print("  " + (head + rows).replace("\n", "\n  "))
    md += f"\n## Behaviour as shipped\n\n{head}{rows}"

    hr("EXPECTED-ROLE RANK PER CASE AS SHIPPED ('-' = not recommended)")
    cn = list(configs)
    line = f"  {'case':44s} {'exp':>4s} " + " ".join(f"{c[:1]:>4s}" for c in cn)
    print(line)
    md += "\n## Expected-role rank per case (as shipped)\n\n| case | expected rank | " + " | ".join(c[:1] for c in cn) + " |\n|---|---|" + "---|" * len(cn) + "\n"
    for i, c in enumerate(cases):
        rks = [expected_rank_of(shipped[x][i][0], c) for x in cn]
        er = c["expected_rank"] if c["expected_rank"] is not None else "none"
        print(f"  {c['name'][:44]:44s} {str(er):>4s} " + " ".join(f"{(r if r else '-'):>4}" for r in rks))
        md += f"| {c['name']} | {er} | " + " | ".join(str(r) if r else "-" for r in rks) + " |\n"

    hr("PAIRED DIFFERENCES over the 21 cases, corrected scoring (mean diff [95% CI]; * = CI excludes 0)")
    md += "\n## Paired differences (corrected scoring)\n\n| comparison | basis | diff [95% CI] |\n|---|---|---|\n"
    comps = [("B fine-tuned + current", "A pretrained + current", "model effect, current template"),
             ("C fine-tuned + compact", "D pretrained + compact", "model effect, compact template"),
             ("D pretrained + compact", "A pretrained + current", "template effect, pretrained"),
             ("C fine-tuned + compact", "A pretrained + current", "best new vs shipped")]
    results["paired"] = []
    for a, b, label in comps:
        for basis, key in (("as shipped", "shipped"), ("LOO-tuned", "loo")):
            da = {"score": {i: float(v) for i, v in enumerate(per_case[(a, 'corrected')][key])}}
            db = {"score": {i: float(v) for i, v in enumerate(per_case[(b, 'corrected')][key])}}
            d = paired_bootstrap(da, db, "score", n_boot=args.n_boot)
            if d:
                star = "*" if d["significant"] else " "
                print(f"  {star} {a[:1]} - {b[:1]}  {label:34s} {basis:10s} {d['mean_diff']:+.3f} "
                      f"[{d['lo']:+.3f}, {d['hi']:+.3f}]")
                md += f"| {a[:1]} - {b[:1]} ({label}) | {basis} | {d['mean_diff']:+.3f} [{d['lo']:+.3f}, {d['hi']:+.3f}]{star} |\n"
                results["paired"].append({"a": a, "b": b, "basis": basis, **d})

    hr("DECISION RULE (fixed in advance)")
    print("  Switch the app default to the fine-tuned model ONLY if, on LOO-tuned corrected scoring, a fine-tuned")
    print("  configuration beats A (what ships today) with a paired CI above 0 AND no safety regression")
    print("  (negative cases must not recommend the wrong role at a high score). Otherwise keep pretrained as the")
    print("  default and publish the fine-tuned model as an opt-in flag with these results in the README.")
    md += ("\n## Decision rule (fixed in advance)\n\nSwitch the default only if, on LOO-tuned corrected scoring, a "
           "fine-tuned configuration beats A with a paired CI above 0 and shows no safety regression on the "
           "negative cases. Otherwise keep pretrained as the default.\n")

    results["behaviour"] = behav
    with open(args.out + ".md", "w", encoding="utf-8") as f:
        f.write(md)
    with open(args.out + ".json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=float)
    print(f"\nSaved: {args.out}.md / .json")


if __name__ == "__main__":
    main()

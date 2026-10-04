#!/usr/bin/env python
"""
Validation-suite audit. Run from the repo root:

    py -3.12 -X utf8 ml\\suite_audit.py                       # part 1 only (no model, a few seconds)
    py -3.12 -X utf8 ml\\suite_audit.py --prev-ref a1e0976    # part 1 + part 2 (needs the model)

PART 1 - is each case well-posed?  (independent of the embedding model and of the scoring formula)
  For every case, compare the INTENDED role's core-skill coverage with every other role's:
    UNIQUE MAX  the intended role strictly covers the candidate best  -> rank 1 is a fair expectation
    TIED MAX    k roles share the top coverage                        -> the case cannot fairly demand rank 1
    BELOW MAX   another role covers the candidate's skills better     -> the case contradicts the role profiles
    NEGATIVE    expected_rank is None: the unwanted role should be absent; its coverage is shown
  Coverage is shown with exact string matching (old behaviour) and with level words stripped (current).

PART 2 - what did the matcher change do?  (--prev-ref REF)
  Runs every case under role_matcher.py as of git REF and under the current file, with the same model,
  and the suite's corrected scoring (correct absence on negative cases = pass). The original cases and the
  cases added later are summarised separately, so the effect on cases written AFTER the change is visible.
"""

import argparse
import importlib.util
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

import role_matcher as rm  # noqa: E402
from validation_test_suite import TEST_CASES  # noqa: E402


def exact_coverage(skills, profile):
    core = {s.lower() for s in profile["core_skills"]}
    have = {s.lower() for s in skills}
    return len(have & core) / len(core) if core else 0.0


def audit_case(case):
    exp = case["expected_role"].lower()
    cands = [r for r in rm.ROLE_PROFILES if exp in r.lower()]
    norm = {r: rm.compute_skill_overlap(case["skills"], p) for r, p in rm.ROLE_PROFILES.items()}
    intended = max(cands, key=lambda r: norm[r]) if cands else None
    best = max(norm.values())
    top = [r for r, v in norm.items() if v == best]
    others = sorted(((v, r) for r, v in norm.items() if r != intended), reverse=True)
    best_other = others[0] if others else (0.0, "-")
    if case["expected_rank"] is None:
        verdict = "NEGATIVE: unwanted role coverage %.0f%% (%s)" % (
            100 * norm[intended], "below the 15% gate, absent" if norm[intended] < 0.15 else "could appear")
    elif intended is None:
        verdict = "NO ROLE NAMED '%s'" % case["expected_role"]
    elif norm[intended] == best and len(top) == 1:
        verdict = "UNIQUE MAX"
    elif norm[intended] == best:
        verdict = "TIED MAX (%d roles: %s)" % (len(top), ", ".join(sorted(top)))
    else:
        verdict = "BELOW MAX (%s covers %.0f%%)" % (best_other[1], 100 * best_other[0])
    exact = exact_coverage(case["skills"], rm.ROLE_PROFILES[intended]) if intended else 0.0
    return intended, exact, norm.get(intended, 0.0), best_other, verdict


def status(recs, case):
    """Same rules as validation_test_suite.py (corrected: correct absence on negative cases = pass)."""
    er, exp = case["expected_rank"], case["expected_role"].lower()
    if not recs:
        return ("PASSED", None) if er is None else ("SKIPPED", None)
    hits = [i + 1 for i, r in enumerate(recs) if exp in r["role"].lower()]
    if not hits:
        return ("PASSED", None) if er is None else ("FAILED", None)
    rank = hits[0]
    if er is None:
        return ("PARTIAL", rank) if rank <= 5 else ("PASSED", rank)
    return ("PASSED", rank) if rank <= er else ("PARTIAL", rank)


def effectiveness(statuses):
    pts = {"PASSED": 1.0, "PARTIAL": 0.5}
    return 100.0 * sum(pts.get(s, 0.0) for s in statuses) / len(statuses) if statuses else float("nan")


def load_prev(ref):
    src = subprocess.run(["git", "show", f"{ref}:role_matcher.py"], cwd=ROOT, capture_output=True, text=True,
                         encoding="utf-8")
    if src.returncode != 0:
        print(f"ERROR: could not read role_matcher.py at git ref '{ref}': {src.stderr.strip()}")
        sys.exit(1)
    path = os.path.join(ROOT, "_role_matcher_prev.py")
    with open(path, "w", encoding="utf-8", newline="") as f:
        f.write(src.stdout)
    try:
        spec = importlib.util.spec_from_file_location("_role_matcher_prev", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    finally:
        os.remove(path)
    return mod


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prev-ref", default=None, help="git ref holding the previous role_matcher.py (enables part 2)")
    ap.add_argument("--n-original", type=int, default=21, help="how many cases were in the suite originally")
    args = ap.parse_args()
    n_orig = args.n_original

    print("=" * 100)
    print("PART 1: is each case well-posed? (core-skill coverage of the intended role vs every other role)")
    print("=" * 100)
    print(f"{'#':>2} {'case':50s} {'exact':>6} {'now':>5}  verdict")
    counts = {}
    for i, c in enumerate(TEST_CASES, 1):
        intended, ex, nm, best_other, verdict = audit_case(c)
        key = verdict.split(" (")[0].split(":")[0]
        counts[key] = counts.get(key, 0) + 1
        mark = "   <-- added later" if i > n_orig else ""
        print(f"{i:2d} {c['name'][:50]:50s} {100 * ex:5.0f}% {100 * nm:4.0f}%  {verdict}{mark}")
    print("\nverdict counts:", counts)
    print("TIED / BELOW cases cannot fairly demand rank 1 from this matcher; a PARTIAL on them reflects the case, "
          "not a defect.")

    if not args.prev_ref:
        print("\n(Part 2 skipped: pass --prev-ref <git ref> to compare against the previous role_matcher.py)")
        return

    print("\n" + "=" * 100)
    print(f"PART 2: previous role_matcher.py ({args.prev_ref}) vs current, same model, corrected scoring")
    print("=" * 100)
    prev = load_prev(args.prev_ref)
    model = rm._get_model()          # one model instance shared by both versions
    prev._model = model
    for mod in (rm, prev):
        mod.generate_job_links = lambda *a, **k: {}

    rows = []
    for c in TEST_CASES:
        o = prev.match_roles(c["skills"], c["roles"], c["tools"])
        n = rm.match_roles(c["skills"], c["roles"], c["tools"])
        rows.append((c, status(o, c), status(n, c)))

    fmt = lambda s: f"{s[0]}" + (f" #{s[1]}" if s[1] else "")
    print(f"{'#':>2} {'case':50s} {'expected':>8s}  {'previous':16s} {'current':16s}")
    for i, (c, o, n) in enumerate(rows, 1):
        exp = "none" if c["expected_rank"] is None else f"<= {c['expected_rank']}"
        flag = "   <-- changed" if o != n else ""
        print(f"{i:2d} {c['name'][:50]:50s} {exp:>8s}  {fmt(o):16s} {fmt(n):16s}{flag}")

    for label, sl in (("original cases", slice(0, n_orig)), ("cases added later", slice(n_orig, None))):
        part = rows[sl]
        if part:
            print(f"\n{label} (n={len(part)}): previous {effectiveness([o[0] for _, o, _ in part]):.1f}%   "
                  f"current {effectiveness([n[0] for _, _, n in part]):.1f}%")
    print(f"all cases (n={len(rows)}): previous {effectiveness([o[0] for _, o, _ in rows]):.1f}%   "
          f"current {effectiveness([n[0] for _, _, n in rows]):.1f}%")


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""
Phase 3, step 2: data loading, text views and train/val/test splitting for
netsol/resume-score-details (read from the local HuggingFace cache).

Importable:
    from data import load_pairs, make_split
    pairs, dropped = load_pairs()
    splits, n_dropped = make_split(pairs, mode="resume", seed=42)   # {"train": [...], "val": [...], "test": [...]}

Runnable (prints counts only, never resume text):
    py -3.12 ml\\data.py --report
    py -3.12 ml\\data.py --report --token-check

Each pair is a dict with:
    id, group, label_group ('match' | 'mismatch' | 'other'),
    rel            graded relevance in [0, 1] = mean(macro, micro) / 10
    resume_raw, jd_raw                       full texts (truncated by MiniLM at 256 tokens)
    resume_compact, jd_compact               short structured views that fit in 256 tokens
    rh, jh                                   normalized hashes (group keys)

Split modes:
    resume     group by resume only. Test resumes are unseen; JDs may be shared with train.
    component  group by connected component of the resume<->JD graph (strictest, may be one blob).
    strict     split resumes and JDs independently; keep only pairs whose two sides land in the
               same split. No resume or JD appears in two splits, at the cost of dropping pairs.
"""

import argparse
import json
import os
import random
import re
import statistics
import sys
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from inspect_netsol import Tee, find_data_dir, get, hr, load_records, norm_hash  # noqa: E402

SCORE_SCALE = 10.0
SPLIT_NAMES = ("train", "val", "test")
FRACS = (0.8, 0.1, 0.1)


# --------------------------------------------------------------------------- #
# Compact text views
# --------------------------------------------------------------------------- #
def _titles(items, limit=6):
    out = []
    if isinstance(items, list):
        for it in items:
            if isinstance(it, dict):
                for k, v in it.items():
                    if isinstance(v, str) and v.strip() and any(
                            t in k.lower() for t in ("title", "role", "position", "designation")):
                        out.append(v.strip())
                        break
            elif isinstance(it, str) and it.strip():
                out.append(it.strip())
    return out[:limit]


def resume_compact(details):
    if not isinstance(details, dict):
        return ""
    skills = details.get("skills")
    if isinstance(skills, dict):
        flat = []
        for v in skills.values():
            flat.extend(v if isinstance(v, list) else [v])
        skills = flat
    skills = [str(s).strip() for s in skills if str(s).strip()][:25] if isinstance(skills, list) else []
    summary = details.get("executive_summary")
    summary = re.sub(r"\s+", " ", summary).strip()[:400] if isinstance(summary, str) else ""
    titles = _titles(details.get("employment_history"))
    parts = []
    if titles:
        parts.append("Experience: " + "; ".join(titles) + ".")
    if skills:
        parts.append("Skills: " + ", ".join(skills) + ".")
    if summary:
        parts.append("Summary: " + summary)
    return " ".join(parts)


def _jd_title(jd):
    head = (jd or "")[:800]
    m = re.search(r"job\s*title\s*:?\**\s*(.+)", head, flags=re.I)
    if m:
        return re.sub(r"[*#]+", "", m.group(1)).strip()[:120]
    for line in head.splitlines():
        line = re.sub(r"[*#]+", "", line).strip()
        if line:
            return line[:120]
    return ""


def jd_compact(jd, macro_dict, micro_dict, min_req):
    title = _jd_title(jd)
    crit = []
    for dct in (macro_dict, micro_dict):
        if isinstance(dct, dict):
            crit.extend(str(k) for k in dct.keys())
    reqs = [r.strip()[:140] for r in (min_req or []) if isinstance(r, str) and r.strip()][:4] \
        if isinstance(min_req, list) else []
    parts = []
    if title:
        parts.append(f"Role: {title}.")
    if crit:
        parts.append("Key criteria: " + ", ".join(crit) + ".")
    if reqs:
        parts.append("Requirements: " + "; ".join(reqs) + ".")
    return " ".join(parts)


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def load_pairs(folder=None, repo=None, verbose=False):
    if folder is None:
        _, folder = find_data_dir(None, repo, allow_download=False)
    recs, _ = load_records(folder)
    pairs, dropped = [], Counter()
    for name, d in recs:
        prefix = re.sub(r"[_-]?\d+\.json$", "", name)
        resume = get(d, "input", "resume")
        jd = get(d, "input", "job_description")
        agg = get(d, "output", "scores", "aggregated_scores", default={})
        macro = agg.get("macro_scores") if isinstance(agg, dict) else None
        micro = agg.get("micro_scores") if isinstance(agg, dict) else None
        if not isinstance(macro, (int, float)) or not isinstance(micro, (int, float)):
            dropped["no_scores"] += 1
            continue
        if get(d, "output", "valid_resume_and_jd") is not True:
            dropped["valid_flag_not_true"] += 1
            continue
        if not isinstance(resume, str) or not resume.strip() or not isinstance(jd, str) or not jd.strip():
            dropped["missing_text"] += 1
            continue
        rel = max(0.0, min(1.0, (macro + micro) / 2.0 / SCORE_SCALE))
        pairs.append({
            "id": name,
            "group": prefix,
            "label_group": prefix if prefix in ("match", "mismatch") else "other",
            "rel": round(rel, 4),
            "macro": float(macro),
            "micro": float(micro),
            "resume_raw": resume,
            "jd_raw": jd,
            "resume_compact": resume_compact(d.get("details")),
            "jd_compact": jd_compact(jd, get(d, "input", "macro_dict"), get(d, "input", "micro_dict"),
                                     get(d, "input", "minimum_requirements")),
            "rh": norm_hash(resume),
            "jh": norm_hash(jd),
        })
    if verbose:
        print(f"  usable pairs: {len(pairs)}   dropped: {dict(dropped)}")
    return pairs, dict(dropped)


# --------------------------------------------------------------------------- #
# Splitting
# --------------------------------------------------------------------------- #
def _components(pairs):
    parent = {}

    def find(x):
        while parent.setdefault(x, x) != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for p in pairs:
        a, b = find("R:" + p["rh"]), find("J:" + p["jh"])
        if a != b:
            parent[a] = b
    return {i: find("R:" + p["rh"]) for i, p in enumerate(pairs)}


def _greedy_group_split(pairs, group_of, seed):
    groups = defaultdict(list)
    for i, p in enumerate(pairs):
        groups[group_of(i, p)].append(i)
    keys = sorted(groups)
    random.Random(seed).shuffle(keys)
    targets = [f * len(pairs) for f in FRACS]
    filled = [0, 0, 0]
    out = [[], [], []]
    for k in keys:
        s = max(range(3), key=lambda j: targets[j] - filled[j])
        out[s].extend(groups[k])
        filled[s] += len(groups[k])
    return out, 0


def _strict_split(pairs, seed):
    rng = random.Random(seed)

    def assign(keys):
        keys = sorted(set(keys))
        rng.shuffle(keys)
        n = len(keys)
        a, b = int(FRACS[0] * n), int((FRACS[0] + FRACS[1]) * n)
        return {k: (0 if i < a else 1 if i < b else 2) for i, k in enumerate(keys)}

    rs = assign(p["rh"] for p in pairs)
    js = assign(p["jh"] for p in pairs)
    out, dropped = [[], [], []], 0
    for i, p in enumerate(pairs):
        a, b = rs[p["rh"]], js[p["jh"]]
        if a == b:
            out[a].append(i)
        else:
            dropped += 1
    return out, dropped


def make_split(pairs, mode="resume", seed=42):
    """Return ({'train': [pair,...], 'val': [...], 'test': [...]}, n_dropped)."""
    if mode == "resume":
        idx, dropped = _greedy_group_split(pairs, lambda i, p: p["rh"], seed)
    elif mode == "component":
        comp = _components(pairs)
        idx, dropped = _greedy_group_split(pairs, lambda i, p: comp[i], seed)
    elif mode == "strict":
        idx, dropped = _strict_split(pairs, seed)
    else:
        raise ValueError(mode)
    return {name: [pairs[i] for i in ids] for name, ids in zip(SPLIT_NAMES, idx)}, dropped


def kfold_splits(pairs, k=5, seed=42):
    """Grouped (by resume) K-fold, stratified by pairs-per-resume so every fold gets resumes with
    several scored JDs. Returns a list of (train_idx, test_idx) index lists into `pairs`.
    Every resume lands in exactly one test fold -> out-of-fold predictions for all resumes."""
    by_resume = defaultdict(list)
    for i, p in enumerate(pairs):
        by_resume[p["rh"]].append(i)
    rng = random.Random(seed)
    keys = sorted(by_resume)
    rng.shuffle(keys)
    keys.sort(key=lambda r: -len(by_resume[r]))  # stable: ties keep the random order
    folds = [[] for _ in range(k)]
    for n, r in enumerate(keys):  # snake-deal so fold sizes and pair-counts stay balanced
        lap, pos = divmod(n, k)
        folds[pos if lap % 2 == 0 else k - 1 - pos].append(r)
    out = []
    for f in range(k):
        test_resumes = set(folds[f])
        test_idx = [i for i, p in enumerate(pairs) if p["rh"] in test_resumes]
        train_idx = [i for i, p in enumerate(pairs) if p["rh"] not in test_resumes]
        out.append((train_idx, test_idx))
    return out


# --------------------------------------------------------------------------- #
# Report
# --------------------------------------------------------------------------- #
def report(args):
    sys.stdout = Tee(args.out + ".txt")
    out = {}
    hr("LOAD")
    pairs, dropped = load_pairs(verbose=True)
    out["usable"] = len(pairs)
    out["dropped"] = dropped
    if not pairs:
        return

    rels = [p["rel"] for p in pairs]
    print(f"  relevance (mean(macro,micro)/10): mean={statistics.mean(rels):.3f} "
          f"median={statistics.median(rels):.3f}")
    print(f"  label groups: {dict(Counter(p['label_group'] for p in pairs))}")
    nr, nj = len({p['rh'] for p in pairs}), len({p['jh'] for p in pairs})
    print(f"  unique resumes={nr}  unique JDs={nj}")

    hr("COMPACT VIEW COVERAGE (how often the structured fields yielded text)")
    cov = {
        "resume_compact non-empty": sum(bool(p["resume_compact"]) for p in pairs) / len(pairs),
        "resume has 'Skills:'": sum("Skills:" in p["resume_compact"] for p in pairs) / len(pairs),
        "resume has 'Experience:'": sum("Experience:" in p["resume_compact"] for p in pairs) / len(pairs),
        "resume has 'Summary:'": sum("Summary:" in p["resume_compact"] for p in pairs) / len(pairs),
        "jd_compact non-empty": sum(bool(p["jd_compact"]) for p in pairs) / len(pairs),
        "jd has 'Role:'": sum("Role:" in p["jd_compact"] for p in pairs) / len(pairs),
        "jd has 'Key criteria:'": sum("Key criteria:" in p["jd_compact"] for p in pairs) / len(pairs),
        "jd has 'Requirements:'": sum("Requirements:" in p["jd_compact"] for p in pairs) / len(pairs),
    }
    for k, v in cov.items():
        print(f"  {k:28s} {v:6.1%}")
    out["coverage"] = cov
    if cov["resume has 'Experience:'"] < 0.5:
        print("  >> 'Experience:' is rarely filled: employment_history item keys probably don't "
              "contain title/role/position/designation. Run the key probe below.")
    # structure probe: KEY NAMES only, never values
    ex = next((p for p in pairs), None)
    if ex:
        recs, _ = load_records(find_data_dir(None, None, False)[1])
        for name, d in recs:
            eh = get(d, "details", "employment_history")
            if isinstance(eh, list) and eh and isinstance(eh[0], dict):
                print(f"  employment_history[0] keys (names only): {sorted(eh[0].keys())}")
                break
        for name, d in recs:
            pr = get(d, "details", "projects")
            if isinstance(pr, list) and pr and isinstance(pr[0], dict):
                print(f"  projects[0] keys (names only):           {sorted(pr[0].keys())}")
                break

    if args.token_check:
        hr("TOKEN LENGTHS: raw vs compact (all-MiniLM-L6-v2, max_seq_length=256)")
        try:
            from sentence_transformers import SentenceTransformer
            m = SentenceTransformer("all-MiniLM-L6-v2")
            tok, ml = m.tokenizer, m.max_seq_length
            seen = {}
            for field in ("resume_raw", "resume_compact", "jd_raw", "jd_compact"):
                texts = list({p[field] for p in pairs})[:500]
                lens = [len(tok.encode(t, truncation=False)) for t in texts]
                over = sum(l > ml for l in lens) / max(len(lens), 1)
                print(f"  {field:15s} median={statistics.median(lens):6.0f} max={max(lens):6d} "
                      f"over_{ml}={over:6.1%}")
                seen[field] = {"median": statistics.median(lens), "over": round(over, 3)}
            out["tokens"] = seen
        except Exception as e:
            print(f"  skipped: {type(e).__name__}: {e}")

    hr("CONNECTED COMPONENTS of the resume<->JD graph")
    comp = _components(pairs)
    sizes = Counter(comp.values())
    ordered = sorted(sizes.values(), reverse=True)
    print(f"  components: {len(sizes)}   largest sizes (pairs): {ordered[:8]}")
    print(f"  largest component holds {ordered[0] / len(pairs):.1%} of all pairs")
    if ordered[0] / len(pairs) > 0.5:
        print("  >> One giant component: 'component' split cannot give a real test set. "
              "Use 'resume' (honest about JD sharing) and/or 'strict' (drops pairs).")
    out["components"] = {"n": len(sizes), "top": ordered[:8]}

    hr("SPLIT STRATEGIES (seed=%d)" % args.seed)
    out["splits"] = {}
    for mode in ("resume", "component", "strict"):
        splits, dropped_pairs = make_split(pairs, mode, args.seed)
        print(f"\n  mode = {mode}   (pairs dropped to avoid leakage: {dropped_pairs})")
        rs = {n: {p["rh"] for p in s} for n, s in splits.items()}
        js = {n: {p["jh"] for p in s} for n, s in splits.items()}
        info = {}
        for n, s in splits.items():
            jd_top = Counter(p["jh"] for p in s).most_common(1)
            top_share = jd_top[0][1] / len(s) if s else 0
            mean_rel = statistics.mean(p["rel"] for p in s) if s else float("nan")
            print(f"    {n:5s} pairs={len(s):4d}  resumes={len(rs[n]):3d}  JDs={len(js[n]):3d}  "
                  f"top-JD share={top_share:5.1%}  mean rel={mean_rel:.3f}")
            info[n] = {"pairs": len(s), "resumes": len(rs[n]), "jds": len(js[n])}
        print(f"    resume overlap train&test={len(rs['train'] & rs['test'])}  "
              f"JD overlap train&test={len(js['train'] & js['test'])}")
        info["resume_overlap_train_test"] = len(rs["train"] & rs["test"])
        info["jd_overlap_train_test"] = len(js["train"] & js["test"])
        info["dropped"] = dropped_pairs
        out["splits"][mode] = info

    hr("HOW TO READ THIS")
    print("""  - Prefer a mode where resume overlap is 0. If JD overlap stays >0 ('resume' mode), say so in the
    README: the test measures generalisation to unseen resumes, not unseen JDs.
  - A split whose val/test has only a handful of resumes or JDs gives noisy metrics; check the counts.
  - Watch 'top-JD share': if one JD dominates the test split, retrieval metrics collapse to that JD.
  - If 'strict' keeps enough pairs and all three splits have a reasonable JD count, it is the most
    defensible headline number; otherwise report 'resume' mode with the JD caveat.
""")
    with open(args.out + ".json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"Saved: {args.out}.txt and {args.out}.json")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--token-check", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=os.path.join("ml", "results", "data_report"))
    a = ap.parse_args()
    if a.report:
        report(a)
    else:
        ap.print_help()

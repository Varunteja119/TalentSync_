#!/usr/bin/env python
"""
Phase 3, step 3 (revised): baseline on ALL usable pairs, with paired comparisons.

Why this replaces baseline.py: the old fixed split left only ~18 evaluable test resumes.
A pretrained model needs no held-out set, so it is scored on every pair. The JD prior (which
is fitted on scores) uses grouped 5-fold cross-validation so each resume is scored by a prior
that never saw it. The same K-fold will be used for the fine-tuned model.

    py -3.12 ml\\baseline_cv.py

Outputs ml/results/baseline_cv_results.md / .json (metrics only, no resume text).
"""

import argparse
import json
import os
import sys
from collections import Counter, defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data import kfold_splits, load_pairs  # noqa: E402
from eval import evaluate, format_cell, paired_bootstrap, random_baseline, summarize  # noqa: E402
from inspect_netsol import Tee, hr  # noqa: E402

VIEWS = {"raw": ("resume_raw", "jd_raw"), "compact": ("resume_compact", "jd_compact")}
BY_RESUME_COLS = ["pairwise", "spearman", "top1", "ndcg"]
BY_JD_COLS = ["pairwise", "spearman", "top1", "ndcg", "ndcg@10"]


def cosine_preds(model, pairs, rf, jf):
    uniq = sorted({p[rf] for p in pairs} | {p[jf] for p in pairs})
    vecs = model.encode(uniq, batch_size=32, normalize_embeddings=True, show_progress_bar=False)
    emb = dict(zip(uniq, vecs))
    return [float(np.dot(emb[p[rf]], emb[p[jf]])) for p in pairs]


def jd_prior_oof(pairs, folds):
    preds = [0.0] * len(pairs)
    for train_idx, test_idx in folds:
        sums, cnt = defaultdict(float), defaultdict(int)
        for i in train_idx:
            sums[pairs[i]["jh"]] += pairs[i]["rel"]
            cnt[pairs[i]["jh"]] += 1
        glob = sum(sums.values()) / max(sum(cnt.values()), 1)
        for i in test_idx:
            j = pairs[i]["jh"]
            preds[i] = sums[j] / cnt[j] if cnt[j] else glob
    return preds


def table(rows, cols):
    head = "| model | " + " | ".join(cols) + " | n |\n|---|" + "---|" * (len(cols) + 1) + "\n"
    body = ""
    for name, summ in rows:
        n = max((s["n"] for s in summ.values()), default=0)
        body += f"| {name} | " + " | ".join(format_cell(summ[c]) if c in summ else "n/a" for c in cols) \
                + f" | {n} |\n"
    return head + body


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="all-MiniLM-L6-v2")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--out", default=os.path.join("ml", "results", "baseline_cv_results"))
    args = ap.parse_args()

    sys.stdout = Tee(args.out + ".txt")
    from sentence_transformers import SentenceTransformer

    pairs, _ = load_pairs()
    folds = kfold_splits(pairs, args.folds, args.seed)

    hr("DATA")
    per_resume = Counter(p["rh"] for p in pairs)
    per_jd = Counter(p["jh"] for p in pairs)
    print(f"  pairs={len(pairs)}  resumes={len(per_resume)}  JDs={len(per_jd)}")
    print(f"  resumes with >=2 scored JDs: {sum(c >= 2 for c in per_resume.values())}")
    print(f"  JDs with >=2 scored resumes: {sum(c >= 2 for c in per_jd.values())}")
    print(f"  fold test sizes (pairs): {[len(t) for _, t in folds]}   "
          f"(resumes): {[len({pairs[i]['rh'] for i in t}) for _, t in folds]}")

    model = SentenceTransformer(args.model)
    model_preds = {f"MiniLM pretrained, {v}": cosine_preds(model, pairs, rf, jf)
                   for v, (rf, jf) in VIEWS.items()}
    prior = jd_prior_oof(pairs, folds)

    results, md = {}, f"# Baseline (all pairs; JD prior via {args.folds}-fold CV) - {args.model}\n"
    md += "\nRelevance = mean(macro, micro)/10 from GPT-4o. 95% bootstrap CIs over resumes (or JDs).\n"
    per_group_store = {}
    for by, cols in (("resume", BY_RESUME_COLS), ("jd", BY_JD_COLS)):
        hr(f"RANK {'JDs per resume' if by == 'resume' else 'resumes per JD'}")
        per = {name: evaluate(pairs, p, by) for name, p in model_preds.items()}
        per["random (chance)"] = random_baseline(pairs, by)
        if by == "resume":
            per["JD prior (no resume info)"] = evaluate(pairs, prior, by)
        per_group_store[by] = per
        rows = [(name, summarize(d, args.n_boot)) for name, d in per.items()]
        results[by] = {name: s for name, s in rows}
        t = table(rows, cols)
        print("  " + t.replace("\n", "\n  "))
        md += f"\n## Rank {'JDs per resume' if by == 'resume' else 'resumes per JD'}\n\n{t}"

    hr("PAIRED COMPARISONS (mean difference, 95% bootstrap CI over resumes; * = CI excludes 0)")
    md += "\n## Paired differences (by resume)\n\n| comparison | metric | diff [95% CI] | n |\n|---|---|---|---|\n"
    comps = [("MiniLM pretrained, compact", "MiniLM pretrained, raw"),
             ("MiniLM pretrained, compact", "JD prior (no resume info)"),
             ("MiniLM pretrained, raw", "JD prior (no resume info)"),
             ("MiniLM pretrained, compact", "random (chance)")]
    results["paired"] = []
    for a, b in comps:
        for metric in ("pairwise", "spearman", "top1"):
            d = paired_bootstrap(per_group_store["resume"][a], per_group_store["resume"][b], metric)
            if not d:
                continue
            star = "*" if d["significant"] else " "
            line = f"{a} - {b} | {metric}"
            print(f"  {star} {a:28s} vs {b:27s} {metric:9s} {d['mean_diff']:+.3f} "
                  f"[{d['lo']:+.3f}, {d['hi']:+.3f}]  n={d['n']}")
            md += f"| {a} - {b} | {metric} | {d['mean_diff']:+.3f} [{d['lo']:+.3f}, {d['hi']:+.3f}]{star} | {d['n']} |\n"
            results["paired"].append({"a": a, "b": b, "metric": metric, **d})

    md += ("\nNotes: NDCG has a high chance floor with ~4 items per pool; lead with pairwise/Spearman. "
           "JD prior ignores the resume; a model must beat it to claim it reads resumes.\n")
    with open(args.out + ".md", "w", encoding="utf-8") as f:
        f.write(md)
    with open(args.out + ".json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=float)
    print(f"\nSaved: {args.out}.md / .json")


if __name__ == "__main__":
    main()

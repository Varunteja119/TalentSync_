#!/usr/bin/env python
"""
Decisive ablation: does fine-tuning do more than "pretrained cosine + each JD's typical score"?

    hybrid = z(JD prior from training folds) + alpha * z(pretrained cosine, JD-centered)

No training and no tuned weights: alpha = 1 is the pre-specified primary; 0.5 and 2 are shown
as a sensitivity check, not for picking the best one. Uses the same folds as train_cv.py for a
given --seed, so the JD-prior and pretrained rows should reproduce the earlier baseline numbers
(a built-in sanity check).

    py -3.12 ml\\hybrid_baseline.py --seed 42
    py -3.12 ml\\hybrid_baseline.py --seed 43 --oof ml\\results\\train_cv_cosent_compact_seed43_oof.json

With --oof (written by train_cv.py), the fine-tuned model is compared PAIRED against the hybrid.
"""

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from baseline_cv import jd_prior_oof, table  # noqa: E402
from data import kfold_splits, load_pairs  # noqa: E402
from eval import evaluate, paired_bootstrap, random_baseline, summarize  # noqa: E402
from inspect_netsol import Tee, hr  # noqa: E402
from train_cv import VIEWS, predict_fold  # noqa: E402

COLS = ["pairwise", "spearman", "top1"]
ALPHAS = (0.5, 1.0, 2.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="all-MiniLM-L6-v2")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--oof", default=None, help="OOF predictions JSON from train_cv.py")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    out = args.out or os.path.join("ml", "results", f"hybrid_seed{args.seed}")
    sys.stdout = Tee(out + ".txt")
    from sentence_transformers import SentenceTransformer

    pairs, _ = load_pairs()
    folds = kfold_splits(pairs, args.folds, args.seed)
    rf, jf = VIEWS["compact"]
    model = SentenceTransformer(args.model)

    cos_raw, cos_cen = {}, {}
    for _, test_idx in folds:
        r, c = predict_fold(model, pairs, test_idx, rf, jf)
        cos_raw.update(r)
        cos_cen.update(c)
    n = len(pairs)
    raw = [cos_raw[i] for i in range(n)]
    cen = np.array([cos_cen[i] for i in range(n)])
    prior = np.array(jd_prior_oof(pairs, folds))
    zp = (prior - prior.mean()) / prior.std()
    zc = cen / cen.std()

    per = {"pretrained cosine (raw)": evaluate(pairs, raw, "resume"),
           "JD prior only": evaluate(pairs, list(prior), "resume")}
    for a in ALPHAS:
        per[f"hybrid (prior + {a} x centered cosine)"] = evaluate(pairs, list(zp + a * zc), "resume")
    per["random (chance)"] = random_baseline(pairs, "resume")

    ft_name = None
    if args.oof:
        with open(args.oof, "r", encoding="utf-8") as f:
            oof = json.load(f)
        by_id = {pid: k for k, pid in enumerate(oof["ids"])}
        if set(by_id) != {p["id"] for p in pairs}:
            print("  [!] OOF ids do not match the current pair set; skipping fine-tuned comparison.")
        else:
            final = str(max(int(k) for k in oof["raw"]))
            vals = oof["raw"][final]
            ft_name = f"fine-tuned (epoch {final}, seed {oof['seed']})"
            per[ft_name] = evaluate(pairs, [vals[by_id[p["id"]]] for p in pairs], "resume")

    hr(f"RANK JDs PER RESUME  (seed {args.seed}; sanity: first two rows should match baseline_cv)")
    rows = [(k, summarize(v, args.n_boot)) for k, v in per.items()]
    t = table(rows, COLS)
    print("  " + t.replace("\n", "\n  "))
    md = f"# Hybrid ablation (seed {args.seed})\n\n{t}"

    results = {k: v for k, v in rows}
    primary = "hybrid (prior + 1.0 x centered cosine)"
    if ft_name:
        hr(f"PAIRED: {ft_name}  minus  {primary}   (* = CI excludes 0)")
        md += f"\n## Paired: fine-tuned minus hybrid\n\n| metric | diff [95% CI] | n |\n|---|---|---|\n"
        results["paired_ft_minus_hybrid"] = []
        for m in COLS:
            d = paired_bootstrap(per[ft_name], per[primary], m)
            if d:
                star = "*" if d["significant"] else " "
                print(f"  {star} {m:9s} {d['mean_diff']:+.3f} [{d['lo']:+.3f}, {d['hi']:+.3f}]  n={d['n']}")
                md += f"| {m} | {d['mean_diff']:+.3f} [{d['lo']:+.3f}, {d['hi']:+.3f}]{star} | {d['n']} |\n"
                results["paired_ft_minus_hybrid"].append({"metric": m, **d})
        print("\n  CI includes 0  -> fine-tuning is no better than pretrained cosine + per-JD offset.\n"
              "  CI above 0     -> fine-tuning learned something beyond a JD offset.")
    else:
        print("\n  (No --oof given: compare the hybrid rows with the fine-tuned numbers from "
              "train_cv unpaired, i.e. by overlapping CIs only. Pass --oof for a paired test.)")

    with open(out + ".md", "w", encoding="utf-8") as f:
        f.write(md)
    with open(out + ".json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=float)
    print(f"\nSaved: {out}.md / .json")


if __name__ == "__main__":
    main()

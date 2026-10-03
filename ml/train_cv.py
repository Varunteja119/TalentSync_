#!/usr/bin/env python
"""
Phase 3, step 4: fine-tune MiniLM with grouped 5-fold CV and evaluate OUT-OF-FOLD.

Every resume is scored by a model that never saw it. Epoch 0 (pretrained) is evaluated in the
same loop and code path, so the comparison is apples to apples.

    py -3.12 ml\\train_cv.py --smoke                      # ~1 minute: checks the whole pipeline runs
    py -3.12 ml\\train_cv.py                              # full run (compact view, CoSENT loss)
    py -3.12 ml\\train_cv.py --center-target              # ablation: train on JD-centered targets
    py -3.12 ml\\train_cv.py --loss cosine --view raw     # other ablations

Design choices (fixed in advance, not tuned on the test folds)
  * loss      CoSENT on the graded GPT-4o relevance (rank-based; fits graded labels). In-batch
              negatives (MultipleNegativesRankingLoss) are NOT used as the main loss: only 26 distinct
              JDs exist, so a batch contains the same JD paired with other resumes, and those would be
              wrongly treated as negatives.
  * primary   the FINAL epoch is the pre-specified headline. Earlier epochs are shown to reveal
              overfitting, not for picking the best one (that would be selection on the test folds).
  * metrics   standard by-resume / by-JD, plus JD-centered by-resume (resume-specific fit only).

Outputs: ml/results/train_cv_<loss>[_centered]_<view>.md / .json (metrics only).
"""

import argparse
import json
import math
import os
import random
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from baseline_cv import jd_prior_oof, table  # noqa: E402
from data import kfold_splits, load_pairs  # noqa: E402
from eval import (PAIRWISE_EPS, center_rel_by_jd, evaluate, evaluate_centered,  # noqa: E402
                  paired_bootstrap, random_baseline, summarize)
from inspect_netsol import Tee, hr  # noqa: E402

VIEWS = {"raw": ("resume_raw", "jd_raw"), "compact": ("resume_compact", "jd_compact")}


# --------------------------------------------------------------------------- #
# Loss + training (torch imported lazily so the rest of the file stays importable)
# --------------------------------------------------------------------------- #
def cosent_loss(cos, y, scale=20.0):
    """CoSENT: log(1 + sum_{y_i > y_j} exp(scale * (cos_j - cos_i))). Pushes items with higher
    relevance above items with lower relevance; ignores near-ties (|dy| <= PAIRWISE_EPS)."""
    import torch
    diff = (cos[None, :] - cos[:, None]) * scale
    mask = ((y[:, None] - y[None, :]) > PAIRWISE_EPS).to(cos.dtype)
    diff = diff - (1.0 - mask) * 1e12
    diff = torch.cat([torch.zeros(1, device=cos.device, dtype=cos.dtype), diff.reshape(-1)])
    return torch.logsumexp(diff, dim=0)


def _to_device(features, device):
    """Newer sentence-transformers versions return some non-tensor entries (e.g. strings) in the
    tokenized features. Move tensors to the device and pass everything else through unchanged."""
    return {k: (v.to(device) if hasattr(v, "to") else v) for k, v in features.items()}


def train_epoch(model, pairs, rf, jf, targets, args, optimizer, scheduler, rng, device):
    import torch
    import torch.nn.functional as F
    model.train()
    order = list(range(len(pairs)))
    rng.shuffle(order)
    losses = []
    for s in range(0, len(order), args.batch_size):
        ids = order[s:s + args.batch_size]
        if len(ids) < 2:
            continue
        fa = _to_device(model.tokenize([pairs[i][rf] for i in ids]), device)
        fb = _to_device(model.tokenize([pairs[i][jf] for i in ids]), device)
        cos = F.cosine_similarity(model(fa)["sentence_embedding"], model(fb)["sentence_embedding"])
        y = torch.tensor([targets[i] for i in ids], dtype=cos.dtype, device=device)
        loss = cosent_loss(cos, y) if args.loss == "cosent" else F.mse_loss(cos, y)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        losses.append(float(loss))
    return float(np.mean(losses)) if losses else float("nan")


def predict_fold(model, pairs, test_idx, rf, jf):
    """Cosine for each test pair, plus a JD-centered version. The JD offset is the mean cosine of
    that JD over ALL resumes in the test fold (every resume x every JD): label-free and out-of-fold."""
    res_texts = sorted({pairs[i][rf] for i in test_idx})
    jd_texts = sorted({p[jf] for p in pairs})
    er = np.asarray(model.encode(res_texts, batch_size=32, normalize_embeddings=True,
                                 show_progress_bar=False))
    ej = np.asarray(model.encode(jd_texts, batch_size=32, normalize_embeddings=True,
                                 show_progress_bar=False))
    S = er @ ej.T
    offset = S.mean(axis=0)
    ri = {t: i for i, t in enumerate(res_texts)}
    ji = {t: i for i, t in enumerate(jd_texts)}
    raw, cen = {}, {}
    for i in test_idx:
        r, j = ri[pairs[i][rf]], ji[pairs[i][jf]]
        raw[i] = float(S[r, j])
        cen[i] = float(S[r, j] - offset[j])
    return raw, cen


def train_targets(train_pairs, center):
    t = [p["rel"] for p in train_pairs]
    if not center:
        return t
    sums, cnt = {}, {}
    for p in train_pairs:
        sums[p["jh"]] = sums.get(p["jh"], 0.0) + p["rel"]
        cnt[p["jh"]] = cnt.get(p["jh"], 0) + 1
    return [p["rel"] - sums[p["jh"]] / cnt[p["jh"]] for p in train_pairs]


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="all-MiniLM-L6-v2")
    ap.add_argument("--view", choices=list(VIEWS), default="compact")
    ap.add_argument("--loss", choices=["cosent", "cosine"], default="cosent")
    ap.add_argument("--center-target", action="store_true",
                    help="train on rel minus the JD's training mean (discourages memorising JD strictness)")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--smoke", action="store_true", help="1 fold, 64 training pairs, 1 epoch")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    if args.smoke:
        args.epochs, args.n_boot = 1, 50
    tag = f"{args.loss}{'_centered' if args.center_target else ''}_{args.view}" + ("_smoke" if args.smoke else "")
    out = args.out or os.path.join("ml", "results", f"train_cv_{tag}")

    sys.stdout = Tee(out + ".txt")
    import torch
    from sentence_transformers import SentenceTransformer
    from transformers import get_linear_schedule_with_warmup

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    pairs, _ = load_pairs()
    folds = kfold_splits(pairs, args.folds, args.seed)
    run_folds = folds[:1] if args.smoke else folds
    rf, jf = VIEWS[args.view]
    E = args.epochs
    hr(f"CONFIG  {vars(args)}  device={device}")
    print(f"  pairs={len(pairs)}  folds run={len(run_folds)}")

    raw = {e: {} for e in range(E + 1)}
    cen = {e: {} for e in range(E + 1)}
    for k, (train_idx, test_idx) in enumerate(run_folds):
        t0 = time.time()
        if args.smoke:
            train_idx = train_idx[:64]
        train_pairs = [pairs[i] for i in train_idx]
        targets = train_targets(train_pairs, args.center_target)
        model = SentenceTransformer(args.model, device=device)
        r, c = predict_fold(model, pairs, test_idx, rf, jf)
        raw[0].update(r)
        cen[0].update(c)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
        steps = max(1, E * math.ceil(len(train_pairs) / args.batch_size))
        scheduler = get_linear_schedule_with_warmup(optimizer, int(0.1 * steps), steps)
        rng = random.Random(args.seed * 100 + k)
        for e in range(1, E + 1):
            loss = train_epoch(model, train_pairs, rf, jf, targets, args, optimizer, scheduler, rng, device)
            r, c = predict_fold(model, pairs, test_idx, rf, jf)
            raw[e].update(r)
            cen[e].update(c)
            print(f"  fold {k + 1}/{len(run_folds)} epoch {e}/{E}  train loss={loss:.4f}  "
                  f"({time.time() - t0:.0f}s elapsed)")
        del model, optimizer, scheduler
        if device == "cuda":
            torch.cuda.empty_cache()

    # ------------------------------------------------------------------ evaluate
    eval_idx = sorted(raw[0].keys())
    pe = [pairs[i] for i in eval_idx]
    prior_all = jd_prior_oof(pairs, folds)
    prior = [prior_all[i] for i in eval_idx]

    names = {0: "pretrained (epoch 0)"}
    names.update({e: f"fine-tuned, epoch {e}" for e in range(1, E + 1)})
    per = {}
    for e in range(E + 1):
        pr = [raw[e][i] for i in eval_idx]
        pc = [cen[e][i] for i in eval_idx]
        per[names[e]] = {"res": evaluate(pe, pr, "resume"), "jd": evaluate(pe, pr, "jd"),
                         "cen": evaluate_centered(pe, pc, "resume")}
    rc = random_baseline(center_rel_by_jd(pe), "resume")
    rc.pop("ndcg", None)
    per["random (chance)"] = {"res": random_baseline(pe, "resume"), "jd": random_baseline(pe, "jd"), "cen": rc}
    per["JD prior (no resume info)"] = {"res": evaluate(pe, prior, "resume")}

    sections = [("res", "RANK JDs PER RESUME (the app's task)", ["pairwise", "spearman", "top1"]),
                ("jd", "RANK RESUMES PER JD", ["pairwise", "spearman", "ndcg@10"]),
                ("cen", "JD-CENTERED, JDs per resume (resume-specific fit only)", ["pairwise", "spearman", "top1"])]
    md = f"# Fine-tuning CV results ({tag}, {args.folds}-fold by resume, seed {args.seed})\n"
    md += (f"\nConfig: {args.model}, view={args.view}, loss={args.loss}, epochs={E}, lr={args.lr}, "
           f"batch={args.batch_size}, center_target={args.center_target}. Primary row = final epoch "
           f"(pre-specified); other epochs shown for transparency only.\n")
    results = {}
    for key, title, cols in sections:
        hr(title)
        rows = [(n, summarize(d[key], args.n_boot)) for n, d in per.items() if key in d]
        t = table(rows, cols)
        print("  " + t.replace("\n", "\n  "))
        md += f"\n## {title}\n\n{t}"
        results[key] = {n: s for n, s in rows}

    hr(f"PAIRED COMPARISONS for the pre-specified final model: '{names[E]}'  (* = CI excludes 0)")
    md += f"\n## Paired differences for '{names[E]}'\n\n| vs | direction | metric | diff [95% CI] | n |\n|---|---|---|---|---|\n"
    results["paired"] = []
    comps = [("res", "pretrained (epoch 0)"), ("res", "JD prior (no resume info)"),
             ("cen", "pretrained (epoch 0)"), ("jd", "pretrained (epoch 0)")]
    for key, other in comps:
        for metric in ("pairwise", "spearman", "top1"):
            if other not in per or key not in per[other]:
                continue
            d = paired_bootstrap(per[names[E]][key], per[other][key], metric)
            if not d:
                continue
            star = "*" if d["significant"] else " "
            print(f"  {star} {key:3s} vs {other:27s} {metric:9s} {d['mean_diff']:+.3f} "
                  f"[{d['lo']:+.3f}, {d['hi']:+.3f}]  n={d['n']}")
            md += f"| {other} | {key} | {metric} | {d['mean_diff']:+.3f} [{d['lo']:+.3f}, {d['hi']:+.3f}]{star} | {d['n']} |\n"
            results["paired"].append({"vs": other, "direction": key, "metric": metric, **d})

    md += ("\nReading guide: beating 'pretrained' with a CI excluding 0 = fine-tuning helps. Beating the "
           "JD prior = more than memorising which JDs score high. The JD-centered table is the cleanest "
           "test of resume-specific fit (the JD prior is a null model there by construction).\n")
    # Out-of-fold predictions (pair ids + floats only, no text) so later analyses can pair models.
    oof = {"ids": [pairs[i]["id"] for i in eval_idx], "seed": args.seed, "config": vars(args),
           "raw": {str(e): [raw[e][i] for i in eval_idx] for e in range(E + 1)},
           "cen": {str(e): [cen[e][i] for i in eval_idx] for e in range(E + 1)}}
    with open(out + "_oof.json", "w", encoding="utf-8") as f:
        json.dump(oof, f)
    with open(out + ".md", "w", encoding="utf-8") as f:
        f.write(md)
    with open(out + ".json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=float)
    print(f"\nSaved: {out}.md / .json")


if __name__ == "__main__":
    main()

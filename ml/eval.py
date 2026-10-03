#!/usr/bin/env python
"""
Shared evaluation harness. Any model (pretrained, fine-tuned, trivial) only has to produce
one predicted score per pair; this module turns that into ranking metrics.

    per_group = evaluate(pairs, preds, by="resume")   # {metric: {group_key: value}}
    summary   = summarize(per_group, n_boot=1000)     # {metric: {mean, lo, hi, n}}
    diff      = paired_bootstrap(per_group_a, per_group_b, "ndcg")

Directions
    by="resume": for each resume, rank the JDs it was scored against.   (the app's task)
    by="jd":     for each JD, rank the resumes it was scored against.

Metrics (relevance = graded GPT-4o score in [0, 1])
    ndcg      NDCG over the whole pool (pool sizes are small: ~4 JDs per resume)
    ndcg@10   only reported for groups with more than 10 items (mainly by="jd")
    top1      1 if the model's top item has the maximum true relevance, else 0
    pairwise  fraction of item pairs with |rel diff| > PAIRWISE_EPS ordered correctly
              (prediction ties count 0.5)
    spearman  rank correlation between predicted and true relevance (skips constant groups)
"""

import random
from collections import defaultdict

import numpy as np
from scipy.stats import rankdata

PAIRWISE_EPS = 0.05  # 0.5 points on the 0-10 GPT-4o scale


def ndcg(rels, preds, k=None):
    r = np.asarray(rels, dtype=float)
    order = np.argsort(-np.asarray(preds, dtype=float), kind="stable")
    k = len(r) if k is None else min(k, len(r))
    disc = 1.0 / np.log2(np.arange(2, k + 2))
    dcg = float((r[order][:k] * disc).sum())
    ideal = float((np.sort(r)[::-1][:k] * disc).sum())
    return dcg / ideal if ideal > 0 else None


def top1(rels, preds):
    r = np.asarray(rels, dtype=float)
    i = int(np.argmax(np.asarray(preds, dtype=float)))
    return float(r[i] >= r.max() - 1e-9)


def pairwise_acc(rels, preds):
    n, good, tot = len(rels), 0.0, 0
    for i in range(n):
        for j in range(i + 1, n):
            d = rels[i] - rels[j]
            if abs(d) <= PAIRWISE_EPS:
                continue
            tot += 1
            pd = preds[i] - preds[j]
            good += 1.0 if pd * d > 0 else (0.5 if pd == 0 else 0.0)
    return good / tot if tot else None


def spearman(rels, preds):
    a, b = np.asarray(rels, dtype=float), np.asarray(preds, dtype=float)
    if np.ptp(a) == 0 or np.ptp(b) == 0:
        return None
    return float(np.corrcoef(rankdata(a), rankdata(b))[0, 1])


def group_metrics(rels, preds):
    m = {}
    if len(rels) < 2:
        return m
    for name, val in (("ndcg", ndcg(rels, preds)), ("top1", top1(rels, preds) if max(rels) > min(rels) else None),
                      ("pairwise", pairwise_acc(rels, preds)), ("spearman", spearman(rels, preds))):
        if val is not None:
            m[name] = val
    if len(rels) > 10:
        v = ndcg(rels, preds, 10)
        if v is not None:
            m["ndcg@10"] = v
    return m


def evaluate(pairs, preds, by="resume"):
    key = "rh" if by == "resume" else "jh"
    groups = defaultdict(list)
    for i, p in enumerate(pairs):
        groups[p[key]].append(i)
    per_group = defaultdict(dict)
    for g, idx in groups.items():
        rels = [pairs[i]["rel"] for i in idx]
        pr = [float(preds[i]) for i in idx]
        for metric, val in group_metrics(rels, pr).items():
            per_group[metric][g] = val
    return dict(per_group)


def summarize(per_group, n_boot=1000, seed=0):
    rng = np.random.default_rng(seed)
    out = {}
    for metric, d in per_group.items():
        vals = np.array(list(d.values()), dtype=float)
        if len(vals) == 0:
            continue
        boots = [vals[rng.integers(0, len(vals), len(vals))].mean() for _ in range(n_boot)]
        out[metric] = {"mean": float(vals.mean()), "lo": float(np.percentile(boots, 2.5)),
                       "hi": float(np.percentile(boots, 97.5)), "n": int(len(vals))}
    return out


def paired_bootstrap(per_group_a, per_group_b, metric, n_boot=2000, seed=0):
    """Mean(a - b) over groups both models were scored on, with a 95% bootstrap CI.
    A CI that excludes 0 means the difference is unlikely to be noise at this sample size."""
    da, db = per_group_a.get(metric, {}), per_group_b.get(metric, {})
    common = sorted(set(da) & set(db))
    if not common:
        return None
    diff = np.array([da[g] - db[g] for g in common], dtype=float)
    rng = np.random.default_rng(seed)
    boots = [diff[rng.integers(0, len(diff), len(diff))].mean() for _ in range(n_boot)]
    return {"mean_diff": float(diff.mean()), "lo": float(np.percentile(boots, 2.5)),
            "hi": float(np.percentile(boots, 97.5)), "n": int(len(diff)),
            "significant": bool(np.percentile(boots, 2.5) > 0 or np.percentile(boots, 97.5) < 0)}


def random_baseline(pairs, by="resume", n_draws=100, seed=0):
    """Per-group metrics averaged over many random score assignments (expected chance level)."""
    rng = random.Random(seed)
    acc = defaultdict(lambda: defaultdict(list))
    for _ in range(n_draws):
        preds = [rng.random() for _ in pairs]
        for metric, d in evaluate(pairs, preds, by).items():
            for g, v in d.items():
                acc[metric][g].append(v)
    return {m: {g: float(np.mean(vs)) for g, vs in d.items()} for m, d in acc.items()}


def format_cell(s):
    return f"{s['mean']:.3f} [{s['lo']:.3f}, {s['hi']:.3f}]"


# --------------------------------------------------------------------------- #
# JD-centered evaluation: isolates resume-specific fit from "which JD is easy to score high".
# --------------------------------------------------------------------------- #
def center_rel_by_jd(pairs):
    """Copy of `pairs` with rel replaced by rel minus that JD's mean rel (over the pairs given).
    This only DEFINES the target for evaluation; no model ever sees it."""
    sums, cnt = defaultdict(float), defaultdict(int)
    for p in pairs:
        sums[p["jh"]] += p["rel"]
        cnt[p["jh"]] += 1
    return [dict(p, rel=p["rel"] - sums[p["jh"]] / cnt[p["jh"]]) for p in pairs]


def evaluate_centered(pairs, centered_preds, by="resume"):
    """Rank-based metrics only (NDCG needs non-negative gains). `centered_preds` should already
    have each JD's label-free offset removed (see train_cv.predict_fold)."""
    per = evaluate(center_rel_by_jd(pairs), centered_preds, by)
    per.pop("ndcg", None)
    per.pop("ndcg@10", None)
    return per

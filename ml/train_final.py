#!/usr/bin/env python
"""
Train the FINAL model on all scored pairs with the pre-specified config and save it.

    py -3.12 ml\\train_final.py

Config is the one evaluated in train_cv.py (compact view, CoSENT, 3 epochs, lr 2e-5, batch 16,
seed 42). Nothing here is tuned. There is no held-out set for THIS model: its trustworthy numbers
are the cross-validated ones from train_cv.py. Use it for (a) the Natalia human-labeled check
and (b) the in-app validation-suite comparison.

Output: models\\minilm-resume-jd-ft\\  (about 90 MB: keep it out of git; publish via the HF Hub
only if the dataset's license allows redistribution of models trained on it).
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
from data import load_pairs  # noqa: E402
from train_cv import VIEWS, train_epoch, train_targets  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="all-MiniLM-L6-v2")
    ap.add_argument("--view", choices=list(VIEWS), default="compact")
    ap.add_argument("--loss", choices=["cosent", "cosine"], default="cosent")
    ap.add_argument("--center-target", action="store_true")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-dir", default=os.path.join("models", "minilm-resume-jd-ft"))
    args = ap.parse_args()

    import torch
    from sentence_transformers import SentenceTransformer
    from transformers import get_linear_schedule_with_warmup

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    pairs, _ = load_pairs()
    rf, jf = VIEWS[args.view]
    targets = train_targets(pairs, args.center_target)
    print(f"training on {len(pairs)} pairs, {len({p['rh'] for p in pairs})} resumes, "
          f"{len({p['jh'] for p in pairs})} JDs   device={device}")

    model = SentenceTransformer(args.model, device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    steps = max(1, args.epochs * math.ceil(len(pairs) / args.batch_size))
    scheduler = get_linear_schedule_with_warmup(optimizer, int(0.1 * steps), steps)
    rng = random.Random(args.seed * 100)

    losses, t0 = [], time.time()
    for e in range(1, args.epochs + 1):
        loss = train_epoch(model, pairs, rf, jf, targets, args, optimizer, scheduler, rng, device)
        losses.append(loss)
        print(f"  epoch {e}/{args.epochs}  train loss={loss:.4f}  ({time.time() - t0:.0f}s elapsed)")

    os.makedirs(args.out_dir, exist_ok=True)
    model.save(args.out_dir)
    meta = {"base_model": args.model, "view": args.view, "loss": args.loss,
            "center_target": args.center_target, "epochs": args.epochs, "lr": args.lr,
            "batch_size": args.batch_size, "seed": args.seed, "n_pairs": len(pairs),
            "epoch_losses": losses,
            "note": "Trained on netsol/resume-score-details compact views; labels are GPT-4o scores."}
    with open(os.path.join(args.out_dir, "training_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    # reload check
    reloaded = SentenceTransformer(args.out_dir, device=device)
    emb = reloaded.encode(["Role: Data Analyst. Skills: sql, python."], normalize_embeddings=True)
    print(f"\nSaved to {args.out_dir}; reload OK, embedding dim = {emb.shape[1]}")


if __name__ == "__main__":
    main()

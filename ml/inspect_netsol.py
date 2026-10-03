#!/usr/bin/env python
"""
Phase 3, step 1b: inspect netsol/resume-score-details by reading the raw JSON files.

Why this exists: datasets.load_dataset() fails on this dataset because the ~1,000 nested
JSON files do not share one schema. This script reads them directly (from the HuggingFace
cache you already populated) and answers:

  * How many schema variants are there? (explains the load failure)
  * File-name groups (match / mismatch / invalid / ...) vs the dataset card's counts
  * Unique resumes vs unique JDs vs files -> the GROUP KEY for train/val/test splits
  * Do the same resumes appear in both matched and mismatched files? (leakage risk)
  * Distribution of the aggregated GPT-4o scores per group (graded relevance?)
  * Text lengths (and, with --token-check, how much MiniLM would truncate)

PRIVACY: files contain personal info (names, emails). This script never prints or saves
resume text. It prints structure, counts, scores and the first 200 chars of one JD only.

Usage (from the repo root):
    py -3.12 ml\\inspect_netsol.py
    py -3.12 ml\\inspect_netsol.py --token-check
    py -3.12 ml\\inspect_netsol.py --dir "C:\\path\\to\\folder\\with\\json"   # explicit folder
"""

import argparse
import glob
import hashlib
import json
import os
import re
import statistics
import sys
from collections import Counter, defaultdict

REPO_CANDIDATES = ["saptarshideveloper/resume-score-details", "netsol/resume-score-details"]
CARD_COUNTS = {"mismatched": 201, "matched": 648, "invalid": 142, "missing additional info": 40,
               "total": 1031}


class Tee:
    def __init__(self, path):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self.f = open(path, "w", encoding="utf-8")

    def write(self, s):
        sys.__stdout__.write(s)
        self.f.write(s)

    def flush(self):
        sys.__stdout__.flush()
        self.f.flush()


def hr(title):
    print("\n" + "=" * 78 + f"\n{title}\n" + "=" * 78)


def get(d, *path, default=None):
    for k in path:
        if not isinstance(d, dict) or k not in d:
            return default
        d = d[k]
    return d


def norm_hash(text):
    t = re.sub(r"\s+", " ", str(text).lower()).strip()
    return hashlib.md5(t.encode("utf-8")).hexdigest()


def quantiles(vals, qs=(0.1, 0.25, 0.5, 0.75, 0.9)):
    if not vals:
        return {}
    s = sorted(vals)
    return {f"p{int(q*100)}": round(s[min(int(q * (len(s) - 1)), len(s) - 1)], 3) for q in qs}


def find_data_dir(explicit, repo_id, allow_download):
    if explicit:
        return explicit, explicit
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("ERROR: pip install huggingface_hub (it comes with `datasets`).")
        sys.exit(1)
    repos = [repo_id] if repo_id else REPO_CANDIDATES
    for rid in repos:
        try:
            p = snapshot_download(repo_id=rid, repo_type="dataset", local_files_only=True)
            if glob.glob(os.path.join(p, "**", "*.json"), recursive=True):
                return rid, p
        except Exception:
            continue
    if allow_download:
        rid = repo_id or REPO_CANDIDATES[-1]
        print(f"Not in cache; downloading {rid} (may be rate limited; set HF_TOKEN to help)...")
        return rid, snapshot_download(repo_id=rid, repo_type="dataset")
    print("Could not find a cached copy. Re-run with --download, or pass --dir.")
    sys.exit(1)


def load_records(folder):
    paths = sorted(glob.glob(os.path.join(folder, "**", "*.json"), recursive=True))
    recs, bad = [], []
    for p in paths:
        name = os.path.basename(p)
        if name in ("dataset_info.json", "config.json"):
            continue
        try:
            with open(p, "r", encoding="utf-8") as f:
                d = json.load(f)
        except Exception as e:
            bad.append((name, f"{type(e).__name__}: {e}"))
            continue
        recs.append((name, d))
    return recs, bad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=None)
    ap.add_argument("--repo", default=None)
    ap.add_argument("--download", action="store_true")
    ap.add_argument("--token-check", action="store_true")
    ap.add_argument("--out", default=os.path.join("ml", "results", "netsol_inspection"))
    args = ap.parse_args()

    sys.stdout = Tee(args.out + ".txt")
    report = {}

    repo, folder = find_data_dir(args.dir, args.repo, args.download)
    hr(f"SOURCE: {repo}\n{folder}")
    recs, bad = load_records(folder)
    print(f"JSON files parsed: {len(recs)}   unparseable: {len(bad)}")
    for n, e in bad[:5]:
        print(f"  [!] {n}: {e}")
    report["n_files"] = len(recs)
    report["n_unparseable"] = len(bad)

    # ---------------- schema variants ----------------
    hr("SCHEMA VARIANTS (why load_dataset failed)")
    top_variants = Counter(tuple(sorted(d.keys())) if isinstance(d, dict) else ("<not a dict>",)
                           for _, d in recs)
    in_variants = Counter(tuple(sorted(get(d, "input", default={}).keys()))
                          if isinstance(get(d, "input"), dict) else ("<no input>",) for _, d in recs)
    out_variants = Counter(tuple(sorted(get(d, "output", default={}).keys()))
                           if isinstance(get(d, "output"), dict) else ("<no output>",) for _, d in recs)
    for label, c in (("top-level keys", top_variants), ("input keys", in_variants),
                     ("output keys", out_variants)):
        print(f"\n  {label}: {len(c)} variant(s)")
        for k, v in c.most_common(6):
            print(f"    {v:5d} x {list(k)}")
    report["schema_variants"] = {"top": len(top_variants), "input": len(in_variants),
                                 "output": len(out_variants)}

    # ---------------- per-file rows ----------------
    rows = []
    for name, d in recs:
        prefix = re.sub(r"[_-]?\d+\.json$", "", name)
        resume = get(d, "input", "resume")
        jd = get(d, "input", "job_description")
        agg = get(d, "output", "scores", "aggregated_scores", default={})
        rows.append({
            "name": name,
            "prefix": prefix,
            "resume": resume if isinstance(resume, str) else (json.dumps(resume) if resume else ""),
            "jd": jd if isinstance(jd, str) else (json.dumps(jd) if jd else ""),
            "macro": agg.get("macro_scores") if isinstance(agg, dict) else None,
            "micro": agg.get("micro_scores") if isinstance(agg, dict) else None,
            "valid": get(d, "output", "valid_resume_and_jd"),
            "has_pii_block": bool(get(d, "output", "personal_info")),
        })

    # ---------------- groups vs card ----------------
    hr("FILE-NAME GROUPS vs DATASET CARD")
    groups = Counter(r["prefix"] for r in rows)
    for k, v in groups.most_common():
        print(f"  {k:30s} {v}")
    print(f"\n  Dataset card says: {CARD_COUNTS}")
    print("  (If your group counts differ a lot from the card, this copy is not identical.)")
    report["groups"] = dict(groups)

    print(f"\n  valid_resume_and_jd values: {dict(Counter(str(r['valid']) for r in rows))}")
    print(f"  files containing a personal_info block (contents NOT printed): "
          f"{sum(r['has_pii_block'] for r in rows)}")

    # ---------------- uniqueness / group key ----------------
    hr("UNIQUE RESUMES / JDs  (GROUP KEY for splitting)")
    with_text = [r for r in rows if r["resume"] and r["jd"]]
    print(f"  files with both resume and JD text: {len(with_text)}/{len(rows)}")
    res_files, jd_files, pairs = defaultdict(list), defaultdict(list), set()
    for r in with_text:
        rh, jh = norm_hash(r["resume"]), norm_hash(r["jd"])
        r["rh"], r["jh"] = rh, jh
        res_files[rh].append(r)
        jd_files[jh].append(r)
        pairs.add((rh, jh))
    print(f"  unique resumes: {len(res_files)}   unique JDs: {len(jd_files)}   "
          f"unique (resume,JD) pairs: {len(pairs)}")
    rc = Counter(len(v) for v in res_files.values())
    jc = Counter(len(v) for v in jd_files.values())
    print(f"  files-per-resume distribution: {dict(sorted(rc.items()))}")
    print(f"  files-per-JD distribution:     {dict(sorted(jc.items()))}")
    mixed = sum(1 for v in res_files.values() if len({x['prefix'] for x in v}) > 1)
    print(f"  resumes that appear under more than one file-name group: {mixed}")
    if len(res_files) < len(with_text):
        print("  >> Resumes repeat across files. Split by resume hash, NOT by file, or the "
              "test set leaks.")
    if len(jd_files) < len(with_text):
        print("  >> JDs repeat across files. Group by JD too (connected components of "
              "resume/JD) to be safe.")
    report["unique"] = {"files_with_text": len(with_text), "resumes": len(res_files),
                        "jds": len(jd_files), "pairs": len(pairs), "mixed_group_resumes": mixed}

    # ---------------- scores ----------------
    hr("AGGREGATED GPT-4o SCORES BY GROUP")
    by_group = defaultdict(list)
    for r in rows:
        by_group[r["prefix"]].append(r)
    report["scores"] = {}
    for g, rs in sorted(by_group.items()):
        macro = [x["macro"] for x in rs if isinstance(x["macro"], (int, float))]
        micro = [x["micro"] for x in rs if isinstance(x["micro"], (int, float))]
        print(f"\n  [{g}]  files={len(rs)}  with macro={len(macro)}  with micro={len(micro)}")
        for label, vals in (("macro", macro), ("micro", micro)):
            if vals:
                print(f"    {label}: min={min(vals):.2f} mean={statistics.mean(vals):.2f} "
                      f"max={max(vals):.2f}  {quantiles(vals)}")
        report["scores"][g] = {"n": len(rs), "n_macro": len(macro), "n_micro": len(micro),
                               "macro_mean": round(statistics.mean(macro), 3) if macro else None,
                               "micro_mean": round(statistics.mean(micro), 3) if micro else None}
    print("\n  What to look for: do match-groups score clearly higher than mismatch-groups?\n"
          "  Is there overlap (hard cases) or a clean gap (easy task)? A clean gap means the\n"
          "  baseline will likely score high already and hard-negative mining matters more.")

    # ---------------- lengths ----------------
    hr("TEXT LENGTHS (chars)")
    for label, key in (("resume", "resume"), ("job description", "jd")):
        lens = [len(r[key]) for r in rows if r[key]]
        if lens:
            print(f"  {label}: n={len(lens)} median={statistics.median(lens):.0f} "
                  f"p95={quantiles(lens, (0.95,))['p95']:.0f} max={max(lens)}")
            report[f"len_{key}"] = {"median": statistics.median(lens), "max": max(lens)}

    if args.token_check:
        hr("TOKEN LENGTH CHECK (all-MiniLM-L6-v2)")
        try:
            from sentence_transformers import SentenceTransformer
            m = SentenceTransformer("all-MiniLM-L6-v2")
            tok, ml = m.tokenizer, m.max_seq_length
            print(f"  max_seq_length = {ml}")
            for label, key in (("resume", "resume"), ("job description", "jd")):
                lens = [len(tok.encode(r[key], truncation=False)) for r in rows if r[key]][:600]
                over = sum(l > ml for l in lens) / max(len(lens), 1)
                print(f"  {label}: median={statistics.median(lens):.0f} tokens, max={max(lens)}, "
                      f"{over:.0%} exceed {ml} and would be truncated")
                report[f"tok_over_{key}"] = round(over, 3)
        except Exception as e:
            print(f"  skipped: {type(e).__name__}: {e}")

    # ---------------- one safe sample ----------------
    hr("ONE STRUCTURAL SAMPLE (no resume text, no personal info)")
    for name, d in recs:
        if isinstance(d, dict) and get(d, "input", "macro_dict") and get(d, "output", "scores"):
            print(f"  file: {name}")
            print(f"  macro_dict (criteria weights): {get(d, 'input', 'macro_dict')}")
            print(f"  micro_dict (criteria weights): {get(d, 'input', 'micro_dict')}")
            print(f"  aggregated_scores: {get(d, 'output', 'scores', 'aggregated_scores')}")
            jd = get(d, "input", "job_description") or ""
            print(f"  JD (first 200 chars): {str(jd)[:200].replace(chr(10), ' ')}")
            print(f"  details keys: {list(get(d, 'details', default={}).keys())[:15]}")
            break

    hr("WHAT THIS DECIDES")
    print("""  1. Group key for splits: resume hash (and JD hash if JDs repeat).
  2. Relevance labels: aggregated macro/micro scores (graded) + match/mismatch group.
  3. If match-vs-mismatch scores barely overlap, the in-distribution test is easy; the
     human-labeled Natalia set and the in-app validation suite carry the real evidence.
  4. Never commit the raw data or resume text (personal info). Commit only metrics.
""")
    with open(args.out + ".json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"Saved: {args.out}.txt and {args.out}.json")


if __name__ == "__main__":
    main()

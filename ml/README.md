# ML: fine-tuning and evaluating a resume-to-role bi-encoder

Phase 3 of TalentSync: fine-tune a bi-encoder on GPT-4o-scored resume/job-description pairs, benchmark it against the
pretrained `all-MiniLM-L6-v2` that the app uses, and swap it into `role_matcher.py` only if it measurably wins.

## TL;DR

- **Fine-tuning works on the training task.** In grouped 5-fold cross-validation (3 seeds), pairwise accuracy for ranking
  a resume's job descriptions rose from 0.58 (pretrained) to 0.74 - 0.78. Paired bootstrap CIs on the gain exclude 0 in every seed.
- **Most of that gain is not resume understanding.** A per-JD score lookup that ignores the resume recovers about two thirds of
  it (0.69 - 0.71). The resume-specific part is small and not reliably significant.
- **It did not transfer.** On a small human-labeled set it was indistinguishable from the pretrained model, and on the app's
  own validation suite it changed nothing (identical effectiveness).
- **Decision: the pretrained model stays the default.** `role_matcher.py` is unchanged. The decision rule was fixed before the in-app run.

## Data

| | |
|---|---|
| Training / CV data | `netsol/resume-score-details` (loaded from a HuggingFace copy, `saptarshideveloper/resume-score-details`; group counts match the original card). 1,031 files, of which **851 usable scored pairs** (142 have no scores, 35 flagged invalid, 3 missing text) covering **199 resumes and 26 job descriptions**. Labels are GPT-4o scores. |
| Relevance label | `mean(macro, micro) / 10`, in [0, 1] |
| Held-out human check | Vanetik & Kogan (2023) dataset: 30 resumes x 5 vacancies, two human annotators. 28 usable CVs (annotator 1's rows for CV 9 and 28 are not valid permutations). |
| In-app check | The repo's `validation_test_suite.py` (21 hand-written cases, 39 roles). |

Inputs are short "compact" views built from fields in each record so they fit MiniLM's 256-token window
(the raw texts exceed it for 96.5% of resumes and 80.8% of JDs):
resume = `Experience: <titles>. Skills: <skills>. Summary: <summary>`; JD = `Role: <title>. Key criteria: <criteria>. Requirements: <requirements>`.

## Method

- Base model `all-MiniLM-L6-v2`; **CoSENT loss** on the graded relevance (not in-batch negatives: with only 26 distinct JDs a batch
  contains the same JD paired with other resumes, which would be treated as false negatives).
- 3 epochs, lr 2e-5, batch 16, AdamW (weight decay 0.01), 10% linear warmup. **Nothing was tuned**; the final epoch is the pre-specified headline.
- **Grouped 5-fold CV by resume** (stratified by pairs per resume), so every resume is scored by a model that never saw it.
  Seeds 42, 43, 44 change both the fold split and training randomness. 131 resumes have two or more scored JDs and are evaluable.
- Metrics per resume, over its scored JDs: pairwise ordering accuracy, Spearman, top-1; bootstrap CIs over resumes; **paired**
  bootstrap for model comparisons. NDCG is not reported as a headline: with about 4 items per pool its chance level is already 0.93.
- Baselines: pretrained cosine, random, a **JD prior** (each JD's mean training score, no resume information), and a **hybrid**
  (JD prior + pretrained cosine with its JD offset removed; no training).

## Results

### 1. Cross-validated ranking of JDs per resume (n = 131 resumes)

Pairwise accuracy (chance 0.50):

| Seed | Pretrained | JD prior | Hybrid | **Fine-tuned** | Gain over pretrained (paired 95% CI) |
|---|---|---|---|---|---|
| 42 | 0.579 | 0.693 | 0.718 | **0.759** | +0.180 [+0.131, +0.230] |
| 43 | 0.579 | 0.705 | 0.706 | **0.741** | +0.162 [+0.114, +0.212] |
| 44 | 0.579 | 0.700 | 0.705 | **0.780** | +0.201 [+0.151, +0.256] |

Spearman for the fine-tuned model: 0.546 / 0.515 / 0.586 (pretrained 0.158). Top-1: 0.473 / 0.382 / 0.450 (pretrained 0.244).

### 2. What the gain is made of (pairwise accuracy, paired 95% CI)

| Comparison | Seed 42 | Seed 43 | Seed 44 |
|---|---|---|---|
| Fine-tuned minus JD prior | +0.066 [+0.017, +0.112] | +0.035 [-0.012, +0.083] | +0.080 [+0.034, +0.127] |
| Fine-tuned minus hybrid | not paired (0.759 vs 0.718) | +0.035 [+0.003, +0.067] | +0.075 [+0.032, +0.118] |
| JD-centered (resume-specific fit), gain over pretrained | +0.011 [-0.032, +0.058] | +0.014 [-0.020, +0.049] | +0.011 [-0.027, +0.048] |
| Ranking resumes within a JD, gain over pretrained | +0.074 [+0.019, +0.142] | -0.011 [-0.084, +0.047] | +0.014 [-0.095, +0.094] |

Reading: roughly two thirds of the gain is reproducible by a lookup table. Fine-tuning still beats the hybrid on pairwise and Spearman in
both paired seeds, so it is not *only* a JD lookup, but the resume-specific component is small and the within-JD result from seed 42 did not replicate.
An ablation that trains on JD-centered targets (seed 42) gave 0.651 pairwise: significantly above pretrained (+0.072 [+0.038, +0.111]) but not above the JD prior (-0.042 [-0.104, +0.020]).

### 3. Human-labeled check (Vanetik & Kogan 2023, n = 28 CVs)

The two annotators agree with each other at Spearman 0.14 (chance +/- 0.09), so these labels cannot separate models.

| | Pairwise | Spearman |
|---|---|---|
| Pretrained | 0.291 | -0.470 |
| Fine-tuned | 0.274 | -0.501 |
| TF-IDF cosine | 0.514 | 0.020 |
| Human vs human | 0.546 | 0.143 |
| Chance | 0.504 | 0.012 |

Fine-tuned minus pretrained: pairwise -0.018 [-0.070, +0.025]. The negative numbers come from a vacancy-level effect (the humans'
average ordering of the 5 vacancies is opposite to both models'); after removing each vacancy's mean (post hoc), pretrained and fine-tuned
score 0.466 and 0.453 pairwise against chance 0.505. **Inconclusive.**

### 4. In-app: validation suite, 2 x 2 (model x text template)

Effectiveness on the 21 cases (pass = 1, partial = 0.5). "Corrected" counts correct absence as a pass for the two negative cases
(see note below). "LOO-tuned" re-tunes weights and thresholds on a shared grid with leave-one-case-out.

| Configuration | Suite scoring, as shipped | Corrected, as shipped | Corrected, LOO-tuned |
|---|---|---|---|
| A pretrained + current templates (ships today) | 81.0% | 90.5% | 90.5% |
| B fine-tuned + current templates | 81.0% | 90.5% | 90.5% |
| C fine-tuned + compact templates | 81.0% | 90.5% | 90.5% |
| D pretrained + compact templates | 83.3% | 92.9% | 92.9% |

Fine-tuned minus pretrained is exactly 0.000 [0.000, 0.000] with either template. The one-case gap for D comes from a single case (Compliance Officer, rank 3 to 1).
The fine-tuned model's cosines are about 0.19 higher on the same texts (mean 0.386 to 0.578), so the app's 0.25 threshold and
Strong/Good/Potential score bands would need re-tuning if it were ever adopted.

## Decision

Pre-specified rule: switch only if a fine-tuned configuration beats the shipped one on LOO-tuned, corrected scoring with a paired CI above 0
and no safety regression. It does not, so **the app keeps `all-MiniLM-L6-v2`** and `role_matcher.py` is untouched.

## Limitations

- Labels are GPT-4o scores, not recruiter judgments.
- Only 26 distinct JDs: test resumes are new but most test JDs were seen in training. Nothing here shows generalisation to unseen JDs.
- Compact inputs come from GPT-parsed fields, which TalentSync's own extractor does not produce.
- 199 resumes in total. Seeds vary the split and training randomness, not the underlying resumes.
- The human-labeled set is tiny and noisy, IT-only, and assumes CSV row order = vacancy numbering (unverified).
- The validation suite is small, hand-written and was likely used to tune the current weights, which favours the shipped configuration.
- One scoring bug in the suite was found and fixed: cases with `expected_rank: None` scored correct absence as FAILED. Numbers
  recorded before the fix are not comparable (the shipped configuration moves from 81.0% to 90.5%).

## Reproduce

```
py -3.12 -m pip install datasets sentence-transformers python-docx scikit-learn
py -3.12 ml\inspect_netsol.py --token-check        # schema and group checks
py -3.12 ml\data.py --report --token-check          # splits, compact-view coverage
py -3.12 ml\baseline_cv.py                          # pretrained, JD prior, chance
py -3.12 ml\train_cv.py --seed 42                   # also 43, 44 (use --out to keep results apart)
py -3.12 ml\hybrid_baseline.py --seed 43 --oof ml\results\train_cv_cosent_compact_seed43_oof.json
py -3.12 ml\train_final.py                          # all 851 pairs -> models\minilm-resume-jd-ft
py -3.12 ml\natalia_eval.py                         # needs ..\vacancy-resume-matching-dataset
py -3.12 ml\compare_in_app.py                       # needs the trained model
```

| File | Purpose |
|---|---|
| `data.py` | loader, compact views, split strategies, grouped K-fold |
| `eval.py` | ranking metrics, bootstrap CIs, paired bootstrap, JD-centered evaluation |
| `baseline_cv.py`, `hybrid_baseline.py` | baselines and the hybrid ablation |
| `train_cv.py`, `train_final.py` | cross-validated fine-tuning; final model |
| `natalia_eval.py`, `compare_in_app.py` | human-labeled check; in-app comparison |
| `results/` | metrics only (no resume or job-description text) |

## Data, privacy and citation

The training data contains personal information. No raw records are committed, `results/` holds only metrics and file ids, and the
trained weights (about 90 MB) are not committed. Check each dataset's license before redistributing anything derived from it.

- `netsol/resume-score-details` (HuggingFace).
- Vanetik, N. and Kogan, G. (2023). *Job Vacancy Ranking with Sentence Embeddings, Keywords, and Named Entities.* Information 14(8):468. https://doi.org/10.3390/info14080468

# In-app comparison (validation suite, 21 cases)

Effectiveness = mean case score (pass 1, partial 0.5, fail 0). 'suite' = the suite's own logic; 'corrected' counts correct absence as a pass for expected_rank=None cases. One case = 4.8 points. best-of-grid is optimistic (tuned on the suite); LOO-tuned is the honest tuned estimate.

## Effectiveness (suite scoring)

| config | as shipped | best-of-grid (optimistic) | LOO-tuned |
|---|---|---|---|
| A pretrained + current | 81.0% | 92.9% | 92.9% |
| B fine-tuned + current | 81.0% | 92.9% | 92.9% |
| C fine-tuned + compact | 81.0% | 92.9% | 92.9% |
| D pretrained + compact | 83.3% | 95.2% | 95.2% |

## Effectiveness (corrected scoring)

| config | as shipped | best-of-grid (optimistic) | LOO-tuned |
|---|---|---|---|
| A pretrained + current | 90.5% | 92.9% | 90.5% |
| B fine-tuned + current | 90.5% | 92.9% | 90.5% |
| C fine-tuned + compact | 90.5% | 92.9% | 90.5% |
| D pretrained + compact | 92.9% | 95.2% | 92.9% |

## Behaviour as shipped

| config | median # recs per case | top final score on negative cases (max) |
|---|---|---|
| A pretrained + current | 6 | 58% |
| B fine-tuned + current | 6 | 65% |
| C fine-tuned + compact | 6 | 70% |
| D pretrained + compact | 6 | 66% |

## Expected-role rank per case (as shipped)

| case | expected rank | A | B | C | D |
|---|---|---|---|---|---|
| Perfect Match - Software Engineer | 1 | 1 | 1 | 1 | 1 |
| Perfect Match - Data Scientist | 1 | 2 | 2 | 4 | 4 |
| Perfect Match - Full Stack Developer | 1 | 1 | 1 | 1 | 1 |
| Perfect Match - DevOps Engineer | 1 | 1 | 1 | 1 | 1 |
| Perfect Match - Accounts Executive | 1 | 1 | 1 | 1 | 1 |
| Cross-Domain - Military to Backend | 5 | - | - | - | - |
| Cross-Domain - Military to Data Analysis | 5 | 1 | 1 | 1 | 1 |
| Partial Match - Junior Software Dev | 10 | 2 | 3 | 2 | 2 |
| Complete Mismatch - Accountant vs Backend | none | - | - | - | - |
| Multi-skilled - Full Stack Developer | 2 | 2 | 2 | 1 | 1 |
| Perfect Match - Backend Developer | 1 | 1 | 1 | 1 | 1 |
| Perfect Match - Software Developer | 1 | 1 | 1 | 1 | 1 |
| Perfect Match - Data Analyst | 1 | 1 | 1 | 1 | 1 |
| Perfect Match - Machine Learning Engineer | 1 | 1 | 1 | 1 | 1 |
| Perfect Match - Technical Support Engineer | 1 | 1 | 1 | 1 | 1 |
| Perfect Match - Operations Manager | 1 | 1 | 1 | 1 | 1 |
| Perfect Match - Compliance Officer | 1 | 3 | 3 | 2 | 1 |
| Perfect Match - Team Lead | 1 | 1 | 1 | 1 | 1 |
| Cross-Domain - Support to DevOps | 8 | 1 | 1 | 1 | 1 |
| Cross-Domain - Finance to Data Analyst | 6 | 1 | 1 | 1 | 1 |
| Low Information Profile | none | - | - | - | - |

## Paired differences (corrected scoring)

| comparison | basis | diff [95% CI] |
|---|---|---|
| B - A (model effect, current template) | as shipped | +0.000 [+0.000, +0.000]  |
| B - A (model effect, current template) | LOO-tuned | +0.000 [+0.000, +0.000]  |
| C - D (model effect, compact template) | as shipped | -0.024 [-0.071, +0.000]  |
| C - D (model effect, compact template) | LOO-tuned | -0.024 [-0.071, +0.000]  |
| D - A (template effect, pretrained) | as shipped | +0.024 [+0.000, +0.071]  |
| D - A (template effect, pretrained) | LOO-tuned | +0.024 [+0.000, +0.071]  |
| C - A (best new vs shipped) | as shipped | +0.000 [+0.000, +0.000]  |
| C - A (best new vs shipped) | LOO-tuned | +0.000 [+0.000, +0.000]  |

## Decision rule (fixed in advance)

Switch the default only if, on LOO-tuned corrected scoring, a fine-tuned configuration beats A with a paired CI above 0 and shows no safety regression on the negative cases. Otherwise keep pretrained as the default.

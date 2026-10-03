# Fine-tuning CV results (cosent_compact, 5-fold by resume, seed 42)

Config: all-MiniLM-L6-v2, view=compact, loss=cosent, epochs=3, lr=2e-05, batch=16, center_target=False. Primary row = final epoch (pre-specified); other epochs shown for transparency only.

## RANK JDs PER RESUME (the app's task)

| model | pairwise | spearman | top1 | n |
|---|---|---|---|---|
| pretrained (epoch 0) | 0.579 [0.534, 0.623] | 0.158 [0.058, 0.256] | 0.244 [0.176, 0.321] | 131 |
| fine-tuned, epoch 1 | 0.742 [0.699, 0.780] | 0.512 [0.427, 0.586] | 0.420 [0.336, 0.504] | 131 |
| fine-tuned, epoch 2 | 0.756 [0.716, 0.793] | 0.543 [0.459, 0.616] | 0.443 [0.359, 0.527] | 131 |
| fine-tuned, epoch 3 | 0.759 [0.719, 0.797] | 0.546 [0.461, 0.617] | 0.473 [0.389, 0.557] | 131 |
| random (chance) | 0.500 [0.496, 0.504] | -0.002 [-0.011, 0.006] | 0.218 [0.196, 0.240] | 131 |
| JD prior (no resume info) | 0.693 [0.648, 0.733] | 0.419 [0.327, 0.505] | 0.313 [0.237, 0.389] | 131 |

## RANK RESUMES PER JD

| model | pairwise | spearman | ndcg@10 | n |
|---|---|---|---|---|
| pretrained (epoch 0) | 0.682 [0.581, 0.772] | 0.400 [0.173, 0.595] | 0.782 [0.734, 0.830] | 24 |
| fine-tuned, epoch 1 | 0.645 [0.580, 0.708] | 0.372 [0.218, 0.511] | 0.848 [0.820, 0.871] | 24 |
| fine-tuned, epoch 2 | 0.720 [0.658, 0.781] | 0.503 [0.380, 0.612] | 0.833 [0.801, 0.862] | 24 |
| fine-tuned, epoch 3 | 0.756 [0.693, 0.820] | 0.556 [0.423, 0.672] | 0.845 [0.817, 0.870] | 24 |
| random (chance) | 0.496 [0.486, 0.504] | -0.013 [-0.034, 0.006] | 0.677 [0.632, 0.713] | 24 |

## JD-CENTERED, JDs per resume (resume-specific fit only)

| model | pairwise | spearman | top1 | n |
|---|---|---|---|---|
| pretrained (epoch 0) | 0.576 [0.531, 0.618] | 0.172 [0.082, 0.259] | 0.305 [0.221, 0.389] | 131 |
| fine-tuned, epoch 1 | 0.596 [0.550, 0.640] | 0.230 [0.145, 0.312] | 0.328 [0.244, 0.412] | 131 |
| fine-tuned, epoch 2 | 0.587 [0.538, 0.634] | 0.228 [0.142, 0.315] | 0.344 [0.267, 0.420] | 131 |
| fine-tuned, epoch 3 | 0.587 [0.536, 0.632] | 0.235 [0.148, 0.322] | 0.366 [0.282, 0.450] | 131 |
| random (chance) | 0.499 [0.495, 0.504] | -0.003 [-0.011, 0.006] | 0.215 [0.193, 0.237] | 131 |

## Paired differences for 'fine-tuned, epoch 3'

| vs | direction | metric | diff [95% CI] | n |
|---|---|---|---|---|
| pretrained (epoch 0) | res | pairwise | +0.180 [+0.131, +0.230]* | 128 |
| pretrained (epoch 0) | res | spearman | +0.388 [+0.288, +0.491]* | 131 |
| pretrained (epoch 0) | res | top1 | +0.229 [+0.145, +0.313]* | 131 |
| JD prior (no resume info) | res | pairwise | +0.066 [+0.017, +0.112]* | 128 |
| JD prior (no resume info) | res | spearman | +0.113 [+0.022, +0.215]* | 122 |
| JD prior (no resume info) | res | top1 | +0.160 [+0.061, +0.260]* | 131 |
| pretrained (epoch 0) | cen | pairwise | +0.011 [-0.032, +0.058]  | 126 |
| pretrained (epoch 0) | cen | spearman | +0.064 [-0.015, +0.151]  | 131 |
| pretrained (epoch 0) | cen | top1 | +0.061 [-0.008, +0.130]  | 131 |
| pretrained (epoch 0) | jd | pairwise | +0.074 [+0.019, +0.142]* | 24 |
| pretrained (epoch 0) | jd | spearman | +0.156 [+0.037, +0.307]* | 24 |
| pretrained (epoch 0) | jd | top1 | +0.125 [+0.000, +0.250]  | 24 |

Reading guide: beating 'pretrained' with a CI excluding 0 = fine-tuning helps. Beating the JD prior = more than memorising which JDs score high. The JD-centered table is the cleanest test of resume-specific fit (the JD prior is a null model there by construction).

# Fine-tuning CV results (cosent_centered_compact, 5-fold by resume, seed 42)

Config: all-MiniLM-L6-v2, view=compact, loss=cosent, epochs=3, lr=2e-05, batch=16, center_target=True. Primary row = final epoch (pre-specified); other epochs shown for transparency only.

## RANK JDs PER RESUME (the app's task)

| model | pairwise | spearman | top1 | n |
|---|---|---|---|---|
| pretrained (epoch 0) | 0.579 [0.534, 0.623] | 0.158 [0.058, 0.256] | 0.244 [0.176, 0.321] | 131 |
| fine-tuned, epoch 1 | 0.645 [0.602, 0.686] | 0.293 [0.200, 0.385] | 0.351 [0.275, 0.435] | 131 |
| fine-tuned, epoch 2 | 0.650 [0.606, 0.692] | 0.298 [0.204, 0.393] | 0.351 [0.275, 0.435] | 131 |
| fine-tuned, epoch 3 | 0.651 [0.605, 0.695] | 0.302 [0.204, 0.399] | 0.359 [0.282, 0.443] | 131 |
| random (chance) | 0.500 [0.496, 0.504] | -0.002 [-0.011, 0.006] | 0.218 [0.196, 0.240] | 131 |
| JD prior (no resume info) | 0.693 [0.648, 0.733] | 0.419 [0.327, 0.505] | 0.313 [0.237, 0.389] | 131 |

## RANK RESUMES PER JD

| model | pairwise | spearman | ndcg@10 | n |
|---|---|---|---|---|
| pretrained (epoch 0) | 0.682 [0.581, 0.772] | 0.400 [0.173, 0.595] | 0.782 [0.734, 0.830] | 24 |
| fine-tuned, epoch 1 | 0.670 [0.587, 0.740] | 0.422 [0.256, 0.559] | 0.860 [0.828, 0.888] | 24 |
| fine-tuned, epoch 2 | 0.750 [0.690, 0.806] | 0.563 [0.471, 0.655] | 0.844 [0.812, 0.874] | 24 |
| fine-tuned, epoch 3 | 0.757 [0.690, 0.818] | 0.578 [0.452, 0.686] | 0.857 [0.826, 0.887] | 24 |
| random (chance) | 0.496 [0.486, 0.504] | -0.013 [-0.034, 0.006] | 0.677 [0.632, 0.713] | 24 |

## JD-CENTERED, JDs per resume (resume-specific fit only)

| model | pairwise | spearman | top1 | n |
|---|---|---|---|---|
| pretrained (epoch 0) | 0.576 [0.531, 0.618] | 0.172 [0.082, 0.259] | 0.305 [0.221, 0.389] | 131 |
| fine-tuned, epoch 1 | 0.572 [0.524, 0.619] | 0.178 [0.086, 0.271] | 0.298 [0.214, 0.374] | 131 |
| fine-tuned, epoch 2 | 0.587 [0.539, 0.634] | 0.226 [0.134, 0.311] | 0.336 [0.252, 0.412] | 131 |
| fine-tuned, epoch 3 | 0.596 [0.547, 0.643] | 0.245 [0.158, 0.332] | 0.336 [0.252, 0.420] | 131 |
| random (chance) | 0.499 [0.495, 0.504] | -0.003 [-0.011, 0.006] | 0.215 [0.193, 0.237] | 131 |

## Paired differences for 'fine-tuned, epoch 3'

| vs | direction | metric | diff [95% CI] | n |
|---|---|---|---|---|
| pretrained (epoch 0) | res | pairwise | +0.072 [+0.038, +0.111]* | 128 |
| pretrained (epoch 0) | res | spearman | +0.145 [+0.062, +0.233]* | 131 |
| pretrained (epoch 0) | res | top1 | +0.115 [+0.038, +0.191]* | 131 |
| JD prior (no resume info) | res | pairwise | -0.042 [-0.104, +0.020]  | 128 |
| JD prior (no resume info) | res | spearman | -0.115 [-0.239, +0.015]  | 122 |
| JD prior (no resume info) | res | top1 | +0.046 [-0.061, +0.153]  | 131 |
| pretrained (epoch 0) | cen | pairwise | +0.020 [-0.023, +0.062]  | 126 |
| pretrained (epoch 0) | cen | spearman | +0.074 [-0.006, +0.160]  | 131 |
| pretrained (epoch 0) | cen | top1 | +0.031 [-0.038, +0.099]  | 131 |
| pretrained (epoch 0) | jd | pairwise | +0.074 [-0.015, +0.181]  | 24 |
| pretrained (epoch 0) | jd | spearman | +0.178 [+0.024, +0.380]* | 24 |
| pretrained (epoch 0) | jd | top1 | +0.125 [-0.042, +0.292]  | 24 |

Reading guide: beating 'pretrained' with a CI excluding 0 = fine-tuning helps. Beating the JD prior = more than memorising which JDs score high. The JD-centered table is the cleanest test of resume-specific fit (the JD prior is a null model there by construction).

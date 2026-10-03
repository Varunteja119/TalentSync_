# Baseline (all pairs; JD prior via 5-fold CV) - all-MiniLM-L6-v2

Relevance = mean(macro, micro)/10 from GPT-4o. 95% bootstrap CIs over resumes (or JDs).

## Rank JDs per resume

| model | pairwise | spearman | top1 | ndcg | n |
|---|---|---|---|---|---|
| MiniLM pretrained, raw | 0.553 [0.509, 0.601] | 0.128 [0.027, 0.225] | 0.282 [0.206, 0.359] | 0.945 [0.936, 0.953] | 131 |
| MiniLM pretrained, compact | 0.579 [0.534, 0.623] | 0.158 [0.058, 0.256] | 0.244 [0.176, 0.321] | 0.945 [0.935, 0.953] | 131 |
| random (chance) | 0.500 [0.496, 0.504] | -0.002 [-0.011, 0.006] | 0.218 [0.196, 0.240] | 0.932 [0.926, 0.938] | 131 |
| JD prior (no resume info) | 0.693 [0.648, 0.733] | 0.419 [0.327, 0.505] | 0.313 [0.237, 0.389] | 0.962 [0.955, 0.968] | 131 |

## Rank resumes per JD

| model | pairwise | spearman | top1 | ndcg | ndcg@10 | n |
|---|---|---|---|---|---|---|
| MiniLM pretrained, raw | 0.621 [0.531, 0.701] | 0.302 [0.084, 0.494] | 0.250 [0.083, 0.417] | 0.950 [0.936, 0.963] | 0.804 [0.758, 0.845] | 24 |
| MiniLM pretrained, compact | 0.682 [0.581, 0.772] | 0.400 [0.173, 0.595] | 0.292 [0.125, 0.458] | 0.943 [0.922, 0.962] | 0.782 [0.734, 0.830] | 24 |
| random (chance) | 0.496 [0.486, 0.504] | -0.013 [-0.034, 0.006] | 0.126 [0.073, 0.173] | 0.911 [0.896, 0.927] | 0.677 [0.632, 0.713] | 24 |

## Paired differences (by resume)

| comparison | metric | diff [95% CI] | n |
|---|---|---|---|
| MiniLM pretrained, compact - MiniLM pretrained, raw | pairwise | +0.026 [-0.028, +0.077]  | 128 |
| MiniLM pretrained, compact - MiniLM pretrained, raw | spearman | +0.031 [-0.095, +0.159]  | 124 |
| MiniLM pretrained, compact - MiniLM pretrained, raw | top1 | -0.038 [-0.137, +0.061]  | 131 |
| MiniLM pretrained, compact - JD prior (no resume info) | pairwise | -0.114 [-0.184, -0.049]* | 128 |
| MiniLM pretrained, compact - JD prior (no resume info) | spearman | -0.271 [-0.398, -0.135]* | 122 |
| MiniLM pretrained, compact - JD prior (no resume info) | top1 | -0.069 [-0.168, +0.023]  | 131 |
| MiniLM pretrained, raw - JD prior (no resume info) | pairwise | -0.140 [-0.207, -0.074]* | 128 |
| MiniLM pretrained, raw - JD prior (no resume info) | spearman | -0.305 [-0.439, -0.149]* | 122 |
| MiniLM pretrained, raw - JD prior (no resume info) | top1 | -0.031 [-0.130, +0.069]  | 131 |
| MiniLM pretrained, compact - random (chance) | pairwise | +0.079 [+0.028, +0.125]* | 128 |
| MiniLM pretrained, compact - random (chance) | spearman | +0.160 [+0.058, +0.258]* | 131 |
| MiniLM pretrained, compact - random (chance) | top1 | +0.027 [-0.044, +0.100]  | 131 |

Notes: NDCG has a high chance floor with ~4 items per pool; lead with pairwise/Spearman. JD prior ignores the resume; a model must beat it to claim it reads resumes.

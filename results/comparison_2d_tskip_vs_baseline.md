Baseline: `2d_baseline_lam0.25_disp`  (split: test)

| run | metric | baseline | candidate | Δ | better on | paired t p | wilcoxon p |
|---|---|---|---|---|---|---|---|
| 2d_cross_attn_lam0.1_svf_tskip | dice | 0.7544 | 0.7450 | -0.0094 ** | 43/100 | 6.91e-03 | 6.43e-02 |
| 2d_cross_attn_lam0.1_svf_tskip | folding_fraction | 0.0001 | 0.0000 | -0.0001 | 4/100 | 1.91e-01 | 6.79e-02 |
| 2d_cross_attn_lam0.1_disp_tskip | dice | 0.7544 | 0.7395 | -0.0149 ** | 33/100 | 1.29e-06 | 9.49e-06 |
| 2d_cross_attn_lam0.1_disp_tskip | folding_fraction | 0.0001 | 0.0013 | +0.0012 ** | 0/100 | 5.46e-07 | 5.15e-12 |

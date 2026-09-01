Baseline: `2d_baseline_lam0.25_disp`  (split: test)

| run | metric | baseline | candidate | Δ | better on | paired t p | wilcoxon p |
|---|---|---|---|---|---|---|---|
| 2d_cross_attn_gated_lam0.25_disp | dice | 0.7544 | 0.7550 | +0.0006 | 56/100 | 4.30e-01 | 2.83e-01 |
| 2d_cross_attn_gated_lam0.25_disp | folding_fraction | 0.0001 | 0.0001 | +0.0000 | 2/100 | 5.47e-01 | 4.63e-01 |
| 2d_cross_attn_gated_lam0.25_svf | dice | 0.7544 | 0.7547 | +0.0002 | 55/100 | 7.67e-01 | 5.38e-01 |
| 2d_cross_attn_gated_lam0.25_svf | folding_fraction | 0.0001 | 0.0000 | -0.0001 | 4/100 | 1.91e-01 | 6.79e-02 |
| 2d_cross_attn_lam0.1_disp | dice | 0.7544 | 0.6934 | -0.0611 ** | 0/100 | 9.93e-41 | 3.90e-18 |
| 2d_cross_attn_lam0.1_disp | folding_fraction | 0.0001 | 0.0004 | +0.0003 ** | 2/100 | 4.30e-03 | 1.62e-04 |
| 2d_cross_attn_gated_lam0.1_disp | dice | 0.7544 | 0.7510 | -0.0034 | 57/100 | 3.66e-01 | 6.90e-01 |
| 2d_cross_attn_gated_lam0.1_disp | folding_fraction | 0.0001 | 0.0011 | +0.0010 ** | 0/100 | 1.02e-06 | 5.15e-12 |

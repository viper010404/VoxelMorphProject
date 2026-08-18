Baseline: `2d_baseline_lam0.25_disp`  (split: test)

| run | metric | baseline | candidate | Δ | better on | paired t p | wilcoxon p |
|---|---|---|---|---|---|---|---|
| 2d_lambda_field_lam0.25_disp | dice | 0.7544 | 0.7513 | -0.0031 | 62/100 | 3.63e-01 | 2.22e-01 |
| 2d_lambda_field_lam0.25_disp | folding_fraction | 0.0001 | 0.0011 | +0.0010 ** | 0/100 | 4.23e-07 | 1.62e-11 |
| 2d_lambda_field_lam0.25_svf | dice | 0.7544 | 0.7496 | -0.0048 | 63/100 | 1.99e-01 | 3.86e-01 |
| 2d_lambda_field_lam0.25_svf | folding_fraction | 0.0001 | 0.0000 | -0.0001 | 4/100 | 1.91e-01 | 6.79e-02 |
| 2d_cross_attn_lam0.1_disp | dice | 0.7544 | 0.6934 | -0.0611 ** | 0/100 | 9.93e-41 | 3.90e-18 |
| 2d_cross_attn_lam0.1_disp | folding_fraction | 0.0001 | 0.0004 | +0.0003 ** | 2/100 | 4.30e-03 | 1.62e-04 |
| 2d_baseline_lam0.1_disp | dice | 0.7544 | 0.7495 | -0.0049 | 59/100 | 1.90e-01 | 7.78e-01 |
| 2d_baseline_lam0.1_disp | folding_fraction | 0.0001 | 0.0012 | +0.0011 ** | 0/100 | 1.87e-06 | 3.49e-11 |

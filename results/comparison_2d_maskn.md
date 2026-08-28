Baseline: `2d_baseline_lam0.25_disp`  (split: test)

| run | metric | baseline | candidate | Δ | better on | paired t p | wilcoxon p |
|---|---|---|---|---|---|---|---|
| 2d_lambda_field_lam0.25_disp_maskn | dice | 0.7544 | 0.7574 | +0.0030 | 73/100 | 1.32e-01 | 3.40e-03 |
| 2d_lambda_field_lam0.25_disp_maskn | folding_fraction | 0.0001 | 0.0010 | +0.0008 ** | 0/100 | 2.94e-08 | 5.18e-13 |
| 2d_lambda_field_lam0.25_svf_maskn | dice | 0.7544 | 0.7568 | +0.0023 | 69/100 | 2.64e-01 | 5.71e-04 |
| 2d_lambda_field_lam0.25_svf_maskn | folding_fraction | 0.0001 | 0.0000 | -0.0001 | 4/100 | 1.91e-01 | 6.79e-02 |

Baseline: `2d_baseline_lam0.25_disp`  (split: test)

| run | metric | baseline | candidate | Δ | better on | paired t p | wilcoxon p |
|---|---|---|---|---|---|---|---|
| 2d_lambda_structure_lam0.25_disp | dice | 0.7544 | 0.7604 | +0.0060 ** | 68/100 | 1.74e-08 | 5.14e-08 |
| 2d_lambda_structure_lam0.25_disp | folding_fraction | 0.0001 | 0.0005 | +0.0003 ** | 1/100 | 8.46e-05 | 8.55e-09 |
| 2d_lambda_structure_lam0.25_svf | dice | 0.7544 | 0.7559 | +0.0015 | 55/100 | 1.80e-01 | 2.76e-01 |
| 2d_lambda_structure_lam0.25_svf | folding_fraction | 0.0001 | 0.0000 | -0.0001 | 4/100 | 1.91e-01 | 6.79e-02 |
| 2d_lambda_structure_lam0.1_disp | dice | 0.7544 | 0.7516 | -0.0028 | 53/100 | 3.32e-01 | 6.93e-01 |
| 2d_lambda_structure_lam0.1_disp | folding_fraction | 0.0001 | 0.0033 | +0.0032 ** | 0/100 | 1.02e-14 | 2.60e-17 |
| 2d_lambda_structure_lam0.1_svf | dice | 0.7544 | 0.7510 | -0.0034 | 57/100 | 3.81e-01 | 9.67e-01 |
| 2d_lambda_structure_lam0.1_svf | folding_fraction | 0.0001 | 0.0000 | -0.0001 | 4/100 | 1.91e-01 | 6.79e-02 |
| 2d_lambda_field_lam0.25_disp_maskn | dice | 0.7544 | 0.7574 | +0.0030 | 73/100 | 1.32e-01 | 3.40e-03 |
| 2d_lambda_field_lam0.25_disp_maskn | folding_fraction | 0.0001 | 0.0010 | +0.0008 ** | 0/100 | 2.94e-08 | 5.18e-13 |
| 2d_lambda_field_lam0.25_svf_maskn | dice | 0.7544 | 0.7568 | +0.0023 | 69/100 | 2.64e-01 | 5.71e-04 |
| 2d_lambda_field_lam0.25_svf_maskn | folding_fraction | 0.0001 | 0.0000 | -0.0001 | 4/100 | 1.91e-01 | 6.79e-02 |

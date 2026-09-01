Baseline: `2d_baseline_lam0.25_disp`  (split: test)

| run | metric | baseline | candidate | Δ | better on | paired t p | wilcoxon p |
|---|---|---|---|---|---|---|---|
| 2d_lambda_prior_lam0.25_disp | dice | 0.7544 | 0.7624 | +0.0080 ** | 78/100 | 2.20e-10 | 2.29e-10 |
| 2d_lambda_prior_lam0.25_disp | folding_fraction | 0.0001 | 0.0007 | +0.0006 ** | 1/100 | 5.17e-07 | 7.69e-10 |
| 2d_lambda_shuf0_lam0.25_disp | dice | 0.7544 | 0.7548 | +0.0004 | 50/100 | 6.54e-01 | 4.77e-01 |
| 2d_lambda_shuf0_lam0.25_disp | folding_fraction | 0.0001 | 0.0004 | +0.0003 ** | 0/100 | 1.60e-05 | 2.38e-09 |
| 2d_lambda_shuf1_lam0.25_disp | dice | 0.7544 | 0.7558 | +0.0014 | 50/100 | 1.93e-01 | 1.15e-01 |
| 2d_lambda_shuf1_lam0.25_disp | folding_fraction | 0.0001 | 0.0003 | +0.0002 ** | 1/100 | 4.70e-04 | 4.40e-06 |
| 2d_lambda_shuf2_lam0.25_disp | dice | 0.7544 | 0.7534 | -0.0010 | 46/100 | 2.71e-01 | 2.94e-01 |
| 2d_lambda_shuf2_lam0.25_disp | folding_fraction | 0.0001 | 0.0004 | +0.0002 ** | 2/100 | 3.02e-04 | 1.99e-06 |

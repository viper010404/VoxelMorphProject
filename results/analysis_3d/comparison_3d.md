Baseline: `3d_baseline_lam0.05_svf`  (split: test)

| run | metric | baseline | candidate | Δ | better on | paired t p | wilcoxon p |
|---|---|---|---|---|---|---|---|
| 3d_baseline_lam0.05_disp | dice | 0.7977 | 0.7925 | -0.0052 ** | 14/100 | 2.64e-15 | 2.23e-13 |
| 3d_baseline_lam0.05_disp | folding_fraction | 0.0000 | 0.0009 | +0.0009 ** | 0/100 | 7.30e-24 | 3.90e-18 |
| 3d_baseline_lam0.1_disp | dice | 0.7977 | 0.7892 | -0.0084 ** | 18/100 | 1.79e-08 | 7.79e-09 |
| 3d_baseline_lam0.1_disp | folding_fraction | 0.0000 | 0.0001 | +0.0001 ** | 0/100 | 3.92e-08 | 3.90e-18 |
| 3d_baseline_lam0.1_svf | dice | 0.7977 | 0.7892 | -0.0084 ** | 14/100 | 3.15e-10 | 2.29e-10 |
| 3d_baseline_lam0.1_svf | folding_fraction | 0.0000 | 0.0000 | -0.0000 | 3/100 | 9.18e-02 | 1.09e-01 |
| 3d_baseline_lam0.1_svf | inverse_consistency | 0.0189 | 0.0117 | -0.0071 ** | 100/100 | 1.05e-79 | 3.90e-18 |
| 3d_baseline_lam0.25_disp | dice | 0.7977 | 0.7669 | -0.0308 ** | 9/100 | 7.19e-23 | 1.11e-14 |
| 3d_baseline_lam0.25_disp | folding_fraction | 0.0000 | 0.0000 | +0.0000 ** | 0/100 | 2.45e-02 | 2.36e-10 |
| 3d_baseline_lam0.25_svf | dice | 0.7977 | 0.7692 | -0.0284 ** | 10/100 | 3.05e-22 | 1.49e-14 |
| 3d_baseline_lam0.25_svf | folding_fraction | 0.0000 | 0.0000 | -0.0000 | 3/100 | 9.18e-02 | 1.09e-01 |
| 3d_baseline_lam0.25_svf | inverse_consistency | 0.0189 | 0.0073 | -0.0116 ** | 100/100 | 1.66e-93 | 3.90e-18 |
| 3d_cross_attn_lam0.1_disp | dice | 0.7977 | 0.6683 | -0.1294 ** | 0/100 | 5.36e-60 | 3.90e-18 |
| 3d_cross_attn_lam0.1_disp | folding_fraction | 0.0000 | 0.0001 | +0.0001 ** | 0/100 | 7.90e-03 | 8.32e-18 |
| 3d_cross_attn_lam0.1_svf | dice | 0.7977 | 0.6699 | -0.1277 ** | 0/100 | 2.55e-58 | 3.90e-18 |
| 3d_cross_attn_lam0.1_svf | folding_fraction | 0.0000 | 0.0000 | -0.0000 | 3/100 | 9.18e-02 | 1.09e-01 |
| 3d_cross_attn_lam0.1_svf | inverse_consistency | 0.0189 | 0.0083 | -0.0105 ** | 100/100 | 3.03e-74 | 3.90e-18 |
| 3d_lambda_field_lam0.05_disp | dice | 0.7977 | 0.7789 | -0.0188 ** | 1/100 | 8.17e-32 | 4.27e-18 |
| 3d_lambda_field_lam0.05_disp | folding_fraction | 0.0000 | 0.0061 | +0.0061 ** | 0/100 | 1.16e-48 | 3.90e-18 |
| 3d_lambda_field_lam0.05_svf | dice | 0.7977 | 0.7832 | -0.0144 ** | 3/100 | 4.99e-26 | 5.77e-18 |
| 3d_lambda_field_lam0.05_svf | folding_fraction | 0.0000 | 0.0000 | +0.0000 ** | 0/100 | 2.52e-04 | 4.33e-12 |
| 3d_lambda_field_lam0.05_svf | inverse_consistency | 0.0189 | 0.0373 | +0.0185 ** | 0/100 | 6.53e-83 | 3.90e-18 |
| 3d_lambda_field_lam0.1_disp | dice | 0.7977 | 0.7849 | -0.0128 ** | 0/100 | 3.47e-30 | 3.90e-18 |
| 3d_lambda_field_lam0.1_disp | folding_fraction | 0.0000 | 0.0021 | +0.0021 ** | 0/100 | 2.42e-32 | 3.90e-18 |
| 3d_lambda_field_lam0.1_svf | dice | 0.7977 | 0.7908 | -0.0069 ** | 8/100 | 1.35e-21 | 6.17e-17 |
| 3d_lambda_field_lam0.1_svf | folding_fraction | 0.0000 | 0.0000 | +0.0000 | 1/100 | 1.01e-01 | 1.52e-02 |
| 3d_lambda_field_lam0.1_svf | inverse_consistency | 0.0189 | 0.0234 | +0.0046 ** | 0/100 | 1.58e-58 | 3.90e-18 |
| 3d_lambda_field_lam0.25_disp | dice | 0.7977 | 0.7858 | -0.0118 ** | 13/100 | 7.04e-15 | 3.34e-12 |
| 3d_lambda_field_lam0.25_disp | folding_fraction | 0.0000 | 0.0002 | +0.0002 ** | 0/100 | 3.32e-09 | 3.90e-18 |
| 3d_lambda_field_lam0.25_svf | dice | 0.7977 | 0.7906 | -0.0070 ** | 20/100 | 4.63e-09 | 1.24e-09 |
| 3d_lambda_field_lam0.25_svf | folding_fraction | 0.0000 | 0.0000 | -0.0000 | 3/100 | 9.18e-02 | 1.09e-01 |
| 3d_lambda_field_lam0.25_svf | inverse_consistency | 0.0189 | 0.0128 | -0.0061 ** | 100/100 | 2.93e-69 | 3.90e-18 |

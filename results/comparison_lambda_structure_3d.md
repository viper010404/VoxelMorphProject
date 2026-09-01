Baseline: `3d_baseline_lam0.05_svf`  (split: test)

| run | metric | baseline | candidate | Δ | better on | paired t p | wilcoxon p |
|---|---|---|---|---|---|---|---|
| 3d_lambda_structure_lam0.1_svf | dice | 0.7977 | 0.7797 | -0.0180 ** | 12/100 | 7.95e-15 | 2.31e-12 |
| 3d_lambda_structure_lam0.1_svf | folding_fraction | 0.0000 | 0.0000 | +0.0000 | 2/100 | 7.52e-01 | 1.00e+00 |
| 3d_lambda_structure_lam0.1_svf | inverse_consistency | 0.0189 | 0.0172 | -0.0016 ** | 90/100 | 9.68e-20 | 1.10e-16 |
| 3d_lambda_field_lam0.1_svf | dice | 0.7977 | 0.7908 | -0.0069 ** | 8/100 | 1.35e-21 | 6.17e-17 |
| 3d_lambda_field_lam0.1_svf | folding_fraction | 0.0000 | 0.0000 | +0.0000 | 1/100 | 1.01e-01 | 1.52e-02 |
| 3d_lambda_field_lam0.1_svf | inverse_consistency | 0.0189 | 0.0234 | +0.0046 ** | 0/100 | 1.58e-58 | 3.90e-18 |

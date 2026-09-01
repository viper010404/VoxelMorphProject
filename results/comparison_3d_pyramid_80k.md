Baseline: `3d_baseline_lam0.05_svf_s80k`  (split: test)

| run | metric | baseline | candidate | Δ | better on | paired t p | wilcoxon p |
|---|---|---|---|---|---|---|---|
| 3d_pyramid_lam0.05_svf_nods_s80k | dice | 0.8054 | 0.8144 | +0.0090 ** | 87/100 | 3.95e-18 | 7.73e-14 |
| 3d_pyramid_lam0.05_svf_nods_s80k | folding_fraction | 0.0000 | 0.0000 | -0.0000 | 7/100 | 1.30e-01 | 2.44e-02 |
| 3d_pyramid_lam0.05_svf_nods_s80k | inverse_consistency | 0.0223 | 0.0268 | +0.0045 ** | 0/100 | 2.03e-21 | 3.90e-18 |

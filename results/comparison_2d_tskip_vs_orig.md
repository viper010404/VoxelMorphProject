Baseline: `2d_cross_attn_lam0.1_svf`  (split: test)

| run | metric | baseline | candidate | Δ | better on | paired t p | wilcoxon p |
|---|---|---|---|---|---|---|---|
| 2d_cross_attn_lam0.1_svf_tskip | dice | 0.6686 | 0.7450 | +0.0765 ** | 98/100 | 3.51e-37 | 4.96e-18 |
| 2d_cross_attn_lam0.1_svf_tskip | folding_fraction | 0.0000 | 0.0000 | +0.0000 | 0/100 | 1.00e+00 | 1.00e+00 |
| 2d_cross_attn_lam0.1_svf_tskip | inverse_consistency | 0.0138 | 0.0190 | +0.0051 ** | 3/100 | 2.37e-23 | 7.79e-18 |

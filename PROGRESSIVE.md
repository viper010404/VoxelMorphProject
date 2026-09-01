# Progressive coarse-to-fine VoxelMorph

Where the progressive extension stands. Numbers are mean Dice on the fixed 100 test pairs.

## The model

`variant='pyramid'`, `pyramid_progressive=True`, `deep_supervision=False`. Same UNet as the
baseline — identical encoder, decoder and widths. Two changes, both in how the field is produced:

1. **A flow head per decoder level.** Each of the 5 levels emits its own displacement field.
   The running field is upsampled to the next level and the level's residual is added.
   Magnitudes are rescaled on upsample, since a field is measured in voxels.
2. **Progressive warping.** Before a level's decoder block runs, its skip features are warped by
   the field accumulated so far. Each level only has to explain the residual left by the coarser
   ones.

The loss is unchanged: MSE + λ·gradient smoothness, applied once, to the final field.
Deep supervision is off — it was tested and it hurts.

Cost: +2,276 parameters in 2D (+2.0%), +10,137 in 3D (+3.0%). Roughly +15% training time in 3D.

## baseline vs heads-only vs progressive

`heads-only` (`pyramid_progressive=False`) keeps all five flow heads and every extra parameter,
and removes *only* the skip warping. It is the control that says which of the two changes matters.

3D, 80k steps, single seed, zero folding everywhere:

| | λ=0.025 | λ=0.05 |
|---|---|---|
| baseline | 0.8066 | 0.8054 |
| heads-only | 0.8072 (+0.0006) | 0.8090 (+0.0036) |
| **progressive** | **0.8128 (+0.0062)** | **0.8144 (+0.0090)** |

The extra heads on their own buy little. The warping carries the effect.

## 2D

λ=0.25, 60k steps, 4 seeds per arm:

| | mean | sd | runs |
|---|---|---|---|
| baseline | 0.7566 | 0.0033 | 0.7516, 0.7582, 0.7583, 0.7584 |
| progressive | 0.7597 | 0.0003 | 0.7595, 0.7595, 0.7596, 0.7602 |

Mean difference **+0.0031, p = 0.161** — not significant across runs.

At λ=0.05 there is nothing: 0.7369 vs 0.7368, p = 0.924.

## The variance signal

The 2D mean is not significant, but the variance difference is:

**92× lower run-to-run variance, F-test p = 0.0038.**

The baseline's four seeds span 0.7516–0.7584; progressive's span 0.7595–0.7602. The low baseline
run is not a crash — it converged normally and landed in a worse solution. Almost the whole
+0.0031 mean gap is the removal of that bad tail rather than a lift of the good runs.

This does not appear at λ=0.05, where the baseline is already stable (sd 0.0007). The effect
shows up exactly where the baseline is unstable, which is what an optimisation fix looks like.

## Wall time

Progressive costs ~15% more per step in 3D but reaches a given quality in far fewer steps.
Timed back to back on one idle L40S: baseline 0.1311 s/step, progressive 0.1503 s/step.

At λ=0.025, the baseline's *final* score after the full 80k is 0.8066. Progressive's test trace
crosses that at about 22.5k steps:

| | steps | wall time | Dice |
|---|---|---|---|
| baseline, full run | 80,000 | 2.91 h | 0.8066 |
| progressive, to equal quality | ~22,500 | **0.94 h** | 0.8066 |
| progressive, full run | 80,000 | 3.34 h | 0.8128 |

So matching the baseline takes **about a third of the wall time (3.1×)**, and spending the full
budget instead buys +0.0062 on top. Useful when compute is the binding constraint.

## Reading the two dimensionalities together

3D gains more than 2D. 3D at 80k steps has seen far fewer training pairs than 2D at 20k
(batch 1 vs batch 16), so the two are measured at different points on the convergence curve.
Progressive's advantage is largest early and shrinks with training — in 3D it fell from +0.027
at 4k steps to about +0.011 by 40k, then held.

## Open issues

- **3D is single-seed.** +0.0090 is roughly 5× the baseline's seed sd, but that sd is borrowed
  from a different config and progressive's own seed variance is unmeasured. This is the main
  gap. 2–3 seeds at λ=0.05/80k would close it; measuring seed sd at 10k gives a conservative
  upper bound for ~1/8 the compute.
- **Per-pair p-values are not a valid test between architectures.** Two baseline runs differing
  only in seed give p<0.05 on the paired per-pair test in 8 of 10 comparisons, once at p=1.3e-13.
  The unit of replication is the run. Use run-level tests with n≥4.
- **2D's mean gain is not established** (p=0.161 at n=4). Only the variance result is.
- λ interacts with everything; λ=0.025 and λ=0.05 give different answers in 3D.

## Reproducing

```bash
./scripts/restore_dataset.sh both     # download, cache, and verify the splits still match
./.venv/bin/python -m project.run_experiments --ndim 3 --variants pyramid \
    --sweep-variants pyramid --lambdas 0.05 --integration-steps 7 --steps 80000 \
    --no-deep-supervision --test-every 10000 --gpus 0
```

`notebooks/voxelmorph_progressive.ipynb` is the self-contained 2D version and runs on Colab.

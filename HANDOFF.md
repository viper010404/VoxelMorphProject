# VoxelMorph extension — status and run guide

Operational handoff: where the project is, how to prepare data, and what to run.
**Read §5 (Traps) before debugging anything** — several entries there cost hours to find.

---

## 1. What this is

**Course:** Technion 336033, *Medical Images Processing with Deep Learning*.
**Paper:** Balakrishnan et al., *VoxelMorph*, IEEE TMI 2019 — [`1809.05231v3.pdf`](1809.05231v3.pdf).
**Deadline:** 2026-08-20. Submit `id1_id2.zip` with code and a max 6-page PDF report. The
implementation must run on Google Colab; proof-of-concept scale is explicitly fine.

The paper learns an amortised `g_θ(f,m) = u`: a UNet takes the concatenated fixed and moving
images, emits a displacement field, a spatial transformer warps the moving image, and the loss is
`L_sim(f, m∘φ) + λ·L_smooth(φ)`. Reference results: Dice ≈ 0.75, 0.2–0.4% of voxels with
non-positive Jacobian determinant, and a flat sensitivity to λ.

**Three extensions, each measured against the unmodified baseline:**

| | branch | idea |
|---|---|---|
| A | **λ-field** | per-voxel regularisation weight instead of one global λ |
| B | **cross-attention** | two-stream shared encoder, cross-attention at the bottleneck |
| C | **uncertainty** | 5-seed ensemble, per-voxel displacement variance |

`--integration-steps` (SVF / diffeomorphic) is an **orthogonal flag on all branches**, not a
branch of its own.

**Evaluation rule that governs everything:** all models are scored on one identical, persisted
list of `(fixed, moving)` pairs, and every comparison is **paired** against the baseline with a
significance test. Between-pair variance exceeds the effect size, so unpaired means would hide it.

---

## 2. Where we are

Upstream `voxelmorph` (branch `dev`, v0.3.3) is **unmodified**; all project code is additive under
`project/`. **155 tests pass** (45 project + 110 upstream), `pycodestyle` clean.

```
project/
  prepare_data.py     neurite-OASIS -> cache (.npz for 2D, memory-mapped .npy dir for 3D)
  data.py             dataset, training pair sampler, FIXED eval pair list, label policies
  configs.py          ExperimentConfig + the sweep matrix
  models.py           VxmBaseline / VxmLambdaField / VxmCrossAttention
  losses.py           per-voxel-weighted smoothness + loss assembly
  metrics.py          Dice, folding (|J_φ|<=0), inverse consistency
  rangefind.py        locate the useful λ range for a given dimensionality
  train.py            one run; model selection on validation
  evaluate.py         checkpoint -> per-pair results JSON
  compare.py          paired stats vs baseline
  analyse.py          per-structure table, λ-map profile, Dice/folding trade-off
  run_experiments.py  queue the matrix across GPUs (supports --skip-existing)
```

All models share one interface — `forward(source, target) -> dict` with `displacement`,
`warped_source`, and optionally `velocity` / `lambda_map` — so nothing downstream knows which
branch it is scoring.

### Done

- **2D bake-off complete**, all 14 runs, evaluated on 100 fixed test pairs (14 structures).
  Initial Dice 0.6747, identical across runs.
- **Both datasets prepared**: `data/oasis2d.npz` (9.5 MB) and `data/oasis3d_cache/` (17 GB).
- **3D path smoke-tested end to end**: 160 ms/iter, eval 3.0 min per run, 33 structures.

| variant | best 2D run | Dice | Δ vs initial | folding % |
|---|---|---|---|---|
| baseline | λ=0.25, disp | **0.7544** | +0.0798 | 0.013 |
| λ-field | λ=0.25, disp | 0.7513 | +0.0767 | 0.112 |
| cross-attn | λ=0.1, disp | 0.6934 | +0.0187 | 0.039 |

Paired against the best baseline (n=100): λ-field **−0.0031, p=0.36 (not significant)**, winning
on 62/100 pairs; cross-attention **−0.0611, p≈1e-40 (significantly worse)**. The baseline's own
λ=0.1 vs λ=0.25 difference is also insignificant (p=0.19), reproducing the paper's λ-insensitivity.

Per-structure, the λ-field redistributes strongly and significantly (all p<5e-3): **better** on
lateral ventricles (+0.024/+0.019) and brain stem (+0.021), **worse** on putamen (−0.054),
ventral DC (−0.020/−0.016) and thalamus (−0.015/−0.013). Full tables via `project.analyse`.

### Not done

- 3D runs for all three branches ← **in progress, see §4**
- 5-seed ensembles (branch C) — not started
- Cross-attention evaluation stratified by initial misalignment (its actual hypothesis)
- Colab notebook (2D, self-contained) — a submission requirement
- Report

### Artifacts

- `results/<run>/` — `config.json`, `best.pt`, `final.pt`, `history.json`, `eval_test.json`
  (per-pair rows, not just aggregates)
- `results/comparison.md` — paired comparison table
- `results_failed_meanonly/` — evidence from a superseded λ-field formulation (see §5, trap 13)

---

## 3. Setup and data

### Environment

```bash
python3 -m venv .venv

# CRITICAL: plain `pip install torch` pulls a CUDA 13 build that a 12.8 driver rejects
# ("The NVIDIA driver on your system is too old"). Match the index to `nvidia-smi`.
./.venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cu128
./.venv/bin/pip install neurite scikit-image nibabel scipy h5py packaging pytest pycodestyle

./.venv/bin/python -c "import torch, neurite; print(torch.__version__, torch.cuda.is_available())"
./.venv/bin/python -m pytest tests/ -q     # expect 155 passed, 1 skipped
```

`data/` and `results/` are gitignored — `rsync` the tree or re-fetch as below.

### Data

Both are neurite-OASIS: OASIS-1 reprocessed by the paper's lab, already skull-stripped,
bias-corrected and affinely aligned (VoxelMorph assumes affine alignment and learns only the
nonlinear residual). 414 subjects; splits train 100 / val 50 / test 100, seeded.

**2D (24 MB)** — the bake-off and the Colab deliverable:

```bash
mkdir -p data && cd data
curl -sSLO https://surfer.nmr.mgh.harvard.edu/ftp/data/neurite/data/neurite-oasis.2d.v1.0.tar
md5sum neurite-oasis.2d.v1.0.tar    # c9ae5864f250c7e4b8d83a104e51ae8e
tar -xf neurite-oasis.2d.v1.0.tar && cd ..
./.venv/bin/python -m project.prepare_data --data-dir data --out data/oasis2d.npz
```

**3D (6.6 GB)** — 160×192×224, the paper's exact crop, 35 labels:

```bash
cd data
curl -sSLO https://surfer.nmr.mgh.harvard.edu/ftp/data/neurite/data/neurite-oasis.v1.0.tar
md5sum neurite-oasis.v1.0.tar       # 081392a8150ff99ab7a64a9ded377835
mkdir -p oasis3d
# only template-space files are needed: 1.2 GB instead of 6.6 GB
tar -xf neurite-oasis.v1.0.tar -C oasis3d --wildcards \
    '*/aligned_norm.nii.gz' '*/aligned_seg35.nii.gz' '*/aligned_seg4.nii.gz' \
    'seg35_labels.txt' 'subjects.txt'
cd ..
./.venv/bin/python -m project.prepare_data --ndim 3 \
    --data-dir data/oasis3d --out data/oasis3d_cache
```

Cite the HyperMorph paper and reference the OASIS Data Use Agreement (oasis-brains.org).

---

## 4. What to run

### Measured timings

| config | ms/iter | 20k steps | + eval (100 pairs) |
|---|---|---|---|
| 2D 160×192, b=16 | 4.9 | ~1.6 min | ~0.5 min |
| 3D 160×192×224, b=1 | 160 | ~53 min | ~3 min |
| 3D + SVF (7 steps) | ~180 | ~60 min | ~3 min |

An L40S is the same AD102 silicon as the RTX 6000 Ada with ~10% less memory bandwidth — expect
parity or marginally slower per GPU, so scale by GPU count, not by card.

### 3D, all three branches (the current goal)

**Step 1 — range-find λ first.** λ≈0.25 was tuned in 2D; the 3D smoothness term averages over
three gradient components instead of two, so the optimum shifts. Do not skip this (trap 2).

```bash
# split across two GPUs
CUDA_VISIBLE_DEVICES=0 ./.venv/bin/python -m project.rangefind --ndim 3 \
    --lambdas 0.05 0.1 0.25 --steps 3000 --eval-pairs 12 --out results/rangefind_3d_a.json &
CUDA_VISIBLE_DEVICES=1 ./.venv/bin/python -m project.rangefind --ndim 3 \
    --lambdas 0.5 1.0 2.0  --steps 3000 --eval-pairs 12 --out results/rangefind_3d_b.json &
```

~25 min. Pick the **knee**: the smallest λ whose folding is still near the paper's 0.2–0.4%,
not simply the highest Dice.

**Step 2 — the matrix.** Edit the `lambdas` default in `configs.build_matrix` to the three values
bracketing the knee, then:

```bash
./.venv/bin/python -m project.run_experiments --ndim 3 --gpus 0 1 --dry-run   # preview
./.venv/bin/python -m project.run_experiments --ndim 3 --gpus 0 1 --skip-existing
```

14 runs (baseline ×6, λ-field ×6, cross-attn ×2) at ~60 min each including validation and
evaluation:

| GPUs | wall clock |
|---|---|
| 2 | ~7.0 h — fits an overnight window |
| 6 | ~2.4 h |

Add GPU indices to widen: `--gpus 0 1 2 3 4 5`. One run per GPU, independent processes (not DDP —
these are different configurations, so this scales perfectly with no gradient sync).

`--skip-existing` makes the matrix resumable: interrupted runs are retried, finished ones skipped.

### Ensembles (branch C, not yet started)

```bash
./.venv/bin/python -m project.run_experiments \
    --ensemble-of 2d_baseline_lam0.25_disp --seeds 0 1 2 3 4 --gpus 0 1
```

~30 min in 2D. Gives per-voxel uncertainty, and `analyse lambda-map` reports the across-seed
correlation of the λ profile (i.e. whether the learned map is reproducible).

### Analysis

```bash
./.venv/bin/python -m project.compare --baseline results/<baseline_run> \
    --runs results/<other_runs...> --split test
./.venv/bin/python -m project.analyse tradeoff      --runs results/3d_*
./.venv/bin/python -m project.analyse per-structure --baseline results/A --candidate results/B
./.venv/bin/python -m project.analyse lambda-map    --runs results/*lambda_field*seed*
```

---

## 5. Traps

1. **torch CUDA build.** `pip install torch` defaults to CUDA 13; on a 12.8 driver
   `torch.cuda.is_available()` is False with a misleading "driver too old" warning. Use
   `--index-url .../cu128`.

2. **λ does not transfer** — not from the paper, not from 2D to 3D. Our smoothness is a *mean*
   over voxels and components, a different normalisation to the paper's quoted 0.01. In 2D the
   optimum is λ≈0.1–0.25 (0.01 → 3.25% folding; 0.005 → 5.5% folding and *worse than doing
   nothing*; 5.0 → too stiff). **Always run `project.rangefind` for a new dimensionality.**

3. **Evaluate on ≥100 pairs.** Per-pair Dice std is ~0.15 in 2D. A 20-pair evaluation reported
   **+0.042** for a model whose true effect at 100 pairs was **−0.011**.

4. **Label policy differs by dimensionality.** The paper's rule (≥100 voxels in *every* subject)
   keeps 31/35 structures in 3D — used unchanged there. On a 2D slice it keeps only 8/24,
   discarding hippocampus, ventricles and putamen, so 2D uses a **median ≥100 px** rule
   (14 structures). Handled automatically by `data.default_label_policy`.

5. **`compose` batch-detection bug (upstream).** `nn/functional.py::compose` infers a batch axis
   via `shape[0] != ndim - 1`, misreading batch size **3 in 2D** and **4 in 3D**.
   `metrics.inverse_consistency` loops per sample to avoid it.

6. **`coords_to_disp` raises `NotImplementedError`** upstream. Inverse consistency goes through
   `compose([fwd, bwd])`, whose result *is* the deviation from identity.

7. **`jacobian_determinant` is numpy, channels-LAST, unbatched** — the opposite layout to
   `nn/functional`. `metrics.folding` permutes and loops.

8. **`py/utils.py::dice` returns 0.0, not NaN**, for a label absent from *both* inputs.
   `metrics.dice_per_structure` masks those to None.

9. **UNet spatial dims must be divisible by `2**len(nb_features)`** (32 by default) or the decoder
   skip concat fails. 160×192(×224), 96³, 64³ fine; 80×96×112 not.

10. **`neurite` has no dropout** in `BasicUNet`/`ConvBlock` — hence ensembles rather than
    MC-dropout (which would also need dropout active during training to be valid).

11. **`ne.nn.modules.NCC` returns *squared* correlation, higher-is-better** — negate if used as a
    loss. `MSE` and `SpatialGradient(reduction=None)` cannot yield per-voxel maps (the latter
    returns a *list* of differently-shaped tensors), which is why `losses.spatial_smoothness` is
    implemented directly; a test asserts it equals `SpatialGradient` when unweighted.

12. **`scripts/register.py` is broken upstream** (nonexistent `vxm.networks.VxmDense`) and
    `scripts/train.py` hardcodes an unreachable cluster path. `project/` uses neither.

13. **λ-field: bound the weight before normalising.** Constraining only `mean(w)=1` is evadable —
    minimising `sum(w_i·g_i)` under a mean constraint drives `w→0` exactly where the gradient is
    largest. Measured: weight spanning 1e-5 to 6.4 with mean exactly 1, folding 5.0% vs the
    baseline's 0.56%. Current code bounds via sigmoid to [0.5, 2] *then* normalises. Evidence in
    `results_failed_meanonly/`.

14. **Don't pipe the launcher to `head`.** `run_experiments ... | head -12` sends SIGPIPE and
    kills the orchestrator mid-queue (child runs continue orphaned). Redirect to a file instead.

---

## 6. Open question — the λ-field

At matched λ the λ-field is statistically tied with the baseline overall, but redistributes
smoothing strongly per structure — and in the *opposite* direction to the original hypothesis. We
expected it to protect rigid deep grey matter; it instead relaxes smoothing on ventricles and
brain stem and sacrifices thalamus, putamen and ventral DC.

Likely cause: training optimises **intensity MSE**, so the field relaxes smoothing where intensity
error is largest (large, high-contrast ventricles). Nothing in the unsupervised loss encodes which
anatomy should be rigid — Dice is only ever an evaluation metric.

Also note that at equal λ the two models do **not** spend equal effective regularisation (the
λ-field buys itself a weaker regulariser), so equal-λ is arguably the wrong comparison.

Things to try, in order of promise:

1. **Matched-folding comparison** — sweep λ for both and compare Dice at equal folding rather
   than equal λ. Cheap; the runs mostly exist.
2. **Add the auxiliary segmentation loss** (paper eq. 10) so the weight field gets an anatomical
   signal. Directly addresses the diagnosed cause; `py/generators.py::semisupervised` exists
   upstream as a starting point.
3. **Tighten the bound** to e.g. [0.8, 1.25] so the effective budget is nearly fixed.
4. **Condition the weight on the fixed image only**, so it cannot co-adapt to the field it is
   producing.
5. **Normalise by the penalty-weighted mean**, holding the total penalty constant rather than the
   mean weight.

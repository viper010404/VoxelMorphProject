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
`project/`. **171 tests pass** (61 project + 110 upstream), `pycodestyle` clean.

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
  uncertainty.py      branch C: ensemble spread, ensemble-vs-member, uncertainty-vs-error
  run_experiments.py  queue the matrix across GPUs (supports --skip-existing)
```

All models share one interface — `forward(source, target) -> dict` with `displacement`,
`warped_source`, and optionally `velocity` / `lambda_map` — so nothing downstream knows which
branch it is scoring.

### Done

- **2D bake-off complete**, all 14 runs, evaluated on 100 fixed test pairs (14 structures).
  Initial Dice 0.6747, identical across runs.
- **Both datasets prepared**: `data/oasis2d.npz` (9.5 MB) and `data/oasis3d_cache/` (17 GB).
  Note the 3D cache is gitignored — rebuild it on a new machine per §3, ~10 min.
- **3D path smoke-tested end to end**: 160 ms/iter, eval 3.0 min per run, 33 structures.
- **3D λ range-find complete: λ=0.1** (see §4). The default sweep bracket needs no change.

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

### In flight — launched 2026-08-18 03:41, unattended

Four detached jobs (all `PPID 1`, so they survive a lost shell). Logs are in the repo root.

| job | log | what it does |
|---|---|---|
| 3D chain | `overnight.log` | 3D matrix (14) → baseline ensemble (5) → λ-field ensemble (5) |
| 2D mask-norm sweep | `maskn2d.log` | 6 runs of the λ-field fix below |
| post-chain analysis | `analysis3d.log` | waits for the chain, then runs every §4 analysis into `results/analysis_3d/` |
| 3D mask-norm sweep | `maskn3d.log` | waits for the analysis, then 6 runs of the fix in 3D |

Measured 3D throughput on 6×L40S: **112 ms/iter**, ~45 min per run including eval (SVF ~25%
slower), so the matrix is ~1.8 h rather than the 2.4 h estimated below. At step 4000 the ordering
is already λ=0.1 (0.6996) > λ=0.05 (0.6878) > λ=0.25 (0.6731), consistent with the range-find.

**2D mask-norm sweep finished 04:14, all 6 runs ok** — and the fix flips the sign of the effect.
The two best runs in the whole 2D table are now the mask-normalised λ-field at λ=0.25:

| run | Dice | folding % |
|---|---|---|
| `2d_lambda_field_lam0.25_disp_maskn` | **0.7574** | 0.096 |
| `2d_lambda_field_lam0.25_svf_maskn` | **0.7568** | **0.000** |
| `2d_baseline_lam0.25_disp` (previous best) | 0.7544 | 0.013 |
| `2d_lambda_field_lam0.25_disp` (whole-image norm) | 0.7513 | 0.112 |

so the λ-field goes from −0.0031 *below* the baseline to +0.0030 *above* it, and the SVF variant
gets there with zero folding. **Do not report this as a win yet:** +0.0030 is inside the ±0.0054
seed noise measured from the 5-seed baseline ensemble, so it needs `project.compare` for the
paired test and ideally its own 5 seeds. Folding at matched λ is lower than the whole-image
λ-field everywhere, but still above the baseline — check brain-masked folding per the caveat in §6.

**Tomorrow:** read `results/analysis_3d/`, run the paired test on the 2D mask-norm runs, then see
whether the 3D mask-norm sweep confirms it. Nothing needs relaunching unless a log shows `FAILED`.

### Not done

- Cross-attention evaluation stratified by initial misalignment (its actual hypothesis) —
  still unimplemented; needs the pre-warping step
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

**The download host blocks curl's default user-agent** — see trap 15. Every `curl` below
therefore sets `-A "$UA"`:

```bash
UA="Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120 Safari/537.36"
```

**2D (24 MB)** — the bake-off and the Colab deliverable:

```bash
mkdir -p data && cd data
curl -sSLO -A "$UA" https://surfer.nmr.mgh.harvard.edu/ftp/data/neurite/data/neurite-oasis.2d.v1.0.tar
md5sum neurite-oasis.2d.v1.0.tar    # c9ae5864f250c7e4b8d83a104e51ae8e
tar -xf neurite-oasis.2d.v1.0.tar && cd ..
./.venv/bin/python -m project.prepare_data --data-dir data --out data/oasis2d.npz
```

**3D (6.6 GB)** — 160×192×224, the paper's exact crop, 35 labels:

```bash
cd data
curl -sSLO -A "$UA" https://surfer.nmr.mgh.harvard.edu/ftp/data/neurite/data/neurite-oasis.v1.0.tar
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

**Step 1 — λ range-find: DONE, λ=0.1.** No need to repeat it.

Measured with `project.rangefind --ndim 3` (3000 steps, 12 val pairs, 33 structures, initial
Dice 0.5363):

| λ | Dice | Δ vs initial | folding % | \|disp\| |
|---|---|---|---|---|
| 0.05 | 0.6438 | +0.1075 | 0.193 | 1.015 |
| **0.10** | **0.6523** | **+0.1161** | **0.059** | 0.936 |
| 0.25 | 0.6350 | +0.0988 | 0.006 | 0.808 |
| 0.50 | 0.6282 | +0.0919 | 0.000 | 0.714 |
| 1.00 | 0.6047 | +0.0684 | 0.000 | 0.607 |

A clean interior optimum at **λ=0.1**, so `build_matrix`'s default bracket of (0.05, 0.1, 0.25)
straddles the peak and **needs no edit**. Note the optimum differs from 2D's 0.25 — confirming
that λ does not transfer between dimensionalities (trap 2).

Also note 3D improves Dice by **+0.116** against 2D's best of +0.080, from a lower starting point
(0.5363 vs 0.6747). A single 2D slice cannot match anatomy that moves through the slice plane, so
the 3D numbers are the ones to report.

**Step 2 — run the matrix.**

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

Per-voxel variance across independently seeded models gives the uncertainty estimate, and for the
λ-field it also answers whether the learned weight map is *reproducible* rather than an artefact
of one initialisation — `analyse lambda-map` reports the across-seed correlation of the
per-structure profile.

```bash
# 2D (~30 min on 2 GPUs)
./.venv/bin/python -m project.run_experiments \
    --ensemble-of 2d_baseline_lam0.25_disp --seeds 0 1 2 3 4 --gpus 0 1

# 3D (run after the matrix, so the base run exists)
./.venv/bin/python -m project.run_experiments \
    --ensemble-of 3d_lambda_field_lam0.1_disp --seeds 0 1 2 3 4 --gpus 0 1 2 3 4 5
```

---

## 4b. Overnight run plan (6-GPU box)

Ordered, unattended, resumable. Steps 1–2 are prerequisites; step 3 onward is the actual run.
Every stage takes `--skip-existing`, so if anything dies the same command resumes it.

**1. Rebuild the 3D cache (~15 min).** `data/` is gitignored and does not travel with the repo —
see §3. This must finish before anything else.

**2. Verify the environment (~1 min).**

```bash
./.venv/bin/python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
./.venv/bin/python -m pytest tests/ -q          # expect 155 passed, 1 skipped
```

If `cuda` is False, it is the CUDA-13 wheel trap — §5 trap 1.

**3. The overnight chain.** λ=0.1 is already established (§4 step 1), so this needs no tuning:

```bash
GPUS="0 1 2 3 4 5"
PY=./.venv/bin/python

nohup sh -c "
  $PY -m project.run_experiments --ndim 3 --gpus $GPUS --skip-existing &&
  $PY -m project.run_experiments --ensemble-of 3d_baseline_lam0.1_disp \
      --seeds 0 1 2 3 4 --gpus $GPUS &&
  $PY -m project.run_experiments --ensemble-of 3d_lambda_field_lam0.1_disp \
      --seeds 0 1 2 3 4 --gpus $GPUS
" > overnight.log 2>&1 &
```

Do **not** pipe this to `head`/`tail` — that kills the orchestrator mid-queue (§5 trap 14).
Redirect to a file, as above.

| stage | runs | 6 GPUs | 2 GPUs |
|---|---|---|---|
| 3D matrix (baseline ×6, λ-field ×6, cross-attn ×2) | 14 | ~2.4 h | ~7.0 h |
| baseline ensemble, 5 seeds | 5 | ~0.9 h | ~2.5 h |
| λ-field ensemble, 5 seeds | 5 | ~0.9 h | ~2.5 h |
| **total** | **24** | **~4.2 h** | **~12 h** |

**4. Monitor.**

```bash
tail -f overnight.log                      # orchestrator progress
ls results/3d_*/eval_test.json | wc -l     # completed runs
grep val_dice results/logs/3d_*.log | tail # per-run training curves
```

**5. In the morning**, run the analyses in §4 Analysis below. Start with `tradeoff` for the
overview, then `compare` for the paired statistics against the best baseline.

**Not included and not runnable yet:** the cross-attention evaluation stratified by initial
misalignment. That needs a pre-warping step (warp the moving image by a synthetic field of known
magnitude, then register) which is **not implemented**. Cross-attention was significantly worse in
2D, so this is the branch to drop if time is short.

---

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

15. **The dataset host 403s curl's default user-agent.** `HEAD` returns 200 while `GET` returns a
    199-byte 403 page, so the download "succeeds" and leaves an HTML file named `.tar`. Pass `-A`
    with a browser user-agent (§3). Verify the md5 rather than trusting the exit status.

16. **`compare.py --out` defaults to `results/comparison.md`** and silently overwrites it, so
    running it with a non-canonical baseline destroys the 2D table. Always pass an explicit
    `--out`. Recoverable with `git checkout results/comparison.md` — the per-pair eval JSONs
    under `results/` are tracked, so the experimental record itself survives.

17. **The 2D matrix runs have no checkpoints.** `**/*.pt` is gitignored, so only `config.json`,
    `history.json` and `eval_test.json` travel between machines. Anything checkpoint-based
    (`analyse lambda-map`, `project.uncertainty`) needs the run retrained first — ~2 min each in
    2D. `compare` and `tradeoff` read the eval JSONs and work unchanged.

18. **`run_experiments --python` used to default to an absolute scratchpad path** left over from
    whichever machine last ran it, so the §4b chain failed instantly on a new box. It now
    defaults to `sys.executable`: launch the orchestrator with `./.venv/bin/python -m
    project.run_experiments ...` and the child processes inherit that venv.

19. **A rebuilt `data/` can silently invalidate comparability.** The fixed evaluation pairs live
    in `data/eval_pairs_<split>.json`, which is gitignored, and are regenerated from
    `splits.json` with seed 1234. After a full rebuild they were verified identical to the pairs
    recorded in the historical `eval_test.json` — but re-check after any change to
    `prepare_data.py`, or paired comparisons quietly compare different pairs.

---

## 6. Open question — the λ-field

At matched λ the λ-field is statistically tied with the baseline overall, but redistributes
smoothing strongly per structure — and in the *opposite* direction to the original hypothesis. We
expected it to protect rigid deep grey matter; it instead relaxes smoothing on ventricles and
brain stem and sacrifices thalamus, putamen and ventral DC.

**Diagnosed 2026-08-18 — the λ-field was evading the regulariser a second way.** Bounding the
weight to [0.5, 2] (trap 13) stops `w→0`, but the map is normalised to mean 1 over the *whole
image*, and ~60% of an OASIS slice is background air where smoothing the displacement costs
nothing. The field satisfies the constraint by parking weight in the background:

| region | mean weight |
|---|---|
| outside the brain (59.5% of voxels) | **1.38** |
| inside the brain | **0.42** |

a 3.3× split that nearly saturates the bound ratio. The per-structure profile is otherwise
**flat** — every one of the 14 structures sits at 0.415–0.429, so there is essentially no
redistribution *within* anatomy. The brain runs at an effective λ of 0.42·λ, i.e. **0.105 at a
nominal 0.25**.

That predicts the λ-field at λ=0.25 is just the baseline at λ=0.1, and it is — paired, n=100:
Dice **+0.0018 (p=0.11)**, folding **−0.0001 (p=0.17, Wilcoxon p=0.93)**. Both indistinguishable.
See `results/comparison_matched_effective_lambda.md`. So the per-structure redistribution reported
above is largely an artefact of comparing at mismatched effective regularisation, and the honest
statement is that **the λ-field as originally formulated is not a different model, just a weaker
λ**.

**Fix implemented:** `ExperimentConfig.lambda_mask_norm` / `--lambda-mask-norm` normalises the
weight to mean 1 over a brain mask derived from the *input images* (`source > 0 | target > 0` —
the data is skull-stripped, and using the segmentation instead would leak labels and make the
method semi-supervised). Verified on a trained checkpoint: brain mean **1.000**, background
**0.520** — the optimiser now pushes background to the floor, since raising it only adds penalty.
The brain's budget is pinned to the baseline's and the field can only redistribute within
anatomy, which is the hypothesis actually under test. Off by default, so every earlier run
reproduces unchanged.

**Caveat to check when reading the mask-norm results:** the penalty still averages over all
voxels, so the background is now *less* regularised than the baseline's. `metrics.folding` is
computed over the whole volume, so a higher folding % for a `_maskn` run may be background
folding rather than a brain-level problem — compare folding restricted to the brain mask before
concluding anything from it.

The remaining note below still stands: nothing in the unsupervised loss encodes which anatomy
should be rigid, since training optimises intensity MSE and Dice is only ever an evaluation
metric.

Things to try, in order of promise:

1. **Matched-folding comparison** — sweep λ for both and compare Dice at equal folding rather
   than equal λ. Cheap; the runs mostly exist. (Partly done: the matched-*effective*-λ comparison
   above is the same argument and already shows a tie.)
2. **Add the auxiliary segmentation loss** (paper eq. 10) so the weight field gets an anatomical
   signal. Directly addresses the diagnosed cause; `py/generators.py::semisupervised` exists
   upstream as a starting point.
3. **Tighten the bound** to e.g. [0.8, 1.25] so the effective budget is nearly fixed. Less
   necessary now that mask normalisation pins the brain's budget directly.
4. **Condition the weight on the fixed image only**, so it cannot co-adapt to the field it is
   producing.
5. **Normalise by the penalty-weighted mean**, holding the total penalty constant rather than the
   mean weight. — superseded by mask normalisation, which fixes the same confound more directly.

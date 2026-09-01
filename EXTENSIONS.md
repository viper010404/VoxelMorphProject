# Extension ledger

One line per direction: what it claims, what it measured, what to do next.
Companion to `HANDOFF.md` (operational) — this file is the *decision* record.
Every run's per-pair numbers live in `results/<run>/eval_test.json`; the tables below are
summaries, not the source of truth.

---

## 1. Measured — completed directions

All Dice figures are means over the same 100 fixed test pairs; `p` is a paired t-test vs the
baseline named in the row group. `**` = significant.

### 2D (best baseline `2d_baseline_lam0.25_disp`, Dice 0.7544, folding 0.013%)

| variant | best run | Dice | Δ | p | folding % | verdict |
|---|---|---|---|---|---|---|
| baseline | λ=0.25 disp | 0.7544 | — | — | 0.013 | reference |
| λ-field (mask-norm) | λ=0.25 disp | 0.7574 | +0.0030 | not yet tested | 0.096 | **tie** — inside ±0.0054 seed noise |
| λ-field (mask-norm, SVF) | λ=0.25 svf | 0.7568 | +0.0024 | not yet tested | **0.000** | **tie**, zero folding |
| λ-field (whole-image norm) | λ=0.25 disp | 0.7513 | −0.0031 | 0.36 | 0.112 | no effect |
| cross-attention | λ=0.1 disp | 0.6934 | −0.0611 | ~1e-40 ** | 0.039 | **fails**, worse on 100/100 pairs |

### 3D (best baseline `3d_baseline_lam0.05_svf`, Dice 0.7977, folding 0.000%)

| variant | best run | Dice | Δ | p | verdict |
|---|---|---|---|---|---|
| baseline | λ=0.05 svf | 0.7977 | — | — | reference |
| λ-field | λ=0.1 svf | 0.7908 | −0.0069 | 1e-21 ** | **worse** |
| cross-attention | λ=0.1 svf | 0.6699 | −0.1277 | 3e-58 ** | **fails badly** |

### Cross-cutting findings

- **λ does not transfer between dimensionalities**: 2D optimum 0.25, 3D optimum 0.1.
- **λ-insensitivity reproduced**: baseline λ=0.1 vs λ=0.25 is insignificant in 2D (p=0.19).
- **3D is the number to report**: +0.116 Dice over initial vs 2D's +0.080.
- **SVF buys inverse consistency for free**: −0.006 to −0.012, 100/100 pairs, every variant.
- **Every extension so far trades folding for Dice.** No branch improves folding.
- **Superseded formulation preserved** in `results_failed_meanonly/`: constraining only the
  *mean* of the weight field was evaded — the optimiser drove weight to ~0 exactly where the
  displacement gradient was largest. Bounding before normalising closed it.

---

## 2. Decisions taken

| # | decision | reason |
|---|---|---|
| D1 | Bound the weight map *before* mean-normalising | mean-only constraint is evadable (above) |
| D2 | Normalise λ-field within the brain mask, not whole-image | 60% of the volume is background; the field parked weight there and relaxed the brain to the floor, making the variant just a weaker λ |
| D3 | Score every model on one persisted pair list, compare paired | between-pair variance exceeds the effect size |
| D4 | Select on validation, report on test | many variants tried; selecting and reporting on the same split biases upward |
| D5 | Treat `--integration-steps` as an orthogonal flag, not a branch | it composes with all three variants |

---

## 3. Proposed — not yet measured

Ranked by expected payoff per unit effort. Nothing below has a number yet.

| # | direction | claim | why it might work | effort |
|---|---|---|---|---|
| P1 | **Per-structure λ** | one scalar per structure (24/35) instead of per-voxel | the per-voxel field's *only* coherent behaviour was per-region; 24 d.o.f. is the hypothesis class that behaviour actually needs | S — **in progress** |
| P2 | **Anti-folding penalty** | add `mean(relu(-\|J\|))` to the loss | folding is the metric *every* variant regresses, and the diffusion regulariser never penalises it directly | S |
| P3 | **Gated-residual cross-attention** | gate the attention branch by a scalar init at 0 | current design has no init at which it equals the baseline, so it starts by damaging a working model — a plausible cause of the −13% | S |
| P4 | **Uncertainty calibration** | report AUROC for "will this voxel misregister" + sparsification curve | turns the ensemble from a computed variance into an actionable claim | M |
| P5 | **Test-time instance optimisation** | fine-tune the field per test pair | measures the amortisation gap — the premise of the whole paper, which no branch currently asks | M |
| P6 | **Auxiliary Dice loss (semi-supervised)** | add label overlap to the training objective | the paper reports a large gain (§VI); low-risk insurance if novel branches stay flat | M |
| P7 | **Cross-attention as a refinement stage** | freeze baseline, train a second net for Δu | composed result cannot be worse than baseline at init; also lets cross-attention be tested against *initial misalignment*, its actual hypothesis | L |

### Known gaps (carried from HANDOFF)

- Cross-attention stratified by initial misalignment — its stated hypothesis is still untested.
- Colab notebook (submission requirement).
- Report.

---

## 3b. P3 — gated cross-attention: MEASURED, question settled

**Design.** The original `cross_attn` replaces the baseline's input path with a two-stream
encoder whose streams each see a single image, and follows its bottleneck residual with a
LayerNorm that rescales the representation. There is therefore **no initialisation at which it
equals the baseline** — it begins by damaging a working model. `cross_attn_gated` fixes exactly
that and nothing else:

* main path is a plain UNet on `cat(source, target)` — the baseline's own formulation;
* the attention branch is a second pass of the **same encoder weights** over the swapped input
  `cat(target, source)`, so it adds no encoder parameters (0.33M both — capacity matched);
* combined as `h + tanh(gate) * attend(...)` with `gate` a learned scalar **initialised to 0**.

At init `tanh(0)=0`, so the model is bit-for-bit the plain UNet path — pinned by
`test_at_init_matches_ungated_path`.

**2D result** (single seed, 100 test pairs) — `results/comparison_cross_attn_gated_2d.md`:

| run | Dice | Δ vs baseline 0.7544 | p |
|---|---|---|---|
| `2d_cross_attn_lam0.1_disp` (original) | 0.6934 | **−0.0611** | 1e-40 ** |
| `2d_cross_attn_gated_lam0.1_disp` | 0.7510 | −0.0034 | 0.37 |
| `2d_cross_attn_gated_lam0.25_disp` | 0.7550 | +0.0006 | 0.43 |
| `2d_cross_attn_gated_lam0.25_svf` | 0.7547 | +0.0002 | 0.77, **0% folding** |

**The collapse is entirely an optimisation artefact.** At matched λ the gate recovers **+0.058
Dice** (0.6934 → 0.7510); at λ=0.25 the variant is statistically indistinguishable from the
baseline.

**The gate quantifies what attention is worth.** Trained values, all six runs:
`tanh(gate) = 0.086, 0.100, 0.106, 0.114, 0.137, 0.153`. Consistently positive — so attention is
not actively harmful — but small, and worth ~0.000 Dice. The branch was free to open and barely
did.

**Reportable claim.** *Cross-attention at the bottleneck does not help VoxelMorph on this task.
Added as a zero-initialised gated residual, so the model may ignore it at no cost, the learned
gate settles near 0.1 and produces no significant Dice change. The −6% (2D) / −13% (3D) collapse
of the naive two-stream formulation measures a broken initialisation, not attention.* This is
stronger than the original negative result: one experiment identifies the confound, the other
controls for it.

## 4. In progress

### P1 — per-structure λ (`lambda_structure`)

**Design.** UNet head emits `n_labels` channels, global-average-pooled to one scalar per
structure, sigmoid-bounded to `(0.5, 2.0)`, scattered onto the voxel grid through the *source*
segmentation, then normalised to mean 1 **over the brain mask** (D2) so the regularisation
budget matches the baseline exactly.

**Caveat, must be reported.** This variant needs the **moving image's segmentation at inference
time**, not only during training — without it the model cannot produce a field at all. The
baseline and λ-field need no labels ever. So this is *not* a like-for-like comparison, and any
win is contingent on a segmentation being available at registration time.

It is not, however, metric leakage: the model sees only the *moving* segmentation, while Dice is
scored between the warped moving segmentation and the *fixed* one, which the model never sees.
The evaluation protocol already had the moving segmentation in hand to warp it.

**Guards.** Label ids are read from the NIfTI unremapped, so an id ≥ `n_labels` raises instead
of clamping — clamping would silently merge structures into one weight and return a plausible
wrong answer. Tests pin: unit mean in mask, piecewise-constant on structures, gradient reaches
the head, and both failure modes above.

**2D result (20k steps, 100 fixed test pairs, seed 0).** Paired vs `2d_baseline_lam0.25_disp`
(0.7544). Full table: `results/comparison_lambda_structure_2d.md`.

| run | Dice | Δ | p | better on | folding % |
|---|---|---|---|---|---|
| `2d_lambda_structure_lam0.25_disp` | **0.7604** | **+0.0060** | **1.7e-08** ** | 68/100 | 0.046 |
| `2d_lambda_structure_lam0.25_svf` | 0.7559 | +0.0015 | 0.18 | 55/100 | **0.000** |
| `2d_lambda_structure_lam0.1_disp` | 0.7516 | −0.0028 | 0.33 | 53/100 | 0.332 |
| `2d_lambda_structure_lam0.1_svf` | 0.7510 | −0.0034 | 0.38 | 57/100 | 0.000 |
| `2d_lambda_structure_lam0.05_disp` | 0.7397 | — | — | — | 0.914 |
| `2d_lambda_structure_lam0.05_svf` | 0.7319 | — | — | — | 0.000 |

Best result in the 2D table so far, and the **first extension in this project to beat the
baseline significantly**. Two reasons not to report it as a win yet:

1. The paired test controls for *pair* variance, not *seed* variance. +0.0060 is barely outside
   the ±0.0054 seed band measured from the baseline ensemble. A 5-seed ensemble is running.
2. Folding is 3.5x the baseline's (0.046% vs 0.013%, p=8e-05). The SVF variant reaches
   +0.0015 at **zero** folding, which may be the more defensible configuration.

**Learned allocation is binary, and saturates the bound.** Averaged over the val pairs, the
per-structure weights take exactly two values:

| structure | learned λ multiplier |
|---|---|
| Lateral-Ventricle (L+R), Brain-Stem | **0.337** — free to deform |
| cortex, white matter, thalamus, hippocampus, putamen, ventral DC (11 others) | **1.349** — held stiff |

The ratio is 4.00, exactly the span of the default `weight_range` (0.5, 2.0). Every structure's
sigmoid is fully saturated, so the model wants *more* contrast than the bound permits and the
bound — not the model — is setting the allocation. It also independently rediscovers the
λ-field's per-structure finding (ventricles and brain stem want freedom), which is evidence the
effect is real rather than an artefact of either parameterisation.

Follow-up running: `weight_range` widened to (0.25, 4.0) and (0.125, 8.0), suffixes `_wr8` /
`_wr16`.

**Head-to-head vs the per-voxel field it replaces** (`2d_lambda_field_lam0.25_disp_maskn`,
0.7574) — `results/comparison_struct_vs_field_2d.md`:

| metric | Δ | p | better on |
|---|---|---|---|
| dice | +0.0030 | 0.13 | 43/100 |
| folding_fraction | **−0.0005** | **7.6e-05** ** | 56/100 |

Read this carefully: on Dice the per-structure form is **not** significantly better than the
per-voxel one, and it loses on the *majority* of pairs (43/100, Wilcoxon p=0.52) while its mean
is higher — the gain is carried by a few large wins, not a broad shift. Where it does win
cleanly is **folding, halved at equal Dice**. So the honest claim for the restriction is
"same accuracy, better-behaved deformations", not "more accurate".

### P1 verdict — CONFIRMED at 5 seeds

Averaging each pair's Dice over 5 seeds, then testing paired (n=100). This controls pair
variance *and* seed noise, and is the strongest test available here.

| variant | Dice | Δ vs baseline | wins | p |
|---|---|---|---|---|
| baseline (5-seed) | 0.7521 ± 0.0054 | — | — | — |
| **λ-structure (5-seed)** | **0.7578 ± 0.0025** | **+0.0058** | **80/100** | **1.3e-12** |
| λ-field (5-seed) | 0.7471 ± 0.0018 | −0.0049 | 51/100 | 0.12 (none) |

+0.0058 now sits **above** the ±0.0054 seed band and survives seed averaging, so this is a real
effect, not a lucky initialisation. Note the single-seed numbers reported earlier were both
optimistic: λ-structure seed 0 gave 0.7604 (top of a 0.7541–0.7602 spread) and the baseline's
single run gave 0.7544 against a 0.7521 ensemble mean.

λ-structure is also **more stable across seeds** than the baseline (std 0.0025 vs 0.0054), which
is a second, independent benefit of shrinking the hypothesis class.

**Cost:** folding is worse than baseline (0.058% vs 0.013%, p=5.6e-07; worse on 66/100 pairs),
though better than the per-voxel λ-field's 0.103%. The SVF variant removes folding entirely at a
smaller Dice gain.

### P1 follow-up — widening `weight_range` FAILS

Hypothesis was that saturation meant the 4:1 bound was the binding constraint. It is not —
widening it monotonically hurts both metrics:

| weight_range | Dice (disp) | folding % (disp) | Dice (svf) |
|---|---|---|---|
| (0.5, 2.0) — default, 4:1 | **0.7604** | **0.046** | **0.7559** |
| (0.25, 4.0) — 16:1 | 0.7550 | 0.290 | 0.7550 |
| (0.125, 8.0) — 64:1 | 0.7525 | 0.853 | 0.7510 |

So saturation is the model **exploiting a well-chosen constraint**, not fighting it. The bound is
doing real work: it is what stops the optimiser reproducing the `results_failed_meanonly/`
collapse, where unconstrained weights deleted the regulariser where the displacement gradient was
largest. Keep 4:1.

**Note for whoever reruns this.** `weight_range` is now an `ExperimentConfig` field, so it also
applies to `lambda_field`. Existing runs are unaffected — the default is the old value.

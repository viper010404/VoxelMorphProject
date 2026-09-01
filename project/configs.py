"""
Experiment configuration.

Every run is fully described by an `ExperimentConfig`, which is serialised to `config.json`
beside the checkpoint. Evaluation reconstructs the model from that file alone, so scoring is
decoupled from training: any checkpoint can be re-scored later without knowing which branch
produced it.
"""

import json
from pathlib import Path
from dataclasses import dataclass, asdict, field
from typing import List, Optional, Sequence


DEFAULT_FEATURES = (16, 32, 32, 32, 32)

VARIANTS = ('baseline', 'lambda_field', 'lambda_structure', 'cross_attn',
            'cross_attn_gated', 'coarse_to_fine', 'pyramid', 'fathead', 'msf')


@dataclass
class ExperimentConfig:
    """
    Full description of one training run.

    Attributes
    ----------
    name : str
        Unique run name; also the results directory name.
    variant : str
        Which model to build, one of `VARIANTS`.
    ndim : int
        Spatial dimensionality, 2 or 3.
    lambda_reg : float
        Smoothness weight. For `lambda_field` this is the *mean* of the learned weight map, so
        the total regularisation budget matches the baseline exactly and only its spatial
        distribution differs.
    integration_steps : int
        Scaling-and-squaring steps. 0 gives a plain displacement field (the CVPR formulation);
        >0 integrates a stationary velocity field, giving a diffeomorphism. Orthogonal to the
        choice of variant.
    nb_features : sequence of int
        UNet features per level. Spatial dimensions must be divisible by 2**len(nb_features).
    steps : int
        Number of training iterations.
    batch_size : int
        Pairs per iteration. Avoid 3 in 2D and 4 in 3D anywhere `voxelmorph.nn.functional.compose`
        is reachable: its batch detection misreads those shapes.
    lr : float
        Adam learning rate. The paper uses 1e-4.
    seed : int
        Seed for model init and pair sampling; varied to build the ensemble.
    val_every : int
        Validation interval in steps. The best-on-validation checkpoint is the one evaluated,
        so the test split is only touched once.
    test_every : int
        Trace mean Dice on the fixed test pairs every this many steps; 0 disables it. The trace
        is for monitoring a long run only and never selects a checkpoint -- see `train.train`.
    test_pairs : int
        Number of fixed test pairs the trace scores. Must match what `evaluate.py` uses, or the
        trace and the final number are measuring different things.
    val_pairs : int
        Number of validation pairs used for model selection.
    attn_heads : int
        Attention heads for the `cross_attn` variant.
    cross_attn_target_skips : bool
        For `cross_attn`, fuse the target stream's skip connections into the decoder. Off by
        default so the original formulation reproduces exactly; see
        `models.VxmCrossAttention` for the measured effect.
    cross_attn_use_attention : bool
        For `cross_attn`, whether to cross-attend at the bottleneck. False is the ablation
        control isolating what the attention itself contributes.
    cross_attn_window_level : int
        Skip level for local windowed cross-attention, or -1 to disable. Adds a `_win<level>`
        suffix to the run name.
    cross_attn_window_radius : int
        Neighbourhood radius in tokens for that attention.
    stage_scales : sequence of int
        For `cascade`, the downsampling factor each stage runs at, finest last. (2, 1) is
        coarse-to-fine; (1, 1) is the same-resolution control that isolates iteration from
        multi-resolution. Adds a `_c2f<scales>` suffix.
    pyramid_progressive : bool
        For `pyramid`, warp each level's skip features by the field accumulated so far. False
        keeps the per-level flow heads but predicts levels independently, isolating deep
        supervision from progressive warping. Adds a `_noprog` suffix.
    deep_supervision : bool
        Score every pyramid level, not just the final field. False trains the pyramid on the
        final field alone, isolating the architecture from the objective. Adds `_nods`.
    misalign_magnitude : float
        Mean magnitude, in voxels, of a synthetic smooth deformation applied to the moving image
        during training. 0 leaves the dataset as shipped. neurite-OASIS is affinely pre-aligned,
        so its residual motion is ~1.7 voxels; raising this creates the large-displacement regime
        in which explicit correspondence matching is supposed to help. Adds a `_mis<n>` suffix.
    lambda_mask_norm : bool
        For `lambda_field`, normalise the weight map to mean 1 over the brain mask rather than
        over the whole image. Off by default so earlier runs reproduce exactly; see
        `models.VxmLambdaField._normalise_weights` for why it matters.
    stage_scales : sequence of int
        Downsampling factor per stage for `coarse_to_fine`, coarsest first. (2, 1) is the
        coarse-to-fine cascade; (1, 1) is the same-resolution control that holds capacity fixed.
    bidirectional : bool
        Train each pair in both directions with the same weights, adding the reverse similarity
        term. Costs a second forward pass but no parameters.
    beta_inv : float
        Weight on the inverse-consistency penalty. Requires `bidirectional`; 0 disables it.
    lambda_fold : float
        Weight on the anti-folding penalty `relu(margin - |J|)`, which targets non-invertible
        deformation directly rather than relying on the diffusion term to discourage it
        indirectly. 0 disables it and reproduces the paper's objective. Orthogonal to `variant`.
    fold_margin : float
        Determinant value the anti-folding penalty pushes above. 0 penalises only actual folding.
    structure_lambda : sequence of float or None
        Fixed per-structure weights for `lambda_structure`, bypassing the learned head. Used for
        the control experiments: freezing the learned allocation, and shuffling it across
        structures to test whether the *specific* assignment matters or merely the fact of
        having some heterogeneity. None means learn it.
    weight_range : sequence of float
        `(low, high)` bounds on the learned weight before mean-normalisation, for the weighted
        variants. The *ratio* caps how strongly regularisation may be redistributed; the default
        4:1 was chosen for `lambda_field` and is saturated by `lambda_structure`, so widening it
        is a live experiment rather than a tuning detail.
    n_labels : int
        Size of the per-structure weight table for the `lambda_structure` variant. Must exceed
        the largest label id present, which includes background: the 2D `seg24` holds ids 0-24
        (25 slots) and the 3D `seg35` holds ids 0-35 (36 slots). Verified at forward time.
    data_path : str
        Path to the `.npz` cache.
    output_root : str
        Directory under which `<name>/` is created.
    """

    name: str
    variant: str = 'baseline'
    ndim: int = 2
    lambda_reg: float = 0.01
    integration_steps: int = 0
    nb_features: Sequence[int] = (16, 32, 32, 32, 32)
    steps: int = 20000
    batch_size: int = 16
    lr: float = 1e-4
    seed: int = 0
    val_every: int = 2000
    # Per-pair Dice varies a great deal between subject pairs (standard deviation ~0.1), so a
    # small validation set selects on noise. 64 pairs keeps the standard error near 0.01 while
    # costing only a few seconds per check.
    val_pairs: int = 64
    test_every: int = 0
    test_pairs: int = 100
    head_hidden: Optional[int] = None
    head_kernel: int = 3
    head_image_skip: bool = False
    msf_per_level: int = 4
    head_lr_mult: float = 1.0
    attn_heads: int = 4
    lambda_mask_norm: bool = False
    cross_attn_target_skips: bool = False
    cross_attn_use_attention: bool = True
    cross_attn_window_level: int = -1
    cross_attn_window_radius: int = 2
    misalign_magnitude: float = 0.0
    pyramid_progressive: bool = True
    deep_supervision: bool = True
    structure_lambda: Optional[Sequence[float]] = None
    stage_scales: Sequence[int] = (2, 1)
    bidirectional: bool = False
    beta_inv: float = 0.0
    lambda_fold: float = 0.0
    fold_margin: float = 0.0
    weight_range: Sequence[float] = (0.5, 2.0)
    n_labels: int = 25
    data_path: str = 'data/oasis2d.npz'
    output_root: str = 'results'

    def __post_init__(self) -> None:
        if self.variant not in VARIANTS:
            raise ValueError(f"variant must be one of {VARIANTS}, got '{self.variant}'")
        if self.ndim not in (2, 3):
            raise ValueError(f'ndim must be 2 or 3, got {self.ndim}')
        self.nb_features = tuple(self.nb_features)
        self.weight_range = tuple(self.weight_range)
        self.stage_scales = tuple(self.stage_scales)

    @property
    def output_dir(self) -> Path:
        """Directory holding this run's checkpoints, config and results."""
        return Path(self.output_root) / self.name

    def save(self) -> Path:
        """Write `config.json` into the run directory and return its path."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        path = self.output_dir / 'config.json'
        with open(path, 'w') as f:
            json.dump(asdict(self), f, indent=2)
        return path

    @classmethod
    def load(cls, path) -> 'ExperimentConfig':
        """Rebuild a config from a `config.json` written by `save`."""
        with open(path) as f:
            return cls(**json.load(f))


def build_matrix(
    ndim: int = 2,
    # Centred on 0.1, located empirically: at 0.01 the deformation folds on 3.3% of voxels
    # (the paper reports 0.2-0.4%) while at 5.0 it is too stiff to align anything. The
    # smoothness term here is a mean over voxels, so the paper's quoted 0.01 does not transfer.
    lambdas: Sequence[float] = (0.05, 0.1, 0.25),
    integration_steps: Sequence[int] = (0, 7),
    variants: Sequence[str] = VARIANTS,
    steps: Optional[int] = None,
    data_path: Optional[str] = None,
    batch_size: Optional[int] = None,
    lambda_mask_norm: bool = False,
    cross_attn_target_skips: bool = False,
    cross_attn_use_attention: bool = True,
    cross_attn_window_level: int = -1,
    cross_attn_window_radius: int = 2,
    nb_features: Optional[Sequence[int]] = None,
    misalign_magnitude: float = 0.0,
    stage_scales: Sequence[int] = (2, 1),
    pyramid_progressive: bool = True,
    deep_supervision: bool = True,
    n_labels: Optional[int] = None,
    test_every: int = 0,
    head_hidden: Optional[int] = None,
    head_kernel: int = 3,
    head_image_skip: bool = False,
    msf_per_level: int = 4,
    head_lr_mult: float = 1.0,
    seeds: Sequence[int] = (0,),
    sweep_variants: Sequence[str] = ('baseline', 'lambda_field'),
) -> List[ExperimentConfig]:
    """
    Build the sweep of configurations for the bake-off.

    The baseline and `lambda_field` variants are swept over lambda so the comparison is against
    the *best* baseline rather than an arbitrary one -- otherwise any improvement could just be
    a better-tuned regulariser. `cross_attn` is run at the middle lambda only, to keep the
    schedule affordable.

    Parameters
    ----------
    ndim : int, optional
        Spatial dimensionality of the sweep.
    lambdas : sequence of float, optional
        Smoothness weights to sweep.
    integration_steps : sequence of int, optional
        Displacement (0) and/or diffeomorphic (>0) settings.
    variants : sequence of str, optional
        Which model variants to include.
    steps : int or None, optional
        Training iterations; defaults to 20000. Any other value adds an `_s<n>k` suffix, keeping
        each training budget in its own namespace -- runs at different budgets are not comparable
        and must not collide.
    data_path : str or None, optional
        Override the dataset cache path.
    batch_size : int or None, optional
        Override the batch size. Defaults to 16 in 2D and 1 in 3D.
    lambda_mask_norm : bool, optional
        Normalise the lambda-field weight map within the brain mask. Adds a `_maskn` suffix to
        the run name so the two formulations never collide in `results/`.
    cross_attn_target_skips : bool, optional
        Fuse the target encoder pyramid into the cross-attention decoder. Adds a `_tskip` suffix
        to the run name.
    cross_attn_use_attention : bool, optional
        Set False for the ablation that keeps the target skips but removes the attention. Adds a
        `_noattn` suffix.
    nb_features : sequence of int or None, optional
        UNet width per level. The *length* sets the depth, and so the bottleneck's downsampling
        factor: five levels give 32x, where every displacement in this dataset is smaller than
        one token. Shorter is coarser-to-finer. Adds a `_d<levels>` suffix unless it is the
        five-level default. Spatial dims must stay divisible by `2 ** len(nb_features)`.
    sweep_variants : sequence of str, optional
        Which variants get the full lambda sweep; the rest run at the middle value only. Defaults
        to the two whose regularisation is under test, keeping the schedule affordable.

    Returns
    -------
    list of ExperimentConfig
    """
    if data_path is None:
        data_path = 'data/oasis2d.npz' if ndim == 2 else 'data/oasis3d_cache'
    if batch_size is None:
        batch_size = 16 if ndim == 2 else 1
    if steps is None:
        steps = 20000
    if nb_features is None:
        nb_features = DEFAULT_FEATURES
    nb_features = tuple(nb_features)
    if n_labels is None:
        n_labels = 25 if ndim == 2 else 36

    # A 3D validation pass costs ~1.8 s per pair (Dice over 33 structures on 6.9M voxels, plus
    # the Jacobian determinant), so the 2D settings would add ~19 min to every run. Validate
    # less often and on fewer pairs; between-pair variance is lower in 3D than on a single slice.
    val_pairs = 64 if ndim == 2 else 32
    val_every = 2000 if ndim == 2 else 4000

    configs: List[ExperimentConfig] = []
    for variant in variants:
        # Sweep lambda only where it is the quantity under test; cross-attention runs at the
        # middle value so the schedule stays affordable.
        if variant in sweep_variants:
            sweep = tuple(lambdas)
        else:
            sweep = (lambdas[len(lambdas) // 2],)
        for lam in sweep:
            for isteps in integration_steps:
                tag = 'svf' if isteps > 0 else 'disp'
                suffix = ''
                if lambda_mask_norm and variant == 'lambda_field':
                    suffix = '_maskn'
                if variant == 'cross_attn':
                    suffix = '_tskip' if cross_attn_target_skips else ''
                    if not cross_attn_use_attention:
                        suffix += '_noattn'
                    if cross_attn_window_level >= 0:
                        suffix += f'_win{cross_attn_window_level}'
                if len(nb_features) != 5:
                    suffix += f'_d{len(nb_features)}'
                # Depth alone is not enough to identify the architecture: a *widened* UNet of the
                # same depth would otherwise reuse the plain baseline's directory and overwrite
                # its results with a different model's numbers. Capacity controls need their own
                # namespace precisely because they are meant to be compared against it.
                if tuple(nb_features) != DEFAULT_FEATURES[:len(nb_features)]:
                    suffix += '_f' + '-'.join(str(width) for width in nb_features)
                if variant == 'coarse_to_fine':
                    suffix += '_c2f' + ''.join(str(x) for x in stage_scales)
                if variant == 'fathead':
                    # The head's width and kernel are the quantities under test here, so they
                    # have to be in the directory name or the sweep overwrites itself.
                    if head_hidden is not None:
                        suffix += f'_h{head_hidden}'
                    if head_kernel != 3:
                        suffix += f'_k{head_kernel}'
                    if head_image_skip:
                        suffix += '_imgskip'
                if head_lr_mult != 1.0:
                    suffix += f'_hlr{head_lr_mult:g}'
                if variant == 'msf' and msf_per_level != 4:
                    suffix += f'_p{msf_per_level}'
                if variant == 'pyramid':
                    if not pyramid_progressive:
                        suffix += '_noprog'
                    if not deep_supervision:
                        suffix += '_nods'
                if misalign_magnitude > 0:
                    suffix += f'_mis{misalign_magnitude:g}'
                # A different training budget is a different experiment, not a newer version of
                # the same one: without this the 80k runs would overwrite the 20k matrix in place
                # and --skip-existing would skip them entirely, silently leaving no reference.
                if steps != 20000:
                    suffix += f'_s{steps // 1000}k'
                for seed in seeds:
                    # Seed 0 keeps the bare name so an existing single-seed run is reused as a
                    # member of its own ensemble rather than retrained under a new name.
                    seed_suffix = '' if seed == 0 else f'_seed{seed}'
                    configs.append(ExperimentConfig(
                        seed=seed,
                        name=f'{ndim}d_{variant}_lam{lam}_{tag}{suffix}{seed_suffix}',
                        variant=variant,
                        ndim=ndim,
                        nb_features=nb_features,
                        misalign_magnitude=misalign_magnitude,
                        stage_scales=stage_scales,
                        pyramid_progressive=pyramid_progressive,
                        deep_supervision=deep_supervision,
                        lambda_reg=lam,
                        integration_steps=isteps,
                        steps=steps,
                        batch_size=batch_size,
                        val_pairs=val_pairs,
                        val_every=val_every,
                        test_every=test_every,
                        head_hidden=head_hidden,
                        head_kernel=head_kernel,
                        head_image_skip=head_image_skip,
                        msf_per_level=msf_per_level,
                        head_lr_mult=head_lr_mult,
                        data_path=data_path,
                        lambda_mask_norm=lambda_mask_norm,
                        cross_attn_target_skips=cross_attn_target_skips,
                        cross_attn_use_attention=cross_attn_use_attention,
                        cross_attn_window_level=cross_attn_window_level,
                        cross_attn_window_radius=cross_attn_window_radius,
                        n_labels=n_labels,
                    ))
    return configs


def ensemble_configs(
    base: ExperimentConfig,
    seeds: Sequence[int] = (0, 1, 2, 3, 4),
) -> List[ExperimentConfig]:
    """
    Replicate a configuration across seeds to form an ensemble.

    Per-voxel variance across independently trained models gives the uncertainty estimate, and
    -- for the lambda-field variant -- lets us check whether the learned weight map is
    reproducible rather than an artefact of one initialisation.

    Parameters
    ----------
    base : ExperimentConfig
        Configuration to replicate.
    seeds : sequence of int, optional
        Seeds to train.

    Returns
    -------
    list of ExperimentConfig
    """
    members = []
    for seed in seeds:
        member = ExperimentConfig(**{**asdict(base), 'name': f'{base.name}_seed{seed}',
                                     'seed': seed})
        members.append(member)
    return members

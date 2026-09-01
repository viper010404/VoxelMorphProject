"""
Tests for the project's registration models.

The critical test here is `test_lambda_map_has_unit_mean`. The lambda-field variant is only a
meaningful experiment if the network cannot escape the regularisation budget by driving the
weight map to zero; the normalisation makes the mean exactly 1 by construction, and that is what
this asserts. If it ever fails, the comparison against the baseline stops being like-for-like.
"""

import pytest
import torch

from project.configs import ExperimentConfig
from project.models import (LocalCrossAttention, VxmBaseline, VxmCrossAttention,
                            VxmFatHead, VxmLambdaField, VxmMultiScaleFeatures,
                            VxmPyramid, build_model)


CASES = [
    ('baseline', 2, (64, 64), 2),
    ('fathead', 2, (64, 64), 2),
    ('fathead', 3, (32, 32, 32), 1),
    ('msf', 2, (64, 64), 2),
    ('msf', 3, (32, 32, 32), 1),
    ('lambda_field', 2, (64, 64), 2),
    ('cross_attn', 2, (64, 64), 2),
    ('baseline', 3, (32, 32, 32), 1),
    ('lambda_field', 3, (32, 32, 32), 1),
    ('cross_attn', 3, (32, 32, 32), 1),
]


def _inputs(batch, shape):
    torch.manual_seed(0)
    return torch.rand(batch, 1, *shape), torch.rand(batch, 1, *shape)


@pytest.mark.parametrize('variant,ndim,shape,batch', CASES)
@pytest.mark.parametrize('integration_steps', [0, 7])
def test_forward_shapes(variant, ndim, shape, batch, integration_steps):
    """Every variant returns a displacement and a warped source of the expected shape."""
    config = ExperimentConfig(name='t', variant=variant, ndim=ndim,
                              integration_steps=integration_steps)
    source, target = _inputs(batch, shape)
    outputs = build_model(config)(source, target)

    assert outputs['displacement'].shape == (batch, ndim, *shape)
    assert outputs['warped_source'].shape == source.shape
    assert ('velocity' in outputs) == (integration_steps > 0)


@pytest.mark.parametrize('ndim,shape', [(2, (64, 64)), (3, (32, 32, 32))])
def test_lambda_map_has_unit_mean(ndim, shape):
    """
    The learned weight map must have mean exactly 1 for every sample.

    This is the anti-collapse guarantee: the model may redistribute regularisation but cannot
    reduce its total, so it spends the same smoothness budget as the baseline.
    """
    model = VxmLambdaField(ndim=ndim)
    source, target = _inputs(2, shape)
    lambda_map = model(source, target)['lambda_map']

    per_sample_mean = lambda_map.flatten(1).mean(1)
    assert torch.allclose(per_sample_mean, torch.ones_like(per_sample_mean), atol=1e-5)


@pytest.mark.parametrize('ndim,shape', [(2, (64, 64)), (3, (32, 32, 32))])
def test_lambda_map_strictly_positive(ndim, shape):
    """A negative weight would turn the regulariser into a reward for roughness."""
    model = VxmLambdaField(ndim=ndim)
    source, target = _inputs(2, shape)
    assert (model(source, target)['lambda_map'] > 0).all()


@pytest.mark.parametrize('bias', [-50.0, 50.0])
def test_lambda_map_survives_extreme_inputs(bias):
    """
    A saturated head must degrade to a uniform field, never to a vanishing one.

    Saturating in either direction should reduce the model to the baseline (a constant weight of
    1), not produce a degenerate map.
    """
    model = VxmLambdaField(ndim=2)
    with torch.no_grad():
        model.unet.out_layer.conv0.bias.fill_(bias)
    source, target = _inputs(2, (64, 64))
    lambda_map = model(source, target)['lambda_map']

    assert torch.isfinite(lambda_map).all()
    per_sample_mean = lambda_map.flatten(1).mean(1)
    assert torch.allclose(per_sample_mean, torch.ones_like(per_sample_mean), atol=1e-4)
    assert torch.allclose(lambda_map, torch.ones_like(lambda_map), atol=1e-3)


@pytest.mark.parametrize('ndim,shape', [(2, (64, 64)), (3, (32, 32, 32))])
def test_lambda_map_cannot_evade_the_regulariser(ndim, shape):
    """
    The weight field must stay bounded away from zero, whatever the head outputs.

    This is the property that unit-mean normalisation alone does *not* provide. Minimising a
    weighted sum under only a mean constraint drives the weight towards zero exactly where the
    displacement gradient is largest -- measured on a real run, a map spanning 1e-5 to 6.4 with
    mean 1, which folded 5.0% of voxels against the baseline's 0.56%. Bounding before
    normalising caps how far the field can redistribute.
    """
    low, high = 0.5, 2.0
    model = VxmLambdaField(ndim=ndim, weight_range=(low, high))
    source, target = _inputs(2, shape)

    # Drive the head to extreme values in both directions within one batch.
    with torch.no_grad():
        model.unet.out_layer.conv0.weight.mul_(1000.0)
    lambda_map = model(source, target)['lambda_map']

    # After normalisation the attainable range is bounded by the ratio of the two bounds.
    assert float(lambda_map.min()) >= low / high
    assert float(lambda_map.max()) <= high / low


def test_lambda_field_capacity_matches_baseline():
    """
    The lambda-field model must differ from the baseline by only the extra output channel.

    If it were substantially larger, an improvement could be attributed to capacity rather than
    to the spatial weighting.
    """
    baseline = sum(p.numel() for p in VxmBaseline(ndim=2).parameters())
    lambda_field = sum(p.numel() for p in VxmLambdaField(ndim=2).parameters())
    assert 0 < lambda_field - baseline <= 64


def test_cross_attention_uses_both_images():
    """
    Changing only the target must change the predicted field.

    A two-stream model that silently ignored the target stream would still produce plausible
    output shapes, so this checks the target genuinely reaches the output.

    The assertion is made at the bottleneck as well as on the displacement, because at
    initialisation the flow layer has standard deviation 1e-5 and squashes every output to
    ~1e-5. At that scale a real relative difference of 1e-5 is smaller than the default absolute
    tolerance of `torch.allclose`, so a naive comparison would report "identical" for a model
    that is working correctly.
    """
    model = VxmCrossAttention(ndim=2)
    model.eval()
    torch.manual_seed(0)
    source = torch.rand(1, 1, 64, 64)
    target_a = torch.rand(1, 1, 64, 64)
    target_b = torch.rand(1, 1, 64, 64)

    with torch.no_grad():
        source_feat, _ = model._encode(source)
        fused_a = model._cross_attend(source_feat, model._encode(target_a)[0])
        fused_b = model._cross_attend(source_feat, model._encode(target_b)[0])

        first = model(source, target_a)['displacement']
        second = model(source, target_b)['displacement']

    # The fused bottleneck must differ substantially between the two targets.
    relative = (fused_a - fused_b).abs().max() / fused_a.abs().max()
    assert relative > 1e-4

    # And that difference must survive to the output, however small the init scale makes it.
    assert (first - second).abs().max() > 0


def test_gradients_flow_to_all_parameters():
    """Every parameter of every variant must receive gradient from the registration loss."""
    from project.losses import registration_loss

    for variant in ('baseline', 'lambda_field', 'cross_attn'):
        config = ExperimentConfig(name='t', variant=variant, ndim=2)
        model = build_model(config)
        source, target = _inputs(2, (64, 64))

        outputs = model(source, target)
        losses = registration_loss(target, outputs['warped_source'],
                                   outputs['displacement'], 0.01,
                                   weight=outputs.get('lambda_map'))
        losses['total'].backward()

        missing = [name for name, param in model.named_parameters() if param.grad is None]
        assert not missing, f'{variant}: parameters without gradient: {missing}'


def test_unknown_variant_rejected():
    config = ExperimentConfig(name='t')
    config.variant = 'nonsense'
    with pytest.raises(ValueError, match='unknown variant'):
        build_model(config)


def _brain_mask_inputs(shape=(64, 64)):
    """Skull-stripped-style inputs: a central positive box on an exactly-zero background."""
    source = torch.zeros(1, 1, *shape)
    target = torch.zeros(1, 1, *shape)
    centre = tuple(slice(s // 4, 3 * s // 4) for s in shape)
    source[(slice(None), slice(None)) + centre] = 0.8
    target[(slice(None), slice(None)) + centre] = 0.9
    return source, target


def test_foreground_mask_is_the_union_of_both_images():
    source = torch.tensor([[[[0.0, 0.5], [0.0, 0.0]]]])
    target = torch.tensor([[[[0.0, 0.0], [0.3, 0.0]]]])
    mask = VxmLambdaField._foreground_mask(source, target)
    assert mask.tolist() == [[[[False, True], [True, False]]]]


def test_masked_normalisation_gives_unit_mean_inside_the_mask():
    """The brain's regularisation budget must equal the baseline's, exactly."""
    raw = torch.randn(2, 1, 16, 16)
    mask = torch.zeros(2, 1, 16, 16, dtype=torch.bool)
    mask[:, :, 4:12, 4:12] = True
    weights = VxmLambdaField._normalise_weights(raw, (0.5, 2.0), mask)
    for sample in range(2):
        inside = weights[sample][mask[sample]]
        assert float(inside.mean()) == pytest.approx(1.0, abs=1e-5)


def test_masked_normalisation_closes_the_background_parking_evasion():
    """
    Weight parked in background air must not buy the brain a weaker regulariser.

    Normalising over the whole image lets the field satisfy mean(w)=1 by loading the ~60% of
    voxels that are background, where smoothing the displacement costs nothing, and dropping
    every brain voxel to the floor. Measured on a trained 2D run that gave 1.38 outside the
    brain against 0.42 inside. Masked normalisation must leave the brain at mean 1 regardless
    of what the background does.
    """
    raw = torch.full((1, 1, 16, 16), -4.0)
    mask = torch.zeros(1, 1, 16, 16, dtype=torch.bool)
    mask[:, :, 6:10, 6:10] = True
    raw[~mask] = 4.0                       # background driven to the ceiling

    unmasked = VxmLambdaField._normalise_weights(raw, (0.5, 2.0), None)
    masked = VxmLambdaField._normalise_weights(raw, (0.5, 2.0), mask)

    # Whole-image normalisation: the brain is relaxed far below the baseline's budget.
    assert float(unmasked[mask].mean()) < 0.5
    # Masked normalisation: the brain gets exactly the baseline's budget back.
    assert float(masked[mask].mean()) == pytest.approx(1.0, abs=1e-5)


@pytest.mark.parametrize('ndim, shape', [(2, (64, 64)), (3, (32, 32, 32))])
def test_mask_normalised_model_pins_the_brain_budget(ndim, shape):
    """End to end through `forward`, the mean weight over brain voxels is 1."""
    model = VxmLambdaField(ndim=ndim, mask_normalise=True)
    source = torch.zeros(1, 1, *shape)
    target = torch.zeros(1, 1, *shape)
    centre = tuple(slice(s // 4, 3 * s // 4) for s in shape)
    source[(slice(None), slice(None)) + centre] = 0.8
    target[(slice(None), slice(None)) + centre] = 0.9

    with torch.no_grad():
        model.unet.out_layer.conv0.weight.mul_(1000.0)
    lambda_map = model(source, target)['lambda_map']

    mask = (source > 0) | (target > 0)
    assert float(lambda_map[mask].mean()) == pytest.approx(1.0, abs=1e-4)


def test_mask_normalisation_is_off_by_default():
    """Earlier runs must reproduce exactly, so the default has to stay whole-image."""
    assert VxmLambdaField(ndim=2).mask_normalise is False
    assert ExperimentConfig(name='x', variant='lambda_field').lambda_mask_norm is False


def test_build_model_passes_the_mask_normalisation_flag():
    config = ExperimentConfig(name='x', variant='lambda_field', ndim=2, lambda_mask_norm=True)
    assert build_model(config).mask_normalise is True


def test_config_without_the_new_field_still_loads(tmp_path):
    """`config.json` files written before the flag existed must still rebuild."""
    import json
    path = tmp_path / 'config.json'
    path.write_text(json.dumps({'name': 'legacy', 'variant': 'lambda_field', 'ndim': 2}))
    config = ExperimentConfig.load(path)
    assert config.lambda_mask_norm is False


@pytest.mark.parametrize('ndim, shape', [(2, (64, 64)), (3, (32, 32, 32))])
def test_target_skip_fusion_is_the_identity_at_initialisation(ndim, shape):
    """
    Turning on target skips must not perturb the model before it has learned anything.

    The fusion is zero-initialised on the target half and identity on the source half, so the
    fused skip is bit-identical to the source skip at step 0. That makes the target pyramid
    something the network switches on if it helps, rather than a distribution shift dropped on a
    decoder that has not learned to use it.
    """
    model = VxmCrossAttention(ndim=ndim, target_skips=True).eval()
    source = torch.rand(1, 1, *shape)
    target = torch.rand(1, 1, *shape)

    with torch.no_grad():
        _, source_skips = model._encode(source)
        _, target_skips = model._encode(target)
        fused = model._fuse_skips(source_skips, target_skips)

    for fused_skip, source_skip in zip(fused, source_skips):
        assert torch.equal(fused_skip, source_skip)


def test_target_skip_fusion_is_actually_connected_to_the_target():
    """A zero-initialised branch is worthless if the target never reaches it."""
    model = VxmCrossAttention(ndim=2, target_skips=True).eval()
    source = torch.rand(1, 1, 64, 64)
    target = torch.rand(1, 1, 64, 64)

    with torch.no_grad():
        _, source_skips = model._encode(source)
        _, target_skips = model._encode(target)
        width = model.skip_fusion[0].out_channels
        model.skip_fusion[0].weight[:, width:].fill_(0.1)
        fused = model._fuse_skips(source_skips, target_skips)

    assert not torch.equal(fused[0], source_skips[0])


def test_target_skips_keep_the_decoder_widths_unchanged():
    """
    The 1x1 fusion maps 2C -> C, so the decoder is untouched and capacity stays comparable.

    Concatenating the pyramids directly would double every decoder input and inflate the
    parameter count, which would confound the comparison against the baseline.
    """
    plain = VxmCrossAttention(ndim=3)
    fused = VxmCrossAttention(ndim=3, target_skips=True)
    extra = sum(p.numel() for p in fused.parameters()) - sum(p.numel() for p in plain.parameters())
    baseline = sum(p.numel() for p in VxmBaseline(ndim=3).parameters())
    assert extra == 8848
    assert extra / baseline < 0.03


def test_target_skips_are_off_by_default():
    assert VxmCrossAttention(ndim=2).target_skips is False
    assert ExperimentConfig(name='x', variant='cross_attn').cross_attn_target_skips is False


def test_build_model_passes_the_target_skip_flag():
    config = ExperimentConfig(name='x', variant='cross_attn', ndim=2,
                              cross_attn_target_skips=True)
    assert build_model(config).target_skips is True


@pytest.mark.parametrize('ndim, shape', [(2, (8, 8)), (3, (6, 6, 6))])
def test_local_attention_starts_as_a_no_op(ndim, shape):
    """Zero-initialised output projection, so enabling it cannot perturb a fresh network."""
    module = LocalCrossAttention(channels=8, ndim=ndim, radius=1, heads=2).eval()
    with torch.no_grad():
        out = module(torch.rand(1, 8, *shape), torch.rand(1, 8, *shape))
    assert torch.all(out == 0)


@pytest.mark.parametrize('ndim, shape', [(2, (8, 8)), (3, (6, 6, 6))])
def test_local_attention_only_sees_its_neighbourhood(ndim, shape):
    """
    The locality is the whole point -- it is what makes fine-resolution attention affordable.

    Perturbing the target at one corner must change the output there and nowhere far away. If
    this leaks, the module is doing global attention at a resolution whose cost we did not pay
    for, and the complexity argument for windowing collapses.
    """
    module = LocalCrossAttention(channels=8, ndim=ndim, radius=1, heads=2).eval()
    torch.nn.init.normal_(module.project.weight, std=0.5)

    source = torch.rand(1, 8, *shape)
    target = torch.rand(1, 8, *shape)
    corner = tuple(size - 1 for size in shape)
    origin = (0,) * ndim

    perturbed = target.clone()
    perturbed[(0, slice(None)) + corner] += 10.0

    with torch.no_grad():
        before = module(source, target)
        after = module(source, perturbed)

    delta = (after - before).abs()
    assert float(delta[(0, slice(None)) + corner].max()) > 1e-3
    assert float(delta[(0, slice(None)) + origin].max()) == pytest.approx(0.0, abs=1e-6)


def test_local_attention_rejects_indivisible_head_count():
    with pytest.raises(ValueError, match='divisible'):
        LocalCrossAttention(channels=10, ndim=2, radius=1, heads=4)


@pytest.mark.parametrize('ndim, shape', [(2, (160, 192)), (3, (32, 32, 32))])
def test_windowed_model_matches_its_control_at_initialisation(ndim, shape):
    """Turning windowing on must leave an untrained network numerically unchanged."""
    control = VxmCrossAttention(ndim=ndim, target_skips=True, use_attention=False).eval()
    windowed = VxmCrossAttention(ndim=ndim, target_skips=True, use_attention=False,
                                 window_level=3, window_radius=2).eval()
    windowed.load_state_dict(control.state_dict(), strict=False)

    source = torch.rand(1, 1, *shape)
    target = torch.rand(1, 1, *shape)
    with torch.no_grad():
        assert torch.equal(control(source, target)['displacement'],
                           windowed(source, target)['displacement'])


def test_windowing_is_disabled_by_default():
    assert VxmCrossAttention(ndim=2).window_level == -1
    assert ExperimentConfig(name='x', variant='cross_attn').cross_attn_window_level == -1


def test_build_model_passes_the_window_settings():
    config = ExperimentConfig(name='x', variant='cross_attn', ndim=2,
                              cross_attn_window_level=3, cross_attn_window_radius=2)
    model = build_model(config)
    assert model.window_level == 3
    assert model.local_attention.radius == 2


@pytest.mark.parametrize('ndim, shape', [(2, (160, 192)), (3, (32, 32, 32))])
def test_pyramid_emits_one_field_per_decoder_level(ndim, shape):
    from project.models import VxmPyramid
    model = VxmPyramid(ndim=ndim).eval()
    with torch.no_grad():
        out = model(torch.rand(1, 1, *shape), torch.rand(1, 1, *shape))
    assert len(out['pyramid']) == len(model.unet.upsampling_conv_blocks)
    # coarsest first, each level twice the previous, finest matching the input
    sizes = [tuple(f.shape[2:]) for f in out['pyramid']]
    assert sizes[-1] == tuple(shape)
    for coarse, fine in zip(sizes, sizes[1:]):
        assert all(f == 2 * c for c, f in zip(coarse, fine))
    assert out['displacement'].shape == out['pyramid'][-1].shape


@pytest.mark.parametrize('ndim', [2, 3])
def test_upsampling_a_field_rescales_its_magnitudes(ndim):
    """
    A displacement is measured in voxels, so doubling the grid doubles every displacement.

    Skipping the rescale is silent: the field still has the right shape and the model still
    trains, it just quietly discards half of every coarse level's contribution.
    """
    from project.models import VxmPyramid
    model = VxmPyramid(ndim=ndim)
    coarse = torch.ones(1, ndim, *([8] * ndim))
    fine = model._upsample_field(coarse, [16] * ndim)
    assert fine.shape[2:] == torch.Size([16] * ndim)
    assert float(fine.mean()) == pytest.approx(2.0, abs=1e-4)


def test_progressive_warping_changes_the_computation():
    """With identical weights, warping the skips must not be a no-op."""
    from project.models import VxmPyramid
    warped = VxmPyramid(ndim=2, progressive=True).eval()
    plain = VxmPyramid(ndim=2, progressive=False).eval()
    plain.load_state_dict(warped.state_dict())
    source, target = torch.rand(1, 1, 160, 192), torch.rand(1, 1, 160, 192)
    with torch.no_grad():
        assert not torch.equal(warped(source, target)['displacement'],
                               plain(source, target)['displacement'])


def test_deep_supervision_trains_every_level():
    """
    Every flow head must receive gradient, including the coarsest.

    This is the whole point of the objective: with a single loss at full resolution the coarse
    levels can free-ride, which is how the baseline's bottleneck ended up worth only 0.007 Dice.
    """
    from project.losses import pyramid_loss
    from project.models import VxmPyramid
    model = VxmPyramid(ndim=2)
    source, target = torch.rand(2, 1, 160, 192), torch.rand(2, 1, 160, 192)
    out = model(source, target)
    pyramid_loss(target, source, out['pyramid'], lambda_reg=0.1,
                 transformer=model.spatial_transformer)['total'].backward()
    for index, head in enumerate(model.flow_heads):
        assert head.weight.grad is not None, f'level {index} got no gradient'
        assert float(head.weight.grad.abs().sum()) > 0, f'level {index} gradient is zero'


def test_pyramid_loss_weights_the_finest_level_most():
    """
    Coarse terms guide the objective; they must not take it over.

    Asserted against the per-level values the loss reports, rather than by perturbing a field:
    a fixed displacement is proportionally much larger on a coarse grid, so a naive perturbation
    test measures the grid size rather than the weighting.
    """
    from project.losses import pyramid_loss
    torch.manual_seed(0)
    target = torch.rand(1, 1, 32, 32)
    source = torch.rand(1, 1, 32, 32)
    fields = [torch.zeros(1, 2, 32 // 2 ** k, 32 // 2 ** k) for k in (2, 1, 0)]

    result = pyramid_loss(target, source, fields, lambda_reg=0.0)
    levels = result['similarity_levels']
    expected = sum(v / 2 ** (len(levels) - 1 - i) for i, v in enumerate(levels))

    assert float(result['total']) == pytest.approx(float(expected), rel=1e-6)
    assert float(result['similarity']) == pytest.approx(float(levels[-1]), rel=1e-6)
    # weights must be strictly increasing towards the finest level
    weights = [1 / 2 ** (len(levels) - 1 - i) for i in range(len(levels))]
    assert weights == sorted(weights)
    assert weights[-1] == 1.0


def test_build_model_creates_the_pyramid_variant():
    config = ExperimentConfig(name='x', variant='pyramid', ndim=2, pyramid_progressive=False)
    model = build_model(config)
    assert model.progressive is False
    assert len(model.flow_heads) == 5


def test_cascade_composes_transforms_rather_than_adding_them():
    """
    `u = u2 + warp(u1, u2)` must beat naive addition against a true two-stage warp.

    With the convention `moved(x) = src(x + u(x))`, applying stage 1 then stage 2 gives
    `u(x) = u2(x) + u1(x + u2(x))`. Adding the fields is wrong whenever the first stage moves
    anything, because the second stage's correction lives in the frame the first stage created.
    Tested on a smooth image: white noise cannot distinguish these, since double interpolation
    smooths it and single interpolation does not, swamping the effect being measured.
    """
    import voxelmorph as vxm
    from project.misalign import random_displacement

    transformer = vxm.nn.modules.SpatialTransformer()
    grid_y, grid_x = torch.meshgrid(torch.linspace(0, 1, 64), torch.linspace(0, 1, 64),
                                    indexing='ij')
    smooth = (torch.sin(6 * grid_x) * torch.cos(5 * grid_y)).unsqueeze(0).unsqueeze(0)

    first = random_displacement((64, 64), 2, 3.0, seed=1)
    second = random_displacement((64, 64), 2, 2.0, seed=2)

    two_stage = transformer(transformer(smooth, first), second)
    composed = second + transformer(first, second)
    added = first + second

    composed_error = float((transformer(smooth, composed) - two_stage).abs().mean())
    added_error = float((transformer(smooth, added) - two_stage).abs().mean())
    assert composed_error < added_error / 3


@pytest.mark.parametrize('scales, expected', [((2, 1), 194800), ((1, 1), 222512)])
def test_cascade_parameter_counts(scales, expected):
    """
    A stage at scale `s` drops log2(s) UNet levels, which is what makes (2,1) cheaper than (1,1).

    Pinned because the parameter count is the whole basis of the capacity control: the claim is
    that the cascade beats a baseline widened *past* its own size, so its size must be known.
    """
    config = ExperimentConfig(name='x', variant='coarse_to_fine', ndim=2, stage_scales=scales)
    assert sum(p.numel() for p in build_model(config).parameters()) == expected


def test_cascade_rejects_non_power_of_two_scales():
    from project.models import VxmCoarseToFine
    with pytest.raises(ValueError, match='powers of two'):
        VxmCoarseToFine(ndim=2, stage_scales=(3, 1))


def test_cascade_warped_source_matches_its_composed_field():
    from project.models import VxmCoarseToFine
    model = VxmCoarseToFine(ndim=2).eval()
    source, target = torch.rand(1, 1, 160, 192), torch.rand(1, 1, 160, 192)
    with torch.no_grad():
        out = model(source, target)
        assert torch.allclose(model.spatial_transformer(source, out['displacement']),
                              out['warped_source'], atol=1e-6)


def test_widened_baseline_gets_its_own_run_name():
    """
    A capacity control must not reuse the plain baseline's directory.

    Depth alone does not identify an architecture: a widened UNet of the same depth would collide
    with the baseline and overwrite the very result it exists to be compared against.
    """
    from project.configs import build_matrix
    plain = build_matrix(ndim=2, lambdas=(0.25,), integration_steps=(0,),
                         variants=('baseline',))[0]
    wide = build_matrix(ndim=2, lambdas=(0.25,), integration_steps=(0,), variants=('baseline',),
                        nb_features=(24, 44, 44, 44, 44))[0]
    assert plain.name == '2d_baseline_lam0.25_disp'
    assert wide.name != plain.name
    assert wide.nb_features == (24, 44, 44, 44, 44)


@pytest.mark.parametrize('ndim', [2, 3])
def test_fathead_matches_the_pyramid_parameter_budget(ndim):
    """
    The control exists to separate capacity from multi-resolution structure.

    It is only able to do that if the budgets really match: the pyramid's win is ~2% of the
    model, so a control that drifted by even a few hundred parameters would be answering a
    different question. The control is also required to be no *larger* than what it controls
    for, so a win can never be explained by it having been handed more.
    """
    pyramid = sum(p.numel() for p in VxmPyramid(ndim=ndim, progressive=False).parameters())
    fathead = sum(p.numel() for p in VxmFatHead(ndim=ndim).parameters())
    assert fathead <= pyramid
    assert abs(fathead - pyramid) / pyramid < 0.001


@pytest.mark.parametrize('ndim', [2, 3])
def test_fathead_spends_its_budget_at_one_resolution(ndim):
    """
    The whole point is that the extra parameters sit at full resolution, not across scales.

    A regression that reintroduced per-level heads would still match on parameter count while
    silently making the control a copy of the model it controls for.
    """
    model = VxmFatHead(ndim=ndim)
    assert not hasattr(model, 'flow_heads')
    assert model.hidden_conv.in_channels == model.unet.up_actual_channels[-1]
    assert model.flow_head.out_channels == ndim


def test_fathead_starts_at_the_identity():
    """
    Every variant must start from the identity, or a comparison measures the starting point.
    """
    model = VxmFatHead(ndim=2).eval()
    source, target = _inputs(1, (64, 64))
    assert model(source, target)['displacement'].abs().max().item() < 1e-3


def test_fathead_gradients_reach_the_extra_convolution():
    """A control whose extra parameters were dead would understate what capacity alone buys."""
    model = VxmFatHead(ndim=2)
    source, target = _inputs(1, (64, 64))
    model(source, target)['displacement'].pow(2).sum().backward()
    assert model.hidden_conv.weight.grad is not None
    assert model.hidden_conv.weight.grad.abs().sum().item() > 0


@pytest.mark.parametrize('ndim', [2, 3])
def test_matched_hidden_holds_the_budget_across_kernel_sizes(ndim):
    """
    The 1x1 head is the control that separates capacity from receptive field.

    It only separates them if it still costs the same. A 1x1 kernel is cheaper per channel, so
    matching on parameters must widen the hidden layer to compensate -- if `matched_hidden`
    ignored `kernel`, the 1x1 arm would quietly be a much smaller model and any loss would be
    read as "receptive field matters" when it only meant "fewer parameters".
    """
    pyramid = sum(p.numel() for p in VxmPyramid(ndim=ndim, progressive=False).parameters())
    for kernel in (1, 3):
        model = VxmFatHead(ndim=ndim, kernel=kernel)
        total = sum(p.numel() for p in model.parameters())
        assert total <= pyramid
        assert abs(total - pyramid) / pyramid < 0.001
    assert (VxmFatHead(ndim=ndim, kernel=1).hidden_conv.out_channels
            > VxmFatHead(ndim=ndim, kernel=3).hidden_conv.out_channels)


def test_one_by_one_head_really_has_no_spatial_extent():
    """A 1x1 head that silently kept padding would still see neighbours and prove nothing."""
    model = VxmFatHead(ndim=2, kernel=1)
    assert model.hidden_conv.kernel_size == (1, 1)
    assert model.flow_head.kernel_size == (1, 1)
    assert model.hidden_conv.padding == (0, 0)


def test_image_skip_actually_reaches_the_head():
    """
    The head must consume the raw images, not just be built wider and ignore them.

    Changing only the source outside the region the UNet could propagate from would still
    change the output through the encoder, so this checks the wiring by channel count -- the
    one thing a forward-pass test cannot distinguish from ordinary sensitivity.
    """
    plain = VxmFatHead(ndim=2, image_skip=False)
    skipped = VxmFatHead(ndim=2, image_skip=True)
    assert skipped.hidden_conv.in_channels == plain.hidden_conv.in_channels + 2


def test_multiscale_features_start_at_the_identity_and_use_every_level():
    """Each decoder level must be projected and reach the head, or 'multi-scale' is a misnomer."""
    model = VxmMultiScaleFeatures(ndim=2).eval()
    assert len(model.projections) == len(model.unet.up_actual_channels)
    assert model.flow_head.in_channels == 4 * len(model.projections)
    source, target = _inputs(1, (64, 64))
    assert model(source, target)['displacement'].abs().max().item() < 1e-3


def test_multiscale_features_does_not_rescale_feature_maps():
    """
    Upsampling a *field* must rescale magnitudes; upsampling *features* must not.

    Getting this backwards is invisible -- the model still trains and still produces plausible
    fields -- so it is pinned here: a constant feature map must survive upsampling unchanged.
    """
    model = VxmMultiScaleFeatures(ndim=2)
    coarse = torch.full((1, 4, 8, 8), 3.0)
    up = torch.nn.functional.interpolate(coarse, size=(64, 64), mode='bilinear',
                                         align_corners=True)
    assert up.min().item() == pytest.approx(3.0)
    assert up.max().item() == pytest.approx(3.0)


def test_seed_zero_keeps_the_bare_run_name():
    """
    Seed 0 must not gain a suffix, or every existing single-seed run is orphaned.

    The whole matrix was trained at seed 0 under unsuffixed names. If seeding renamed them, an
    ensemble would retrain work already done and, worse, compare against a *different* run than
    the one every earlier conclusion was drawn from.
    """
    from project.configs import build_matrix
    configs = build_matrix(ndim=2, variants=('baseline',), lambdas=(0.25,),
                           integration_steps=(7,), seeds=(0, 1, 2))
    names = [c.name for c in configs]
    assert '2d_baseline_lam0.25_svf' in names
    assert '2d_baseline_lam0.25_svf_seed1' in names
    assert sorted(c.seed for c in configs) == [0, 1, 2]


def test_seeds_default_to_a_single_run():
    """Adding the option must not silently triple every sweep that does not ask for it."""
    from project.configs import build_matrix
    configs = build_matrix(ndim=2, variants=('baseline',), lambdas=(0.25,),
                           integration_steps=(7,))
    assert len(configs) == 1 and configs[0].seed == 0


def test_sweep_stays_single_seed_unless_seeds_are_requested():
    """
    An ordinary sweep must produce one run per configuration.

    `--seeds` serves two paths: `--ensemble-of` wants five members by default, while the
    experiment matrix must stay single-seed. A truthy argparse default made `if args.seeds`
    always true, which silently multiplied every sweep by five -- five times the GPU time, and
    a matrix nobody asked for.
    """
    from project.configs import build_matrix
    assert len(build_matrix(ndim=2, variants=('baseline',), lambdas=(0.25,),
                            integration_steps=(7,))) == 1


def test_run_names_are_stable_across_the_merge():
    """
    Existing result directories are addressed by generated name.

    If a name changes, `--skip-existing` stops recognising finished work and every earlier
    result is orphaned from the config that produced it. These are real directories under
    results/, so a rename breaks the experimental record rather than just a test.
    """
    from project.configs import build_matrix
    cases = [
        (dict(variants=('baseline',), lambdas=(0.25,)), '2d_baseline_lam0.25_svf'),
        (dict(variants=('pyramid',), sweep_variants=('pyramid',), lambdas=(0.25,),
              pyramid_progressive=False, deep_supervision=False),
         '2d_pyramid_lam0.25_svf_noprog_nods'),
        (dict(variants=('baseline',), lambdas=(0.25,), head_lr_mult=100),
         '2d_baseline_lam0.25_svf_hlr100'),
    ]
    for kwargs, expected in cases:
        names = [c.name for c in build_matrix(ndim=2, integration_steps=(7,), **kwargs)]
        assert expected in names, f'{expected} not in {names}'

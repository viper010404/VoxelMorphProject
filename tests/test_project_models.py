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
from project.models import VxmBaseline, VxmCrossAttention, VxmLambdaField, build_model


CASES = [
    ('baseline', 2, (64, 64), 2),
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

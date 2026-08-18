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

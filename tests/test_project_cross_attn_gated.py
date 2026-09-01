"""
Tests for the gated cross-attention variant.

The defining property -- and the entire reason this variant exists -- is that at initialisation
it is *exactly* the plain UNet path, so adding attention cannot damage a working model. If
`test_gate_is_closed_at_init` or `test_at_init_matches_ungated_path` ever fails, the variant has
silently become the thing it was built to fix.
"""

import pytest
import torch

from project.configs import ExperimentConfig
from project.models import VxmCrossAttentionGated, build_model


def _inputs(batch, shape):
    torch.manual_seed(0)
    return torch.rand(batch, 1, *shape), torch.rand(batch, 1, *shape)


@pytest.mark.parametrize('ndim,shape,batch', [(2, (64, 64), 2), (3, (32, 32, 32), 1)])
@pytest.mark.parametrize('integration_steps', [0, 7])
def test_forward_shapes(ndim, shape, batch, integration_steps):
    """Shapes match the shared interface."""
    model = VxmCrossAttentionGated(ndim=ndim, integration_steps=integration_steps)
    source, target = _inputs(batch, shape)
    outputs = model(source, target)

    assert outputs['displacement'].shape == (batch, ndim, *shape)
    assert outputs['warped_source'].shape == source.shape
    assert ('velocity' in outputs) == (integration_steps > 0)


def test_gate_is_closed_at_init():
    """The gate must start at exactly zero, or the branch perturbs the model from step 1."""
    assert float(VxmCrossAttentionGated(ndim=2)(*_inputs(1, (64, 64)))['gate']) == 0.0


def test_at_init_matches_ungated_path():
    """
    At init the output equals the plain UNet path with the attention branch removed.

    This is the property the original `VxmCrossAttention` lacks: its post-residual LayerNorm
    rescales the representation, so no initialisation reproduces the baseline.
    """
    model = VxmCrossAttentionGated(ndim=2).eval()
    source, target = _inputs(1, (64, 64))

    with torch.no_grad():
        gated = model(source, target)['displacement']

        # The ungated path: decode the forward bottleneck with no attention contribution.
        bottleneck, skips = model._encode(torch.cat([source, target], dim=1))
        plain = model.flow_layer(model._decode(bottleneck, skips))

    assert torch.allclose(gated, plain, atol=1e-6)


def test_gate_opens_only_by_training():
    """The gate is a learned parameter and must receive gradient, or it can never open."""
    model = VxmCrossAttentionGated(ndim=2)
    model(*_inputs(1, (64, 64)))['displacement'].sum().backward()

    assert model.gate.grad is not None


def test_encoder_weights_are_shared_between_streams():
    """
    The swapped stream must reuse the encoder, adding no encoder parameters.

    Otherwise a measured gain could be extra capacity rather than the attention mechanism.
    """
    model = VxmCrossAttentionGated(ndim=2)
    baseline_unet = build_model(ExperimentConfig(name='t', variant='lambda_field', ndim=2)).unet

    gated_encoder = sum(p.numel() for p in model.unet.downsampling_conv_blocks.parameters())
    plain_encoder = sum(p.numel() for p in baseline_unet.downsampling_conv_blocks.parameters())
    assert gated_encoder == plain_encoder


def test_build_model_routes_variant():
    """`build_model` recognises the variant."""
    config = ExperimentConfig(name='t', variant='cross_attn_gated', ndim=2)
    assert isinstance(build_model(config), VxmCrossAttentionGated)

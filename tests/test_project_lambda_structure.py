"""
Tests for the per-structure lambda variant.

Two properties make this variant a valid experiment rather than a confound, and both are
asserted here:

* the weight map must have mean 1 over the brain mask, so the regularisation *budget* matches
  the baseline's and the comparison isolates allocation from strength, and
* the map must be piecewise constant on the segmentation -- that is the whole hypothesis. If it
  ever varied within a structure, the variant would silently be a per-voxel field again.
"""

import pytest
import torch

from project.configs import ExperimentConfig
from project.models import VxmLambdaStructure, build_model, forward_model


N_LABELS = 8


def _inputs(batch, shape):
    torch.manual_seed(0)
    source = torch.rand(batch, 1, *shape)
    target = torch.rand(batch, 1, *shape)
    seg = torch.randint(0, N_LABELS, (batch, 1, *shape))
    return source, target, seg


@pytest.mark.parametrize('ndim,shape,batch', [(2, (64, 64), 2), (3, (32, 32, 32), 1)])
@pytest.mark.parametrize('integration_steps', [0, 7])
def test_forward_shapes(ndim, shape, batch, integration_steps):
    """Shapes match the shared interface, so nothing downstream needs to special-case it."""
    model = VxmLambdaStructure(ndim=ndim, n_labels=N_LABELS,
                               integration_steps=integration_steps)
    source, target, seg = _inputs(batch, shape)
    outputs = model(source, target, seg=seg)

    assert outputs['displacement'].shape == (batch, ndim, *shape)
    assert outputs['warped_source'].shape == source.shape
    assert outputs['lambda_map'].shape == (batch, 1, *shape)
    assert outputs['structure_lambda'].shape == (batch, N_LABELS)
    assert ('velocity' in outputs) == (integration_steps > 0)


def test_lambda_map_has_unit_mean_in_mask():
    """The budget guarantee: mean 1 inside the brain mask, matching the baseline's spend."""
    model = VxmLambdaStructure(ndim=2, n_labels=N_LABELS)
    source, target, seg = _inputs(2, (64, 64))
    lambda_map = model(source, target, seg=seg)['lambda_map']

    mask = ((source > 0) | (target > 0)).float()
    masked_mean = (lambda_map * mask).flatten(1).sum(1) / mask.flatten(1).sum(1)
    assert torch.allclose(masked_mean, torch.ones_like(masked_mean), atol=1e-5)


def test_map_is_piecewise_constant_on_structures():
    """
    Every voxel of a structure carries the same weight.

    This is the defining difference from `VxmLambdaField`; without it the variant would have
    per-voxel freedom again and the parameter-count argument would be false.
    """
    model = VxmLambdaStructure(ndim=2, n_labels=N_LABELS)
    source, target, seg = _inputs(1, (64, 64))
    lambda_map = model(source, target, seg=seg)['lambda_map']

    for label in range(N_LABELS):
        values = lambda_map[0, 0][seg[0, 0] == label]
        if values.numel():
            assert torch.allclose(values, values[0].expand_as(values), atol=1e-6)


def test_label_id_beyond_table_raises():
    """
    An id larger than `n_labels` must fail loudly.

    Label ids are read from the NIfTI unremapped, so a too-small table would otherwise merge
    distinct structures into one weight and return a plausible but meaningless result.
    """
    model = VxmLambdaStructure(ndim=2, n_labels=4)
    source, target, _ = _inputs(1, (64, 64))
    seg = torch.full((1, 1, 64, 64), 9)

    with pytest.raises(ValueError, match='n_labels'):
        model(source, target, seg=seg)


def test_missing_segmentation_raises():
    """The variant is semi-supervised; it cannot silently fall back to an unsupervised path."""
    model = VxmLambdaStructure(ndim=2, n_labels=N_LABELS)
    source, target, _ = _inputs(1, (64, 64))

    with pytest.raises(ValueError, match='segmentation'):
        model(source, target)


def test_structure_lambda_receives_gradient():
    """
    The per-structure weights must be trainable through the scatter.

    `gather` is differentiable but easy to break with an in-place scatter, so this pins it.
    """
    torch.manual_seed(0)
    model = VxmLambdaStructure(ndim=2, n_labels=N_LABELS)
    source, target, seg = _inputs(1, (64, 64))
    lambda_map = model(source, target, seg=seg)['lambda_map']
    # A plain .sum() is very nearly invariant to the weights, because the map is normalised to
    # unit mean over the brain mask -- so the gradient is floating-point residue and is exactly
    # zero for some seeds (2 of 10 measured). Project onto a fixed random direction instead,
    # which depends on the weights and gives this test something real to detect.
    projection = torch.randn_like(lambda_map)
    (lambda_map * projection).sum().backward()

    weight_head_grads = [p.grad for p in model.unet.out_layer.parameters() if p.grad is not None]
    assert weight_head_grads
    assert any(g.abs().sum() > 0 for g in weight_head_grads)


def test_build_model_routes_variant():
    """`build_model` and `forward_model` handle the variant without caller-side special casing."""
    config = ExperimentConfig(name='t', variant='lambda_structure', ndim=2, n_labels=N_LABELS)
    model = build_model(config)
    assert isinstance(model, VxmLambdaStructure)

    source, target, seg = _inputs(1, (64, 64))
    assert 'lambda_map' in forward_model(model, source, target, seg)

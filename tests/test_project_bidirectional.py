"""
Tests for the inverse-consistency loss used by the bidirectional variant.

The property that matters is that this is the *differentiable counterpart of the reported
metric*: `metrics.inverse_consistency` scores every run using `voxelmorph`'s numpy composition,
while training needs a batched torch version. `test_agrees_with_reported_metric` checks the two
rank the same fields the same way, so the penalty reduces the number that actually gets reported
rather than a near neighbour of it.
"""

import torch
import voxelmorph as vxm

from project.losses import compose_displacements, inverse_consistency_loss
from project.metrics import inverse_consistency


def _transformer():
    return vxm.nn.modules.SpatialTransformer()


def _smooth_field(scale=3.0, seed=0):
    torch.manual_seed(seed)
    return torch.nn.functional.avg_pool2d(torch.randn(1, 2, 64, 64), 15, 1, 7) * scale


def test_zero_for_identity():
    """Two zero transforms invert each other exactly."""
    zero = torch.zeros(1, 2, 64, 64)
    assert float(inverse_consistency_loss(zero, zero, _transformer())) == 0.0


def test_small_for_approximate_inverse_large_otherwise():
    """A field and its negation nearly invert; a field and itself do not."""
    field = _smooth_field()
    transformer = _transformer()

    approximate = float(inverse_consistency_loss(field, -field, transformer))
    unrelated = float(inverse_consistency_loss(field, field, transformer))

    assert approximate < 0.01
    assert unrelated > 50 * approximate


def test_is_symmetric_in_its_arguments():
    """
    Both composition orders are penalised, so swapping the arguments changes nothing.

    Constraining only one direction would let the other drift while the loss still read zero.
    """
    a, b = _smooth_field(seed=0), _smooth_field(seed=1)
    transformer = _transformer()
    forward = float(inverse_consistency_loss(a, b, transformer))
    backward = float(inverse_consistency_loss(b, a, transformer))
    assert abs(forward - backward) < 1e-6


def test_agrees_with_reported_metric():
    """
    The training penalty must order fields the same way the reported metric does.

    `metrics.inverse_consistency` is numpy, detached and per-sample; this loss is torch and
    batched. They need not be numerically identical, but a field pair that is more
    inverse-consistent by one must be more inverse-consistent by the other.
    """
    transformer = _transformer()
    field = _smooth_field()

    good_loss = float(inverse_consistency_loss(field, -field, transformer))
    bad_loss = float(inverse_consistency_loss(field, _smooth_field(seed=7), transformer))

    good_metric = inverse_consistency(field, -field)[0]
    bad_metric = inverse_consistency(field, _smooth_field(seed=7))[0]

    assert (good_loss < bad_loss) == (good_metric < bad_metric)


def test_is_differentiable():
    """The penalty must reach both fields, or only one direction would be trained."""
    a = _smooth_field(seed=0).requires_grad_(True)
    b = _smooth_field(seed=1).requires_grad_(True)
    inverse_consistency_loss(a, b, _transformer()).backward()

    assert a.grad is not None and a.grad.abs().sum() > 0
    assert b.grad is not None and b.grad.abs().sum() > 0


def test_composition_is_not_addition():
    """
    Guards the shared composition helper against being simplified to `a + b`.

    Addition is *nearly* right for small fields, which is exactly why it needs pinning.
    """
    a, b = _smooth_field(seed=0), _smooth_field(seed=1)
    composed = compose_displacements(a, b, _transformer())
    assert not torch.allclose(composed, a + b, atol=1e-3)

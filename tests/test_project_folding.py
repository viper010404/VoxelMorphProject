"""
Tests for the differentiable Jacobian determinant and the anti-folding penalty.

The load-bearing test is `test_matches_upstream_numpy_implementation`. The training penalty and
the reported metric must be the *same* quantity: `metrics.folding` delegates to
`voxelmorph.py.utils.jacobian_determinant` (numpy, `np.gradient`, i.e. central differences),
while the loss needs a torch version. If the two ever diverge, training would be optimising a
near neighbour of the reported number and the results would silently stop meaning what they say.
"""

import numpy as np
import pytest
import torch
import voxelmorph as vxm

from project.losses import folding_penalty, jacobian_determinant


def _random_field(ndim, shape, scale=0.6, seed=0):
    torch.manual_seed(seed)
    # Smooth the noise so the field resembles a real deformation rather than white noise;
    # an unsmoothed field folds essentially everywhere and would make the comparison trivial.
    field = torch.randn(1, ndim, *shape)
    pool = torch.nn.AvgPool2d(5, 1, 2) if ndim == 2 else torch.nn.AvgPool3d(5, 1, 2)
    return pool(field) * scale


@pytest.mark.parametrize('ndim,shape', [(2, (48, 56)), (3, (24, 28, 32))])
def test_matches_upstream_numpy_implementation(ndim, shape):
    """The torch determinant equals the upstream numpy one on the shared interior."""
    disp = _random_field(ndim, shape)

    mine = jacobian_determinant(disp)[0].numpy()

    field = disp[0].permute(*range(1, ndim + 1), 0).numpy()
    theirs = vxm.py.utils.jacobian_determinant(field)
    # `np.gradient` uses one-sided differences on the boundary rim; compare the interior, which
    # is what `jacobian_determinant` returns.
    interior = tuple(slice(1, -1) for _ in range(ndim))

    assert mine.shape == theirs[interior].shape
    np.testing.assert_allclose(mine, theirs[interior], rtol=1e-4, atol=1e-5)


@pytest.mark.parametrize('ndim,shape', [(2, (32, 32)), (3, (16, 16, 16))])
def test_identity_has_unit_determinant(ndim, shape):
    """A zero displacement is the identity transform, whose Jacobian determinant is 1."""
    determinant = jacobian_determinant(torch.zeros(1, ndim, *shape))
    assert torch.allclose(determinant, torch.ones_like(determinant), atol=1e-6)


def test_penalty_is_zero_without_folding():
    """The penalty is one-sided: a well-behaved field must incur no cost at all."""
    assert float(folding_penalty(torch.zeros(1, 2, 32, 32))) == 0.0


def test_penalty_is_positive_when_folded():
    """A field that folds must be penalised."""
    # A displacement of -x cancels the identity and inverts orientation beyond it.
    disp = torch.zeros(1, 2, 32, 32)
    disp[0, 0] = -2.0 * torch.arange(32).float().reshape(-1, 1)
    assert float(folding_penalty(disp)) > 0.0


def test_penalty_is_differentiable():
    """The penalty must produce gradients, or it cannot influence training."""
    disp = torch.zeros(1, 2, 32, 32, requires_grad=True)
    disp2 = disp - 2.0 * torch.arange(32).float().reshape(-1, 1)
    folding_penalty(disp2).backward()
    assert disp.grad is not None and disp.grad.abs().sum() > 0


def test_margin_penalises_near_singular_fields():
    """A positive margin also penalises a compressive but still-invertible field."""
    disp = torch.zeros(1, 2, 32, 32)
    disp[0, 0] = -0.7 * torch.arange(32).float().reshape(-1, 1)   # det ~ 0.3 > 0
    assert float(folding_penalty(disp, margin=0.0)) == 0.0
    assert float(folding_penalty(disp, margin=0.5)) > 0.0

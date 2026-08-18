"""
Tests for the project's smoothness and similarity losses.

The important property here is that the weighted penalty reduces exactly to the standard one
when the weight is uniform. If that failed, the baseline and lambda-field runs would differ by
two things at once -- the weighting *and* an inconsistent gradient implementation -- and no
conclusion could be drawn from the comparison.
"""

import pytest
import torch
import neurite as ne

from project.losses import image_similarity, registration_loss, spatial_smoothness


@pytest.fixture
def displacement_2d():
    """A reproducible non-trivial 2D displacement field."""
    torch.manual_seed(0)
    return torch.randn(2, 2, 32, 32)


def test_matches_neurite_spatial_gradient(displacement_2d):
    """Unweighted smoothness must equal neurite's SpatialGradient."""
    ours = spatial_smoothness(displacement_2d, weight=None, penalty='l2')
    theirs = ne.nn.modules.SpatialGradient('l2')(displacement_2d)
    assert torch.allclose(ours, theirs, atol=1e-6)


def test_uniform_weight_reduces_to_unweighted(displacement_2d):
    """A weight map of all ones must reproduce the unweighted penalty exactly."""
    weight = torch.ones(2, 1, 32, 32)
    assert torch.allclose(
        spatial_smoothness(displacement_2d, weight=weight),
        spatial_smoothness(displacement_2d, weight=None),
        atol=1e-6,
    )


def test_zero_displacement_has_zero_penalty():
    """A constant field has no spatial variation and so no smoothness cost."""
    assert spatial_smoothness(torch.zeros(1, 2, 16, 16)) == 0.0
    assert spatial_smoothness(torch.full((1, 2, 16, 16), 3.0)) == pytest.approx(0.0)


def test_weight_concentrates_penalty():
    """
    Weighting must actually redistribute cost, not merely rescale it.

    A field that varies only in one half is penalised more when the weight is concentrated on
    that half than when it is concentrated on the smooth half -- at equal mean weight.
    """
    disp = torch.zeros(1, 2, 16, 16)
    disp[0, 0, :8] = torch.arange(8, dtype=torch.float32).view(8, 1)

    heavy_on_rough = torch.full((1, 1, 16, 16), 0.5)
    heavy_on_rough[0, 0, :8] = 1.5
    heavy_on_smooth = torch.full((1, 1, 16, 16), 1.5)
    heavy_on_smooth[0, 0, :8] = 0.5

    assert heavy_on_rough.mean() == pytest.approx(heavy_on_smooth.mean())
    assert (spatial_smoothness(disp, weight=heavy_on_rough)
            > spatial_smoothness(disp, weight=heavy_on_smooth))


def test_l1_and_l2_penalties_differ(displacement_2d):
    """Both penalty types are available and are not accidentally the same code path."""
    assert not torch.allclose(
        spatial_smoothness(displacement_2d, penalty='l1'),
        spatial_smoothness(displacement_2d, penalty='l2'),
    )


def test_invalid_penalty_rejected(displacement_2d):
    with pytest.raises(ValueError, match="penalty must be"):
        spatial_smoothness(displacement_2d, penalty='l3')


def test_image_similarity_zero_for_identical_images():
    image = torch.rand(2, 1, 16, 16)
    assert image_similarity(image, image.clone()) == pytest.approx(0.0)


def test_registration_loss_components_combine():
    """Total must equal similarity + lambda * smoothness."""
    torch.manual_seed(0)
    target = torch.rand(1, 1, 16, 16)
    warped = torch.rand(1, 1, 16, 16)
    disp = torch.randn(1, 2, 16, 16)
    lambda_reg = 0.02

    losses = registration_loss(target, warped, disp, lambda_reg)
    expected = losses['similarity'] + lambda_reg * losses['smoothness']
    assert torch.allclose(losses['total'], expected)


def test_smoothness_rejects_wrong_rank():
    with pytest.raises(ValueError, match='must be'):
        spatial_smoothness(torch.zeros(4, 4))

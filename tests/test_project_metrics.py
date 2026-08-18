"""
Tests for the evaluation metrics.

These use known-answer cases rather than regression values, because the metrics are the
instrument the whole project is measured with: if Dice or the folding count is subtly wrong,
every conclusion drawn from the comparison table is wrong with it.
"""

import numpy as np
import pytest
import torch

import voxelmorph.nn.modules as vxm_modules

from project.metrics import (
    dice_per_structure,
    folding,
    inverse_consistency,
    mean_dice,
    warp_segmentation,
)


@pytest.fixture
def segmentation():
    """A 2D label map containing structures 1 and 2 but not 3."""
    seg = np.zeros((32, 32), dtype=np.int16)
    seg[8:20, 8:20] = 1
    seg[22:28, 5:12] = 2
    return seg


def test_identical_segmentations_score_one(segmentation):
    scores = dice_per_structure(segmentation, segmentation.copy(), labels=[1, 2])
    assert scores[1] == pytest.approx(1.0)
    assert scores[2] == pytest.approx(1.0)


def test_structure_absent_from_both_is_none_not_zero(segmentation):
    """
    A structure present in neither segmentation must be excluded, not scored 0.

    The upstream numpy dice returns 0.0 here (0 divided by an epsilon). Averaging that in would
    penalise a model for failing to align anatomy that was never in the slice.
    """
    scores = dice_per_structure(segmentation, segmentation.copy(), labels=[1, 2, 3])
    assert scores[3] is None
    assert mean_dice(scores) == pytest.approx(1.0)


def test_structure_present_in_only_one_scores_zero(segmentation):
    """A genuine total miss must score 0 -- that is a real failure, not a missing structure."""
    other = segmentation.copy()
    other[other == 2] = 0
    scores = dice_per_structure(segmentation, other, labels=[1, 2, 3])
    assert scores[2] == pytest.approx(0.0)
    assert scores[3] is None
    assert mean_dice(scores) == pytest.approx(0.5)


def test_mean_dice_of_all_absent_is_nan():
    empty = np.zeros((8, 8), dtype=np.int16)
    assert np.isnan(mean_dice(dice_per_structure(empty, empty, labels=[1, 2])))


def test_partial_overlap_between_zero_and_one():
    """A shifted structure must land strictly between total miss and perfect overlap."""
    a = np.zeros((32, 32), dtype=np.int16)
    a[10:20, 10:20] = 1
    b = np.zeros((32, 32), dtype=np.int16)
    b[13:23, 10:20] = 1
    score = dice_per_structure(a, b, labels=[1])[1]
    assert 0.0 < score < 1.0


@pytest.mark.parametrize('shape,ndim', [((32, 32), 2), ((16, 16, 16), 3)])
def test_zero_displacement_does_not_fold(shape, ndim):
    """The identity transform is everywhere invertible."""
    results = folding(torch.zeros(2, ndim, *shape))
    assert all(r['count'] == 0 and r['fraction'] == 0.0 for r in results)


def test_folded_field_is_detected():
    """A field with opposing displacements must fold where they meet."""
    disp = torch.zeros(1, 2, 32, 32)
    disp[0, 0, :16, :] = 6.0
    disp[0, 0, 16:, :] = -6.0
    result = folding(disp)[0]
    assert result['count'] > 0
    assert 0.0 < result['fraction'] < 1.0


def test_inverse_consistency_better_for_true_inverse():
    """
    Integrating -v must invert integrating +v far better than repeating +v does.

    This is the property that makes the diffeomorphic setting worth reporting: the inverse is
    available for free and is genuinely an inverse.
    """
    integrator = vxm_modules.IntegrateVelocityField(steps=7)
    torch.manual_seed(0)
    velocity = torch.randn(2, 2, 32, 32) * 0.3

    true_inverse = np.mean(inverse_consistency(integrator(velocity), integrator(-velocity)))
    wrong_inverse = np.mean(inverse_consistency(integrator(velocity), integrator(velocity)))

    assert true_inverse < wrong_inverse


def test_inverse_consistency_zero_for_zero_field():
    zero = torch.zeros(1, 2, 16, 16)
    assert inverse_consistency(zero, zero)[0] == pytest.approx(0.0, abs=1e-6)


def test_warp_segmentation_invents_no_labels(segmentation):
    """
    Nearest-neighbour warping must not interpolate label values into new ones.

    Linear interpolation of categorical labels would produce values between 1 and 2, which are
    not anatomy.
    """
    disp = torch.zeros(1, 2, 32, 32)
    disp[0, 0] = 2.5
    seg = torch.from_numpy(segmentation).view(1, 1, 32, 32)
    warped = warp_segmentation(seg, disp)

    assert set(np.unique(warped.numpy())).issubset({0, 1, 2})
    assert warped.dtype == seg.dtype


def test_warp_segmentation_with_zero_field_is_identity(segmentation):
    seg = torch.from_numpy(segmentation).view(1, 1, 32, 32)
    warped = warp_segmentation(seg, torch.zeros(1, 2, 32, 32))
    assert torch.equal(warped, seg)

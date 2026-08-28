"""
Tests for the ensemble uncertainty analysis.

The summary statistics here are what branch C reports, so they are tested on constructed inputs
with known answers rather than on a trained ensemble: if the uncertainty-versus-error
correlation is computed the wrong way round, the conclusion inverts while still looking
plausible.
"""

import numpy as np
import pytest

from project.uncertainty import (
    _structure_means,
    summarise,
    uncertainty_error_correlation,
)


def _row(uncertainty, dice, members=(0.5, 0.6), ensemble=0.55):
    """Build a per-pair row with the fields the summary functions read."""
    return {
        'dice_ensemble_mean_field': ensemble,
        'dice_members': list(members),
        'dice_member_mean': float(np.mean(members)),
        'dice_member_std': float(np.std(members, ddof=1)),
        'uncertainty_mean': 1.0,
        'uncertainty_max': 2.0,
        'folding_ensemble_mean_field': 0.0,
        'per_structure_uncertainty': {str(k): v for k, v in uncertainty.items()},
        'per_structure_dice_ensemble': {str(k): v for k, v in dice.items()},
    }


def test_structure_means_averages_within_the_mask():
    values = np.array([[1.0, 1.0], [3.0, 5.0]])
    seg = np.array([[1, 1], [2, 2]])
    means = _structure_means(values, seg, [1, 2])
    assert means[1] == pytest.approx(1.0)
    assert means[2] == pytest.approx(4.0)


def test_structure_absent_from_the_map_is_none():
    """A structure with no voxels must be None, so it is dropped rather than averaged as 0."""
    values = np.ones((2, 2))
    seg = np.ones((2, 2), dtype=int)
    means = _structure_means(values, seg, [1, 7])
    assert means[7] is None


def test_correlation_is_positive_when_uncertainty_tracks_error():
    """
    High disagreement on the structures that register worst must give a positive correlation.

    This is the sign convention the whole branch rests on: positive means "uncertain where
    wrong", which is what makes the estimate usable as a failure detector.
    """
    rows = [_row(
        uncertainty={1: 0.1, 2: 0.5, 3: 0.9, 4: 1.3},
        dice={1: 0.9, 2: 0.7, 3: 0.5, 4: 0.3},
    )]
    result = uncertainty_error_correlation(rows)
    assert result['mean_spearman'] == pytest.approx(1.0)
    assert result['n_pairs'] == 1


def test_correlation_is_negative_when_uncertainty_is_anticorrelated():
    rows = [_row(
        uncertainty={1: 1.3, 2: 0.9, 3: 0.5, 4: 0.1},
        dice={1: 0.9, 2: 0.7, 3: 0.5, 4: 0.3},
    )]
    assert uncertainty_error_correlation(rows)['mean_spearman'] == pytest.approx(-1.0)


def test_correlation_skips_pairs_with_too_few_structures():
    """Two structures give a Spearman correlation of +-1 by construction; it carries no signal."""
    rows = [_row(uncertainty={1: 0.1, 2: 0.5}, dice={1: 0.9, 2: 0.7})]
    result = uncertainty_error_correlation(rows)
    assert result['n_pairs'] == 0
    assert result['mean_spearman'] is None


def test_correlation_ignores_structures_missing_from_either_map():
    rows = [_row(
        uncertainty={1: 0.1, 2: 0.5, 3: 0.9, 4: None},
        dice={1: 0.9, 2: 0.7, 3: 0.5, 4: 0.3},
    )]
    result = uncertainty_error_correlation(rows)
    assert result['mean_spearman'] == pytest.approx(1.0)


def test_summary_reports_the_paired_ensemble_advantage():
    """The ensemble-versus-member test must be paired, and count the pairs it wins."""
    rows = [
        _row({1: 1.0, 2: 2.0, 3: 3.0}, {1: 0.5, 2: 0.4, 3: 0.3}, members=(0.50, 0.52),
             ensemble=0.60),
        _row({1: 1.0, 2: 2.0, 3: 3.0}, {1: 0.5, 2: 0.4, 3: 0.3}, members=(0.60, 0.62),
             ensemble=0.70),
    ]
    summary = summarise(rows)
    assert summary['dice_ensemble_mean_field'] == pytest.approx(0.65)
    assert summary['dice_member_mean'] == pytest.approx(0.56)
    assert summary['delta_ensemble_vs_member_mean'] == pytest.approx(0.09)
    assert summary['delta_better_on'] == 2


def test_best_member_is_an_oracle_upper_bound():
    """
    `dice_best_member_mean` picks the best member per pair, which needs the ground truth.

    It is reported as an unreachable ceiling, so it must be at least the member average -- if it
    were computed as a per-run maximum instead it could fall below on some pairs.
    """
    rows = [
        _row({1: 1.0, 2: 2.0, 3: 3.0}, {1: 0.5, 2: 0.4, 3: 0.3}, members=(0.4, 0.8)),
        _row({1: 1.0, 2: 2.0, 3: 3.0}, {1: 0.5, 2: 0.4, 3: 0.3}, members=(0.9, 0.3)),
    ]
    summary = summarise(rows)
    assert summary['dice_best_member_mean'] == pytest.approx(0.85)
    assert summary['dice_best_member_mean'] >= summary['dice_member_mean']

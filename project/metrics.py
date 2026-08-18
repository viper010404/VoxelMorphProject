"""
Evaluation metrics, shared by every branch.

Three quantities are reported, mirroring the paper:

* **Dice** per anatomical structure (eq. 8) -- registration accuracy.
* **Folding**, the count and fraction of voxels with non-positive Jacobian determinant
  (§V-A-2) -- deformation plausibility. A deformation that scores well on Dice while folding
  tissue onto itself is not anatomically meaningful, so the two must be read together.
* **Inverse consistency**, only meaningful for the diffeomorphic setting where the inverse is
  obtained for free by integrating the negated velocity field.

Everything here is computed per *pair* and returned per *structure*, never pre-aggregated. That
granularity is what allows a paired comparison against the baseline: the same pair registered by
two models gives two numbers whose difference cancels the (large) between-pair variance.
"""

from typing import Dict, List, Optional, Sequence

import numpy as np
import torch

import voxelmorph as vxm
import voxelmorph.nn.functional as vxf


def warp_segmentation(seg: torch.Tensor, disp: torch.Tensor) -> torch.Tensor:
    """
    Warp an integer label map with nearest-neighbour interpolation.

    Linear interpolation of categorical labels would invent labels that do not exist (halfway
    between label 3 and label 7 is not label 5), so nearest-neighbour is required here even
    though images are warped linearly.

    Parameters
    ----------
    seg : torch.Tensor
        Integer label map of shape (B, 1, *spatial).
    disp : torch.Tensor
        Displacement field of shape (B, ndim, *spatial).

    Returns
    -------
    torch.Tensor
        Warped label map of shape (B, 1, *spatial), same dtype as the input.
    """
    transformer = vxm.nn.modules.SpatialTransformer(interpolation_mode='nearest')
    warped = transformer(seg.float(), disp)
    return warped.round().to(seg.dtype)


def dice_per_structure(
    seg_fixed: np.ndarray,
    seg_moved: np.ndarray,
    labels: Sequence[int],
) -> Dict[int, Optional[float]]:
    """
    Dice overlap for each anatomical structure of a single pair.

    Structures absent from *both* segmentations are reported as None rather than 0.0. The
    upstream `voxelmorph.py.utils.dice` returns 0.0 in that case (it divides 0 by an epsilon),
    which would silently drag the mean down for every subject missing a small structure --
    penalising a model for failing to align something that was never there.

    Parameters
    ----------
    seg_fixed : np.ndarray
        Reference label map.
    seg_moved : np.ndarray
        Warped moving label map, same shape.
    labels : sequence of int
        Structure ids to score.

    Returns
    -------
    dict
        Mapping of label id to Dice score, or None where the structure is absent from both.
    """
    labels = list(labels)
    scores = vxm.py.utils.dice(seg_fixed, seg_moved, labels=labels)

    result: Dict[int, Optional[float]] = {}
    for index, label in enumerate(labels):
        present = (seg_fixed == label).any() or (seg_moved == label).any()
        result[label] = float(scores[index]) if present else None
    return result


def mean_dice(scores: Dict[int, Optional[float]]) -> float:
    """
    Average the per-structure Dice scores, ignoring absent structures.

    Parameters
    ----------
    scores : dict
        Output of `dice_per_structure`.

    Returns
    -------
    float
        Mean over present structures, or NaN if none are present.
    """
    values = [v for v in scores.values() if v is not None]
    return float(np.mean(values)) if values else float('nan')


def folding(disp: torch.Tensor) -> List[Dict[str, float]]:
    """
    Count voxels where the deformation folds, per sample in the batch.

    A non-positive Jacobian determinant means the transform is locally non-invertible: tissue is
    folded over itself. The paper reports this alongside Dice (Table I) because a model can
    otherwise buy overlap with physically impossible deformations.

    Reuses `voxelmorph.py.utils.jacobian_determinant`, which is numpy, unbatched, and expects a
    **channels-last** displacement field -- the opposite layout to everything in
    `voxelmorph.nn.functional` -- so the field is permuted and looped here.

    Parameters
    ----------
    disp : torch.Tensor
        Displacement field of shape (B, ndim, *spatial).

    Returns
    -------
    list of dict
        One entry per sample with keys `count` and `fraction`.
    """
    disp = disp.detach().cpu()
    results = []

    for sample in disp:
        # (ndim, *spatial) -> (*spatial, ndim) for the upstream numpy implementation.
        field = sample.permute(*range(1, sample.dim()), 0).numpy()
        determinant = vxm.py.utils.jacobian_determinant(field)
        count = int((determinant <= 0).sum())
        results.append({
            'count': count,
            'fraction': count / determinant.size,
        })

    return results


def inverse_consistency(forward_disp: torch.Tensor, backward_disp: torch.Tensor) -> List[float]:
    """
    Mean residual displacement after composing a transform with its inverse.

    For a perfect inverse pair the composition is the identity, which in displacement terms is
    the zero field -- so the magnitude of `compose([forward, backward])` is itself the error and
    no grid subtraction is needed. `voxelmorph.nn.functional.coords_to_disp` raises
    `NotImplementedError`, so composition is the only available route.

    Composition is applied one sample at a time because `compose` infers the presence of a batch
    axis from `shape[0] != ndim - 1`, which misclassifies batch size 3 in 2D and 4 in 3D.

    Parameters
    ----------
    forward_disp : torch.Tensor
        Displacement field of shape (B, ndim, *spatial).
    backward_disp : torch.Tensor
        Its putative inverse, same shape.

    Returns
    -------
    list of float
        Mean residual magnitude in voxels, one per sample.
    """
    results = []
    for forward, backward in zip(forward_disp.detach(), backward_disp.detach()):
        residual = vxf.compose([forward, backward])
        magnitude = residual.pow(2).sum(dim=0).sqrt()
        results.append(float(magnitude.mean()))
    return results

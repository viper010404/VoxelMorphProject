"""
Losses for registration training.

The smoothness penalty is implemented here rather than taken from `neurite.nn.modules
.SpatialGradient` for one specific reason: the lambda-field variant needs the penalty *before*
it is reduced, so it can be weighted per voxel. `SpatialGradient(reduction=None)` returns a
Python list of per-dimension tensors of differing shapes (each shrunk by one along its own axis
by `torch.diff`), which cannot be weighted by a single map without alignment work.

Implementing it once here means the baseline and the lambda-field variant share the exact same
penalty code and differ only by the presence of the weight map -- so any measured difference is
attributable to the weighting, not to two subtly different implementations of a gradient.
`tests/test_project_losses.py` asserts this function matches `SpatialGradient` when unweighted.
"""

from typing import Literal, Optional, Sequence

import torch

import voxelmorph as vxm


def spatial_smoothness(
    disp: torch.Tensor,
    weight: Optional[torch.Tensor] = None,
    penalty: Literal['l1', 'l2'] = 'l2',
) -> torch.Tensor:
    """
    Diffusion regulariser on a displacement field, optionally weighted per voxel.

    Implements the paper's smoothness term (eq. 7), approximating spatial gradients by forward
    differences between neighbouring voxels. With `weight` supplied, each voxel's contribution is
    scaled, which is what turns the single global regularisation weight into a spatial field.

    Parameters
    ----------
    disp : torch.Tensor
        Displacement field of shape (B, ndim, *spatial).
    weight : torch.Tensor or None, optional
        Per-voxel weight of shape (B, 1, *spatial). When None the penalty is unweighted, which
        reproduces the standard formulation. The weight is cropped -- not interpolated -- onto
        each forward-difference grid, so weight voxel p scales the difference between p and p+1.
    penalty : {'l1', 'l2'}, optional
        Whether to penalise absolute or squared differences. The paper uses L2.

    Returns
    -------
    torch.Tensor
        Scalar penalty, averaged over spatial dimensions and vector components.
    """
    if disp.dim() < 3:
        raise ValueError(f'disp must be (B, ndim, *spatial), got shape {tuple(disp.shape)}')

    num_spatial = disp.dim() - 2
    total = disp.new_zeros(())

    for axis in range(num_spatial):
        dim = 2 + axis
        difference = torch.diff(disp, dim=dim)

        if penalty == 'l2':
            per_voxel = difference.pow(2)
        elif penalty == 'l1':
            per_voxel = difference.abs()
        else:
            raise ValueError(f"penalty must be 'l1' or 'l2', got '{penalty}'")

        # Average over the vector components so the result is one value per voxel.
        per_voxel = per_voxel.mean(dim=1, keepdim=True)

        if weight is not None:
            # torch.diff drops one element along `dim`; narrow the weight onto the same grid.
            cropped = weight.narrow(dim, 0, per_voxel.shape[dim])
            per_voxel = per_voxel * cropped

        total = total + per_voxel.mean()

    return total / num_spatial


def jacobian_determinant(disp: torch.Tensor) -> torch.Tensor:
    """
    Differentiable Jacobian determinant of the transform, on the interior grid.

    The evaluation metric (`metrics.folding`) delegates to
    `voxelmorph.py.utils.jacobian_determinant`, which is numpy and detached and so cannot appear
    in a loss. This reimplements it in torch **using the same convention** -- central differences
    of `disp + grid`, matching `np.gradient` -- so that a penalty applied during training reduces
    the very quantity that is later reported, rather than a near neighbour of it.
    `tests/test_project_folding.py` asserts agreement with the upstream function.

    The interior is used because `np.gradient` switches to one-sided differences at the
    boundary. Reproducing that exactly would add a special case for a rim of voxels that is
    background in every neurite-OASIS image, so instead each spatial axis is cropped by one
    voxel at each end and the penalty is defined on what remains.

    Parameters
    ----------
    disp : torch.Tensor
        Displacement field of shape (B, ndim, *spatial).

    Returns
    -------
    torch.Tensor
        Determinant of shape (B, *interior_spatial), where each spatial extent is reduced by 2.
        Values near 1 mean locally volume-preserving; values <= 0 mean the transform folds.
    """
    ndim = disp.shape[1]
    if disp.dim() - 2 != ndim:
        raise ValueError(f'disp must be (B, ndim, *spatial) with matching ndim, '
                         f'got shape {tuple(disp.shape)}')

    # gradient[j][:, i] is d(disp_i)/dx_j by central difference, cropped so that every partial
    # derivative is defined on one common interior grid.
    gradients = []
    for axis in range(ndim):
        dim = 2 + axis
        extent = disp.shape[dim]
        forward = disp.narrow(dim, 2, extent - 2)
        backward = disp.narrow(dim, 0, extent - 2)
        derivative = (forward - backward) / 2.0
        for other in range(ndim):
            if other != axis:
                other_dim = 2 + other
                derivative = derivative.narrow(other_dim, 1, derivative.shape[other_dim] - 2)
        gradients.append(derivative)

    # Jacobian of phi = disp + identity, so the identity contributes 1 on the diagonal.
    def entry(i, j):
        value = gradients[j][:, i]
        return value + 1.0 if i == j else value

    if ndim == 2:
        return entry(0, 0) * entry(1, 1) - entry(0, 1) * entry(1, 0)

    return (
        entry(0, 0) * (entry(1, 1) * entry(2, 2) - entry(1, 2) * entry(2, 1))
        - entry(0, 1) * (entry(1, 0) * entry(2, 2) - entry(1, 2) * entry(2, 0))
        + entry(0, 2) * (entry(1, 0) * entry(2, 1) - entry(1, 1) * entry(2, 0))
    )


def folding_penalty(disp: torch.Tensor, margin: float = 0.0) -> torch.Tensor:
    """
    Penalise locally non-invertible deformation, i.e. a non-positive Jacobian determinant.

    The paper's diffusion regulariser penalises large spatial gradients of the displacement, which
    discourages folding only indirectly -- a field can be smooth on average and still fold. Every
    extension measured in this project trades folding for Dice, so this targets the failing
    quantity directly.

    The penalty is one-sided: `relu(margin - det)` is zero wherever the transform is already
    orientation-preserving, so it never competes with the similarity term in well-behaved
    regions and acts only where the deformation actually folds.

    Parameters
    ----------
    disp : torch.Tensor
        Displacement field of shape (B, ndim, *spatial).
    margin : float, optional
        Determinant value to push above. 0 penalises only actual folding; a small positive
        value also discourages near-singular regions, at the cost of penalising extreme but
        still-valid compression.

    Returns
    -------
    torch.Tensor
        Scalar penalty, averaged over voxels and batch.
    """
    return torch.relu(margin - jacobian_determinant(disp)).mean()


def compose_displacements(
    accumulated: torch.Tensor,
    residual: torch.Tensor,
    transformer,
) -> torch.Tensor:
    """
    Compose two displacement fields: warp by `accumulated`, then by `residual`.

    Displacement fields do not add. The second field is defined on the grid the first has already
    moved, so the composed transform is `residual(x) + accumulated(x + residual(x))` -- the
    residual appears first because it is the outer transform that relocates the sampling point.

    This is easy to get subtly wrong and hard to notice: measured against sequential warping on
    smooth fields, plain addition is ~1.1% off and the reversed ordering ~1.3%, against 0.6% for
    this one. Nothing raises; the model just learns a slightly incoherent transform.
    `tests/test_project_coarse_to_fine.py` pins it numerically.

    Used in preference to `voxelmorph.nn.functional.compose`, which must be applied one sample at
    a time (its batch detection misreads some shapes -- see `metrics.inverse_consistency`) and so
    cannot sit in a training loop.

    Parameters
    ----------
    accumulated, residual : torch.Tensor
        Displacement fields of shape (B, ndim, *spatial).
    transformer : nn.Module
        A `voxelmorph` spatial transformer used to resample one field through the other.

    Returns
    -------
    torch.Tensor
        The composed displacement field, same shape.
    """
    return residual + transformer(accumulated, residual)


def inverse_consistency_loss(
    forward_disp: torch.Tensor,
    backward_disp: torch.Tensor,
    transformer,
) -> torch.Tensor:
    """
    Penalise failure of a transform and its reverse to undo each other.

    Registering A to B and then B back to A should return every voxel to where it started, so the
    composition of the two fields should be the zero displacement. The residual magnitude is
    therefore the error directly, with no grid subtraction needed.

    Both orderings are penalised. Composition is not symmetric, and constraining only one
    direction leaves the other free to drift -- which would let the network satisfy the letter of
    the constraint while still producing a pair that does not invert.

    This is the *differentiable* counterpart of `metrics.inverse_consistency`, which is the
    quantity already reported for every run; training therefore optimises the metric that is
    ultimately scored rather than a near neighbour of it.

    Parameters
    ----------
    forward_disp, backward_disp : torch.Tensor
        Displacement fields for A->B and B->A, shape (B, ndim, *spatial).
    transformer : nn.Module
        Spatial transformer used for composition.

    Returns
    -------
    torch.Tensor
        Scalar penalty; zero when the two transforms invert each other exactly.
    """
    forward_then_back = compose_displacements(forward_disp, backward_disp, transformer)
    back_then_forward = compose_displacements(backward_disp, forward_disp, transformer)
    return forward_then_back.pow(2).mean() + back_then_forward.pow(2).mean()


def structure_weight_map(
    seg: torch.Tensor,
    structure_lambda,
    mask: torch.Tensor,
) -> torch.Tensor:
    """
    Build a per-voxel regularisation weight from a fixed per-structure table.

    The fixed anatomical prior depends only on the segmentation, not on the network, so it does
    not belong inside a model. Computing it here lets any variant be trained under the prior --
    in particular the cascade, whose model emits no weight map of its own.

    Normalisation is over the brain mask, matching `models.VxmLambdaStructure`, so the total
    regularisation budget equals the baseline's and only its distribution differs.

    Parameters
    ----------
    seg : torch.Tensor
        Segmentation of the moving image, shape (B, 1, *spatial), integer label ids.
    structure_lambda : sequence of float
        One weight per label id; must cover the largest id present.
    mask : torch.Tensor
        Boolean brain mask of the same shape as `seg`.

    Returns
    -------
    torch.Tensor
        Weight map of shape (B, 1, *spatial) with masked per-sample mean 1.
    """
    table = torch.as_tensor(list(structure_lambda), dtype=torch.float32, device=seg.device)
    index = seg.reshape(seg.shape[0], -1).long()

    largest = int(index.max())
    if largest >= table.numel():
        raise ValueError(f'segmentation contains label id {largest} but structure_lambda has '
                         f'{table.numel()} entries')

    weights = table[index].reshape_as(seg).to(seg.device)

    spatial_dims = tuple(range(1, weights.dim()))
    indicator = mask.to(weights.dtype)
    count = indicator.sum(dim=spatial_dims, keepdim=True).clamp(min=1.0)
    masked_mean = (weights * indicator).sum(dim=spatial_dims, keepdim=True) / count
    return weights / masked_mean


def image_similarity(target: torch.Tensor, warped_source: torch.Tensor) -> torch.Tensor:
    """
    Mean squared voxelwise difference between the fixed image and the warped moving image.

    This is the paper's MSE similarity term (eq. 5), appropriate here because all images come
    from the same modality and are intensity-normalised to [0, 1] by `prepare_data.py`.

    Parameters
    ----------
    target : torch.Tensor
        Fixed image, shape (B, C, *spatial).
    warped_source : torch.Tensor
        Moving image after warping, same shape as `target`.

    Returns
    -------
    torch.Tensor
        Scalar similarity loss.
    """
    return (target - warped_source).pow(2).mean()


def registration_loss(
    target: torch.Tensor,
    warped_source: torch.Tensor,
    disp: torch.Tensor,
    lambda_reg: float,
    weight: Optional[torch.Tensor] = None,
    lambda_fold: float = 0.0,
    fold_margin: float = 0.0,
) -> dict:
    """
    Assemble the full unsupervised registration objective.

    Computes `L_sim + lambda * L_smooth` (eq. 4). When `weight` is given it already has mean 1
    by construction (see `models.VxmLambdaField`), so `lambda_reg` remains the average
    regularisation strength and the baseline and lambda-field runs spend the same total
    smoothness budget.

    Parameters
    ----------
    target : torch.Tensor
        Fixed image, shape (B, C, *spatial).
    warped_source : torch.Tensor
        Warped moving image, same shape as `target`.
    disp : torch.Tensor
        Displacement field, shape (B, ndim, *spatial).
    lambda_reg : float
        Regularisation trade-off parameter.
    weight : torch.Tensor or None, optional
        Per-voxel regularisation weight of shape (B, 1, *spatial), mean 1.
    lambda_fold : float, optional
        Weight on the anti-folding penalty. 0 disables it, reproducing the paper's objective
        exactly, so existing runs are unaffected.
    fold_margin : float, optional
        Determinant margin for that penalty; see `folding_penalty`.

    Returns
    -------
    dict
        Keys `total`, `similarity`, `smoothness` and `folding`, each a scalar tensor. The
        components are returned separately so training curves can show which term is moving.
    """
    similarity = image_similarity(target, warped_source)
    smoothness = spatial_smoothness(disp, weight=weight)
    total = similarity + lambda_reg * smoothness

    # Skipped entirely when disabled: the determinant is not free, and paying for it on every
    # step of every baseline run would slow the whole matrix for a term worth zero.
    if lambda_fold > 0.0:
        folding = folding_penalty(disp, margin=fold_margin)
        total = total + lambda_fold * folding
    else:
        folding = disp.new_zeros(())

    return {
        'total': total,
        'similarity': similarity,
        'smoothness': smoothness,
        'folding': folding,
    }


def pyramid_loss(
    target: torch.Tensor,
    source: torch.Tensor,
    fields: Sequence[torch.Tensor],
    lambda_reg: float,
    transformer=None,
) -> dict:
    """
    Deeply-supervised objective: score every level of a displacement pyramid, not just the last.

    A single loss at full resolution lets the network route everything through whichever path is
    shortest, which is how the coarse decoder levels ended up contributing almost nothing --
    deleting the bottleneck of a trained baseline costs only 0.007 Dice. Attaching a similarity
    term to each level removes that option: a level cannot free-ride, because it is scored on how
    well its own field aligns the images at its own resolution.

    Both images are downsampled to each level's grid before scoring, and the field is used as-is
    -- its magnitudes are already in that level's voxels, which is why `VxmPyramid` rescales when
    it upsamples.

    Coarse levels are weighted `1 / 2 ** (levels - 1 - i)`, so the finest level dominates and the
    coarse terms act as a guide rather than taking over the objective. The smoothness penalty is
    applied once, to the final field, because the coarse fields are components of it rather than
    independent deformations, and penalising each would regularise the same displacement several
    times over.

    Parameters
    ----------
    target : torch.Tensor
        Fixed image at full resolution, shape (B, C, *spatial).
    source : torch.Tensor
        Moving image at full resolution, same shape.
    fields : sequence of torch.Tensor
        Displacement fields, coarsest first; the last is the model's output.
    lambda_reg : float
        Regularisation trade-off parameter.
    transformer : callable or None, optional
        Warping operator; defaults to a `voxelmorph` spatial transformer.

    Returns
    -------
    dict
        Keys `total`, `similarity`, `smoothness`, plus `similarity_levels` giving the per-level
        similarity values so training curves can show whether the coarse levels are learning.
    """
    if transformer is None:
        transformer = vxm.nn.modules.SpatialTransformer()

    levels = len(fields)
    ndim = target.dim() - 2
    mode = 'bilinear' if ndim == 2 else 'trilinear'

    similarity = target.new_zeros(())
    per_level = []

    for index, field in enumerate(fields):
        shape = tuple(field.shape[2:])
        if shape == tuple(target.shape[2:]):
            level_target, level_source = target, source
        else:
            level_target = torch.nn.functional.interpolate(
                target, size=shape, mode=mode, align_corners=True)
            level_source = torch.nn.functional.interpolate(
                source, size=shape, mode=mode, align_corners=True)

        value = image_similarity(level_target, transformer(level_source, field))
        per_level.append(value)
        similarity = similarity + value / (2 ** (levels - 1 - index))

    smoothness = spatial_smoothness(fields[-1])

    return {
        'total': similarity + lambda_reg * smoothness,
        'similarity': per_level[-1],
        'smoothness': smoothness,
        'similarity_levels': per_level,
    }

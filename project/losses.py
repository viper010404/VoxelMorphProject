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

    Returns
    -------
    dict
        Keys `total`, `similarity` and `smoothness`, each a scalar tensor. The components are
        returned separately so training curves can show which term is moving.
    """
    similarity = image_similarity(target, warped_source)
    smoothness = spatial_smoothness(disp, weight=weight)

    return {
        'total': similarity + lambda_reg * smoothness,
        'similarity': similarity,
        'smoothness': smoothness,
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

#!/usr/bin/env python3
"""
Synthetic misalignment, for testing registration under deformations the dataset does not contain.

Every negative result for the cross-attention branch was measured on neurite-OASIS as shipped,
which is **affinely pre-aligned**: only the nonlinear residual remains, and it is small. Measured
on the trained baseline, displacements inside the brain have a median of 1.7 voxels and a maximum
of 14.2. The bottleneck those models attend over is 32x downsampled, so one token spans 32 voxels
and *every* displacement in the data is smaller than a single token. Global matching therefore has
nothing to match: the long-range correspondence problem attention exists to solve was removed by
the affine step before the network ever saw the data.

This module puts that problem back. A smooth random deformation of controlled magnitude is applied
to the moving image (and its segmentation, so Dice stays well defined), and the network has to
undo it on top of the anatomical difference. That is the regime in which the branch's original
hypothesis -- that explicit correspondence helps when images start far apart -- is actually
testable.

The field is built by sampling white noise on a coarse grid and interpolating it up, which gives a
smooth, invertible-in-practice deformation rather than per-voxel jitter. Magnitude is specified in
voxels and the field is rescaled so its mean displacement matches, making "20-voxel misalignment"
mean the same thing across images and dimensionalities.

Usage
-----
    from project.misalign import random_displacement, apply_displacement
    field = random_displacement(shape, ndim, magnitude=20.0, seed=7, device='cuda')
    moved_image, moved_seg = apply_displacement(image, field, seg)
"""

from typing import Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

import voxelmorph as vxm


def random_displacement(
    shape: Sequence[int],
    ndim: int,
    magnitude: float,
    seed: int,
    device: str = 'cpu',
    control_points: int = 5,
) -> torch.Tensor:
    """
    Build a smooth random displacement field with a prescribed mean magnitude.

    Parameters
    ----------
    shape : sequence of int
        Spatial shape of the field.
    ndim : int
        Spatial dimensionality, 2 or 3.
    magnitude : float
        Target mean displacement in voxels. 0 returns a zero field.
    seed : int
        Seed for the field. Passing the same seed for the same pair makes the perturbation
        identical across models, which is what keeps the comparison paired.
    device : str, optional
        Device to build the field on.
    control_points : int, optional
        Coarse grid resolution per axis before upsampling. Smaller gives smoother, more global
        deformation; this default produces the low-frequency, large-scale motion the experiment
        is about, rather than high-frequency noise a smoothness prior would trivially reject.

    Returns
    -------
    torch.Tensor
        Displacement field of shape (1, ndim, *shape), in voxels.
    """
    if magnitude <= 0:
        return torch.zeros(1, ndim, *shape, device=device)

    generator = torch.Generator(device='cpu').manual_seed(seed)
    coarse = torch.randn(1, ndim, *([control_points] * ndim), generator=generator)

    mode = 'bilinear' if ndim == 2 else 'trilinear'
    field = F.interpolate(coarse.to(device), size=tuple(shape), mode=mode, align_corners=True)

    # Rescale so the *mean* displacement equals `magnitude`, so the number means the same thing
    # regardless of grid size or dimensionality.
    current = field.pow(2).sum(dim=1).sqrt().mean()
    if float(current) > 0:
        field = field * (magnitude / current)
    return field


def apply_displacement(
    image: torch.Tensor,
    field: torch.Tensor,
    seg: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """
    Warp an image, and optionally its segmentation, by a displacement field.

    The segmentation must be warped by the *same* field with nearest-neighbour interpolation, or
    the labels no longer describe the image and every Dice number computed from them is wrong.

    Parameters
    ----------
    image : torch.Tensor
        Shape (B, 1, *spatial).
    field : torch.Tensor
        Displacement field of shape (B, ndim, *spatial) or (1, ndim, *spatial).
    seg : torch.Tensor or None, optional
        Matching label map of shape (B, 1, *spatial).

    Returns
    -------
    tuple
        `(warped_image, warped_seg)`; the second is None when `seg` is None.
    """
    if field.shape[0] == 1 and image.shape[0] > 1:
        field = field.expand(image.shape[0], *field.shape[1:])

    linear = vxm.nn.modules.SpatialTransformer()
    warped_image = linear(image, field)

    warped_seg = None
    if seg is not None:
        nearest = vxm.nn.modules.SpatialTransformer(interpolation_mode='nearest')
        warped_seg = nearest(seg.float(), field).round().to(seg.dtype)

    return warped_image, warped_seg


def pair_seed(fixed_index: int, moving_index: int, magnitude: float, base: int = 4242) -> int:
    """
    Deterministic seed for one (pair, magnitude) combination.

    Every model must face the identical perturbation on the identical pair, otherwise the paired
    comparison is comparing different problems. Deriving the seed from the pair indices and the
    magnitude gives that without storing a field per pair.

    Parameters
    ----------
    fixed_index, moving_index : int
        Subject indices of the evaluation pair.
    magnitude : float
        Misalignment magnitude in voxels.
    base : int, optional
        Base offset for the seed.

    Returns
    -------
    int
        Seed value.
    """
    return base + 1009 * int(fixed_index) + 9176 * int(moving_index) + int(round(magnitude * 100))

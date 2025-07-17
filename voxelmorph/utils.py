"""
Utilities for the `voxelmorph` project.
"""

import torch
from torch import Tensor

__all__ = [
    "affine_to_displacement_field",
]


def affine_to_displacement_field(
    affine: Tensor,
    meshgrid: Tensor,
    rotate_around_center: bool = True
) -> Tensor:
    """
    Convert an affine transformation matrix to a dense displacement field.

    Parameters
    ----------
    affine : Tensor
        Affine transformation matrix. It is expected to be a vox2vox target to source transformation.
    meshgrid : Tensor
        The meshgrid tensor of shape `(W, H[, D], N)`, where N is the spatial dimensionality.
    rotate_around_center : bool, optional
        If True, the rotation will be around the center of the image, otherwise around the origin.

    Returns
    -------
    Tensor
        The generated displacement field of shape `meshgrid.shape[:-1]`.
    """
    ndim = meshgrid.shape[-1]
    shape = meshgrid.shape[:-1]

    # if rotate_around_center is enabled, adjust the meshgrid so that the rotation
    # is around the center of the image instead of the origin
    grid = meshgrid.clone() if rotate_around_center else meshgrid
    if rotate_around_center:
        for d in range(ndim):
            grid[..., d] -= (shape[d] - 1) / 2

    # convert the meshgrid to homogeneous coordinates by appending a column of ones
    coords = grid.view(-1, ndim)
    ones = torch.ones((coords.shape[-2], 1), device=meshgrid.device)
    coords = torch.cat([coords, ones], dim=-1)

    # apply the affine transformation to the coordinates to get the shift vector
    # affine needs to be vox2vox transformation matrix, and mapping from target to source
    # the computed shift is the absolute crs in source space
    shift = (affine @ coords.T)[:ndim].T

    # reshape the shift vector to match the shape of the meshgrid and subtract
    # the original meshgrid to get the displacement field
    shift = shift.view(*shape, ndim) - grid

    return shift

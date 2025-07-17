"""
Utilities for the `voxelmorph` project.
"""

import torch
from torch import Tensor

__all__ = [
    "affine_to_displacement_field",
    "grid_coordinates",
    "spatial_transform",
    "displacement_field_to_coords",
    "integrate_displacement_field"
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


def grid_coordinates(shape, device: torch.device = None) -> Tensor:
    """
    TODOC
    """
    ranges = [torch.arange(s, dtype=torch.float32, device=device) for s in shape]
    meshgrid = torch.stack(torch.meshgrid(*ranges, indexing='ij'), dim=-1)
    return meshgrid


def spatial_transform(
    image: Tensor,
    trf: Tensor,
    method: str = 'linear',
    isdisp: bool = True,
    meshgrid: Tensor = None,
    rotate_around_center: bool = True
) -> Tensor:
    """
    TODOC
    """
    if trf is None:
        return image

    if trf.ndim == 2:
        if meshgrid is None:
            meshgrid = grid_coordinates(image.shape[1:], device=image.device)

        trf = torch.linalg.inv(trf)
        trf = affine_to_displacement_field(
            trf,
            meshgrid,
            rotate_around_center=rotate_around_center
        )
        isdisp = True

    if isdisp:
        # convert the displacement crs to absolute crs scaled to range [-1, 1]
        trf = displacement_field_to_coords(trf, meshgrid=meshgrid)

    method = 'bilinear' if method == 'linear' else method

    reset_type = None
    if not torch.is_floating_point(image):
        if method == 'nearest':
            reset_type = image.dtype
        image = image.type(torch.float32)

    image = image.unsqueeze(0)
    trf = trf.unsqueeze(0)

    # trf is an absolute crs field in the range of [-1, 1]
    interped = torch.nn.functional.grid_sample(image, trf, align_corners=True, mode=method)
    interped = interped.squeeze(0)

    if reset_type is not None:
        interped = interped.type(reset_type)

    return interped


def displacement_field_to_coords(disp, meshgrid=None) -> Tensor:
    """
    Convert the displacement crs to absolute crs scaled to range [-1, 1].

    Parameters:
    -----------
    disp: torch.Tensor
        Displacement crs field
    meshgrid: torch.Tensor, optional
       crs grid for the image shape

    Returns:
    --------
    torch.Tensor:
        The absolute crs field scaled to range [-1, 1].
    """
    if meshgrid is None:
        meshgrid = grid_coordinates(disp.shape[:-1], device=disp.device)

    shape = disp.shape[:-1]
    ndim = disp.shape[-1]

    # compute the absolute crs field
    # scale the field to range [-1, 1], which is expected by torch.nn.functional.grid_sample()
    coords = (meshgrid + disp)
    for d in range(ndim):
        if shape[d] == 1:
            coords[..., d] *= 0
        else:
            coords[..., d] *= 2 / (shape[d] - 1)
            coords[..., d] -= 1

    coords = coords.flip(-1)

    return coords


def integrate_displacement_field(
    disp: Tensor,
    steps: int,
    meshgrid: Tensor = None
) -> Tensor:
    """
    TODOC
    """
    if meshgrid is None:
        # generate a crs grid
        meshgrid = grid_coordinates(disp.shape[:-1], device=disp.device)

    if steps == 0:
        return disp

    disp = disp / (2 ** steps)
    for _ in range(steps):
        disp += spatial_transform(disp.movedim(-1, 0), disp, meshgrid=meshgrid).movedim(0, -1)

    return disp

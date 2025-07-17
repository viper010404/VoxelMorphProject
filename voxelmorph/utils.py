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
    "integrate_displacement_field",
    "angles_to_rotation_matrix",
    "compose_affine"
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


def angles_to_rotation_matrix(
    rotation: Tensor,
    degrees: bool = True
) -> Tensor:
    """
    Compute a rotation matrix from the given rotation angles.

    Parameters
    ----------
    rotation : Tensor
        A tensor containing the rotation angles. If `degrees` is True, the angles
        are in degrees, otherwise they are in radians.
    degrees : bool, optional
        Whether to interpret the rotation angles as degrees.

    Returns
    -------
    Tensor
        The computed `(ndim + 1, ndim + 1)` rotation matrix.
    """
    if degrees:
        rotation = torch.deg2rad(rotation)

    # scalar value allowed for 2D transforms
    rotation = torch.as_tensor(rotation)
    if rotation.ndim == 0:
        rotation = rotation.view(1)
    num_angles = len(rotation)

    # build the matrix
    if num_angles == 1:
        c, s = torch.cos(rotation[0]), torch.sin(rotation[0])
        matrix = torch.tensor([[c, -s], [s, c]], dtype=torch.float64)
    elif num_angles == 3:
        c, s = torch.cos(rotation[0]), torch.sin(rotation[0])
        rx = torch.tensor([[1, 0, 0], [0, c, s], [0, -s, c]], dtype=torch.float64)
        c, s = torch.cos(rotation[1]), torch.sin(rotation[1])
        ry = torch.tensor([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=torch.float64)
        c, s = torch.cos(rotation[2]), torch.sin(rotation[2])
        rz = torch.tensor([[c, s, 0], [-s, c, 0], [0, 0, 1]], dtype=torch.float64)
        matrix = rx @ ry @ rz
    else:
        raise ValueError(f'expected 1 (2D) or 3 (3D) rotation angles, got {num_angles}')

    return matrix.to(rotation.device)


def compose_affine(
    ndim: int,
    translation: Tensor = None,
    rotation: Tensor = None,
    scale: Tensor = None,
    shear: Tensor = None,
    degrees: bool = True,
    device: torch.device = None
) -> Tensor:
    """
    Composes an affine matrix from a set of translation, rotation, scale,
    and shear transform components.

    Parameters
    ----------
    ndim (int):
        The number of dimensions of the affine matrix. Must be 2 or 3.
    translation : Tensor, optional
        The translation vector. Must be a vector of size `ndim`. 
    rotation : Tensor, optional
        The rotation angles. Must be a scalar value for 2D affine matrices, 
        and a tensor of size 3 for 3D affine matrices.
    scale : Tensor, optional
        The scaling factor. Can be scalar or vector of size `ndim`.
    shear : Tensor, optional
        The shearing factor. Must be a scalar value for 2D affine matrices, 
        and a tensor of size 3 for 3D affine matrices.
    degrees : bool, optional
        Whether to interpret the rotation angles as degrees.
    device : torch.device, optional
        The device of the returned matrix.

    Returns
    -------
    Tensor
        The composed affine matrix, as a tensor of shape `(ndim + 1, ndim + 1)`.
    """
    if ndim not in (2, 3):
        raise ValueError(f'affine transform must be 2D or 3D, got ndim {ndim}')

    # check translation
    translation = torch.zeros(ndim) if translation is None else torch.as_tensor(translation)
    if len(translation) != ndim:
        raise ValueError(f'translation must be of shape ({ndim},)')

    # check rotation angles
    expected = 3 if ndim == 3 else 1
    rotation = torch.zeros(expected) if rotation is None else torch.as_tensor(rotation)
    if rotation.ndim == 0 and ndim == 3 or rotation.ndim != 0 and rotation.shape[0] != expected:
        raise ValueError(f'rotation must be of shape ({expected},)')

    # check scaling factor
    scale = torch.ones(ndim) if scale is None else torch.as_tensor(scale)
    if scale.ndim == 0:
        scale = scale.repeat(ndim)
    if scale.shape[0] != ndim:
        raise ValueError(f'scale must be of size {ndim}')

    # check shearing
    expected = 3 if ndim == 3 else 1
    shear = torch.zeros(expected) if shear is None else torch.as_tensor(shear)
    if shear.ndim == 0:
        shear = shear.view(1)
    if shear.shape[0] != expected:
        raise ValueError(f'shear must be of shape ({expected},)')

    # start from translation
    T = torch.eye(ndim + 1, dtype=torch.float64)
    T[:ndim, -1] = translation

    # rotation matrix
    R = torch.eye(ndim + 1, dtype=torch.float64)
    R[:ndim, :ndim] = angles_to_rotation_matrix(rotation, degrees=degrees)

    # scaling
    Z = torch.diag(torch.cat([scale, torch.ones(1, dtype=torch.float64)]))

    # shear matrix
    S = torch.eye(ndim + 1, dtype=torch.float64)
    S[0][1] = shear[0]
    if ndim == 3:
        S[0][2] = shear[1]
        S[1][2] = shear[2]

    # compose component matrices
    matrix = T @ R @ Z @ S

    return torch.as_tensor(matrix, dtype=torch.float32, device=device)

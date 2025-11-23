"""
Single tensor operations (no B, C, dimensions assumption)
"""
from typing import Union, Sequence, Tuple, Literal

import torch

import neurite as ne
import neurite.nn.functional as nef

__all__ = [
    'affine_to_disp',
    'angles_to_rotation_matrix',
    'params_to_affine',
    'disp_to_coords',
    'spatial_transform',
    'integrate_disp',
]


def angles_to_rotation_matrix(
        rotation: torch.Tensor,
        degrees: bool = True
) -> torch.Tensor:
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
    rotation = torch.as_tensor(rotation)
    if degrees:
        rotation = torch.deg2rad(rotation)
    rotation = torch.atleast_1d(rotation)

    # build the matrix
    if len(rotation) == 1:
        c, s = torch.cos(rotation[0]), torch.sin(rotation[0])
        matrix = torch.tensor([[c, -s], [s, c]], dtype=torch.float64)
    elif len(rotation) == 3:
        c, s = torch.cos(rotation[0]), torch.sin(rotation[0])
        rx = torch.tensor([[1, 0, 0], [0, c, s], [0, -s, c]], dtype=torch.float64)
        c, s = torch.cos(rotation[1]), torch.sin(rotation[1])
        ry = torch.tensor([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=torch.float64)
        c, s = torch.cos(rotation[2]), torch.sin(rotation[2])
        rz = torch.tensor([[c, s, 0], [-s, c, 0], [0, 0, 1]], dtype=torch.float64)
        matrix = rx @ ry @ rz
    else:
        raise ValueError(f'expected 1 (2D) or 3 (3D) rotation angles, got {len(rotation)}')

    return matrix.to(rotation.device)


def params_to_affine(
    ndim: int,
    translation: Union[torch.Tensor, None] = None,
    rotation: Union[torch.Tensor, None] = None,
    scale: Union[torch.Tensor, None] = None,
    shear: Union[torch.Tensor, None] = None,
    degrees: bool = True,
    device: Union[torch.device, None] = None
) -> torch.Tensor:
    """
    Makes an affine matrix from translation, rotation, scale, and shear transform components.

    Parameters
    ----------
    ndim : int
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


def affine_to_disp(
    affine: torch.Tensor,
    meshgrid: Union[torch.Tensor, None] = None,
    origin_at_center: bool = True,
    shape: Union[Sequence[int], None] = None,
    warp_right: Union[torch.Tensor, None] = None
) -> torch.Tensor:
    """
    Convert an affine transformation matrix to a displacement field.

    Parameters
    ----------
    affine : Tensor
        Affine transformation matrix of shape (N, N+1) or (N+1, N+1).
        Expected to be a vox2vox target-to-source transformation.
    meshgrid : Tensor, optional
        Pre-computed meshgrid tensor of shape (*spatial_shape, N), where N is the spatial
        dimensionality. If None, will be computed from `shape` parameter.
    origin_at_center : bool, optional
        If True, place the coordinate system origin at the image center. If False, origin
        is at the top-left corner. Default is True.
    shape : Sequence[int], optional
        Spatial shape (N dimensions) to create meshgrid if `meshgrid` is not provided.
        Required if `meshgrid` is None.
    warp_right : Tensor, optional
        Right-compose the affine with this displacement field of shape (*spatial_shape, N).
        Computes affine(x + warp_right(x)) - x. Useful for composing transforms.

    Returns
    -------
    Tensor
        Displacement field of shape (*spatial_shape, N).

    Examples
    --------
    >>> # Basic usage with pre-computed meshgrid
    >>> import neurite as ne
    >>> affine = torch.tensor(
    >>> ... [[1., 0., 5.],
    >>> ... [0., 1., 3.]]
    >>> )
    >>> grid = ne.volshape_to_ndgrid((64, 64), stack=True)
    >>> disp = affine_to_disp(affine, meshgrid=grid)

    >>> # Using shape parameter instead
    >>> disp = affine_to_disp(affine, shape=(64, 64))

    >>> # Compose affine with existing displacement field
    >>> warp = torch.randn(64, 64, 2)
    >>> composed = affine_to_disp(affine, shape=(64, 64), warp_right=warp)
    """
    if meshgrid is None:
        if shape is None:
            raise ValueError("Either `meshgrid` or `shape` must be provided")

        meshgrid = ne.volshape_to_ndgrid(                           # (*spatial, ndim)
            size=shape, device=affine.device, dtype=affine.dtype, stack=True
        )

    assert isinstance(meshgrid, torch.Tensor)
    ndim = meshgrid.shape[-1]
    spatial_shape = meshgrid.shape[:-1]

    if affine.shape[-1] != ndim + 1:
        raise ValueError(
            f'Affine dimensionality ({affine.shape[-1] - 1}D) does not match '
            f'meshgrid dimensionality ({ndim}D)'
        )

    # Adjust meshgrid to center origin if requested
    grid = meshgrid.clone() if origin_at_center else meshgrid
    if origin_at_center:
        for d in range(ndim):
            grid[..., d] -= (spatial_shape[d] - 1) / 2

    # Flatten and transpose grid
    mesh = grid.reshape(-1, ndim).T  # (ndim, num_voxels)
    out = mesh

    # Optionally right-compose with displacement field
    if warp_right is not None:
        if warp_right.shape[-ndim - 1: -1] != spatial_shape:
            raise ValueError(
                f'warp_right spatial shape {warp_right.shape[-ndim - 1:-1]} does not match '
                f'meshgrid shape {spatial_shape}'
            )

        # Flatten and transpose warp
        warp_flat = warp_right.reshape(*warp_right.shape[:-ndim - 1], -1, ndim)
        warp_flat = warp_flat.transpose(-2, -1)

        # Add to coordinates with broadcasting for batch dimension
        out = out + warp_flat

    # Apply affine transformation with broadcasting
    out = affine[..., :ndim, :ndim] @ out + affine[..., :ndim, -1:]
    out = out - mesh  # Subtract original mesh to get displacement
    out = out.transpose(-2, -1)

    disp = out.reshape(*spatial_shape, ndim)

    return disp


def disp_to_coords(
    disp: torch.Tensor,
    meshgrid: Union[torch.Tensor, None] = None,
    non_spatial_dims: Union[Tuple[int, ...], None] = None
) -> torch.Tensor:
    """
    Convert displacement field to normalized coordinates for grid_sample.

    Shape-agnostic implementation. Displacement fields always end with a coordinate
    dimension (*spatial, ndim), and can have optional leading non-spatial dimensions.

    Parameters
    ----------
    disp : torch.Tensor
        Displacement field. Shape depends on non_spatial_dims:
        - (*spatial, ndim) if non_spatial_dims=None
        - (C, *spatial, ndim) if non_spatial_dims=(0,)
        - (B, C, *spatial, ndim) if non_spatial_dims=(0, 1)
        The last dimension contains displacement vectors of length ndim.
    meshgrid : torch.Tensor or None, default=None
        Pre-computed coordinate grid of shape (*spatial, ndim). If None, computed
        from displacement field spatial shape.
    non_spatial_dims : Tuple[int, ...] or None, default=None
        Which leading dimensions are non-spatial (before spatial dims):
        - None: pure spatial displacement field
        - (0,): first dimension is non-spatial (e.g., channel or batch)
        - (0, 1): first two dimensions are non-spatial (e.g., batch and channel)
        - etc..

    Returns
    -------
    torch.Tensor
        Normalized coordinates in range [-1, 1] for torch.nn.functional.grid_sample.
        Same shape as input displacement field.

    Examples
    --------
    >>> # Pure spatial 2D displacement field
    >>> disp = torch.randn(64, 64, 2)
    >>> coords = disp_to_coords(disp)
    >>> coords.shape
    torch.Size([64, 64, 2])

    >>> # With batch and channel dimensions
    >>> disp = torch.randn(2, 3, 64, 64, 2)
    >>> coords = disp_to_coords(disp, non_spatial_dims=(0, 1))
    >>> coords.shape
    torch.Size([2, 3, 64, 64, 2])
    """
    # Parse non spatial -- disp has shape (*non_spatial, *spatial, ndim)
    num_non_spatial, num_spatial = ne.functional._parse_non_spatial_dims(
        non_spatial_dims, disp.ndim - 1
    )
    spatial_shape = disp.shape[num_non_spatial: -1]

    if meshgrid is None:
        meshgrid = ne.volshape_to_ndgrid(size=spatial_shape, device=disp.device, stack=True)
        assert isinstance(meshgrid, torch.Tensor), (
            f'Expected torch.Tensor from volshape_to_ndgrid(stack=True). Got {type(meshgrid)}'
        )

    # Normalize each spatial dimension to [-1, 1] for grid_sample
    coords = meshgrid + disp
    for d in range(num_spatial):
        spatial_idx = num_non_spatial + d
        spatial_size = disp.shape[spatial_idx]

        if spatial_size == 1:
            coords[..., d] *= 0

        else:
            coords[..., d] *= 2 / (spatial_size - 1)
            coords[..., d] -= 1

    # Flip coordinate order for grid_sample (expects reversed spatial dims)
    coords = coords.flip(-1)

    return coords


def spatial_transform(
    image: torch.Tensor,
    trf: Union[torch.Tensor, None],
    mode: Literal['linear', 'nearest'] = 'linear',
    isdisp: bool = True,
    meshgrid: Union[torch.Tensor, None] = None,
    origin_at_center: bool = True,
    non_spatial_dims: Union[Tuple[int, ...], None] = None,
    align_corners: bool = True
) -> torch.Tensor:
    """
    Apply spatial transformation to image using displacement or coordinate field.

    Shape-agnostic implementation that works with any tensor dimensionality.

    Parameters
    ----------
    image : torch.Tensor
        Input image to transform. Shape depends on non_spatial_dims:
        - (*spatial,) if non_spatial_dims=None
        - (C, *spatial) if non_spatial_dims=(0,)
        - (B, C, *spatial) if non_spatial_dims=(0, 1)
    trf : torch.Tensor or None
        Transformation field. Can be:
        - Affine matrix: shape (N+1, N+1) or (N, N+1)
        - Displacement field: shape (*spatial, N)
        - Coordinate field: shape (*spatial, N)
        - None: returns image unchanged
    mode : {'linear', 'nearest'}, default='linear'
        Interpolation mode. 'linear' will auto-detect appropriate mode
        (bilinear/trilinear) based on spatial dimensionality.
    isdisp : bool, default=True
        If True, treat trf as displacement field and convert to coordinates.
        If False, treat trf as normalized coordinates ready for grid_sample.
    meshgrid : torch.Tensor or None, default=None
        Pre-computed coordinate grid. If None, computed from image shape.
    origin_at_center : bool, default=True
        Place origin at image center when converting affine matrices to displacement.
    non_spatial_dims : Tuple[int, ...] or None, default=None
        Which dimensions of image are non-spatial:
        - None: pure spatial tensor
        - (0,): first dimension is non-spatial (e.g., channel)
        - (0, 1): first two dimensions are non-spatial (e.g., batch, channel)
    align_corners : bool, default=True
        Align corners parameter for grid_sample.

    Returns
    -------
    torch.Tensor
        Transformed image with same shape as input.

    Examples
    --------
    >>> # Pure spatial image (H, W)
    >>> image = torch.randn(64, 64)
    >>> disp = torch.randn(64, 64, 2)
    >>> warped = spatial_transform(image, disp)
    >>> warped.shape
    torch.Size([64, 64])

    >>> # Image with channel dimension (C, H, W)
    >>> image = torch.randn(3, 64, 64)
    >>> disp = torch.randn(64, 64, 2)
    >>> warped = spatial_transform(image, disp, non_spatial_dims=(0,))
    >>> warped.shape
    torch.Size([3, 64, 64])

    >>> # Image with batch and channel (B, C, H, W)
    >>> image = torch.randn(2, 3, 64, 64)
    >>> disp = torch.randn(64, 64, 2)
    >>> warped = spatial_transform(image, disp, non_spatial_dims=(0, 1))
    >>> warped.shape
    torch.Size([2, 3, 64, 64])
    """
    # Early return for no transformation
    if trf is None:
        return image

    # Parse image dimensions to understand shape
    num_non_spatial, num_spatial = ne.functional._parse_non_spatial_dims(
        non_spatial_dims, image.ndim
    )
    spatial_shape = image.shape[num_non_spatial:]

    # Convert affine matrix to displacement field if needed
    if trf.ndim == 2:
        # Invert affine to get source-to-target mapping for warping
        trf = torch.linalg.inv(trf)
        trf = affine_to_disp(trf, meshgrid, shape=spatial_shape, origin_at_center=origin_at_center)
        isdisp = True

    if isdisp:
        trf = disp_to_coords(trf, meshgrid=meshgrid)

    if mode == 'linear':
        mode = ne.utils.infer_linear_interpolation_mode(num_spatial)

    # Prepare image for grid_sample (must have B, C)
    original_dtype = None
    if not torch.is_floating_point(image):
        if mode == 'nearest':
            original_dtype = image.dtype
        image = image.type(torch.float32)

    # Add dimensions to reach (B, C, *spatial) format
    num_dims_to_add = 2 - num_non_spatial
    for _ in range(num_dims_to_add):
        image = image.unsqueeze(0)

    # Prepare coordinates for grid_sample (requires batch dimension)
    # Coordinates format: (*spatial, ndim) or (B, *spatial, ndim)
    # Check if batch dimension already exists
    trf_has_batch_dim = trf.ndim > (num_spatial + 1)
    if not trf_has_batch_dim:
        trf = trf.unsqueeze(0)

    # Apply transformation
    transformed = torch.nn.functional.grid_sample(
        image, trf, align_corners=align_corners, mode=mode
    )

    # Restore original format
    for _ in range(num_dims_to_add):
        transformed = transformed.squeeze(0)
    if original_dtype is not None:
        transformed = transformed.type(original_dtype)

    return transformed


def integrate_disp(
    disp: torch.Tensor,
    steps: int,
    meshgrid: Union[torch.Tensor, None] = None
) -> torch.Tensor:
    """
    TODOC
    """
    if meshgrid is None:
        meshgrid = ne.volshape_to_ndgrid(size=disp.shape[:-1], device=disp.device, stack=True)

    if steps == 0:
        return disp

    disp = disp / (2 ** steps)
    for _ in range(steps):
        disp += spatial_transform(
            disp.movedim(-1, 0), disp, meshgrid=meshgrid, non_spatial_dims=(0,)
        ).movedim(0, -1)

    return disp

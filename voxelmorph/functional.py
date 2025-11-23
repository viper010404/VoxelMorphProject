"""
Single tensor operations (no B, C, dimensions assumption)
"""
from typing import Union, Sequence, Tuple, Literal

import torch

import neurite as ne
import neurite.nn.functional as nef
from neurite.functional import _parse_non_spatial_dims


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
        Affine transformation matrix of shape (..., N, N+1) or (..., N+1, N+1).
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
        Right-compose the affine with this displacement field of shape (..., *spatial_shape, N).
        Computes affine(x + warp_right(x)) - x. Useful for composing transforms.

    Returns
    -------
    Tensor
        Displacement field of shape (..., *spatial_shape, N).

    Examples
    --------
    >>> # Basic usage with pre-computed meshgrid
    >>> import neurite.nn.functional as nef
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

        meshgrid = ne.volshape_to_ndgrid(
            size=shape, device=affine.device, dtype=affine.dtype, stack=True
        )

    ndim = meshgrid.shape[-1]
    spatial_shape = meshgrid.shape[:-1]

    if affine.shape[-1] != ndim + 1:
        raise ValueError(
            f'Affine dimensionality ({affine.shape[-1] - 1}D) does not match '
            f'meshgrid dimensionality ({ndim}D)'
        )

    batch_shape = affine.shape[:-2]

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

    if batch_shape:
        disp = out.reshape(*batch_shape, *spatial_shape, ndim)
    else:
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
    num_non_spatial, num_spatial = _parse_non_spatial_dims(non_spatial_dims, disp.ndim - 1)
    spatial_shape = disp.shape[num_non_spatial: -1]

    if meshgrid is None:
        meshgrid = ne.volshape_to_ndgrid(size=spatial_shape, device=disp.device, stack=True)

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

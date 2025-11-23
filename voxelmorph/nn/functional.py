"""
Functions containing the core operations and logic of for image registration for `voxelmorph`
written in PyTorch.
"""

# Standard library imports
from typing import List, Union, Optional, Sequence, Literal

# Third-party imports
import torch
from torch import Tensor
import numpy as np

import neurite as ne
import neurite.nn.functional as nef
import voxelmorph as vxm

__all__ = [
    "spatial_transform",
    "compose",
    "smooth_gaussian",
    "perlin",
    "random_disp",
    "random_transform",
]


def spatial_transform(
    image: Tensor,
    trf: Union[Tensor, None],
    method: Literal['nearest', 'linear'] = 'linear',
    isdisp: bool = True,
    meshgrid: Union[Tensor, None] = None,
    origin_at_center: bool = True
) -> Tensor:
    """
    Apply spatial transformation to image in (B, C, *spatial) format.

    Wrapper around voxelmorph.functional.spatial_transform with non_spatial_dims=(0, 1).

    Parameters
    ----------
    image : Tensor
        Input image with shape (B, C, *spatial).
    trf : Tensor or None
        Transformation field. Can be:
        - Affine matrix: shape (N+1, N+1) or (N, N+1)
        - Displacement field: shape (*spatial, N)
        - Coordinate field: shape (*spatial, N)
        - None: returns image unchanged
    method : str, default='linear'
        Interpolation mode ('linear' or 'nearest').
    isdisp : bool, default=True
        If True, treat trf as displacement field. If False, treat as coordinates.
    meshgrid : Tensor or None, default=None
        Pre-computed coordinate grid.
    origin_at_center : bool, default=True
        Place origin at image center for affine transformations.

    Returns
    -------
    Tensor
        Transformed image with shape (B, C, *spatial).

    Examples
    --------
    >>> # 2D image with batch and channel
    >>> image = torch.randn(2, 3, 64, 64)
    >>> disp = torch.randn(64, 64, 2)
    >>> warped = spatial_transform(image, disp)
    >>> warped.shape
    torch.Size([2, 3, 64, 64])

    >>> # 3D image with batch and channel
    >>> image = torch.randn(1, 1, 64, 64, 64)
    >>> disp = torch.randn(64, 64, 64, 3)
    >>> warped = spatial_transform(image, disp)
    >>> warped.shape
    torch.Size([1, 1, 64, 64, 64])
    """
    return vxm.functional.spatial_transform(
        image=image,
        trf=trf,
        mode=method,
        isdisp=isdisp,
        meshgrid=meshgrid,
        origin_at_center=origin_at_center,
        non_spatial_dims=(0, 1),
        align_corners=True
    )


def smooth_gaussian(
    shape: Sequence[int],
    sigma: float,
    magnitude: float = 1.0,
    device: Union[torch.device, None] = None,
    method: Literal['blur', 'upsample'] = 'blur'
) -> Tensor:
    """
    Generate smooth Gaussian noise in (B, C, *spatial) format.

    Parameters
    ----------
    shape : Sequence[int]
        Desired shape of output tensor in (B, C, *spatial). Must have at least 3 dimensions
        (batch, channel, and spatial). Examples: (1, 1, 64, 64) for 2D, (2, 3, 64, 64, 64) for 3D.
    sigma : float
        Spatial smoothing sigma in voxel coordinates.
    magnitude : float, default=1.0
        Standard deviation of the noise after normalization.
    device : torch.device or None, default=None
        Device for tensor allocation. If None, defaults to CPU.
    method : {'blur', 'upsample'}, default='blur'
        Noise generation method. 'upsample' is faster and more memory efficient for larger sigma
        values, but at the cost of quality.

    Returns
    -------
    Tensor
        Smooth Gaussian noise with shape (B, C, *spatial).

    Examples
    --------
    >>> # Generate 2D noise field
    >>> noise_2d = smooth_gaussian(shape=(1, 1, 64, 64), sigma=3.0)
    >>> noise_2d.shape
    torch.Size([1, 1, 64, 64])

    >>> # Generate 3D noise field with multiple channels
    >>> noise_3d = smooth_gaussian(shape=(2, 3, 32, 32, 32), sigma=5.0, magnitude=2.0)
    >>> noise_3d.shape
    torch.Size([2, 3, 32, 32, 32])

    >>> # Use upsample method for efficiency with large sigma
    >>> noise_fast = smooth_gaussian(shape=(1, 1, 128, 128), sigma=10.0, method='upsample')
    """
    spatial_shape = shape[2:]
    ndim = len(spatial_shape)

    if method == 'blur':
        noise = torch.normal(0, 1, size=shape, device=device)
        noise = nef.gaussian_smoothing(noise, sigma=sigma, truncate=3)

    elif method == 'upsample':
        # Compute downsampled shape for spatial dimensions only
        downshape = tuple([max(int(s // sigma), 2) for s in spatial_shape])
        # Generate downsampled noise with (B, C, *downsampled_spatial)
        noise = torch.normal(0, 1, size=(*shape[:2], *downshape), device=device)
        # Upsample to target spatial shape
        mode = {1: 'linear', 2: 'bilinear', 3: 'trilinear'}[ndim]
        noise = torch.nn.functional.interpolate(noise, size=spatial_shape, mode=mode)

    else:
        raise ValueError(f'unknown smooth gaussian method `{method}`')

    # In-place normalize
    noise -= noise.mean()
    noise *= magnitude / noise.std()
    return noise


def perlin(
    shape: Sequence[int],
    smoothing: Union[float, List[float], None] = None,
    magnitude: float = 1.0,
    weights: Union[List[float], None] = None,
    device: Union[torch.device, None] = None,
    method: Literal['blur', 'upsample'] = 'blur'
) -> Tensor:
    """
    Generate Perlin noise in (B, C, *spatial) format.

    Parameters
    ----------
    shape : Sequence[int]
        Desired shape of output tensor in (B, C, *spatial) format. Must have at least 3
        dimensions (batch, channel, and spatial). Examples: (1, 1, 64, 64) for 2D,
        (2, 3, 64, 64, 64) for 3D.
    smoothing : float, List[float], or None, default=None
        Spatial smoothing sigma(s) in voxel coordinates for each scale. If None, defaults
        to powers of 2 up to max spatial dimension. If scalar, reduces to single-scale
        smooth_gaussian(). If list, each value defines a smoothing scale.
    magnitude : float, default=1.0
        Standard deviation of the final normalized noise.
    weights : List[float] or None, default=None
        Weight for each smoothing scale. If None, uses linearly increasing weights
        [1, 2, 3, ...]. Length must match smoothing if both are lists.
    device : torch.device or None, default=None
        Device for tensor allocation. If None, defaults to CPU.
    method : {'blur', 'upsample'}, default='blur'
        Noise generation method. 'upsample' is faster and more memory efficient for
        larger sigma values, but at the cost of quality.

    Returns
    -------
    Tensor
        Perlin noise with shape (B, C, *spatial).

    Examples
    --------
    >>> # Generate 2D Perlin noise with default scales
    >>> noise_2d = perlin(shape=(1, 1, 64, 64))
    >>> noise_2d.shape
    torch.Size([1, 1, 64, 64])

    >>> # Generate 3D Perlin noise with custom smoothing scales
    >>> noise_3d = perlin(shape=(1, 1, 32, 32, 32), smoothing=[2.0, 4.0, 8.0], magnitude=2.0)

    >>> # Single-scale Perlin (equivalent to smooth_gaussian)
    >>> noise_single = perlin(shape=(1, 1, 64, 64), smoothing=5.0)
    """
    spatial_shape = shape[2:]

    # Default smoothing: powers of 2 up to max spatial dimension
    if smoothing is None:
        smoothing = 2 ** np.arange(np.log2(max(spatial_shape)))[1:]

    # Single-scale case: delegate to smooth_gaussian
    elif np.isscalar(smoothing):
        return smooth_gaussian(
            shape=shape, sigma=smoothing, magnitude=magnitude, device=device, method=method
        )

    # Multi-scale case: combine multiple smoothing levels
    if len(smoothing) == 1:
        weights = [None]
    elif weights is None:
        weights = np.arange(len(smoothing)) + 1

    noise = None
    for s, w in zip(smoothing, weights):
        # Generate smooth field at this scale
        sample = smooth_gaussian(shape=shape, sigma=s, device=device, method=method)
        if w is not None:
            sample *= w

        # Merge with accumulated noise
        if noise is None:
            noise = sample
        else:
            noise += sample

    # In-place normalize
    noise -= noise.mean()
    noise *= magnitude / noise.std()
    return noise


def random_disp(
    shape: List[int],
    smoothing: Union[float, List[float]] = 10,
    magnitude: Union[float, List[float]] = 10,
    integrations: int = 0,
    voxsize: float = 1,
    meshgrid: Tensor = None,
    device: torch.device = None,
    perlin_method: str = 'upsample'
) -> Tensor:
    """
    Generate random displacement field using Perlin noise.

    Creates a displacement field by generating independent Perlin noise for each spatial
    dimension and stacking them. The resulting field has shape (*spatial, ndim).

    Parameters
    ----------
    shape : List[int]
        Spatial shape of the displacement field (e.g., [64, 64] for 2D, [64, 64, 64] for 3D).
    smoothing : float or List[float], default=10
        Spatial smoothing sigma in voxel coordinates, divided by voxsize.
    magnitude : float or List[float], default=10
        Standard deviation of displacement in voxel coordinates, divided by voxsize.
    integrations : int, default=0
        Number of integration steps for diffeomorphic transform. If 0, no integration.
    voxsize : float, default=1
        Voxel size for scaling smoothing and magnitude parameters.
    meshgrid : Tensor or None, default=None
        Coordinate grid for integration. If None and integrations > 0, computed internally.
    device : torch.device or None, default=None
        Device for tensor allocation. If None, defaults to CPU.
    perlin_method : str, default='upsample'
        Noise generation method ('blur' or 'upsample').

    Returns
    -------
    Tensor
        Displacement field with shape (*spatial, ndim).

    Examples
    --------
    >>> # Generate 2D displacement field
    >>> disp_2d = random_disp(shape=[64, 64], smoothing=5.0, magnitude=3.0)
    >>> disp_2d.shape
    torch.Size([64, 64, 2])

    >>> # Generate 3D displacement field with integration
    >>> disp_3d = random_disp(shape=[32, 32, 32], integrations=5)
    >>> disp_3d.shape
    torch.Size([32, 32, 32, 3])
    """
    # Scale parameters by voxel size
    smoothing = smoothing / voxsize
    magnitude = magnitude / voxsize

    # Generate independent Perlin noise for each spatial dimension
    ndim = len(shape)
    disp = [
        perlin(
            shape=(1, 1, *shape),  # Add batch and channel dimensions
            smoothing=smoothing,
            magnitude=magnitude,
            method=perlin_method,
            device=device
        ).squeeze(0).squeeze(0)  # Remove batch and channel dimensions
        for i in range(ndim)
    ]
    disp = torch.stack(disp, dim=-1)

    if integrations > 0:
        disp = vxm.functional.integrate_disp(disp, integrations, meshgrid)

    return disp


def random_transform(
    shape: List[int],
    affine_probability: float = 1.0,
    max_translation: float = 5.0,
    max_rotation: float = 5.0,
    max_scaling: float = 1.1,
    warp_probability: float = 1.0,
    warp_integrations: int = 5,
    warp_smoothing_range: List[int] = [10, 20],
    warp_magnitude_range: List[int] = [1, 2],
    voxsize: int = 1,
    device: torch.device = None,
    isdisp: bool = True,
    perlin_method: str = 'upsample',
    sampling: bool = True,
) -> Tensor:
    """
    generate a randomly sampled transform

    Parameters:
    -----------
    disp: torch.Tensor
        Displacement crs field
    meshgrid: torch.Tensor, optional
       crs grid for the image shape

    Returns:
    --------
    torch.Tensor:
       displacement crs field, or
       absolute crs field scaled to range [-1, 1] if isdisp is False
    """
    ndim = len(shape)
    trf = None

    # generate a random affine
    if ne.utils.bernoulli(p=affine_probability, shape=(1,)).item():

        # compute meshgrid, it is the target crs
        meshgrid = ne.volshape_to_ndgrid(size=shape, device=device, stack=True)

        # convert max_translation from mm to voxel
        # the matrix returned from vxm.random_affine() is vox2vox rotating around the image center.
        # it is used as target to source transformation in affine_to_disp() to covert
        # the vox2vox matrix to dispacement field.
        max_translation = max_translation / voxsize
        matrix = vxm.random_affine(
            ndim=ndim,
            max_translation=max_translation,
            max_rotation=max_rotation,
            max_scaling=max_scaling,
            device=device,
            sampling=sampling)
        trf = vxm.functional.affine_to_disp(matrix, meshgrid)

    # generate a nonlinear transform
    if ne.utils.bernoulli(p=warp_probability, shape=(1,)).item():
        disp = random_disp(
            shape=shape,
            smoothing=np.random.uniform(*warp_smoothing_range),
            magnitude=np.random.uniform(*warp_magnitude_range),
            integrations=warp_integrations,
            voxsize=voxsize,
            device=device,
            perlin_method=perlin_method)

        # merge with the affine transform if necessary
        if trf is None:
            trf = disp
        else:
            trf += vxm.functional.spatial_transform(
                disp.movedim(-1, 0), trf, meshgrid=meshgrid, non_spatial_dims=(0,)
            ).movedim(0, -1)

    # convert to coordinates if specified
    if trf is not None and not isdisp:
        # compute the absolute crs field scaled to range [-1, 1]
        trf = vxm.functional.disp_to_coords(trf)

    return trf


def constant_shift_field(
    shape: tuple = (1, 1, 16, 16),
    shift_size: Union[int, Sequence[Union[int, float]], torch.Tensor] = 1,
    normalize: bool = False,
    device: str = 'cpu',
) -> torch.Tensor:
    """
    Makes a simple flow field for testing registration in N-dimensional space.

    This function generates a flow field with channels that represent the transformations to each
    spatial dimension. E.g. channel 1 represents the dense transformation on the x-axis, channel 2
    represents the dense transformation on the y axis, and so on...

    Parameters
    ----------
    shape : Sequence[int], default=(1, 1, 16, 16)
        Shape of the input tensor, expected as (B, C, *spatial_dims).
    shift_size : int, Sequence[int], or torch.Tensor, default=1
        Shift magnitude for each axis. If int, same shift on all axes. If Sequence[int], length must
        equal number of spatial dims. If Tensor, must have shape (n_spatial_dims,).
    normalize : bool, default=False
        If True, normalize the first spatial channel by (size - 1), where
        size is the extent of that axis.
    device : str, default='cpu'
        Device on which to create the tensor.

    Returns
    -------
    flow_field : torch.Tensor
        A tensor representing the flow field, shaped as (B, n_spatial_dims, *spatial_dims).
        The first spatial dimension is shifted by +1 in a normalized manner.

    Example
    -------
    >>> flow = constant_shift_field((1, 1, 4, 4), device='cpu')
    >>> flow.shape
    torch.Size([1, 2, 4, 4])

    >>> flow_3d = constant_shift_field((1, 1, 4, 4, 4), device='cpu')
    >>> flow_3d.shape
    torch.Size([1, 3, 4, 4, 4])
    """

    spatial_dims = shape[2:]
    n_spatial_dims = len(spatial_dims)

    if isinstance(shift_size, int):
        shift_size: torch.Tensor = torch.tensor([shift_size] * n_spatial_dims)
    elif isinstance(shift_size, (Sequence)):
        shift_size = torch.tensor(shift_size)
    elif isinstance(shift_size, torch.Tensor):
        pass
    else:
        raise ValueError(f'shift_size must be a tensor, got {type(shift_size)}: {shift_size}')

    assert shift_size.shape[0] == n_spatial_dims, (
        f'shift_size must have {n_spatial_dims} elements. Got {shift_size.shape}: {shift_size}')

    flow_field = torch.zeros(shape[0], n_spatial_dims, *spatial_dims, device=device)

    # Reshape shift_size for broadcasting across spatial dimensions
    shift_size = shift_size.view(1, -1, *[1] * n_spatial_dims)
    flow_field += shift_size

    if normalize:
        flow_field[:, 0, ...] /= (spatial_dims[0] - 1)

    return flow_field


def make_square_affine(mat: Tensor) -> Tensor:
    """
    Convert affine matrix from compact form (..., N, N+1) to square form (..., N+1, N+1).

    Adds the homogeneous row [0, 0, ..., 0, 1] to the bottom of the matrix.

    Parameters
    ----------
    mat : Tensor
        Affine matrix of shape (..., M, N+1) where M is N or N+1.

    Returns
    -------
    Tensor
        Square affine matrix of shape (..., N+1, N+1).

    Examples
    --------
    >>> affine = torch.tensor(
    >>> ... [[1., 0., 5.],
    >>> ... [0., 1., 3.]]
    >>> )
    >>> square = make_square_affine(affine)
    >>> square.shape
    torch.Size([3, 3])
    >>> square[-1]
    tensor([0., 0., 1.])
    """
    if not vxm.is_affine_shape(mat.shape):
        raise ValueError(f'Invalid affine shape: {mat.shape}')

    # Already square
    if mat.shape[-2] == mat.shape[-1]:
        return mat

    # Get dimensions
    *batch_dims, rows, cols = mat.shape

    # Create bottom row as [0, 0, ..., 0, 1]
    bottom_row = torch.zeros(*batch_dims, 1, cols, dtype=mat.dtype, device=mat.device)
    bottom_row[..., 0, -1] = 1.0

    return torch.cat([mat, bottom_row], dim=-2)


def compose(
    transforms: Sequence[Tensor],
    interpolation_mode: str = 'bilinear',
    origin_at_center: bool = True,
    shape: Sequence[int] | None = None
) -> Tensor:
    """
    Compose a single transform from a series of transforms.

    Supports both affine matrices and dense displacement fields. Returns a displacement
    field unless all inputs are affine matrices. For transforms [A, B, C], the composed
    transform T satisfies T(x) = A(B(C(x))), meaning C is applied first, then B, then A.

    Parameters
    ----------
    transforms : Sequence[Tensor]
        List or tuple of affine matrices and/or displacement fields to compose.
        - Affine matrices: shape (..., N, N+1) or (..., N+1, N+1)
        - Displacement fields: shape (..., *spatial_shape, N)
    interpolation_mode : str, optional
        Interpolation method for composing displacement fields.
        Options: 'bilinear', 'nearest', 'trilinear'. Default is 'bilinear'.
    origin_at_center : bool, optional
        Shift grid origin to image center when converting affine matrices to displacement fields.
        Default is True.
    shape : Sequence[int], optional
        Spatial shape (N dimensions) for converting affine matrices to displacement fields.
        Only used if the rightmost transform is an affine matrix. If None and the rightmost
        transform is an affine, you must have at least one displacement field in the list.
        Incompatible with origin_at_center=False.

    Returns
    -------
    Tensor
        Composed transform as either:
        - Affine matrix of shape (..., N, N+1) if all inputs are affine
        - Displacement field of shape (..., *spatial_shape, N) otherwise

    Examples
    --------
    >>> # Compose two affine matrices
    >>> translate = torch.tensor([[1., 0., 10.],
    ...                           [0., 1., 5.]])
    >>> scale = torch.tensor([[2., 0., 0.],
    ...                       [0., 2., 0.]])
    >>> composed = compose([translate, scale])
    >>> # Result is affine: scale applied first, then translate

    >>> # Compose affine with displacement field
    >>> disp = torch.randn(64, 64, 2)
    >>> affine = torch.tensor([[1., 0., 5.],
    ...                        [0., 1., 3.]])
    >>> composed = compose([affine, disp])
    >>> # Result is displacement field: disp applied first, then affine

    >>> # Compose multiple displacement fields
    >>> disp1 = torch.randn(64, 64, 2)
    >>> disp2 = torch.randn(64, 64, 2)
    >>> composed = compose([disp1, disp2])
    >>> # Result is displacement field

    Notes
    -----
    The composition uses matrix indexing ('ij') consistently. When composing displacement
    fields, the left field is interpolated using the right field as sampling coordinates.
    """
    if len(transforms) == 0:
        raise ValueError('Cannot compose empty list of transforms')

    if len(transforms) == 1:
        return transforms[0]

    # Convert all to tensors with floating point dtype
    safe_transforms = []
    for transform in transforms:
        if isinstance(transform, Tensor) and not transform.is_floating_point():
            transform = transform.float()
        elif not isinstance(transform, Tensor):
            transform = torch.as_tensor(transform, dtype=torch.float32)
        safe_transforms.append(transform)

    # Start from the rightmost transform (last to be applied)
    curr = transforms[-1]

    # Iterate through remaining transforms in reverse order
    for next_trf in reversed(transforms[:-1]):

        curr_is_affine = vxm.is_affine_shape(curr.shape)

        # Case 1: Dense warp on left, affine on right. Convert affine to disp
        if not vxm.is_affine_shape(next_trf.shape):
            if curr_is_affine:
                curr_shape = next_trf.shape[-next_trf.shape[-1] - 1:-1]
                if shape is not None:
                    curr_shape = shape
                curr = vxm.functional.affine_to_disp(
                    affine=curr,
                    shape=curr_shape,
                    origin_at_center=origin_at_center
                )

            # Now both are displacement fields: warp next using curr
            # This computes: next(x + curr(x))
            # spatial_transform expects (C, *spatial) format, but displacement is (*spatial, N)
            ndim = next_trf.shape[-1]
            next_trf_permuted = next_trf.permute(-1, *range(ndim))  # (*spatial, N) -> (N, *spatial)

            warped = vxm.functional.spatial_transform(
                image=next_trf_permuted,
                trf=curr,
                mode=interpolation_mode,
                isdisp=True,
                non_spatial_dims=(0,)
            )

            # Permute back: (N, *spatial) -> (*spatial, N)
            warped = warped.permute(*range(1, ndim + 1), 0)
            curr = curr + warped

        # Case 2: Affine on left, dense warp on right
        elif not curr_is_affine:
            curr = vxm.functional.affine_to_disp(
                next_trf,
                shape=curr.shape[-curr.shape[-1] - 1: -1],  # Spatial shape from curr
                origin_at_center=origin_at_center,
                warp_right=curr
            )

        # Case 3: Both are affine matrices
        else:
            next_sq = make_square_affine(next_trf)
            curr_sq = make_square_affine(curr)
            curr = (next_sq @ curr_sq)[..., :-1, :]  # Remove last row to return compact form

    return curr

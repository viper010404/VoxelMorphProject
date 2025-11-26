"""
Functions containing the core operations and logic of for image registration for `voxelmorph`
written in PyTorch.
"""

# Standard library imports
from typing import List, Union, Optional, Sequence, Literal

# Third-party imports
import torch
import numpy as np

import neurite as ne
import neurite.nn.functional as nef
import voxelmorph as vxm

__all__ = [
    "spatial_transform",
    "smooth_gaussian",
    "perlin",
    "random_disp",
    "random_transform",
]


def spatial_transform(
    image: torch.Tensor,
    trf: Union[torch.Tensor, None],
    method: Literal['nearest', 'linear'] = 'linear',
    isdisp: bool = True,
    meshgrid: Union[torch.Tensor, None] = None,
    origin_at_center: bool = True
) -> torch.Tensor:
    """
    Apply spatial transformation to image in (B, C, *spatial) format.

    Wrapper around voxelmorph.functional.spatial_transform with non_spatial_dims=(0, 1).

    Parameters
    ----------
    image : torch.Tensor
        Input image with shape (B, C, *spatial).
    trf : torch.Tensor or None
        Transformation field. Can be:
        - Affine matrix: shape (N+1, N+1) or (N, N+1)
        - Displacement field: shape (*spatial, N)
        - Coordinate field: shape (*spatial, N)
        - None: returns image unchanged
    method : str, default='linear'
        Interpolation mode ('linear' or 'nearest').
    isdisp : bool, default=True
        If True, treat trf as displacement field. If False, treat as coordinates.
    meshgrid : torch.Tensor or None, default=None
        Pre-computed coordinate grid.
    origin_at_center : bool, default=True
        Place origin at image center for affine transformations.

    Returns
    -------
    torch.Tensor
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
    scale: float,
    magnitude: float = 1.0,
    device: Union[torch.device, None] = None,
    method: Literal['blur', 'upsample'] = 'blur'
) -> torch.Tensor:
    """
    Generate smooth Gaussian noise in (B, C, *spatial) format.

    Parameters
    ----------
    shape : Sequence[int]
        Desired shape of output tensor in (B, C, *spatial). Must have at least 3 dimensions
        (batch, channel, and spatial). Examples: (1, 1, 64, 64) for 2D, (2, 3, 64, 64, 64) for 3D.
    scale : float
        Spatial smoothing scale in voxel coordinates.
    magnitude : float, default=1.0
        Standard deviation of the noise after normalization.
    device : torch.device or None, default=None
        Device for tensor allocation. If None, defaults to CPU.
    method : {'blur', 'upsample'}, default='blur'
        Noise generation method. 'upsample' is faster and more memory efficient for larger scale
        values, but at the cost of quality.

    Returns
    -------
    torch.Tensor
        Smooth Gaussian noise with shape (B, C, *spatial).

    Examples
    --------
    >>> # Generate 2D noise field
    >>> noise_2d = smooth_gaussian(shape=(1, 1, 64, 64), scale=3.0)
    >>> noise_2d.shape
    torch.Size([1, 1, 64, 64])

    >>> # Generate 3D noise field with multiple channels
    >>> noise_3d = smooth_gaussian(shape=(2, 3, 32, 32, 32), scale=5.0, magnitude=2.0)
    >>> noise_3d.shape
    torch.Size([2, 3, 32, 32, 32])

    >>> # Use upsample method for efficiency with large scale
    >>> noise_fast = smooth_gaussian(shape=(1, 1, 128, 128), scale=10.0, method='upsample')
    """
    if method == 'blur':
        noise = torch.normal(0, 1, size=shape, device=device)
        noise = nef.gaussian_smoothing(noise, sigma=scale, truncate=3)

    elif method == 'upsample':
        noise = vxm.upsample_noise(shape, scale=scale, non_spatial_dims=(0, 1), device=device)

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
) -> torch.Tensor:
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
    torch.Tensor
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
    meshgrid: torch.Tensor = None,
    device: torch.device = None,
    perlin_method: str = 'upsample'
) -> torch.Tensor:
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
    meshgrid : torch.Tensor or None, default=None
        Coordinate grid for integration. If None and integrations > 0, computed internally.
    device : torch.device or None, default=None
        Device for tensor allocation. If None, defaults to CPU.
    perlin_method : str, default='upsample'
        Noise generation method ('blur' or 'upsample').

    Returns
    -------
    torch.Tensor
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
) -> torch.Tensor:
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

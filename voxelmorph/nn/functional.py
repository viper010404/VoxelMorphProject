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
    "upsample_noise",
    "smooth_gaussian",
    "fractal_noise",
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


def upsample_noise(
    shape: Sequence[int],
    scale: Union[float, int],
    device: Union[torch.device, None] = None,
) -> torch.Tensor:
    """
    Generate smooth noise by upsampling from a coarse grid in (B, C, *spatial) format.

    Wrapper around voxelmorph.upsample_noise with non_spatial_dims=(0, 1).

    Parameters
    ----------
    shape : Sequence[int]
        Desired shape of output tensor in (B, C, *spatial). Must have at least 3 dimensions
        (batch, channel, and spatial). Examples: (1, 1, 64, 64) for 2D, (2, 3, 64, 64, 64) for 3D.
    scale : float or int
        Downsampling factor. Larger values produce smoother noise. The coarse grid size along each
        spatial dimension is max(spatial_size // scale, 2).
    device : torch.device or None, default=None
        Device for tensor allocation. If None, defaults to CPU.

    Returns
    -------
    torch.Tensor
        Upsampled noise with shape (B, C, *spatial).

    Examples
    --------
    >>> # Generate 2D noise field
    >>> noise_2d = upsample_noise(shape=(1, 1, 64, 64), scale=8.0)
    >>> noise_2d.shape
    torch.Size([1, 1, 64, 64])

    >>> # Generate 3D noise field with multiple channels
    >>> noise_3d = upsample_noise(shape=(2, 3, 32, 32, 32), scale=4.0)
    >>> noise_3d.shape
    torch.Size([2, 3, 32, 32, 32])
    """
    return vxm.upsample_noise(
        shape=shape,
        scale=scale,
        non_spatial_dims=(0, 1),
        device=device,
    )


def smooth_gaussian(
    shape: Sequence[int],
    scale: float,
    magnitude: float = 1.0,
    device: Union[torch.device, None] = None,
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
    """
    return vxm.smooth_gaussian(
        shape=shape,
        sigma=scale,
        magnitude=magnitude,
        non_spatial_dims=(0, 1),
        device=device,
    )


def fractal_noise(
    shape: Sequence[int],
    scales: Union[float, int, List[float], None] = None,
    magnitude: float = 1.0,
    weights: Union[List[float], None] = None,
    device: Union[torch.device, None] = None,
    method: Literal['blur', 'upsample'] = 'blur'
) -> torch.Tensor:
    """
    Generate Fractal noise in (B, C, *spatial) format.

    Parameters
    ----------
    shape : Sequence[int]
        Desired shape of output tensor in (B, C, *spatial) format. Must have at least 3
        dimensions (batch, channel, and spatial). Examples: (1, 1, 64, 64) for 2D,
        (2, 3, 64, 64, 64) for 3D.
    scales : float, int, List[float], or None, default=None
        Smoothing scale(s) for each octave. Interpretation depends on method:
        - method='blur': sigma values for Gaussian smoothing
        - method='upsample': downsampling factors for upsampled noise
        If None, defaults to powers of 2 up to max spatial dimension.
        If scalar, reduces to single-scale noise generation.
    magnitude : float, default=1.0
        Standard deviation of the final normalized noise.
    weights : List[float] or None, default=None
        Weight for each scale. If None, uses linearly increasing weights
        [1, 2, 3, ...]. Length must match scales if both are lists.
    device : torch.device or None, default=None
        Device for tensor allocation. If None, defaults to CPU.
    method : {'blur', 'upsample'}, default='blur'
        Noise generation method:
        - 'blur': Generate noise at full spatial res and apply Gaussian smoothing (higher quality)
        - 'upsample': Generate coarse noise and upsample (faster, lower memory)

    Returns
    -------
    torch.Tensor
        Fractal noise with shape (B, C, *spatial).

    Examples
    --------
    >>> # Generate 2D Fractal noise with default scales
    >>> noise_2d = fractal_noise(shape=(1, 1, 64, 64))
    >>> noise_2d.shape
    torch.Size([1, 1, 64, 64])

    >>> # Generate 3D Fractal noise with custom scales
    >>> noise_3d = fractal_noise(shape=(1, 1, 32, 32, 32), scales=[2.0, 4.0, 8.0], magnitude=2.0)

    >>> # Single-scale Fractal (equivalent to smooth_gaussian)
    >>> noise_single = fractal_noise(shape=(1, 1, 64, 64), scales=5.0)

    >>> # Using upsample method for faster generation
    >>> noise_fast = fractal_noise(shape=(1, 1, 128, 128), method='upsample', scales=[4, 8, 16])
    """
    return vxm.fractal_noise(
        shape=shape,
        scales=scales,
        magnitude=magnitude,
        weights=weights,
        non_spatial_dims=(0, 1),
        device=device,
        method=method,
    )


def random_disp(
    shape: Sequence[int],
    scales: Union[float, int, List[float]] = 10,
    magnitude: float = 10,
    integrations: int = 0,
    voxsize: float = 1,
    meshgrid: Union[torch.Tensor, None] = None,
    device: Union[torch.device, None] = None,
    method: Literal['blur', 'upsample'] = 'upsample'
) -> torch.Tensor:
    """
    Generate random displacement field for images in (B, C, *spatial) format.

    Takes shape in (B, C, *spatial) format (matching image tensors) but outputs
    displacement field in (B, *spatial, ndim) format for use with grid_sample.
    The channel dimension is ignored since displacement is per-voxel, not per-channel.

    Parameters
    ----------
    shape : Sequence[int]
        Shape in (B, C, *spatial) format matching the image to be transformed.
        Examples: (1, 1, 64, 64) for 2D, (2, 3, 64, 64, 64) for 3D.
    scales : float, int, or List[float], default=10
        Smoothing scale(s) for fractal noise, divided by voxsize. Interpretation depends
        on method:
        - method='blur': sigma values for Gaussian smoothing
        - method='upsample': downsampling factors for upsampled noise
    magnitude : float, default=10
        Standard deviation of displacement in voxel coordinates, divided by voxsize.
    integrations : int, default=0
        Number of integration steps for diffeomorphic transform. If 0, no integration.
    voxsize : float, default=1
        Voxel size for scaling smoothing and magnitude parameters.
    meshgrid : torch.Tensor or None, default=None
        Coordinate grid for integration. If None and integrations > 0, computed internally.
    device : torch.device or None, default=None
        Device for tensor allocation.
    method : {'blur', 'upsample'}, default='upsample'
        Noise generation method:
        - 'blur': Generate noise and apply Gaussian smoothing (higher quality)
        - 'upsample': Generate coarse noise and upsample (faster, lower memory)

    Returns
    -------
    torch.Tensor
        Displacement field with shape (B, *spatial, ndim).

    Examples
    --------
    >>> # Generate displacement for 2D image with shape (B, C, H, W)
    >>> disp = random_disp(shape=(1, 1, 64, 64), scales=5.0, magnitude=3.0)
    >>> disp.shape
    torch.Size([1, 64, 64, 2])

    >>> # Generate displacement for 3D image with shape (B, C, D, H, W)
    >>> disp = random_disp(shape=(2, 3, 32, 32, 32), integrations=5)
    >>> disp.shape
    torch.Size([2, 32, 32, 32, 3])
    """
    # Extract batch and spatial shape, ignoring channel dimension
    batch_size = shape[0]
    spatial_shape = shape[2:]  # Skip B and C

    return vxm.random_disp(
        shape=(batch_size, *spatial_shape),
        scales=scales,
        magnitude=magnitude,
        integrations=integrations,
        voxsize=voxsize,
        meshgrid=meshgrid,
        non_spatial_dims=(0,),
        device=device,
        method=method,
    )


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
    fractal_method: str = 'upsample',
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
            fractal_method=fractal_method)

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

"""
Functions containing the core operations and logic of for image registration for `voxelmorph`
written in PyTorch.
"""

# Core library imports
from typing import List, Union, Sequence, Literal

# Third-party imports
import torch

# Local imports
import voxelmorph as vxm


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
        - Displacement field: shape (N, *spatial) - channels-first format
        - None: returns image unchanged
    method : str, default='linear'
        Interpolation mode ('linear' or 'nearest').
    isdisp : bool, default=True
        If True, treat trf as displacement field (N, *spatial). If False, treat as
        coordinates (*spatial, N) ready for grid_sample.
    meshgrid : torch.Tensor or None, default=None
        Pre-computed coordinate grid of shape (ndim, *spatial).
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
    >>> disp = torch.randn(2, 64, 64)  # (ndim, H, W)
    >>> warped = spatial_transform(image, disp)
    >>> warped.shape
    torch.Size([2, 3, 64, 64])

    >>> # 3D image with batch and channel
    >>> image = torch.randn(1, 1, 64, 64, 64)
    >>> disp = torch.randn(3, 64, 64, 64)  # (ndim, D, H, W)
    >>> warped = spatial_transform(image, disp)
    >>> warped.shape
    torch.Size([1, 1, 64, 64, 64])
    """
    return vxm.spatial_transform(
        image=image,
        trf=trf,
        mode=method,
        isdisp=isdisp,
        meshgrid=meshgrid,
        origin_at_center=origin_at_center,
        non_spatial_dims=(0, 1),
        align_corners=True
    )


def disp_to_coords(
    disp: torch.Tensor,
    meshgrid: Union[torch.Tensor, None] = None
) -> torch.Tensor:
    """
    Convert displacement field to normalized coordinates for (B, ndim, *spatial) format.

    Adds displacement to base meshgrid coordinates and normalizes to [-1, 1] range.

    Parameters
    ----------
    disp : torch.Tensor
        Displacement field with shape (B, ndim, *spatial).
    meshgrid : torch.Tensor or None, default=None
        Pre-computed coordinate grid of shape (ndim, *spatial). If None, computed
        from displacement field spatial shape.

    Returns
    -------
    torch.Tensor
        Normalized coordinates in range [-1, 1] with shape (B, ndim, *spatial).

    Examples
    --------
    >>> # 2D displacement field with batch
    >>> disp = torch.randn(2, 2, 64, 64)  # (B, ndim, H, W)
    >>> coords = disp_to_coords(disp)
    >>> coords.shape
    torch.Size([2, 2, 64, 64])

    >>> # 3D displacement field
    >>> disp = torch.randn(1, 3, 32, 32, 32)  # (B, ndim, D, H, W)
    >>> coords = disp_to_coords(disp)
    >>> coords.shape
    torch.Size([1, 3, 32, 32, 32])
    """
    batch_size = disp.shape[0]

    coords = torch.stack([
        vxm.disp_to_coords(disp[i], meshgrid=meshgrid)
        for i in range(batch_size)
    ])

    return coords


def coords_to_disp(
    coords: torch.Tensor,
    meshgrid: Union[torch.Tensor, None] = None
) -> torch.Tensor:
    """
    Convert normalized coordinates to displacement field for (B, ndim, *spatial) format.

    This is the inverse operation of disp_to_coords().

    Parameters
    ----------
    coords : torch.Tensor
        Normalized coordinates in range [-1, 1] with shape (B, *spatial, ndim).
        Channels-last format as output by grid_sample.
    meshgrid : torch.Tensor or None, default=None
        Pre-computed coordinate grid of shape (ndim, *spatial). If None, computed
        from coordinate field spatial shape.

    Returns
    -------
    torch.Tensor
        Displacement field with shape (B, ndim, *spatial).

    Examples
    --------
    >>> # Round-trip conversion
    >>> import voxelmorph.nn.functional as vxf
    >>> original_disp = torch.randn(2, 2, 64, 64)  # (B, ndim, H, W)
    >>> coords = vxf.disp_to_coords(original_disp)
    >>> reconstructed_disp = vxf.coords_to_disp(coords)
    >>> torch.allclose(original_disp, reconstructed_disp, atol=1e-6)
    True
    """
    batch_size = coords.shape[0]

    disp = torch.stack([
        vxm.coords_to_disp(coords[i], meshgrid=meshgrid)
        for i in range(batch_size)
    ])

    return disp


def integrate_disp(
    disp: torch.Tensor,
    steps: int,
    meshgrid: Union[torch.Tensor, None] = None
) -> torch.Tensor:
    """
    Integrate displacement field via scaling and squaring for (B, ndim, *spatial) format.

    Converts a stationary velocity field into a displacement field through iterative
    composition. The input is scaled by 1/2^steps, then composed with itself `steps` times.

    Parameters
    ----------
    disp : torch.Tensor
        Displacement/velocity field with shape (B, ndim, *spatial).
    steps : int
        Number of integration steps. If 0, returns disp unchanged.
    meshgrid : torch.Tensor or None, default=None
        Pre-computed coordinate grid of shape (ndim, *spatial). If None, computed
        internally from disp spatial shape.

    Returns
    -------
    torch.Tensor
        Integrated displacement field with shape (B, ndim, *spatial).

    Examples
    --------
    >>> # 2D velocity field with batch
    >>> vel = torch.randn(2, 2, 64, 64)  # (B, ndim, H, W)
    >>> disp = integrate_disp(vel, steps=7)
    >>> disp.shape
    torch.Size([2, 2, 64, 64])

    >>> # 3D velocity field
    >>> vel = torch.randn(1, 3, 32, 32, 32)  # (B, ndim, D, H, W)
    >>> disp = integrate_disp(vel, steps=5)
    >>> disp.shape
    torch.Size([1, 3, 32, 32, 32])
    """
    return vxm.integrate_disp(disp, steps=steps, meshgrid=meshgrid, non_spatial_dims=(0,))


def compose(
    transforms: Sequence[torch.Tensor],
    interpolation_mode: str = 'linear',
    origin_at_center: bool = True,
    shape: Union[Sequence[int], None] = None
) -> torch.Tensor:
    """
    Compose transforms for (B, ndim, *spatial) format displacement fields.

    Composes a sequence of transforms into a single transform. For transforms [A, B, C],
    the composed transform T satisfies T(x) = A(B(C(x))), meaning C is applied first,
    then B, then A.

    Parameters
    ----------
    transforms : Sequence[Tensor]
        List of transforms to compose. Each transform should be:
        - Displacement field: shape (B, ndim, *spatial)
        - Affine matrix: shape (N, N+1) or (N+1, N+1) or batched (B, N, N+1)
    interpolation_mode : str, default='linear'
        Interpolation method for composing displacement fields. Options are {'linear', 'nearest'}.
    origin_at_center : bool, default=True
        Place origin at image center when converting affine matrices to displacement.
    shape : Sequence[int] or None, default=None
        Spatial shape for converting affine matrices to displacement fields.
        Required if rightmost transform is an affine matrix.

    Returns
    -------
    torch.Tensor
        Composed transform as either:
        - Affine matrix if all inputs are affine
        - Displacement field with shape (B, ndim, *spatial) otherwise

    Examples
    --------
    >>> import voxelmorph.nn.functional as vxf
    >>> # Compose two batched displacement fields
    >>> disp1 = torch.randn(2, 2, 64, 64)  # (B, ndim, H, W)
    >>> disp2 = torch.randn(2, 2, 64, 64)
    >>> composed = vxf.compose([disp1, disp2])
    >>> composed.shape
    torch.Size([2, 2, 64, 64])

    >>> # Compose affine with batched displacement
    >>> affine = torch.eye(3).unsqueeze(0).repeat(2, 1, 1)  # (B, 3, 3)
    >>> disp = torch.randn(2, 2, 64, 64)
    >>> composed = vxf.compose([affine, disp])
    >>> composed.shape
    torch.Size([2, 2, 64, 64])
    """
    return vxm.compose(
        transforms=transforms,
        interpolation_mode=interpolation_mode,
        origin_at_center=origin_at_center,
        shape=shape
    )


def random_disp(
    shape: Sequence[int],
    scales: Union[float, int, List[float]] = 10,
    magnitude: float = 10,
    integrations: int = 0,
    voxsize: float = 1,
    meshgrid: Union[torch.Tensor, None] = None,
    device: Union[torch.device, None] = None,
    fractal_mode: Literal['blur', 'upsample'] = 'upsample'
) -> torch.Tensor:
    """
    Generate random displacement field for images in (B, C, *spatial) format.

    Takes shape in (B, C, *spatial) format (matching image tensors) and outputs
    displacement field in (B, ndim, *spatial) format - channels-first format.
    The channel dimension is ignored since displacement is per-voxel, not per-channel.

    Parameters
    ----------
    shape : Sequence[int]
        Shape in (B, C, *spatial) format matching the image to be transformed.
        Examples: (1, 1, 64, 64) for 2D, (2, 3, 64, 64, 64) for 3D.
    scales : float, int, or List[float], default=10
        Smoothing scale(s) for fractal noise, divided by voxsize. Interpretation depends
        on fractal_mode:
        - fractal_mode='blur': sigma values for Gaussian smoothing
        - fractal_mode='upsample': downsampling factors for upsampled noise
    magnitude : float, default=10
        Standard deviation of displacement in voxel coordinates, divided by voxsize.
    integrations : int, default=0
        Number of integration steps for diffeomorphic transform. If 0, no integration.
    voxsize : float, default=1
        Voxel size for scaling smoothing and magnitude parameters.
    meshgrid : torch.Tensor or None, default=None
        Coordinate grid of shape (ndim, *spatial) for integration. If None and
        integrations > 0, computed internally.
    device : torch.device or None, default=None
        Device for tensor allocation.
    fractal_mode : {'blur', 'upsample'}, default='upsample'
        Fractal noise generation method:
        - 'blur': Generate noise and apply Gaussian smoothing (higher quality)
        - 'upsample': Generate coarse noise and upsample (faster, lower memory)

    Returns
    -------
    torch.Tensor
        Displacement field with shape (B, ndim, *spatial) - channels-first format.

    Examples
    --------
    >>> # Generate displacement for 2D image with shape (B, C, H, W)
    >>> disp = random_disp(shape=(1, 1, 64, 64), scales=5.0, magnitude=3.0)
    >>> disp.shape
    torch.Size([1, 2, 64, 64])

    >>> # Generate displacement for 3D image with shape (B, C, D, H, W)
    >>> disp = random_disp(shape=(2, 3, 32, 32, 32), integrations=5)
    >>> disp.shape
    torch.Size([2, 3, 32, 32, 32])
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
        fractal_mode=fractal_mode,
    )


def random_transform(
    shape: Sequence[int],
    affine_probability: float = 1.0,
    max_translation: float = 5.0,
    max_rotation: float = 5.0,
    max_scaling: float = 1.1,
    warp_probability: float = 1.0,
    warp_integrations: int = 5,
    warp_scales_range: Sequence[float] = (10, 20),
    warp_magnitude_range: Sequence[float] = (1, 2),
    voxsize: Union[float, int] = 1,
    device: Union[torch.device, None] = None,
    fractal_mode: Literal['blur', 'upsample'] = 'upsample',
    sampling: bool = True,
) -> torch.Tensor:
    """
    Generate random spatial transformation for images in (B, C, *spatial) format.

    Takes shape in (B, C, *spatial) format (matching image tensors) and outputs
    displacement field in (B, ndim, *spatial) format - channels-first format.

    Parameters
    ----------
    shape : Sequence[int]
        Shape in (B, C, *spatial) format matching the image to be transformed.
        Examples: (1, 1, 64, 64) for 2D, (2, 3, 64, 64, 64) for 3D.
    affine_probability : float, default=1.0
        Probability of applying an affine transformation.
    max_translation : float, default=5.0
        Maximum translation in voxel coordinates (before dividing by voxsize).
    max_rotation : float, default=5.0
        Maximum rotation in degrees.
    max_scaling : float, default=1.1
        Maximum scaling factor (min is 1/max_scaling).
    warp_probability : float, default=1.0
        Probability of applying a nonlinear warp.
    warp_integrations : int, default=5
        Number of integration steps for diffeomorphic warp.
    warp_scales_range : Sequence[float], default=(10, 20)
        Range (min, max) to sample smoothing scales for fractal noise.
    warp_magnitude_range : Sequence[float], default=(1, 2)
        Range (min, max) to sample displacement magnitude.
    voxsize : float or int, default=1
        Voxel size for scaling translation, smoothing, and magnitude parameters.
    device : torch.device or None, default=None
        Device for tensor allocation.
    fractal_mode : {'blur', 'upsample'}, default='upsample'
        Fractal noise generation method for nonlinear warp.
    sampling : bool, default=True
        If True, sample random parameters. If False, use maximum values directly.

    Returns
    -------
    torch.Tensor
        Displacement field with shape (B, ndim, *spatial) - channels-first format.

    Examples
    --------
    >>> # Generate transform for 2D image with shape (B, C, H, W)
    >>> trf = random_transform(shape=(1, 1, 64, 64))
    >>> trf.shape
    torch.Size([1, 2, 64, 64])

    >>> # Generate transform for 3D image with shape (B, C, D, H, W)
    >>> trf = random_transform(shape=(2, 3, 32, 32, 32), max_rotation=10.0)
    >>> trf.shape
    torch.Size([2, 3, 32, 32, 32])
    """
    # Extract batch and spatial shape, ignoring channel dimension
    batch_size = shape[0]
    spatial_shape = shape[2:]  # Skip B and C

    return vxm.random_transform(
        shape=(batch_size, *spatial_shape),
        affine_probability=affine_probability,
        max_translation=max_translation,
        max_rotation=max_rotation,
        max_scaling=max_scaling,
        warp_probability=warp_probability,
        warp_integrations=warp_integrations,
        warp_scales_range=warp_scales_range,
        warp_magnitude_range=warp_magnitude_range,
        voxsize=voxsize,
        non_spatial_dims=(0,),
        device=device,
        fractal_mode=fractal_mode,
        sampling=sampling,
    )

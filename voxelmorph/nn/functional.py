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
import neurite.nn.functional as nef

__all__ = [
    "affine_to_disp",
    "spatial_transform",
    "disp_to_coords",
    "integrate_disp",
    "angles_to_rotation_matrix",
    "compose",
    "params_to_affine",
    "gaussian_kernel_1d",
    "gaussian_blur",
    "smooth_gaussian",
    "perlin",
    "random_disp",
    "chance",
    "random_affine",
    "coords_to_disp",
    "random_transform",
]


def affine_to_disp(
    affine: Tensor,
    meshgrid: Tensor | None = None,
    origin_at_center: bool = True,
    shape: Sequence[int] | None = None,
    warp_right: Tensor | None = None
) -> Tensor:
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
    >>> grid = nef.volshape_to_ndgrid((64, 64), stack=True)
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
        meshgrid = nef.volshape_to_ndgrid(
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


def spatial_transform(
    image: Tensor,
    trf: Tensor,
    method: str = 'linear',
    isdisp: bool = True,
    meshgrid: Tensor = None,
    origin_at_center: bool = True
) -> Tensor:
    """
    TODOC
    """
    if trf is None:
        return image

    if trf.ndim == 2:
        if meshgrid is None:
            meshgrid = nef.volshape_to_ndgrid(
                size=image.shape[1:], device=image.device, stack=True
            )

        trf = torch.linalg.inv(trf)
        trf = affine_to_disp(
            trf,
            meshgrid=meshgrid,
            origin_at_center=origin_at_center
        )
        isdisp = True

    if isdisp:
        # convert the displacement crs to absolute crs scaled to range [-1, 1]
        trf = disp_to_coords(trf, meshgrid=meshgrid)

    # Auto-detect interpolation mode for 'linear' based on dimensionality
    if method == 'linear':
        ndim = image.ndim - 1  # Exclude channel dimension
        method = 'trilinear' if ndim == 3 else 'bilinear'

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


def disp_to_coords(disp, meshgrid=None) -> Tensor:
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
        meshgrid = nef.volshape_to_ndgrid(
            size=disp.shape[:-1], device=disp.device, stack=True
        )

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


def integrate_disp(
    disp: Tensor,
    steps: int,
    meshgrid: Tensor = None
) -> Tensor:
    """
    TODOC
    """
    if meshgrid is None:
        # generate a crs grid
        meshgrid = nef.volshape_to_ndgrid(
            size=disp.shape[:-1], device=disp.device, stack=True
        )

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


def params_to_affine(
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


def gaussian_kernel_1d(sigma, truncate: int = 3, device=None, dtype=None):
    """
    Generate a 1D Gaussian kernel with the specified standard deviations.

    Parameters
    ----------
    sigma : float
        A list of standard deviations for each dimension.
    truncate : int, optional
        The number of standard deviations to extend the kernel before truncating.
    device : torch.device, optional
        The device on which to create the kernel.
    dtype : torch.dtype | None, optional
        Data type of the returned kernel.

    Returns
    -------
    Tensor
        A kernel of shape `2 * truncate * sigma + 1`.

    Notes
    -----
    The kernel is truncated when its values drop below `1e-5` of the maximum value.
    """
    r = int(truncate * sigma + 0.5)
    x = torch.arange(-r, r + 1, device=device, dtype=dtype)
    sigma2 = 1 / torch.clip(torch.as_tensor(sigma), min=1e-5).pow(2)
    pdf = torch.exp(-0.5 * (x.pow(2) * sigma2))
    return pdf / pdf.sum()


def gaussian_blur(
    image: Tensor,
    sigma: List[float],
    batched: bool = False,
    truncate: int = 3,
) -> Tensor:
    """
    Apply Gaussian blurring to an image.

    Parameters
    ----------
    image : Tensor
        An input tensor of shape `(C, W, H[, D])` to blur. A batch dimension
        can be included by setting `batched` to `True`.
    sigma : float or List[float]
        Standard deviation(s) of the Gaussian filter along each dimension.
    batched : bool, optional
        Whether the input tensor includes a batch dimension.
    truncate : int, optional
        The number of standard deviations to extend the kernel before truncating.

    Returns
    -------
    Tensor
        The blurred tensor with the same shape as the input tensor.

    Notes
    -----
    The Gaussian filter is applied using convolution. The size of the filter kernel is
    determined by the standard deviation and the truncation factor.
    """
    ndim = image.ndim - (2 if batched else 1)

    # sanity check for common mistake
    if ndim == 4 and not batched:
        raise ValueError(
            f'gaussian blur input has {image.ndim} dims, but batched option is False'
        )

    # normalize sigmas
    if torch.as_tensor(sigma).ndim == 0:
        sigma = [sigma] * ndim
    if len(sigma) != ndim:
        raise ValueError(f'sigma must be {ndim}D, but got length {len(sigma)}')

    blurred = image if batched else image.unsqueeze(0)

    if all(s == sigma[0] for s in sigma):
        # Isotropic, can use the same vector in all directions cases. Since
        # creating the kernel is actually one of the most time intensive steps
        # this is an efficiency gain worth exploiting
        kernel_vec = gaussian_kernel_1d(
            sigma[0],
            truncate,
            device=blurred.device,
            dtype=blurred.dtype,
        )
        kernel_vecs = [kernel_vec] * ndim
    else:
        # Three different kernels, one for each direction
        kernel_vecs = [
            gaussian_kernel_1d(
                s,
                truncate,
                device=blurred.device,
                dtype=blurred.dtype,
            )
            for s in sigma
        ]

    for dim, kernel in enumerate(kernel_vecs):

        # apply the convolution
        slices = [None] * (ndim + 2)
        slices[dim + 2] = slice(None)
        kernel_dim = kernel[slices]
        conv = getattr(torch.nn.functional, f'conv{ndim}d')
        blurred = conv(blurred, kernel_dim, groups=image.shape[0], padding="same")

    if not batched:
        blurred = blurred.squeeze(0)

    return blurred


def smooth_gaussian(shape, sigma, magnitude=1.0, device=None, method='blur'):
    """
    Generates a smooth Gaussian noise image.

    Parameters
    ----------
    shape : List[int]
        The desired shape of the output tensor. Can be 2D or 3D.
    sigma : float
        The spatial smoothing sigma in voxel coordinates.
    magnitude : float
        The standard deviation of the noise.
    device : torch.device or None, optional
        The device on which the output tensor is allocated. If None, defaults to CPU.
    method : 'blur' or 'upsample'
        Method for noise generation. Upsampling is much faster and more memory efficient
        for larger sigma values, but at the cost of quality.

    Returns
    -------
    Tensor
        A smooth Gaussian noise image of shape `shape`.
    """
    if method == 'blur':
        noise = torch.normal(0, 1, size=shape, device=device)
        noise = gaussian_blur(noise.unsqueeze(0), sigma).squeeze(0)
    elif method == 'upsample':
        downshape = tuple([max(int(s // sigma), 2) for s in shape])
        noise = torch.normal(0, 1, size=(1, 1, *downshape), device=device)
        mode = 'trilinear' if len(shape) == 3 else 'bilinear'
        noise = torch.nn.functional.interpolate(noise, shape, mode=mode).view(shape)
    else:
        raise ValueError(f'unknown smooth gaussian method `{method}`')

    # in-place normalize
    noise -= noise.mean()
    noise *= magnitude / noise.std()
    return noise


def perlin(
    shape,
    smoothing: Union[float, List[float]] = None,
    magnitude: Union[float, List[float]] = 1.0,
    weights=None,
    device=None,
    method='blur'
):
    """
    Generates a perlin noise image.

    Parameters
    ----------
    shape : List[int]
        The desired shape of the output tensor. Can be 2D or 3D.
    smoothing : float or List[float]
        The spatial smoothing sigma(s) in voxel coordinates.
    magnitude : float
        The standard deviation of the noise.
    weights : float or List[float]
        The weights of the smoothing components (scales). If None, defaults
        to monotonically increasing weights.
    device : torch.device or None, optional
        The device on which the output tensor is allocated. If None, defaults to CPU.
    method : 'blur' or 'upsample'
        Method for noise generation. Upsampling is much faster and more memory efficient
        for larger sigma values, but at the cost of quality.

    Returns
    -------
    Tensor
        A Perlin noise image of shape `shape`.
    """
    if smoothing is None:
        smoothing = 2 ** np.arange(np.log2(max(shape)))[1:]

    elif np.isscalar(smoothing):
        return smooth_gaussian(
            shape, smoothing, magnitude, device=device, method=method
        )

    if len(smoothing) == 1:
        weights = [None]

    elif weights is None:
        weights = np.arange(len(smoothing)) + 1

    noise = None
    for s, w in zip(smoothing, weights):

        # generate smooth field
        sample = smooth_gaussian(shape, s, device=device, method=method)
        if w is not None:
            sample *= w

        # merge the noise at this scale with the rest
        if noise is None:
            noise = sample

        else:
            noise += sample

    # in-place normalize
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
    TODOC
    """

    # Perlin can take a list so
    smoothing = smoothing / voxsize
    magnitude = magnitude / voxsize

    # randomly sample a displacement crs field of the input shape
    ndim = len(shape)
    disp = [
        perlin(
            shape, smoothing, magnitude, method=perlin_method, device=device
        ) for i in range(ndim)
    ]
    disp = torch.stack(disp, dim=-1)

    if integrations > 0:
        disp = integrate_disp(disp, integrations, meshgrid)

    return disp


def chance(prob: float) -> bool:
    """
    Returns True with given probability.

    Parameters
    ----------
    prob : float
        Probability of returning True. Must be in the range [0, 1].

    Returns
    -------
    bool
        True with probability `prob`.
    """
    if prob < 0.0 or prob > 1.0:
        raise ValueError(f'chance() expected a value in the range [0, 1], but got {prob}')
    return np.random.rand() < prob


def random_affine(
    ndim: int,
    max_translation: float = 0,
    max_rotation: float = 0,
    max_scaling: float = 1,
    device: torch.device = None,
    sampling: bool = True
) -> Tensor:
    """
    Parameters
    ----------
    ndim : int
        Dimensionality of target transform.
    max_translation : float
        Range to sample translation parameters from. Scalar values define the max
        deviation from 0.0 (-max_translation, max_translation).
    max_rotation : float
        Range to sample rotation parameters from. Scalar values define the max
        deviation from 0.0 (-max_rotation, max_rotation).
    max_scaling : float
        Max to sample scale parameters from.
        It is converted into a 2-element array defines the (min, max) deviation from 1.0.

    Returns
    -------
    Tensor
        vox2vox affine matrix rotating around the image center
    """

    #
    if (sampling):
        translation_range = sorted([-max_translation, max_translation])
        translation = np.random.uniform(*translation_range, size=ndim)
    else:
        translation = np.array([max_translation] * ndim)

    #
    if (sampling):
        rotation_range = sorted([-max_rotation, max_rotation])
        rotation = np.random.uniform(*rotation_range, size=(1 if ndim == 2 else 3))
    else:
        rotation = np.array([max_rotation] * (1 if ndim == 2 else 3))

    #
    if (sampling):
        if max_scaling < 1:
            raise ValueError('max scaling to random affine cannot be less than 1, '
                             'see function doc for more info')
        inv = np.random.choice([-1, 1], size=ndim)
        scale = np.random.uniform(1, max_scaling, size=ndim) ** inv
    else:
        scale = np.array(max_scaling * ndim)

    # compose from random paramters
    aff = params_to_affine(
        ndim=ndim,
        translation=translation,
        rotation=rotation,
        scale=scale,
        device=device)
    return aff


def coords_to_disp(coords, meshgrid=None) -> Tensor:
    """
    TODOC
    """
    if meshgrid is None:
        meshgrid = nef.volshape_to_ndgrid(
            size=coords.shape[:-1], device=coords.device, stack=True
        )

    raise NotImplementedError(
        'coords_to_disp is not yet implemented. '
        'contact andrew if you get this... or implement it :)'
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
    if chance(affine_probability):

        # compute meshgrid, it is the target crs
        meshgrid = nef.volshape_to_ndgrid(size=shape, device=device, stack=True)

        # convert max_translation from mm to voxel
        # the matrix returned from random_affine() is vox2vox rotating around the image center.
        # it is used as target to source transformation in affine_to_disp() to covert
        # the vox2vox matrix to dispacement field.
        max_translation = max_translation / voxsize
        matrix = random_affine(
            ndim=ndim,
            max_translation=max_translation,
            max_rotation=max_rotation,
            max_scaling=max_scaling,
            device=device,
            sampling=sampling)
        trf = affine_to_disp(matrix, meshgrid)

    # generate a nonlinear transform
    if chance(warp_probability):
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
            trf += spatial_transform(disp.movedim(-1, 0), trf, meshgrid=meshgrid).movedim(0, -1)

    # convert to coordinates if specified
    if trf is not None and not isdisp:
        # compute the absolute crs field scaled to range [-1, 1]
        trf = disp_to_coords(trf)

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


def is_affine_shape(shape: tuple) -> bool:
    """
    Determine whether the given shape represents an N-dimensional affine matrix.

    An affine matrix has shape (..., M, N+1) where:
    - N is the spatial dimensionality (2 or 3)
    - M is either N or N+1 (compact or square form)

    Parameters
    ----------
    shape : tuple
        Shape of the tensor to check.

    Returns
    -------
    bool
        True if shape represents an affine matrix, False otherwise.
    """
    if len(shape) < 2:
        return False

    rows, cols = shape[-2], shape[-1]

    # Cols should be N+1 where N is 2 or 3
    ndim = cols - 1
    if ndim not in (2, 3):
        return False

    # rows should be N or N+1
    if rows not in (ndim, ndim + 1):
        return False

    return True


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
    if not is_affine_shape(mat.shape):
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

        curr_is_affine = is_affine_shape(curr.shape)

        # Case 1: Dense warp on left, affine on right. Convert affine to disp
        if not is_affine_shape(next_trf.shape):
            if curr_is_affine:
                curr_shape = next_trf.shape[-next_trf.shape[-1] - 1:-1]
                if shape is not None:
                    curr_shape = shape
                curr = affine_to_disp(curr, shape=curr_shape, origin_at_center=origin_at_center)

            # Now both are displacement fields: warp next using curr
            # This computes: next(x + curr(x))
            # spatial_transform expects (C, *spatial) format, but displacement is (*spatial, N)
            ndim = next_trf.shape[-1]
            next_trf_permuted = next_trf.permute(-1, *range(ndim))  # (*spatial, N) -> (N, *spatial)

            warped = spatial_transform(
                image=next_trf_permuted,
                trf=curr,
                method=interpolation_mode,
                isdisp=True
            )

            # Permute back: (N, *spatial) -> (*spatial, N)
            warped = warped.permute(*range(1, ndim + 1), 0)
            curr = curr + warped

        # Case 2: Affine on left, dense warp on right
        elif not curr_is_affine:
            curr = affine_to_disp(
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

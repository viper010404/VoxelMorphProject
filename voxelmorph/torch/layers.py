__all__ = [
    "SpatialTransformer",
    "VecInt",
    "ResizeTransform",
]

from typing import Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as nnf

import neurite as ne


class SpatialTransformer(nn.Module):
    """
    N-D Spatial transformation according to a deformation field.

    Uses a deformation field to transform the moving image.

    References
    ----------
    If you find this helpful, please cite the following paper:

    @ARTICLE{8633930,
    author={Balakrishnan, Guha and Zhao, Amy and Sabuncu, Mert R. and Guttag, John and Dalca,
    Adrian V.},
    journal={IEEE Transactions on Medical Imaging},
    title={VoxelMorph: A Learning Framework for Deformable Medical Image Registration},
    year={2019},
    volume={38},
    number={8},
    pages={1788-1800},
    keywords={Strain;Training;Biomedical imaging;Image segmentation;Optimization;Image registration;
    Three-dimensional displays;Registration;machine learning;convolutional neural networks},
    doi={10.1109/TMI.2019.2897538}}
    """

    def __init__(
        self,
        size: Tuple[int],
        interpolation_mode: str = "bilinear",
        align_corners: bool = False,
        device: Union[str, torch.device] = "cpu",
    ):
        """
        Initialize `SpatialTransformer`.

        Parameters
        ----------
        size : tuple[int]
            Expected size of `moving_image` (input image to be warped) for the forward pass.
        interpolation_mode : str
            Algorithm used for interpolating the warped image. Default is  'bilinear'. Options are:
            'bilinear' | 'nearest' | 'bicubic'.
        align_corners : bool
            Map the corner points of the moving image to the corner points of the warped image.
        device : str
            Device to construct and hold the identity grid.
        """
        super().__init__()

        self.size = size
        self.device = device
        self.interpolation_mode = interpolation_mode
        self.align_corners = align_corners

        # Make identity grid (the grid to later warp with deformation field) and register as a
        # buffer (without saving to `state_dict`: persistent=False)
        self.register_buffer(
            name='identity_grid',
            tensor=ne.utils.make_grid(size=size, device=device),
            persistent=False  # Don't save to this module's state dict!
        )

    def forward(
        self,
        moving_image: torch.Tensor,
        deformation_field: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass of `SpatialTransformer`

        Parameters
        ----------
        moving_image : torch.Tensor
            Tensor to be spatially transformed by `deformation_field`
        deformation_field : torch.Tensor
            Field causing the spatial transformation of `moving_image`.

        Returns
        -------
        torch.Tensor
            Warped `moving_image` according to the `deformation_field`.
        """

        # Validate the dimensions of the input
        if moving_image.dim() < 4 or deformation_field.dim() != moving_image.dim():
            raise ValueError(
                f"Expected `moving_image` to have at least 4 dimensions and for `flow field` to "
                f"match `moving_image` dimensions, got moving_image.dim()={moving_image.dim()}, "
                f"deformation_field.dim()={deformation_field.dim()}"
            )

        # Wow, this is legacy! Neither Adrian nor I know why the dims need to be permuted...
        # Well, at least that's what he said in his code
        deformation_field = deformation_field.moveaxis(1, -1).contiguous()

        # Warp the identity grid with the deformation field
        warped_grid = self.identity_grid + deformation_field

        # Normalize the axes so the range does not exceed the interval [-1, 1]
        warped_grid = self._normalize_warped_grid(warped_grid)

        # Sample grid
        warped_image = nnf.grid_sample(
            input=moving_image,
            grid=warped_grid,
            mode=self.interpolation_mode,
            align_corners=self.align_corners,
            padding_mode="border"
        )

        return warped_image

    def _normalize_warped_grid(
        self,
        warped_grid: torch.Tensor
    ) -> torch.Tensor:
        """
        Normalize a warped grid to make PyTorch `grid_sample()` happy!

        PyTorch's `grid_sample()` requires coordinates in the range [-1, 1].
        This function scales and shifts the warped grid accordingly.

        Parameters
        ----------
        warped_grid : torch.Tensor
            The resultant of the identity grid and the deformation field.

        Returns
        -------
        torch.Tensor
            The warped grid rescaled to the range [-1, 1] for each spatial axis
        """

        for i, dim in enumerate(self.size):

            # Rescale each dimension individually
            warped_grid[..., i] = 2 * (warped_grid[..., i] / (dim - 1) - 0.5)

        return warped_grid


class VecInt(nn.Module):
    """
    Integrates a vector field via scaling and squaring.
    """

    def __init__(self, inshape, nsteps):
        super().__init__()

        assert nsteps >= 0, 'nsteps should be >= 0, found: %d' % nsteps
        self.nsteps = nsteps
        self.scale = 1.0 / (2 ** self.nsteps)
        self.transformer = SpatialTransformer(inshape)

    def forward(self, vec):
        vec = vec * self.scale
        for _ in range(self.nsteps):
            vec = vec + self.transformer(vec, vec)
        return vec


class ResizeTransform(nn.Module):
    """
    Resize a transform, which involves resizing the vector field *and* rescaling it.
    """

    def __init__(self, vel_resize, ndims):
        super().__init__()
        self.factor = 1.0 / vel_resize
        self.mode = 'linear'
        if ndims == 2:
            self.mode = 'bi' + self.mode
        elif ndims == 3:
            self.mode = 'tri' + self.mode

    def forward(self, x):
        if self.factor < 1:
            # resize first to save memory
            x = nnf.interpolate(x, align_corners=True, scale_factor=self.factor, mode=self.mode)
            x = self.factor * x

        elif self.factor > 1:
            # multiply first to save memory
            x = self.factor * x
            x = nnf.interpolate(x, align_corners=True, scale_factor=self.factor, mode=self.mode)

        # don't do anything if resize is 1
        return x

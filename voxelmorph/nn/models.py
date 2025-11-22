"""
Core VoxelMorph models for unsupervised and supervised learning.
"""

# Core library imports
from typing import List, Literal, Sequence, Union, Callable, Tuple

# Third-party imports
import torch
import torch.nn as nn
import neurite as ne

# Local imports
import voxelmorph as vxm

__all__ = [
    "VxmPairwise",
]


class VxmPairwise(nn.Module):
    """
    A network architecture built on `BasicUNet` to perform nD image registration using a flow
    field.

    Parameters
    ----------
    ndim : int
        Number of spatial dimensions (e.g., 2 for 2D, 3 for 3D).
    source_channels : int
        Number of channels in the source image.
    target_channels : int
        Number of channels in the target image.
    spatial_shape : tuple[int]
        The expected shape of the `moving_tensor` input to the forward method of this class.
        without batch or channel dimensions. Used to initialize the `VecInt` integrator.
    out_channels : int
        Number of output channels in the displacement field.
    *args : list
        Additional positional arguments for the `BasicUNet` constructor.
    nb_features : List[int], optional
        List of integers specifying the number of features in each
        level of the UNet architecture. Default is `[16, 16, 16, 16, 16]`.
    normalizations : Union[List[str], str], optional
        Normalization layers for the UNet. Can be a list of normalization
        types or a single normalization type. Default is `None`.
    activations : Union[List[str], str], optional
        Activation functions for the UNet layers. Can be a list of
        activation functions or a single function. Default is `nn.ReLU`.
    order : str, optional
        The order of operations in each UNet block. Default is `'ca'`.
    final_activation : Union[str, nn.Module, None], optional
        The activation applied to the final output of the network. Default is `None`.
    flow_initializer : float, optional
        Standard deviation for initializing the flow layer weights with a
        normal distribution (mean=0). Default is `1e-5`.
    integration_steps : int, optional
        Number of steps to take in integrating the flow field. Default is 5.
    **kwargs : dict
        Additional keyword arguments passed to the `BasicUNet` constructor.

    Attributes
    ----------
    flow_layer : nn.Module
        A custom convolutional block used to generate the flow field
        from the combined source and target features.

    Methods
    -------
    forward(source, target, return_warped_source, return_warped_target, return_field_type)
        Combines source and target images, processes them through the UNet and the flow layer,
        and returns the velocity or displacement field. Optionally returns warped source and/or
        target images.
    """

    def __init__(
        self,
        ndim: int,
        source_channels: int,
        target_channels: int,
        spatial_shape: Tuple[int, ...],
        nb_features: Sequence[int] = (16, 16, 16, 16, 16),
        normalizations: Union[List[Union[Callable, str]], Callable, str, None] = None,
        activations: Union[List[Union[Callable, str]], Callable, str, None] = nn.ReLU,
        order: str = 'ca',
        final_activation: Union[str, nn.Module, None] = None,
        flow_initializer: float = 1e-5,
        integration_steps: int = 5,
        resize_integrated_fields: bool = False,
        device: str = "cpu",
    ):

        """
        Initialize the `VxmPairwise`.

        Parameters
        ----------
        ndim : int
            Dimensionality of the input (1, 2, or 3).
        source_channels : int
            Number of channels in the `source_tensor` input to the forward method of this class.
        target_channels : int
            Number of channels in the `target_tensor` input to the forward method of this class.
        spatial_shape : tuple[int]
            The expected shape of the `moving_tensor` input to the forward method of this class.
            without batch or channel dimensions. Used to initialize the `VecInt` integrator.
        nb_features : List[int]
            Number of features at each level of the unet. Must be a list of
            positive integers.
        normalizations : Union[List[str], str, None], optional
            Normalization layers to use in each block. Can be a string or a list
            of strings specifying normalizations for each layer, or `None` for no norm.
        activations : Union[List[str], str, Callable], optional
            Activation functions to use in each block. Can be a callable,
            a string, or a list of strings/callables.
        order : str, optional
            The order of operations in each convolutional block. Default is 'cna'
            (normalization -> convolution -> activation). Each character in the string represents
            one of the following:
            - `'c'`: Convolution
            - `'n'`: Normalization
            - `'a'`: Activation
        integration_steps : int, optional
            Number of scaling and squaring steps for integrating the flow field.
            Default is 5.
        device : str, optional
            Device identifier (e.g., 'cpu' or 'cuda') to place/run the model on.
        """

        # Initialize the Module
        super().__init__()

        # Set constant attrs
        self.integration_steps = integration_steps
        self.resize_integrated_fields = resize_integrated_fields
        self.device = device
        self.spatial_shape = spatial_shape
        self.out_channels = ndim

        # Set derived attrs
        self._init_flow_layer(ndim, self.out_channels, flow_initializer)
        self.model = ne.nn.models.BasicUNet(
            ndim=ndim, in_channels=(source_channels + target_channels),
            out_channels=self.out_channels,
            nb_features=nb_features,
            normalizations=normalizations, activations=activations, order=order,
            final_activation=final_activation
        )

        # Initialize the velocity field integrator with spatial shape
        if self.integration_steps > 0:
            self.velocity_field_integrator = vxm.nn.modules.IntegrateVelocityField(
                shape=self.spatial_shape, steps=self.integration_steps, device=self.device
            )

        # Initialize the spatial transformer with spatial shape
        self.spatial_transformer = vxm.nn.modules.SpatialTransformer(
            size=self.spatial_shape, device=self.device
        )

    def forward(
        self,
        source: torch.Tensor,
        target: torch.Tensor,
        return_warped_source: bool = False,
        return_warped_target: bool = False,
        return_field_type: Literal['displacement', 'velocity', 'svf'] = 'displacement',
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
        """
        Forward pass of `VxmPairwise`.

        This forward pass concatenates the `source` and `target` images, processes them with a
        `BasicUNet` backbone, and uses a flow layer to predict a velocity field (source -> target).

        By default, this method returns only the predicted displacement field. You can optionally
        request warped versions of the source and/or target images using the return flags.
        The displacement field is obtained by integrating the velocity field when
        `integration_steps > 0`; otherwise, the velocity field is used directly as the
        displacement for warping.

        Parameters
        ----------
        source : torch.Tensor
            Source image tensor with shape (B, C_source, *spatial_dims).
        target : torch.Tensor
            Target image tensor with shape (B, C_target, *spatial_dims).
            Must have the same spatial dimensions as `source`.
        return_warped_source : bool, optional
            If `True`, include the warped source image in the return tuple. Default is `False`.
        return_warped_target : bool, optional
            If `True`, include the warped target image in the return tuple. Default is `False`.
            Requires `integration_steps > 0` to ensure proper inverse transformation via
            stationary velocity field integration.
        return_field_type : str, optional
            Type of field to return. Options are:
            - 'displacement': Return the integrated displacement field.
            - 'velocity' or 'svf': Return the velocity (stationary velocity field).
            Default is 'displacement'.

        Returns
        -------
        Union[torch.Tensor, Tuple[torch.Tensor, ...]]
            Return values depend on the flags that are set:
            - No flags (default): field (velocity or displacement based on return_field_type)
            - `return_warped_source=True` only: (field, warped_source)
            - `return_warped_target=True` only: (field, warped_target)
            - Both flags: (field, warped_source, warped_target)

            Where:
            - field shape (B, ndim, *spatial_dims) - velocity or displacement based on
              return_field_type
            - warped_source shape (B, C_source, *spatial_dims)
            - warped_target shape (B, C_target, *spatial_dims)

        Raises
        ------
        ValueError
            If `return_warped_target=True` but `integration_steps=0`. Returning the warped
            target requires diffeomorphic registration to compute a proper inverse transformation.
        ValueError
            If `return_field_type` is not one of {'velocity', 'svf', 'displacement'}.
        """
        valid_field_types = {'velocity', 'svf', 'displacement'}
        if return_field_type not in valid_field_types:
            raise ValueError(
                f"return_field_type must be one of {valid_field_types}, got '{return_field_type}'"
            )

        if self.integration_steps == 0:
            if return_warped_target:
                raise ValueError("Cannot return warped target image when integration_steps=0.")

        # Pass combined features through the model's backbone & flow layer
        combined_features = torch.cat([source, target], dim=1)
        combined_features = self.model(combined_features)
        velocity = self.flow_layer(combined_features)   # Positive velocity: (source -> target)

        if self.integration_steps > 0:
            self.velocity = velocity

        # Early return if no warped images requested and returning velocity
        if not return_warped_source and not return_warped_target:
            if return_field_type in {'velocity', 'svf'}:
                return velocity

        pos_displacement = velocity
        neg_displacement = None

        if return_warped_source or return_field_type == 'displacement':
            if self.integration_steps > 0:
                # Only need positive displacement
                pos_displacement = self.velocity_field_integrator(velocity)

        if self.integration_steps > 0:

            if return_warped_target:
                # Only need negative displacement
                neg_displacement = self.velocity_field_integrator(-velocity)

        if return_field_type == 'displacement':
            return_field = pos_displacement
        else:
            return_field = velocity

        # Build return tuple starting with the requested field type
        outputs = [return_field]

        if return_warped_source:
            warped_source = self.spatial_transformer(source, pos_displacement)
            outputs.append(warped_source)

        if return_warped_target:
            warped_target = self.spatial_transformer(target, neg_displacement)
            outputs.append(warped_target)

        return tuple(outputs) if len(outputs) > 1 else outputs[0]

    def _init_flow_layer(
        self,
        ndim: int,
        features: int,
        flow_initializer: float = 1e-5
    ):
        """
        Initialize the flow layer with custom weight initialization.

        This layer is a convolutional block that produces a displacement (flow)
        field. The weights of its initial convolution are initialized from a
        normal distribution with mean=0 and std=flow_initializer, and biases
        are set to zero.

        Parameters
        ----------
        ndim : int
            **Spatial** dimensionality of the input (1, 2, or 3).
        features : int
            Number of input and output features for the flow layer.
        flow_initializer : float, optional
            Standard deviation for initializing the flow layer weights with a
            normal distribution (mean=0). Default is `1e-5`.
        """

        # Initialize the conv ("flow") layer with congruent in and out features
        flow_layer = ne.nn.modules.ConvBlock(ndim, features, features).to(self.device)

        # Apply custom initialization using PyTorch's native init
        if flow_initializer is not None:
            # Initialize weights from Normal(mean=0, std=flow_initializer)
            with torch.no_grad():
                torch.nn.init.normal_(flow_layer.conv0.weight, mean=0.0, std=flow_initializer)
                # Set the bias term(s) to zero for the first (and only) conv
                if flow_layer.conv0.bias is not None:
                    flow_layer.conv0.bias.zero_()
        # Register the flow layer as a submodule
        self.add_module("flow_layer", flow_layer)

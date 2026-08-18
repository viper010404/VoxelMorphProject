"""
Registration networks: the baseline and the two architectural extensions.

All three expose the same `forward(source, target) -> dict` interface. Branch-specific outputs
(the learned weight map, the velocity field) appear as extra keys rather than as a different
return signature, so training, evaluation and metrics are written once and every variant flows
through the identical path. That is what makes the comparison against the baseline meaningful:
nothing downstream of the model knows which branch it is scoring.

Returned keys
-------------
displacement : (B, ndim, *spatial)
    Always present. The field used for warping and for every deformation metric.
warped_source : (B, C, *spatial)
    Always present.
velocity : (B, ndim, *spatial)
    Present when `integration_steps > 0`; the field that is integrated to a diffeomorphism.
lambda_map : (B, 1, *spatial)
    Present for the lambda-field variant. Normalised to mean 1 (see `VxmLambdaField`).
"""

from typing import Dict, List, Sequence, Tuple

import torch
import torch.nn as nn
import neurite as ne

import voxelmorph as vxm

from project.configs import ExperimentConfig


def _init_flow_layer(ndim: int, flow_initializer: float = 1e-5) -> nn.Module:
    """
    Build the convolution that emits the flow field, initialised to near-zero.

    A near-zero initialisation makes the network start from (approximately) the identity
    transform, which matters because a large random initial deformation scrambles the moving
    image and gives the similarity term no usable gradient. This mirrors
    `voxelmorph.nn.models.VxmPairwise._init_flow_layer`.

    Parameters
    ----------
    ndim : int
        Spatial dimensionality; also the number of flow channels.
    flow_initializer : float, optional
        Standard deviation of the weight initialisation.

    Returns
    -------
    nn.Module
        A `neurite` convolution block producing `ndim` output channels.
    """
    flow_layer = ne.nn.modules.ConvBlock(ndim, ndim, ndim)
    with torch.no_grad():
        nn.init.normal_(flow_layer.conv0.weight, mean=0.0, std=flow_initializer)
        if flow_layer.conv0.bias is not None:
            flow_layer.conv0.bias.zero_()
    return flow_layer


class VxmBaseline(nn.Module):
    """
    The unmodified VoxelMorph model, wrapped to the shared dict interface.

    This delegates to the upstream `voxelmorph.nn.models.VxmPairwise` rather than reimplementing
    it, so the comparison anchor is genuinely the library's model and not our paraphrase of it.

    Parameters
    ----------
    ndim : int
        Spatial dimensionality.
    nb_features : sequence of int
        UNet features per level.
    integration_steps : int
        Scaling-and-squaring steps; 0 for a plain displacement field.
    """

    def __init__(
        self,
        ndim: int,
        nb_features: Sequence[int] = (16, 32, 32, 32, 32),
        integration_steps: int = 0,
    ) -> None:
        super().__init__()
        self.ndim = ndim
        self.integration_steps = integration_steps
        self.net = vxm.nn.models.VxmPairwise(
            ndim=ndim,
            source_channels=1,
            target_channels=1,
            nb_features=nb_features,
            integration_steps=integration_steps,
        )

    def forward(self, source: torch.Tensor, target: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Register `source` to `target`; see module docstring for returned keys."""
        displacement, warped_source = self.net(
            source, target, return_warped_source=True, return_field_type='displacement'
        )
        outputs = {'displacement': displacement, 'warped_source': warped_source}
        if self.integration_steps > 0:
            outputs['velocity'] = self.net.velocity
        return outputs


class VxmLambdaField(nn.Module):
    """
    VoxelMorph with a spatially varying regularisation weight.

    The paper applies one global lambda to the smoothness term everywhere, but anatomy is not
    uniform: ventricles genuinely differ a great deal between subjects while the brain stem
    barely moves, so any single value is a compromise. Here the network emits an extra channel
    that becomes a per-voxel weight on the smoothness penalty.

    The weight map is **bounded and then normalised to mean 1 per sample**. Both parts are
    necessary, and the reason is worth stating because the obvious design fails:

    Normalising to unit mean alone (`w = softplus(raw) / mean(softplus(raw))`) fixes the average
    regularisation but *not* the penalty itself. The smoothness term is a weighted sum, and to
    minimise `sum(w_i * g_i)` subject to `mean(w) = 1` the optimiser drives `w -> 0` exactly
    where the gradient `g_i` is largest and piles the weight where it is small. Measured on a
    real run this produced a weight map spanning 1e-5 to 6.4 -- a ratio of 6.4e5 -- with the mean
    still exactly 1, and folding on 5.0% of voxels against the baseline's 0.56%. The constraint
    was satisfied and the regulariser was deleted anyway.

    Bounding the weight to `weight_range` before normalisation closes that loophole: the field
    can still redistribute smoothing between regions, but it cannot remove it anywhere. With the
    default range the strongest possible redistribution is a factor of four between two
    locations, which is ample to express "ventricles deform, brain stem does not" while leaving
    the total budget comparable to the baseline's.

    The model therefore cannot escape smoothing, only *reallocate* it -- which is precisely the
    hypothesis under test: at a comparable budget, does a learned allocation beat a uniform one?

    Only one thing differs from the baseline: the UNet emits `ndim + 1` channels instead of
    `ndim`. Capacity is otherwise identical, so a measured difference is attributable to the
    weighting rather than to a bigger model.

    Parameters
    ----------
    ndim : int
        Spatial dimensionality.
    nb_features : sequence of int
        UNet features per level.
    integration_steps : int
        Scaling-and-squaring steps; 0 for a plain displacement field.
    weight_range : tuple of float
        Lower and upper bound on the weight before mean-normalisation. The ratio of the two
        bounds caps how strongly the field may redistribute regularisation.
    """

    def __init__(
        self,
        ndim: int,
        nb_features: Sequence[int] = (16, 32, 32, 32, 32),
        integration_steps: int = 0,
        weight_range: Tuple[float, float] = (0.5, 2.0),
    ) -> None:
        super().__init__()
        self.ndim = ndim
        self.integration_steps = integration_steps
        self.weight_range = weight_range

        self.unet = ne.nn.models.BasicUNet(
            ndim=ndim,
            in_channels=2,
            out_channels=ndim + 1,
            nb_features=nb_features,
        )
        self.flow_layer = _init_flow_layer(ndim)
        self.spatial_transformer = vxm.nn.modules.SpatialTransformer()
        if integration_steps > 0:
            self.integrator = vxm.nn.modules.IntegrateVelocityField(steps=integration_steps)

    def forward(self, source: torch.Tensor, target: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Register `source` to `target`; see module docstring for returned keys."""
        features = self.unet(torch.cat([source, target], dim=1))

        velocity = self.flow_layer(features[:, :self.ndim])
        lambda_map = self._normalise_weights(
            features[:, self.ndim:self.ndim + 1], self.weight_range
        )

        if self.integration_steps > 0:
            displacement = self.integrator(velocity)
        else:
            displacement = velocity

        outputs = {
            'displacement': displacement,
            'warped_source': self.spatial_transformer(source, displacement),
            'lambda_map': lambda_map,
        }
        if self.integration_steps > 0:
            outputs['velocity'] = velocity
        return outputs

    @staticmethod
    def _normalise_weights(
        raw: torch.Tensor,
        weight_range: Tuple[float, float] = (0.5, 2.0),
    ) -> torch.Tensor:
        """
        Map raw network output to a bounded weight field with mean exactly 1.

        A sigmoid maps the unconstrained output into `weight_range`, which guarantees the weight
        is bounded away from zero everywhere; dividing by the per-sample mean then fixes the
        average at 1. Order matters: bounding first and normalising second keeps both properties,
        whereas normalising a softplus output gives unit mean but permits weights of ~1e-5, which
        is enough for the optimiser to evade the regulariser entirely.

        Sigmoid is used rather than softplus precisely because it saturates at both ends, so no
        epsilon guard is needed and the field cannot run away in either direction.

        Parameters
        ----------
        raw : torch.Tensor
            Unconstrained network output of shape (B, 1, *spatial).
        weight_range : tuple of float, optional
            `(low, high)` bounds applied before normalisation.

        Returns
        -------
        torch.Tensor
            Weight field of the same shape, bounded and strictly positive, with per-sample
            mean 1.
        """
        low, high = weight_range
        bounded = low + (high - low) * torch.sigmoid(raw)
        spatial_dims = tuple(range(1, bounded.dim()))
        return bounded / bounded.mean(dim=spatial_dims, keepdim=True)


class VxmCrossAttention(nn.Module):
    """
    VoxelMorph with a two-stream encoder and cross-attention at the bottleneck.

    The baseline concatenates the moving and fixed images into a two-channel input at layer 0
    and leaves the UNet to discover correspondence implicitly. The paper notes (§IV-A) that the
    bottleneck's receptive field must be at least as large as the maximum expected displacement,
    which is exactly the constraint that explicit global matching removes -- and the optical-flow
    literature the paper builds on moved from early fusion to explicit cost volumes long ago.

    Here each image is encoded separately with **shared weights**, and the two bottleneck feature
    maps attend to each other before decoding. Attention cost is negligible at this depth: the
    bottleneck is 5x6 = 30 tokens in 2D and 5x6x7 = 210 tokens at full 3D resolution.

    The `neurite` UNet's submodules are driven directly rather than calling its `forward`, because
    `BasicUNet.forward` discards the skip connections and returns only the final tensor. The
    submodules used (`downsampling_conv_blocks`, `lowest_resolution_conv_block`,
    `upsampling_conv_blocks`, `out_layer`) are public attributes, so no fork of `neurite` is
    needed.

    Skip connections come from the **source** stream, since the decoder's job is to produce a
    field defined on the source grid.

    Parameters
    ----------
    ndim : int
        Spatial dimensionality.
    nb_features : sequence of int
        UNet features per level.
    integration_steps : int
        Scaling-and-squaring steps; 0 for a plain displacement field.
    attn_heads : int
        Number of attention heads.
    """

    def __init__(
        self,
        ndim: int,
        nb_features: Sequence[int] = (16, 32, 32, 32, 32),
        integration_steps: int = 0,
        attn_heads: int = 4,
    ) -> None:
        super().__init__()
        self.ndim = ndim
        self.integration_steps = integration_steps

        # in_channels=1: each stream encodes a single image.
        self.unet = ne.nn.models.BasicUNet(
            ndim=ndim,
            in_channels=1,
            out_channels=ndim,
            nb_features=nb_features,
        )

        bottleneck_channels = self.unet.lowest_resolution_conv_block.out_channels
        self.attention = nn.MultiheadAttention(
            embed_dim=bottleneck_channels,
            num_heads=attn_heads,
            batch_first=True,
        )
        self.attention_norm = nn.LayerNorm(bottleneck_channels)

        self.flow_layer = _init_flow_layer(ndim)
        self.spatial_transformer = vxm.nn.modules.SpatialTransformer()
        if integration_steps > 0:
            self.integrator = vxm.nn.modules.IntegrateVelocityField(steps=integration_steps)

    def _encode(self, image: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Run one image through the encoder, returning bottleneck features and skips.

        Parameters
        ----------
        image : torch.Tensor
            Single-channel image of shape (B, 1, *spatial).

        Returns
        -------
        tuple
            `(bottleneck, skips)` where bottleneck has shape (B, C, *reduced_spatial).
        """
        features = image
        skips: List[torch.Tensor] = []
        for block in self.unet.downsampling_conv_blocks:
            features, skip = block(features)
            skips.append(skip)
        return self.unet.lowest_resolution_conv_block(features), skips

    def _cross_attend(self, source_feat: torch.Tensor, target_feat: torch.Tensor) -> torch.Tensor:
        """
        Let source bottleneck features attend to target bottleneck features.

        Parameters
        ----------
        source_feat, target_feat : torch.Tensor
            Bottleneck features of shape (B, C, *reduced_spatial).

        Returns
        -------
        torch.Tensor
            Source features updated with matched target context, same shape as `source_feat`.
        """
        batch, channels = source_feat.shape[:2]
        spatial = source_feat.shape[2:]

        # (B, C, *spatial) -> (B, N, C) token sequences.
        query = source_feat.flatten(2).transpose(1, 2)
        key = target_feat.flatten(2).transpose(1, 2)

        attended, _ = self.attention(query, key, key, need_weights=False)
        # Residual keeps the source representation intact if attention is unhelpful, so the
        # model can fall back towards the baseline behaviour rather than being forced to use it.
        fused = self.attention_norm(query + attended)

        return fused.transpose(1, 2).reshape(batch, channels, *spatial)

    def _decode(self, features: torch.Tensor, skips: List[torch.Tensor]) -> torch.Tensor:
        """
        Decode bottleneck features back to full resolution using the source skips.

        Parameters
        ----------
        features : torch.Tensor
            Bottleneck features of shape (B, C, *reduced_spatial).
        skips : list of torch.Tensor
            Encoder skip tensors in encoder order; consumed in reverse.

        Returns
        -------
        torch.Tensor
            Full-resolution features of shape (B, ndim, *spatial).
        """
        for block, skip in zip(self.unet.upsampling_conv_blocks, reversed(skips)):
            features = block(features, skip)
        return self.unet.out_layer(features)

    def forward(self, source: torch.Tensor, target: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Register `source` to `target`; see module docstring for returned keys."""
        source_feat, source_skips = self._encode(source)
        target_feat, _ = self._encode(target)

        fused = self._cross_attend(source_feat, target_feat)
        velocity = self.flow_layer(self._decode(fused, source_skips))

        if self.integration_steps > 0:
            displacement = self.integrator(velocity)
        else:
            displacement = velocity

        outputs = {
            'displacement': displacement,
            'warped_source': self.spatial_transformer(source, displacement),
        }
        if self.integration_steps > 0:
            outputs['velocity'] = velocity
        return outputs


def build_model(config: ExperimentConfig) -> nn.Module:
    """
    Instantiate the model described by a configuration.

    Parameters
    ----------
    config : ExperimentConfig
        Run configuration; `config.variant` selects the architecture.

    Returns
    -------
    nn.Module
        A model exposing the shared `forward(source, target) -> dict` interface.
    """
    common = {
        'ndim': config.ndim,
        'nb_features': config.nb_features,
        'integration_steps': config.integration_steps,
    }

    if config.variant == 'baseline':
        return VxmBaseline(**common)
    if config.variant == 'lambda_field':
        return VxmLambdaField(**common)
    if config.variant == 'cross_attn':
        return VxmCrossAttention(attn_heads=config.attn_heads, **common)

    raise ValueError(f"unknown variant '{config.variant}'")

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

import math
from typing import Dict, List, Optional, Sequence, Tuple

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
        mask_normalise: bool = False,
    ) -> None:
        super().__init__()
        self.ndim = ndim
        self.integration_steps = integration_steps
        self.weight_range = weight_range
        self.mask_normalise = mask_normalise

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
        mask = self._foreground_mask(source, target) if self.mask_normalise else None
        lambda_map = self._normalise_weights(
            features[:, self.ndim:self.ndim + 1], self.weight_range, mask
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
    def _foreground_mask(source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Brain mask taken from the images themselves, as the union of the two.

        neurite-OASIS is skull-stripped, so background is exactly zero and a positive-intensity
        test recovers the brain without needing the segmentation. Deriving the mask from the
        segmentation would leak label information into training and turn an unsupervised method
        into a semi-supervised one; deriving it from the input image does not.

        The union of source and target is used because a voxel that is brain in either image is
        somewhere the deformation has real work to do.

        Parameters
        ----------
        source, target : torch.Tensor
            Input images of shape (B, 1, *spatial).

        Returns
        -------
        torch.Tensor
            Boolean mask of shape (B, 1, *spatial).
        """
        return (source > 0) | (target > 0)

    @staticmethod
    def _normalise_weights(
        raw: torch.Tensor,
        weight_range: Tuple[float, float] = (0.5, 2.0),
        mask: Optional[torch.Tensor] = None,
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

        **Normalising over the whole image leaves a second way out**, which bounding does not
        close. Roughly 60% of an OASIS volume is background air, where the displacement field is
        unconstrained by the similarity term and smoothing it costs the optimiser nothing. The
        field can therefore satisfy `mean(w) = 1` by parking weight in the background and
        relaxing every brain voxel to the floor. Measured on a trained 2D run: weight 1.38
        outside the brain against 0.42 inside, a 3.3x split that nearly saturates the bound
        ratio, leaving the brain at an effective lambda of 0.42 * lambda. The variant was then
        not a redistribution of the baseline's budget but simply the baseline at a weaker
        lambda, which is exactly what the measurements showed.

        Passing `mask` normalises to mean 1 **within the mask**, pinning the brain's
        regularisation budget to the baseline's and leaving the field free only to redistribute
        smoothing between anatomical regions -- the hypothesis actually under test. Background
        weight is then unconstrained but harmless: raising it only adds penalty, so the
        optimiser drives it to the floor.

        Parameters
        ----------
        raw : torch.Tensor
            Unconstrained network output of shape (B, 1, *spatial).
        weight_range : tuple of float, optional
            `(low, high)` bounds applied before normalisation.
        mask : torch.Tensor or None, optional
            Region the mean is taken over. None normalises over the whole image.

        Returns
        -------
        torch.Tensor
            Weight field of the same shape, bounded and strictly positive, with per-sample
            mean 1 over the mask (or over the image when no mask is given).
        """
        low, high = weight_range
        bounded = low + (high - low) * torch.sigmoid(raw)
        spatial_dims = tuple(range(1, bounded.dim()))

        if mask is None:
            return bounded / bounded.mean(dim=spatial_dims, keepdim=True)

        weights = mask.to(bounded.dtype)
        # clamp guards a degenerate all-background sample; a real one always has brain voxels.
        count = weights.sum(dim=spatial_dims, keepdim=True).clamp(min=1.0)
        masked_mean = (bounded * weights).sum(dim=spatial_dims, keepdim=True) / count
        return bounded / masked_mean


class LocalCrossAttention(nn.Module):
    """
    Cross-attention restricted to a local neighbourhood, applied at an encoder skip level.

    Global attention at the bottleneck cannot work on this dataset, and the reason is geometric
    rather than architectural. The bottleneck is 32x downsampled, so one token spans 32 voxels,
    while the largest nonlinear displacement anywhere in neurite-OASIS is 14.2 voxels (median
    1.7) -- the data is affinely pre-aligned, so only the fine residual is left. Every match the
    network needs to express is therefore *smaller than a single token*, and no amount of heads
    or positional encoding recovers detail the representation never had.

    Moving attention to a resolution where the motion is visible runs into cost: full attention
    over level 2 (4x) would be 1.1e4 tokens in 2D and 1.1e5 in 3D, i.e. 1.2e10 pairs. Restricting
    each source token to a `radius`-token neighbourhood of the target makes it O(N * K) with
    K = (2*radius+1)**ndim, which is what makes fine-resolution attention tractable -- the same
    move Swin-based registration networks make, and the same structure as an optical-flow cost
    volume with a bounded search range.

    Choose `radius` so the window covers the expected displacement: at 8x downsampling a radius
    of 2 spans +-16 voxels, just above the 14.2-voxel maximum.

    The output projection is zero-initialised, so the module starts as an exact no-op and the
    surrounding network is unchanged at step 0.

    Parameters
    ----------
    channels : int
        Feature width at the level this operates on.
    ndim : int
        Spatial dimensionality.
    radius : int
        Neighbourhood radius in tokens; the window is `2 * radius + 1` per axis.
    heads : int
        Attention heads; `channels` must be divisible by this.
    """

    def __init__(self, channels: int, ndim: int, radius: int = 2, heads: int = 4) -> None:
        super().__init__()
        if channels % heads:
            raise ValueError(f'channels {channels} must be divisible by heads {heads}')
        self.ndim = ndim
        self.radius = radius
        self.heads = heads
        self.head_dim = channels // heads

        conv = nn.Conv2d if ndim == 2 else nn.Conv3d
        self.query = conv(channels, channels, kernel_size=1)
        self.key = conv(channels, channels, kernel_size=1)
        self.value = conv(channels, channels, kernel_size=1)
        self.project = conv(channels, channels, kernel_size=1)
        # Start as a no-op: the level behaves exactly as it did before attention was added.
        nn.init.zeros_(self.project.weight)
        nn.init.zeros_(self.project.bias)

    def _neighbourhoods(self, features: torch.Tensor) -> torch.Tensor:
        """
        Extract each position's local neighbourhood.

        Parameters
        ----------
        features : torch.Tensor
            Shape (B, C, *spatial).

        Returns
        -------
        torch.Tensor
            Shape (B, C, N, K) with N positions and K neighbours each.
        """
        width = 2 * self.radius + 1
        padded = torch.nn.functional.pad(features, (self.radius,) * (2 * self.ndim))
        for axis in range(self.ndim):
            padded = padded.unfold(2 + axis, width, 1)
        # (B, C, *spatial, *window) -> (B, C, N, K)
        batch, channels = padded.shape[:2]
        spatial = padded.shape[2:2 + self.ndim]
        positions = 1
        for size in spatial:
            positions *= size
        return padded.reshape(batch, channels, positions, width ** self.ndim)

    def forward(self, source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Refine `source` with locally matched `target` context.

        Parameters
        ----------
        source, target : torch.Tensor
            Features of shape (B, C, *spatial) from the two encoder streams.

        Returns
        -------
        torch.Tensor
            A residual update for `source`, same shape; zero at initialisation.
        """
        batch = source.shape[0]
        spatial = source.shape[2:]

        query = self.query(source).reshape(batch, self.heads, self.head_dim, -1)
        keys = self._neighbourhoods(self.key(target))
        values = self._neighbourhoods(self.value(target))
        neighbours = keys.shape[-1]

        keys = keys.reshape(batch, self.heads, self.head_dim, -1, neighbours)
        values = values.reshape(batch, self.heads, self.head_dim, -1, neighbours)

        # (B, heads, N, K): each source token scored against its own target neighbourhood only.
        scores = (query.unsqueeze(-1) * keys).sum(dim=2) / (self.head_dim ** 0.5)
        weights = torch.softmax(scores, dim=-1)

        attended = (weights.unsqueeze(2) * values).sum(dim=-1)
        attended = attended.reshape(batch, self.heads * self.head_dim, *spatial)
        return self.project(attended)


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

    **That last choice is the one that breaks it.** With source-only skips the target reaches the
    decoder at exactly one scale -- the bottleneck, 5x6x7 in 3D, where a single token spans 32
    voxels. Measured on this dataset, every displacement the baseline predicts inside the brain
    is smaller than one such token (median 1.7, max 14.2 voxels): neurite-OASIS is affinely
    pre-aligned, so only the fine nonlinear residual is left. The decoder is therefore asked to
    align two images while seeing only one of them at any resolution the motion actually lives
    at, and the trained model under-deforms in response (mean |displacement| 0.68 voxels against
    the baseline's 1.01).

    Setting `target_skips` fuses the target stream's pyramid into the decoder alongside the
    source's, restoring per-scale correspondence. Measured effect at matched training steps:
    +0.062 Dice in 2D and +0.088 in 3D, closing ~89% of the gap to the baseline. Adding
    positional encoding to the attention instead was worth nothing either way (2D: 0.6683
    without skips, 0.7379 with, against 0.7367 for skips alone) -- position cannot help while
    the only shared representation is 32x coarser than the motion.

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
    target_skips : bool
        Fuse the target stream's skip connections into the decoder. Off by default so the
        original formulation reproduces exactly.
    window_level : int
        Encoder skip level at which to apply local windowed cross-attention, or -1 to disable.
        Level 3 is 8x downsampled, where a radius of 2 spans +-16 voxels against a 14.2-voxel
        maximum displacement. See `LocalCrossAttention` for why the bottleneck cannot work.
    window_radius : int
        Neighbourhood radius in tokens for the windowed attention.
    use_attention : bool
        Whether to cross-attend at the bottleneck at all. Setting this False alongside
        `target_skips` is the control that asks whether the attention earns its place: if the
        two score the same, the gain came from giving the decoder both images, not from the
        attention. The attention module is not constructed when disabled, so the ablation is
        also strictly smaller.
    """

    def __init__(
        self,
        ndim: int,
        nb_features: Sequence[int] = (16, 32, 32, 32, 32),
        integration_steps: int = 0,
        attn_heads: int = 4,
        target_skips: bool = False,
        use_attention: bool = True,
        window_level: int = -1,
        window_radius: int = 2,
    ) -> None:
        super().__init__()
        self.ndim = ndim
        self.integration_steps = integration_steps
        self.target_skips = target_skips
        self.use_attention = use_attention
        self.window_level = window_level

        # in_channels=1: each stream encodes a single image.
        self.unet = ne.nn.models.BasicUNet(
            ndim=ndim,
            in_channels=1,
            out_channels=ndim,
            nb_features=nb_features,
        )

        if target_skips:
            self.skip_fusion = self._build_skip_fusion(ndim, self.unet.down_actual_channels)

        if window_level >= 0:
            self.local_attention = LocalCrossAttention(
                channels=self.unet.down_actual_channels[window_level],
                ndim=ndim,
                radius=window_radius,
                heads=attn_heads,
            )

        if use_attention:
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

    @staticmethod
    def _build_skip_fusion(ndim: int, channels: Sequence[int]) -> nn.ModuleList:
        """
        One 1x1 convolution per skip level, fusing concatenated source and target features.

        Each layer maps `2C -> C`, so the decoder's input widths are unchanged and the fix costs
        only 8,848 parameters in 3D (+2.6% against the baseline's 333,241) -- the matched-capacity
        comparison against the baseline survives, and the gain cannot be attributed to size.

        The initialisation matters as much as the layer: the source half is set to the identity
        and the target half to zero, so at step 0 the fused skip *is* the source skip and the
        model is numerically identical to the source-only formulation. Target information is
        therefore something the network switches on if it helps, rather than a distribution shift
        imposed on a decoder that has not yet learned to use it.

        Parameters
        ----------
        ndim : int
            Spatial dimensionality, selecting Conv2d or Conv3d.
        channels : sequence of int
            Encoder width at each skip level.

        Returns
        -------
        nn.ModuleList
            One fusion convolution per level, in encoder order.
        """
        conv = nn.Conv2d if ndim == 2 else nn.Conv3d
        fusion = nn.ModuleList([conv(2 * width, width, kernel_size=1) for width in channels])
        for layer, width in zip(fusion, channels):
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)
            with torch.no_grad():
                identity = torch.eye(width).view(width, width, *([1] * ndim))
                layer.weight[:, :width].copy_(identity)
        return fusion

    def _fuse_skips(
        self,
        source_skips: List[torch.Tensor],
        target_skips: List[torch.Tensor],
    ) -> List[torch.Tensor]:
        """
        Combine the two encoder pyramids level by level.

        Parameters
        ----------
        source_skips, target_skips : list of torch.Tensor
            Encoder features in encoder order, one per level.

        Returns
        -------
        list of torch.Tensor
            Fused skips with the same shapes as `source_skips`.
        """
        return [fusion(torch.cat([source, target], dim=1))
                for fusion, source, target in zip(self.skip_fusion, source_skips, target_skips)]

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
        target_feat, target_skips = self._encode(target)

        if self.window_level >= 0:
            level = self.window_level
            source_skips = list(source_skips)
            source_skips[level] = source_skips[level] + self.local_attention(
                source_skips[level], target_skips[level])

        skips = self._fuse_skips(source_skips, target_skips) if self.target_skips else source_skips

        fused = self._cross_attend(source_feat, target_feat) if self.use_attention \
            else source_feat
        velocity = self.flow_layer(self._decode(fused, skips))

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


class VxmPyramid(nn.Module):
    """
    VoxelMorph with a coarse-to-fine displacement pyramid and optional deep supervision.

    The baseline emits one displacement field from a single output layer at full resolution.
    Every scale of motion has to come out of that one head, and nothing obliges the coarse half
    of the decoder to contribute: measured on a trained model, deleting the bottleneck entirely
    costs only 0.007 Dice, so those layers had drifted into being nearly decorative.

    This variant makes each decoder level accountable for the motion at its own scale, taking
    the two ideas behind pyramid registration networks and applying them to *this* UNet rather
    than replacing it with a stack of separate networks:

    * **Flow head per level.** Each decoder level emits its own displacement field, so the
      hierarchy exists in the *output* and not only in the features.
    * **Progressive warping.** Before a level's decoder block runs, its skip features are warped
      by the field accumulated so far, so each level only has to explain the residual left by
      the ones above it and sees features already brought into rough alignment. This establishes
      correspondence by *moving* features rather than by matching them, which sidesteps the token
      -resolution problem that made attention useless here.

    Fields compose additively in the warped frame -- `field = upsample(field) + residual` after
    warping the skip. That is the standard residual-flow formulation and is exact only to first
    order, but the residuals here are small (median motion is 1.7 voxels) and it avoids the
    upstream `compose` batch-detection bug entirely.

    Upsampling a displacement field must also rescale it: a field is measured in voxels, so
    doubling the grid doubles every displacement. Forgetting that silently halves the coarse
    contribution at every level.

    With `integration_steps > 0` the *composed* field is treated as a stationary velocity field
    and integrated once at the end, rather than integrating at every level.

    Parameters
    ----------
    ndim : int
        Spatial dimensionality.
    nb_features : sequence of int
        UNet features per level.
    integration_steps : int
        Scaling-and-squaring steps applied to the final composed field; 0 for a displacement.
    progressive : bool
        Warp each level's skip features by the field so far. False keeps the per-level flow
        heads but predicts each level independently, which isolates how much of any gain comes
        from the warping rather than from deep supervision alone.
    """

    def __init__(
        self,
        ndim: int,
        nb_features: Sequence[int] = (16, 32, 32, 32, 32),
        integration_steps: int = 0,
        progressive: bool = True,
    ) -> None:
        super().__init__()
        self.ndim = ndim
        self.integration_steps = integration_steps
        self.progressive = progressive

        self.unet = ne.nn.models.BasicUNet(
            ndim=ndim,
            in_channels=2,
            out_channels=ndim,
            nb_features=nb_features,
        )

        conv = nn.Conv2d if ndim == 2 else nn.Conv3d
        self.flow_heads = nn.ModuleList([
            conv(width, ndim, kernel_size=3, padding=1)
            for width in self.unet.up_actual_channels
        ])
        for head in self.flow_heads:
            # Same near-zero start as the baseline's flow layer: the model begins at the
            # identity, so the similarity term has a usable gradient from step one.
            nn.init.normal_(head.weight, mean=0.0, std=1e-5)
            nn.init.zeros_(head.bias)

        self.spatial_transformer = vxm.nn.modules.SpatialTransformer()
        if integration_steps > 0:
            self.integrator = vxm.nn.modules.IntegrateVelocityField(steps=integration_steps)

    def _upsample_field(self, field: torch.Tensor, shape: Sequence[int]) -> torch.Tensor:
        """
        Resample a displacement field to a finer grid, rescaling its magnitudes.

        Parameters
        ----------
        field : torch.Tensor
            Displacement field of shape (B, ndim, *spatial), in voxels.
        shape : sequence of int
            Target spatial shape.

        Returns
        -------
        torch.Tensor
            Field on the new grid, with each component scaled by that axis' size ratio.
        """
        mode = 'bilinear' if self.ndim == 2 else 'trilinear'
        resampled = torch.nn.functional.interpolate(
            field, size=tuple(shape), mode=mode, align_corners=True)
        for axis, (new, old) in enumerate(zip(shape, field.shape[2:])):
            resampled[:, axis] = resampled[:, axis] * (new / old)
        return resampled

    def forward(self, source: torch.Tensor, target: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Register `source` to `target`; returns the per-level fields under `pyramid`."""
        features = torch.cat([source, target], dim=1)

        skips: List[torch.Tensor] = []
        for block in self.unet.downsampling_conv_blocks:
            features, skip = block(features)
            skips.append(skip)
        features = self.unet.lowest_resolution_conv_block(features)

        field = None
        pyramid: List[torch.Tensor] = []

        for head, block, skip in zip(self.flow_heads, self.unet.upsampling_conv_blocks,
                                     reversed(skips)):
            if field is not None:
                field = self._upsample_field(field, skip.shape[2:])
                if self.progressive:
                    skip = self.spatial_transformer(skip, field)

            features = block(features, skip)
            residual = head(features)
            field = residual if field is None else field + residual
            pyramid.append(field)

        velocity = field
        displacement = self.integrator(velocity) if self.integration_steps > 0 else velocity

        outputs = {
            'displacement': displacement,
            'warped_source': self.spatial_transformer(source, displacement),
            'pyramid': pyramid,
        }
        if self.integration_steps > 0:
            outputs['velocity'] = velocity
        return outputs


def _resample_field(field: torch.Tensor, shape: Sequence[int], ndim: int) -> torch.Tensor:
    """
    Resample a displacement field to another grid, rescaling its magnitudes.

    A displacement is measured in voxels, so moving a field to a grid twice as fine doubles every
    displacement. Omitting the rescale is silent -- shapes stay valid and training proceeds -- and
    simply discards part of the coarse stage's contribution.

    Parameters
    ----------
    field : torch.Tensor
        Displacement field of shape (B, ndim, *spatial), in voxels.
    shape : sequence of int
        Target spatial shape.
    ndim : int
        Spatial dimensionality.

    Returns
    -------
    torch.Tensor
        Field on the new grid, each component scaled by that axis' size ratio.
    """
    mode = 'bilinear' if ndim == 2 else 'trilinear'
    resampled = torch.nn.functional.interpolate(
        field, size=tuple(shape), mode=mode, align_corners=True)
    for axis, (new, old) in enumerate(zip(shape, field.shape[2:])):
        resampled[:, axis] = resampled[:, axis] * (new / old)
    return resampled


class VxmFatHead(nn.Module):
    """
    Capacity control for the pyramid: the same parameter budget, spent at one resolution.

    `VxmPyramid` with `progressive=False` beats the baseline by ~0.006 Dice while adding only
    ~2 300 parameters, and it does so across a whole misalignment ladder. Two explanations
    survive that observation and they are not the same claim:

    * the extra parameters and the extra gradient path to the output do the work, in which case
      the branch is a capacity trick and the multi-scale story is decoration; or
    * the *multi-resolution decomposition* does the work -- a residual predicted at 1/16
      resolution and upsampled is band-limited by construction, so the field becomes a sum of
      smooth components and large-scale motion becomes cheap to represent both for the network
      and under the smoothness penalty.

    This class isolates the second. It keeps the pyramid's UNet **exactly** -- same
    `BasicUNet`, same widths, same forward path -- and replaces the five per-level flow heads
    with a two-layer head at the finest resolution only, sized so the parameter counts match to
    within ~0.03%. Any remaining difference is therefore the multi-resolution structure and not
    the budget.

    Matching against `VxmBaseline` instead would not do: that delegates to upstream
    `VxmPairwise`, whose UNet differs from `BasicUNet` by a couple of hundred parameters on its
    own, which is a tenth of the effect under test.

    Parameters
    ----------
    ndim : int
        Spatial dimensionality.
    nb_features : sequence of int
        UNet features per level.
    integration_steps : int
        Scaling-and-squaring steps applied to the field; 0 for a plain displacement.
    hidden : int, optional
        Width of the extra full-resolution convolution. The default is chosen to match
        `VxmPyramid`'s head budget for the default widths; `matched_hidden` computes it for
        any configuration.
    """

    def __init__(
        self,
        ndim: int,
        nb_features: Sequence[int] = (16, 32, 32, 32, 32),
        integration_steps: int = 0,
        hidden: Optional[int] = None,
        kernel: int = 3,
        image_skip: bool = False,
    ) -> None:
        super().__init__()
        self.ndim = ndim
        self.integration_steps = integration_steps
        self.image_skip = image_skip

        self.unet = ne.nn.models.BasicUNet(
            ndim=ndim,
            in_channels=2,
            out_channels=ndim,
            nb_features=nb_features,
        )

        widths = list(self.unet.up_actual_channels)
        if hidden is None:
            hidden = self.matched_hidden(ndim, widths, kernel)

        conv = nn.Conv2d if ndim == 2 else nn.Conv3d
        # The decoder's own features, optionally plus the two raw images. The head currently
        # only ever sees features that have been through the whole UNet; giving it the intensity
        # values directly tests whether the last layer wants information the encoder discarded.
        in_channels = widths[-1] + (2 if image_skip else 0)
        self.hidden_conv = conv(in_channels, hidden, kernel_size=kernel, padding=kernel // 2)
        self.flow_head = conv(hidden, ndim, kernel_size=kernel, padding=kernel // 2)

        # Same near-zero start as every other variant: the model begins at the identity, so a
        # difference against the pyramid cannot come from a different starting point.
        nn.init.normal_(self.flow_head.weight, mean=0.0, std=1e-5)
        nn.init.zeros_(self.flow_head.bias)

        self.spatial_transformer = vxm.nn.modules.SpatialTransformer()
        if integration_steps > 0:
            self.integrator = vxm.nn.modules.IntegrateVelocityField(steps=integration_steps)

    @staticmethod
    def matched_hidden(ndim: int, widths: Sequence[int], kernel: int = 3) -> int:
        """
        Hidden width whose two-layer head costs what `VxmPyramid`'s five heads cost.

        A `k x k` convolution from `a` to `b` channels costs `a*b*k**ndim + b`. The pyramid
        spends one such head per decoder level; this spends `widths[-1] -> hidden -> ndim`. The
        result is rounded down, so the control is never given *more* parameters than the model
        it is controlling for.

        `kernel` matters because a 1x1 head is the control that separates *capacity* from
        *receptive field*: the 3x3 head does not only add parameters, it also lets the output
        layer see a wider neighbourhood. Matching on parameters at kernel 1 buys a much wider
        hidden layer for the same budget, which is the point -- if that does as well, the extra
        spatial context was never what mattered.

        Parameters
        ----------
        ndim : int
            Spatial dimensionality; sets the kernel volume.
        widths : sequence of int
            Decoder output widths, coarsest first.
        kernel : int, optional
            Kernel size of the head's convolutions.

        Returns
        -------
        int
            Hidden width, at least 1.
        """
        pyramid_cost = sum(width * ndim * 3 ** ndim + ndim for width in widths)
        volume = kernel ** ndim
        # cost(hidden) = widths[-1]*hidden*volume + hidden + hidden*ndim*volume + ndim
        per_unit = widths[-1] * volume + 1 + ndim * volume
        return max(1, int((pyramid_cost - ndim) // per_unit))

    def forward(self, source: torch.Tensor, target: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Register `source` to `target`; see module docstring for returned keys."""
        features = torch.cat([source, target], dim=1)

        skips: List[torch.Tensor] = []
        for block in self.unet.downsampling_conv_blocks:
            features, skip = block(features)
            skips.append(skip)
        features = self.unet.lowest_resolution_conv_block(features)

        for block, skip in zip(self.unet.upsampling_conv_blocks, reversed(skips)):
            features = block(features, skip)

        if self.image_skip:
            features = torch.cat([features, source, target], dim=1)

        velocity = self.flow_head(torch.nn.functional.leaky_relu(self.hidden_conv(features), 0.2))
        displacement = self.integrator(velocity) if self.integration_steps > 0 else velocity
        warped_source = self.spatial_transformer(source, displacement)

        outputs = {'displacement': displacement, 'warped_source': warped_source}
        if self.integration_steps > 0:
            outputs['velocity'] = velocity
        return outputs

class VxmMultiScaleFeatures(nn.Module):
    """
    Multi-scale *features* feeding one flow head, rather than multi-scale *fields* summed.

    `VxmPyramid` gives each decoder level its own flow head and adds the resulting fields. That
    turned out to be no better than spending the same parameters on a single fat head at full
    resolution, which raises an obvious question the pyramid does not answer: was the problem
    the multi-scale idea, or the fact that it was applied to the *output* rather than to what
    the output is computed from?

    This variant keeps the multi-scale hierarchy but moves the combination one step earlier.
    Every decoder level is projected to a few channels by a 1x1 convolution, upsampled to full
    resolution, and concatenated; a single head then predicts one field from that stack. So the
    head sees all scales at once and can mix them, instead of each scale independently
    committing to a displacement that is later summed.

    The 1x1 projections are what keep this affordable: concatenating the raw decoder features
    would put ~128 channels at full resolution, and the head over them would dwarf the model it
    is meant to be compared against.

    Unlike a field, a *feature* map does not need rescaling when upsampled -- it carries no
    units. That asymmetry is easy to get backwards, and getting it backwards here would be
    invisible rather than wrong-looking.

    Parameters
    ----------
    ndim : int
        Spatial dimensionality.
    nb_features : sequence of int
        UNet features per level.
    integration_steps : int
        Scaling-and-squaring steps applied to the field; 0 for a plain displacement.
    per_level : int, optional
        Channels each decoder level is projected to before concatenation.
    """

    def __init__(
        self,
        ndim: int,
        nb_features: Sequence[int] = (16, 32, 32, 32, 32),
        integration_steps: int = 0,
        per_level: int = 4,
    ) -> None:
        super().__init__()
        self.ndim = ndim
        self.integration_steps = integration_steps

        self.unet = ne.nn.models.BasicUNet(
            ndim=ndim,
            in_channels=2,
            out_channels=ndim,
            nb_features=nb_features,
        )

        conv = nn.Conv2d if ndim == 2 else nn.Conv3d
        widths = list(self.unet.up_actual_channels)
        self.projections = nn.ModuleList([
            conv(width, per_level, kernel_size=1) for width in widths
        ])
        self.flow_head = conv(per_level * len(widths), ndim, kernel_size=3, padding=1)
        nn.init.normal_(self.flow_head.weight, mean=0.0, std=1e-5)
        nn.init.zeros_(self.flow_head.bias)

        self.spatial_transformer = vxm.nn.modules.SpatialTransformer()
        if integration_steps > 0:
            self.integrator = vxm.nn.modules.IntegrateVelocityField(steps=integration_steps)

    def forward(self, source: torch.Tensor, target: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Register `source` to `target`; see module docstring for returned keys."""
        features = torch.cat([source, target], dim=1)

        skips: List[torch.Tensor] = []
        for block in self.unet.downsampling_conv_blocks:
            features, skip = block(features)
            skips.append(skip)
        features = self.unet.lowest_resolution_conv_block(features)

        levels: List[torch.Tensor] = []
        for block, skip in zip(self.unet.upsampling_conv_blocks, reversed(skips)):
            features = block(features, skip)
            levels.append(features)

        mode = 'bilinear' if self.ndim == 2 else 'trilinear'
        full = levels[-1].shape[2:]
        stacked = torch.cat([
            projection(level) if level.shape[2:] == full else
            torch.nn.functional.interpolate(
                projection(level), size=tuple(full), mode=mode, align_corners=True)
            for projection, level in zip(self.projections, levels)
        ], dim=1)

        velocity = self.flow_head(stacked)
        displacement = self.integrator(velocity) if self.integration_steps > 0 else velocity
        warped_source = self.spatial_transformer(source, displacement)

        outputs = {'displacement': displacement, 'warped_source': warped_source}
        if self.integration_steps > 0:
            outputs['velocity'] = velocity
        return outputs

class VxmCascade(nn.Module):
    """
    Two-stage cascaded refinement: register, warp, then register the residual and compose.

    The single-pass network has to produce the whole deformation in one shot. Running it twice --
    warping the moving image by the first prediction and letting a second network correct the
    now nearly-aligned pair -- means the second stage only ever sees a small residual, which is a
    much easier regression problem.

    Transforms are **composed, not added**. With the convention `moved(x) = source(x + u(x))`,
    applying stage 1 and then stage 2 gives

        u(x) = u2(x) + u1(x + u2(x))

    i.e. `u2 + warp(u1, u2)`. Adding the fields instead would be wrong whenever the first stage
    moves anything appreciably, because the second stage's correction is expressed in the frame
    the first stage created, not the original one.

    `stage_scales` sets the resolution each stage runs at as a downsampling factor, so (2, 1) is
    coarse-to-fine -- a half-resolution first pass then a full-resolution correction -- and (1, 1)
    runs both at full resolution. A stage at scale `s` drops `log2(s)` UNet levels, since the
    grid it sees is that much smaller; this is what keeps the coarse stage's parameter count down
    and makes (2, 1) cheaper than (1, 1) rather than merely different.

    Note the two arms test different hypotheses. (2, 1) is the coarse-to-fine argument from the
    paper's receptive-field discussion; (1, 1) is the control that isolates *iteration* from
    *multi-resolution*, by iterating without ever changing scale. If they score the same, the
    resolution schedule is not what is doing the work.

    With `integration_steps > 0` each stage integrates its own velocity, so every stage is a
    diffeomorphism and so is their composition.

    Parameters
    ----------
    ndim : int
        Spatial dimensionality.
    nb_features : sequence of int
        UNet features per level for the full-resolution stage; coarser stages drop levels.
    integration_steps : int
        Scaling-and-squaring steps applied within each stage.
    stage_scales : sequence of int
        Downsampling factor per stage, finest last. Each must be a power of two.
    """

    def __init__(
        self,
        ndim: int,
        nb_features: Sequence[int] = (16, 32, 32, 32, 32),
        integration_steps: int = 0,
        stage_scales: Sequence[int] = (2, 1),
    ) -> None:
        super().__init__()
        self.ndim = ndim
        self.integration_steps = integration_steps
        self.stage_scales = tuple(stage_scales)

        self.stages = nn.ModuleList()
        for scale in self.stage_scales:
            dropped = int(round(math.log2(scale)))
            if 2 ** dropped != scale:
                raise ValueError(f'stage_scales must be powers of two, got {scale}')
            levels = len(nb_features) - dropped
            if levels < 1:
                raise ValueError(f'scale {scale} leaves no UNet levels for {len(nb_features)}')
            self.stages.append(vxm.nn.models.VxmPairwise(
                ndim=ndim,
                source_channels=1,
                target_channels=1,
                nb_features=tuple(nb_features[:levels]),
                integration_steps=integration_steps,
            ))

        self.spatial_transformer = vxm.nn.modules.SpatialTransformer()

    def forward(self, source: torch.Tensor, target: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Register `source` to `target`; returns each stage's field under `stages`."""
        full = source.shape[2:]
        mode = 'bilinear' if self.ndim == 2 else 'trilinear'

        field = None
        warped = source
        per_stage: List[torch.Tensor] = []

        for stage, scale in zip(self.stages, self.stage_scales):
            if scale > 1:
                small = tuple(size // scale for size in full)
                stage_source = torch.nn.functional.interpolate(
                    warped, size=small, mode=mode, align_corners=True)
                stage_target = torch.nn.functional.interpolate(
                    target, size=small, mode=mode, align_corners=True)
            else:
                stage_source, stage_target = warped, target

            update = stage(stage_source, stage_target, return_field_type='displacement')
            if scale > 1:
                update = _resample_field(update, full, self.ndim)

            # Compose in the frame the earlier stages created, rather than adding.
            field = update if field is None else update + self.spatial_transformer(field, update)
            per_stage.append(field)
            warped = self.spatial_transformer(source, field)

        return {
            'displacement': field,
            'warped_source': warped,
            'stages': per_stage,
        }


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
        return VxmLambdaField(mask_normalise=config.lambda_mask_norm, **common)
    if config.variant == 'cascade':
        return VxmCascade(stage_scales=config.cascade_scales, **common)
    if config.variant == 'pyramid':
        return VxmPyramid(progressive=config.pyramid_progressive, **common)
    if config.variant == 'fathead':
        return VxmFatHead(hidden=config.head_hidden, kernel=config.head_kernel,
                          image_skip=config.head_image_skip, **common)
    if config.variant == 'msf':
        return VxmMultiScaleFeatures(per_level=config.msf_per_level, **common)
    if config.variant == 'cross_attn':
        return VxmCrossAttention(attn_heads=config.attn_heads,
                                 target_skips=config.cross_attn_target_skips,
                                 use_attention=config.cross_attn_use_attention,
                                 window_level=config.cross_attn_window_level,
                                 window_radius=config.cross_attn_window_radius, **common)

    raise ValueError(f"unknown variant '{config.variant}'")

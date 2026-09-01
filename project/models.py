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
from project.losses import compose_displacements


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


class VxmLambdaStructure(nn.Module):
    """
    VoxelMorph with per-structure regularisation weight.

    `VxmLambdaField` gives the weight map one degree of freedom per voxel, and the measured
    per-structure Dice showed it using that freedom coherently by region -- better on the
    lateral ventricles and brain stem, worse on putamen and thalamus. This variant grants only
    the freedom that result actually used: the head emits `n_labels` scalars (25 in 2D, 36 in
    3D, background included), which are broadcast onto the voxel grid through the source segmentation. The
    hypothesis is that "ventricles deform, brain stem does not" needs 24 numbers, not 30k, and
    that the smaller hypothesis class generalises where the per-voxel one did not.

    The weight map is normalised to mean 1 **over the brain mask**, exactly as in the
    mask-normalised lambda-field, so `lambda_reg` remains the average regularisation strength
    and the comparison isolates allocation from strength. Normalisation is applied to the voxel
    map rather than to the structure vector because structures differ enormously in size: an
    unweighted mean over structures would let the tiny ones dominate the budget.

    **This variant is semi-supervised.** The baseline and the lambda-field are trained without
    labels; this one consumes the source segmentation at training time. It is therefore not a
    like-for-like replacement for them, and a win here does not transfer to a setting where
    segmentations are unavailable at registration time. Reported numbers must carry that caveat.

    Parameters
    ----------
    ndim : int
        Spatial dimensionality.
    n_labels : int
        Size of the per-structure weight table; must exceed the largest label id present
        (25 in 2D, 36 in 3D -- both include background as id 0).
    nb_features : sequence of int
        UNet features per level.
    integration_steps : int
        Scaling-and-squaring steps; 0 for a plain displacement field.
    weight_range : tuple of float
        Lower and upper bound on the per-structure weight before mean-normalisation.
    """

    def __init__(
        self,
        ndim: int,
        n_labels: int,
        nb_features: Sequence[int] = (16, 32, 32, 32, 32),
        integration_steps: int = 0,
        weight_range: Tuple[float, float] = (0.5, 2.0),
        structure_lambda: Optional[Sequence[float]] = None,
    ) -> None:
        super().__init__()
        self.ndim = ndim
        self.integration_steps = integration_steps
        self.weight_range = weight_range
        self.n_labels = n_labels
        self.fixed = structure_lambda is not None

        # With a fixed allocation there is no weight head at all, so the UNet is exactly the
        # baseline's. That makes the control a strictly *smaller* model than the learned
        # variant -- if it matches, the learned head is provably doing no work.
        self.unet = ne.nn.models.BasicUNet(
            ndim=ndim,
            in_channels=2,
            out_channels=ndim if self.fixed else ndim + n_labels,
            nb_features=nb_features,
        )
        if self.fixed:
            if len(structure_lambda) != n_labels:
                raise ValueError(f'structure_lambda must have {n_labels} entries, '
                                 f'got {len(structure_lambda)}')
            self.register_buffer('fixed_lambda',
                                 torch.tensor(list(structure_lambda), dtype=torch.float32))
        self.flow_layer = _init_flow_layer(ndim)
        self.spatial_transformer = vxm.nn.modules.SpatialTransformer()
        if integration_steps > 0:
            self.integrator = vxm.nn.modules.IntegrateVelocityField(steps=integration_steps)

    def forward(
        self,
        source: torch.Tensor,
        target: torch.Tensor,
        seg: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Register `source` to `target`.

        Parameters
        ----------
        source, target : torch.Tensor
            Input images of shape (B, 1, *spatial).
        seg : torch.Tensor or None
            Segmentation of the *source* of shape (B, 1, *spatial), integer labels. The
            smoothness penalty lives on the source grid, so the source's labels are the ones
            that say which structure each penalised voxel belongs to.

        Returns
        -------
        dict
            Keys: displacement, warped_source, lambda_map, structure_lambda.
        """
        if seg is None:
            raise ValueError('VxmLambdaStructure requires the source segmentation')

        features = self.unet(torch.cat([source, target], dim=1))

        velocity = self.flow_layer(features[:, :self.ndim])

        if self.fixed:
            structure_lambda = self.fixed_lambda.unsqueeze(0).expand(source.shape[0], -1)
        else:
            # Global average pool the weight channels: the head predicts one scalar per
            # structure, not a spatial field. This is the whole point of the variant --
            # n_labels degrees of freedom instead of one per voxel.
            spatial_dims = tuple(range(2, features.dim()))
            pooled = features[:, self.ndim:self.ndim + self.n_labels].mean(dim=spatial_dims)

            low, high = self.weight_range
            structure_lambda = low + (high - low) * torch.sigmoid(pooled)
        lambda_map = self._scatter_to_voxels(structure_lambda, seg)

        # Normalise the *voxel* map, not the structure vector. Structures differ enormously in
        # size (background dominates), so a plain mean over structures would not hold the
        # regularisation budget equal to the baseline's -- and an equal budget is what makes the
        # comparison a test of allocation rather than of strength.
        mask = (source > 0) | (target > 0)
        lambda_map = self._normalise_to_unit_mean(lambda_map, mask)

        if self.integration_steps > 0:
            displacement = self.integrator(velocity)
        else:
            displacement = velocity

        outputs = {
            'displacement': displacement,
            'warped_source': self.spatial_transformer(source, displacement),
            'lambda_map': lambda_map,
            'structure_lambda': structure_lambda,
        }
        if self.integration_steps > 0:
            outputs['velocity'] = velocity
        return outputs

    def _scatter_to_voxels(
        self,
        structure_lambda: torch.Tensor,
        seg: torch.Tensor,
    ) -> torch.Tensor:
        """
        Broadcast per-structure weights onto the voxel grid.

        Parameters
        ----------
        structure_lambda : torch.Tensor
            Per-structure weights of shape (B, n_labels).
        seg : torch.Tensor
            Segmentation of shape (B, 1, *spatial) holding integer label ids.

        Returns
        -------
        torch.Tensor
            Weight map of shape (B, 1, *spatial), differentiable w.r.t. `structure_lambda`.
        """
        batch = seg.shape[0]
        spatial = seg.shape[2:]

        index = seg.reshape(batch, -1).long()

        # Label ids come straight from the NIfTI with no remapping, so `n_labels` must cover the
        # largest id actually present. Clamping instead would silently merge every structure
        # above the bound into one weight and quietly invalidate the experiment, so this fails
        # loudly rather than returning a plausible wrong answer.
        largest = int(index.max())
        if largest >= self.n_labels:
            raise ValueError(
                f'segmentation contains label id {largest} but n_labels={self.n_labels}; '
                f'set n_labels > {largest}'
            )

        gathered = structure_lambda.gather(1, index)
        return gathered.reshape(batch, 1, *spatial)

    @staticmethod
    def _normalise_to_unit_mean(
        weights: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Rescale a weight map to mean 1 within the brain mask.

        Parameters
        ----------
        weights : torch.Tensor
            Positive weight map of shape (B, 1, *spatial).
        mask : torch.Tensor
            Boolean brain mask of the same shape.

        Returns
        -------
        torch.Tensor
            `weights` rescaled so its masked per-sample mean is 1.
        """
        spatial_dims = tuple(range(1, weights.dim()))
        indicator = mask.to(weights.dtype)
        count = indicator.sum(dim=spatial_dims, keepdim=True).clamp(min=1.0)
        masked_mean = (weights * indicator).sum(dim=spatial_dims, keepdim=True) / count
        return weights / masked_mean


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


class VxmCrossAttentionGated(nn.Module):
    """
    Cross-attention added to the baseline as a zero-initialised gated residual.

    `VxmCrossAttention` loses 6% Dice in 2D and 13% in 3D, worse than the baseline on 100/100
    pairs. The diagnosis is structural rather than a matter of tuning: that variant *replaces*
    the baseline's input path with a two-stream encoder whose streams each see a single image,
    and its bottleneck residual is followed by a LayerNorm, which rescales the representation.
    There is consequently **no initialisation at which it equals the baseline** -- it begins by
    damaging a working model and has to climb back, and it never does.

    This variant fixes exactly that, and changes nothing else:

    * The main path is a plain UNet on `cat(source, target)` -- the baseline's own formulation,
      and the same `BasicUNet` the lambda-field variant uses, so the anchor is like-for-like.
    * The attention branch is a *second* pass of the **same encoder weights** over the swapped
      input `cat(target, source)`. Sharing the encoder means the branch adds no encoder
      parameters, so a measured difference is attributable to the attention and not to capacity.
      The swap gives a genuinely different view -- the pair seen from the target's side -- which
      is what makes attention between the two streams meaningful rather than self-attention.
    * The two bottlenecks are combined as `h = h_forward + tanh(gate) * attend(h_forward,
      h_swapped)` with `gate` a learned scalar **initialised to zero**.

    At initialisation `tanh(0) = 0`, so the attention branch contributes nothing and the model is
    bit-for-bit the plain UNet path. Gradient descent opens the gate only if attention earns it,
    and can close it again. The failure mode of the original -- pay a large upfront cost for a
    mechanism that may not help -- is therefore impossible by construction. This is the
    ReZero/LayerScale trick, used here for the reason it was invented.

    The decoder consumes the **forward** stream's skips, since the field is defined on the source
    grid.

    `gate` is reported in the output dict: if the trained value stays near zero, the honest
    reading is that cross-attention is not useful for this task, and that is a publishable
    negative result rather than a failed run.

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

        self.unet = ne.nn.models.BasicUNet(
            ndim=ndim,
            in_channels=2,
            out_channels=ndim,
            nb_features=nb_features,
        )

        bottleneck_channels = self.unet.lowest_resolution_conv_block.out_channels
        self.attention = nn.MultiheadAttention(
            embed_dim=bottleneck_channels,
            num_heads=attn_heads,
            batch_first=True,
        )
        # The gate is the whole point: zero here means "start as the baseline".
        self.gate = nn.Parameter(torch.zeros(1))

        self.flow_layer = _init_flow_layer(ndim)
        self.spatial_transformer = vxm.nn.modules.SpatialTransformer()
        if integration_steps > 0:
            self.integrator = vxm.nn.modules.IntegrateVelocityField(steps=integration_steps)

    def _encode(self, stacked: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Run a two-channel input through the encoder, returning bottleneck features and skips.

        Parameters
        ----------
        stacked : torch.Tensor
            Two-channel input of shape (B, 2, *spatial).

        Returns
        -------
        tuple
            `(bottleneck, skips)`.
        """
        features = stacked
        skips: List[torch.Tensor] = []
        for block in self.unet.downsampling_conv_blocks:
            features, skip = block(features)
            skips.append(skip)
        return self.unet.lowest_resolution_conv_block(features), skips

    def _attend(self, query_feat: torch.Tensor, key_feat: torch.Tensor) -> torch.Tensor:
        """
        Cross-attend the forward bottleneck to the swapped-view bottleneck.

        Parameters
        ----------
        query_feat, key_feat : torch.Tensor
            Bottleneck features of shape (B, C, *reduced_spatial).

        Returns
        -------
        torch.Tensor
            Attention output, same shape as `query_feat`. Returned *un*-normalised and
            *un*-added, so the caller controls the gate.
        """
        batch, channels = query_feat.shape[:2]
        spatial = query_feat.shape[2:]

        query = query_feat.flatten(2).transpose(1, 2)
        key = key_feat.flatten(2).transpose(1, 2)

        attended, _ = self.attention(query, key, key, need_weights=False)
        return attended.transpose(1, 2).reshape(batch, channels, *spatial)

    def _decode(self, features: torch.Tensor, skips: List[torch.Tensor]) -> torch.Tensor:
        """Decode bottleneck features to full resolution using the forward stream's skips."""
        for block, skip in zip(self.unet.upsampling_conv_blocks, reversed(skips)):
            features = block(features, skip)
        return self.unet.out_layer(features)

    def forward(self, source: torch.Tensor, target: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Register `source` to `target`; see module docstring for returned keys."""
        forward_feat, skips = self._encode(torch.cat([source, target], dim=1))
        swapped_feat, _ = self._encode(torch.cat([target, source], dim=1))

        gated = forward_feat + torch.tanh(self.gate) * self._attend(forward_feat, swapped_feat)
        velocity = self.flow_layer(self._decode(gated, skips))

        if self.integration_steps > 0:
            displacement = self.integrator(velocity)
        else:
            displacement = velocity

        outputs = {
            'displacement': displacement,
            'warped_source': self.spatial_transformer(source, displacement),
            'gate': torch.tanh(self.gate).detach(),
        }
        if self.integration_steps > 0:
            outputs['velocity'] = velocity
        return outputs


class VxmCoarseToFine(nn.Module):
    """
    Cascaded multi-resolution registration: predict, warp, then predict a residual.

    The paper notes (SS IV-A) that the receptive field at the coarsest UNet level must be at least
    as large as the maximum expected displacement. A single full-resolution prediction therefore
    has to solve large and small displacements with the same machinery. This variant splits that:
    an early stage sees a downsampled pair -- where a large displacement is a *small* number of
    voxels and easily within the receptive field -- and later stages only ever see a nearly
    aligned pair and predict a small residual.

    Per stage: warp the source with the transform accumulated so far, predict a residual from
    `(warped_source, target)`, and **compose** rather than add. Composition matters: two
    displacement fields do not sum, because the second is defined on the grid the first has
    already moved. The correct rule is `u_total(x) = u1(x) + u2(x + u1(x))`, evaluated here by
    resampling `u2` through `u1` with the spatial transformer, which stays batched and
    differentiable (unlike `voxelmorph.nn.functional.compose`, which must be looped per sample --
    see `metrics.inverse_consistency`).

    A displacement field upsampled by a factor `s` must also be **scaled by `s`**: it is measured
    in voxels, and voxels get smaller as resolution rises. Omitting that scaling silently halves
    every coarse displacement and is the classic bug in this construction.

    **The capacity confound is real and is controlled by configuration, not by argument.** Two
    stages means two UNets, so any gain could simply be more parameters. Setting `stage_scales`
    to `(1, 1)` gives a cascade at a *single* resolution with identical parameter count, which
    isolates the multi-resolution claim from the extra-capacity one. Report both.

    Parameters
    ----------
    ndim : int
        Spatial dimensionality.
    nb_features : sequence of int
        UNet features per level for the finest stage. Coarser stages drop levels, since a
        downsampled volume cannot be halved as many times.
    integration_steps : int
        Scaling-and-squaring steps applied to each stage's residual field.
    stage_scales : sequence of int
        Downsampling factor per stage, coarsest first. `(2, 1)` is coarse-to-fine; `(1, 1)` is
        the same-resolution cascade control.
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
        self.flow_layers = nn.ModuleList()
        for scale in self.stage_scales:
            # A stage running at 1/scale resolution can afford log2(scale) fewer downsamplings
            # before the spatial extent stops being divisible by two.
            levels = len(nb_features) - int(round(math.log2(scale)))
            self.stages.append(ne.nn.models.BasicUNet(
                ndim=ndim, in_channels=2, out_channels=ndim,
                nb_features=tuple(nb_features[:levels]),
            ))
            self.flow_layers.append(_init_flow_layer(ndim))

        self.spatial_transformer = vxm.nn.modules.SpatialTransformer()
        if integration_steps > 0:
            self.integrator = vxm.nn.modules.IntegrateVelocityField(steps=integration_steps)

    def _resize(self, tensor: torch.Tensor, scale: float, is_field: bool) -> torch.Tensor:
        """
        Resample a tensor by `scale`, rescaling magnitudes when it is a displacement field.

        Parameters
        ----------
        tensor : torch.Tensor
            Image or field of shape (B, C, *spatial).
        scale : float
            Output size relative to input; 0.5 halves each spatial extent.
        is_field : bool
            True for a displacement field, whose values are in voxels and must be scaled too.
        """
        if scale == 1.0:
            return tensor
        mode = 'bilinear' if self.ndim == 2 else 'trilinear'
        resized = nn.functional.interpolate(tensor, scale_factor=scale, mode=mode,
                                            align_corners=False)
        return resized * scale if is_field else resized

    def _compose(self, accumulated: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
        """
        Compose an accumulated transform with a new residual applied *after* it.

        Warping by `accumulated` and then by `residual` is equivalent to warping once by
        `residual(x) + accumulated(x + residual(x))` -- note the residual comes first in the
        composed expression, because it is the outer transform that relocates the sampling
        point. The mirror-image ordering is a natural mistake and is only ~2x worse than this
        one on smooth fields, which is why `tests/test_project_coarse_to_fine.py` checks it
        numerically against sequential warping rather than by inspection. Plain addition,
        `accumulated + residual`, also *nearly* works and is likewise rejected there.
        """
        return compose_displacements(accumulated, residual, self.spatial_transformer)

    def forward(self, source: torch.Tensor, target: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Register `source` to `target`; see module docstring for returned keys."""
        total = torch.zeros(source.shape[0], self.ndim, *source.shape[2:],
                            device=source.device, dtype=source.dtype)
        residuals = []

        for index, scale in enumerate(self.stage_scales):
            warped = self.spatial_transformer(source, total)

            factor = 1.0 / scale
            pair = torch.cat([self._resize(warped, factor, is_field=False),
                              self._resize(target, factor, is_field=False)], dim=1)
            residual = self.flow_layers[index](self.stages[index](pair))

            if self.integration_steps > 0:
                residual = self.integrator(residual)
            residual = self._resize(residual, float(scale), is_field=True)

            total = self._compose(total, residual)
            residuals.append(residual)

        outputs = {
            'displacement': total,
            'warped_source': self.spatial_transformer(source, total),
        }
        if self.integration_steps > 0:
            outputs['velocity'] = residuals[-1]
        return outputs


def forward_model(
    model: nn.Module,
    source: torch.Tensor,
    target: torch.Tensor,
    source_seg: Optional[torch.Tensor] = None,
) -> Dict[str, torch.Tensor]:
    """
    Run a model, supplying the segmentation only to the variant that needs it.

    Keeps the variant check in one place so training, validation and evaluation cannot drift
    apart in how they call the model.

    Parameters
    ----------
    model : nn.Module
        Any of the variants in this module.
    source, target : torch.Tensor
        Input images of shape (B, 1, *spatial).
    source_seg : torch.Tensor or None, optional
        Segmentation of the source, required by `VxmLambdaStructure` and ignored otherwise.

    Returns
    -------
    dict
        The model's output dict.
    """
    if isinstance(model, VxmLambdaStructure):
        return model(source, target, seg=source_seg)
    return model(source, target)


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
        return VxmLambdaField(mask_normalise=config.lambda_mask_norm,
                              weight_range=config.weight_range, **common)
    if config.variant == 'lambda_structure':
        return VxmLambdaStructure(n_labels=config.n_labels,
                                  weight_range=config.weight_range,
                                  structure_lambda=config.structure_lambda, **common)
    if config.variant == 'cross_attn':
        return VxmCrossAttention(attn_heads=config.attn_heads, **common)
    if config.variant == 'coarse_to_fine':
        return VxmCoarseToFine(stage_scales=config.stage_scales, **common)
    if config.variant == 'cross_attn_gated':
        return VxmCrossAttentionGated(attn_heads=config.attn_heads, **common)

    raise ValueError(f"unknown variant '{config.variant}'")

"""
Tests for the cascaded coarse-to-fine variant.

Two of these guard bugs that are silent rather than loud, and both were caught by writing them:

* `test_composition_matches_sequential_warping` -- displacement fields do not add, and the two
  candidate composition orders differ by only about a factor of two in error on smooth fields
  (plain addition sits between them). Nothing crashes if this is wrong; the model simply learns
  a slightly incoherent transform. So the composition is pinned numerically against sequential
  warping, which is the definition it must satisfy.
* `test_upsampled_field_is_rescaled` -- a displacement is measured in voxels, so upsampling a
  field by 2 must double its values. Omitting that halves every coarse displacement and would
  quietly cost the variant exactly the long-range capability it exists to provide.
"""

import pytest
import torch

from project.configs import ExperimentConfig
from project.models import VxmBaseline, VxmCoarseToFine, build_model


def _smooth_field(shape, scale=4.0, seed=0):
    torch.manual_seed(seed)
    field = torch.randn(1, len(shape), *shape)
    return torch.nn.functional.avg_pool2d(field, 15, 1, 7) * scale


def _smooth_image(shape):
    y, x = torch.meshgrid(*[torch.arange(float(n)) for n in shape], indexing='ij')
    return (torch.sin(x / 9) * torch.cos(y / 11)).reshape(1, 1, *shape)


@pytest.mark.parametrize('stage_scales', [(2, 1), (1, 1), (4, 2, 1)])
@pytest.mark.parametrize('integration_steps', [0, 7])
def test_forward_shapes(stage_scales, integration_steps):
    """Every stage configuration returns a full-resolution field."""
    model = VxmCoarseToFine(ndim=2, stage_scales=stage_scales,
                            integration_steps=integration_steps)
    source, target = torch.rand(2, 1, 160, 192), torch.rand(2, 1, 160, 192)
    outputs = model(source, target)

    assert outputs['displacement'].shape == (2, 2, 160, 192)
    assert outputs['warped_source'].shape == source.shape


def test_composition_matches_sequential_warping():
    """
    The composed field must reproduce warping twice in sequence.

    This is the definition of composition and the reason the variant can cascade at all.
    """
    model = VxmCoarseToFine(ndim=2)
    image = _smooth_image((64, 64))
    first = _smooth_field((64, 64), seed=0)
    second = _smooth_field((64, 64), seed=1)

    sequential = model.spatial_transformer(model.spatial_transformer(image, first), second)
    composed = model.spatial_transformer(image, model._compose(first, second))

    error = (composed - sequential).abs().mean()
    # Both the reversed order and plain addition land near 2x this, so the bound discriminates.
    assert error < 0.003, f'composition error {error:.5f} too large'


def test_composition_beats_addition_and_reversed_order():
    """The chosen ordering must be strictly better than the two plausible wrong ones."""
    model = VxmCoarseToFine(ndim=2)
    image = _smooth_image((64, 64))
    a = _smooth_field((64, 64), seed=0)
    b = _smooth_field((64, 64), seed=1)
    warp = model.spatial_transformer

    truth = warp(warp(image, a), b)
    def err(field):
        return float((warp(image, field) - truth).abs().mean())

    assert err(model._compose(a, b)) < err(a + b)
    assert err(model._compose(a, b)) < err(model._compose(b, a))


def test_identity_when_fields_are_zero():
    """Composing zero transforms leaves the image untouched."""
    model = VxmCoarseToFine(ndim=2)
    image = _smooth_image((64, 64))
    zero = torch.zeros(1, 2, 64, 64)
    assert torch.allclose(model.spatial_transformer(image, model._compose(zero, zero)),
                          image, atol=1e-5)


def test_upsampled_field_is_rescaled():
    """Upsampling a displacement field by 2 must double its magnitudes."""
    model = VxmCoarseToFine(ndim=2)
    field = torch.ones(1, 2, 32, 32)
    upsampled = model._resize(field, 2.0, is_field=True)

    assert upsampled.shape == (1, 2, 64, 64)
    assert torch.allclose(upsampled, torch.full_like(upsampled, 2.0))


def test_image_resize_does_not_rescale_values():
    """An image is not a field; its intensities must survive resampling unchanged."""
    model = VxmCoarseToFine(ndim=2)
    image = torch.full((1, 1, 32, 32), 0.7)
    assert torch.allclose(model._resize(image, 2.0, is_field=False),
                          torch.full((1, 1, 64, 64), 0.7))


def test_control_has_no_less_capacity_than_coarse_to_fine():
    """
    The same-resolution control must not be the *smaller* model.

    It is the confound check for the multi-resolution claim, so it has to be at least as large;
    otherwise a win for coarse-to-fine could just be extra parameters.
    """
    fine = sum(p.numel() for p in VxmCoarseToFine(ndim=2, stage_scales=(2, 1)).parameters())
    control = sum(p.numel() for p in VxmCoarseToFine(ndim=2, stage_scales=(1, 1)).parameters())
    assert control >= fine


def test_build_model_routes_variant():
    """`build_model` recognises the variant and honours `stage_scales`."""
    config = ExperimentConfig(name='t', variant='coarse_to_fine', ndim=2, stage_scales=(4, 2, 1))
    model = build_model(config)
    assert isinstance(model, VxmCoarseToFine)
    assert len(model.stages) == 3

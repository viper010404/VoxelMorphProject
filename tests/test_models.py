# Standard library imports
import pytest

# Third-party imports
import torch

# Custom imports
import voxelmorph as vxm


@pytest.fixture
def dummy_input_pair():
    """
    Make a 3D input pair of tensors ~N(0, 1) for source and target images.
    """

    shape = (1, 1, 32, 32, 32)  # (B, C, D, H, W)
    source = torch.rand(*shape)
    target = torch.rand(*shape)
    return source, target


def test_forward_output_shape(dummy_input_pair):
    """
    Test that the forward method returns correct output shapes when without trying registration or
    the bidirectional cost.
    """

    model = vxm.nn.models.VxmDeformable(
        ndim=3,
        source_channels=1,
        target_channels=1,
        out_channels=3,  # Spatial Transformer expects 3 output channels for the 3 spatial dims
        device="cpu"
    )

    source, target = dummy_input_pair
    output = model(source, target)

    assert isinstance(output, list)
    assert output[0].shape == source.shape  # Warped source
    assert output[1].shape[2:] == source.shape[2:]  # Displacement field


def test_register_mode(dummy_input_pair):
    """
    Test that forward pass with registration returns warped source and displacement field.
    """

    model = vxm.nn.models.VxmDeformable(
        ndim=3,
        source_channels=1,
        target_channels=1,
        out_channels=3,
        device="cpu"
    )

    source, target = dummy_input_pair
    warped_source, pos_flow = model(source, target, register=True)

    assert warped_source.shape == source.shape
    assert pos_flow.shape[2:] == source.shape[2:]


def test_spatial_transformer_initialized(dummy_input_pair):
    """
    Test that the spatial transformer module is lazily initialized.
    """

    model = vxm.nn.models.VxmDeformable(
        ndim=3,
        source_channels=1,
        target_channels=1,
        out_channels=3,
        device="cpu"
    )

    # Ensure transformer is not initialized before forward pass
    assert not hasattr(model, "spatial_transformer")

    source, target = dummy_input_pair
    _ = model(source, target)

    # Ensure transformer is initialized after forward
    assert hasattr(model, "spatial_transformer")

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


@pytest.fixture
def vxm_model():
    """
    Create a VxmPairwise model for testing with standard 3D configuration.
    """

    model = vxm.nn.models.VxmPairwise(
        ndim=3,
        source_channels=1,
        target_channels=1,
        spatial_shape=(32, 32, 32),
        device="cpu"
    )
    return model


def test_forward_output_shape(dummy_input_pair, vxm_model):
    """
    Test that the forward method returns correct output shapes when without trying registration or
    the bidirectional cost.
    """

    # Unpack dummy input pair
    source, target = dummy_input_pair

    output = vxm_model(source, target)

    assert isinstance(output, torch.Tensor)
    assert output.shape[2:] == source.shape[2:]

    # Ensure transformer is initialized after forward
    assert hasattr(vxm_model, "flow_layer")
    assert hasattr(vxm_model, "spatial_transformer")
    assert hasattr(vxm_model, "velocity_field_integrator")


def test_backward_compat_return_warped_mode(dummy_input_pair, vxm_model):
    """
    Test that forward pass with registration returns warped source and displacement field.
    """

    source, target = dummy_input_pair
    velocity, warped_source = vxm_model(source, target, return_warped=True)

    assert velocity.shape[2:] == source.shape[2:]
    assert warped_source.shape[2:] == source.shape[2:]

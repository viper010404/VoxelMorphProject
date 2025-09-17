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

    # Unpack dummy input pair
    source, target = dummy_input_pair

    # Initialize model
    model = vxm.nn.models.VxmPairwise(
        ndim=3,
        source_channels=1,
        target_channels=1,
        spatial_shape=(32, 32, 32),
        device="cpu"
    )

    output = model(source, target)

    assert isinstance(output, torch.Tensor)
    assert output.shape[2:] == source.shape[2:]

    # Ensure transformer is initialized after forward
    assert hasattr(model, "flow_layer")
    assert hasattr(model, "spatial_transformer")
    assert hasattr(model, "velocity_field_integrator")


def test_return_warped_mode(dummy_input_pair):
    """
    Test that forward pass with registration returns warped source and displacement field.
    """

    model = vxm.nn.models.VxmPairwise(
        ndim=3,
        source_channels=1,
        target_channels=1,
        spatial_shape=(32, 32, 32),
        device="cpu"
    )

    source, target = dummy_input_pair
    warped_source, pos_flow = model(source, target, return_warped=True)

    assert warped_source.shape[2:] == source.shape[2:]
    assert pos_flow.shape[2:] == source.shape[2:]

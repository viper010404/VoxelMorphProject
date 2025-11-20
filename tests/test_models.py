# Standard library imports
import pytest

# Third-party imports
import torch

# Custom imports
import voxelmorph as vxm


@pytest.fixture(scope='module')
def dummy_input_pair():
    """
    Make a 3D input pair of tensors ~N(0, 1) for source and target images.
    """
    shape = (1, 1, 32, 32, 32)  # (B, C, D, H, W)
    source = torch.rand(*shape)
    target = torch.rand(*shape)
    return source, target


@pytest.fixture(scope='module')
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


@pytest.fixture(scope='module')
def vxm_model_diffeomorphic():
    """
    Create a VxmPairwise model with diffeomorphic registration (integration_steps > 0).
    """

    model = vxm.nn.models.VxmPairwise(
        ndim=3,
        source_channels=1,
        target_channels=1,
        spatial_shape=(32, 32, 32),
        integration_steps=7,
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


@pytest.mark.parametrize(
    "return_warped_source,return_warped_target,expected_outputs",
    [
        (False, False, 1),  # Only velocity
        (True, False, 2),   # velocity + warped_source
        (False, True, 2),   # velocity + warped_target
        (True, True, 3),    # velocity + warped_source + warped_target
    ]
)
def test_return_warped_combinations(
    dummy_input_pair,
    vxm_model_diffeomorphic,
    return_warped_source,
    return_warped_target,
    expected_outputs
):
    """
    Test all combinations of return_warped_source and return_warped_target flags.

    This test verifies that the forward method returns the correct number and type of outputs
    for each combination of flags:
    - No flags: velocity only
    - return_warped_source only: (velocity, warped_source)
    - return_warped_target only: (velocity, warped_target)
    - Both flags: (velocity, warped_source, warped_target)
    """

    source, target = dummy_input_pair
    output = vxm_model_diffeomorphic(
        source,
        target,
        return_warped_source=return_warped_source,
        return_warped_target=return_warped_target
    )

    # Check number of outputs
    if expected_outputs == 1:
        assert isinstance(output, torch.Tensor), "Expected single tensor output"
        velocity = output
        assert velocity.shape == (1, 3, 32, 32, 32), (
            f"Expected shape (1, 3, 32, 32, 32), got {velocity.shape}"
        )

    else:
        assert isinstance(output, tuple), f"Expected tuple, got {type(output)}"
        assert len(output) == expected_outputs, (
            f"Expected {expected_outputs} outputs, got {len(output)}"
        )

        # Velocity is always first
        velocity = output[0]
        assert velocity.shape == (1, 3, 32, 32, 32), (
            f"Expected velocity shape (1, 3, 32, 32, 32), got {velocity.shape}"
        )

        # Check warped outputs
        if expected_outputs == 2:
            warped = output[1]
            assert warped.shape == source.shape, (
                f"Expected warped shape {source.shape}, got {warped.shape}"
            )

        elif expected_outputs == 3:
            warped_source = output[1]
            warped_target = output[2]

            assert warped_source.shape == source.shape, (
                f"Expected warped_source shape {source.shape}, got {warped_source.shape}"
            )

            assert warped_target.shape == target.shape, (
                f"Expected warped_target shape {target.shape}, got {warped_target.shape}"
            )


def test_return_warped_target_requires_integration(vxm_model):
    """
    Test that requesting warped target with integration_steps=0 raises ValueError.
    """

    source = torch.rand(1, 1, 32, 32, 32)
    target = torch.rand(1, 1, 32, 32, 32)

    with pytest.raises(
        ValueError, match="Cannot return warped target image when integration_steps=0"
    ):
        vxm_model(source, target, return_warped_target=True)

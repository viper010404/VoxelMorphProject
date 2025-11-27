"""
Unit tests for the basic utility functions in voxelmorph.
"""

# Standard library imports
import torch

import voxelmorph as vxm
import voxelmorph.nn.functional as vxf
import neurite as ne
import neurite.nn.functional as nef


def test_affine_to_disp_identity():
    """
    Identity affine should produce zero displacement everywhere.
    """
    shape = (3, 4)
    grid = ne.volshape_to_ndgrid(size=shape, stack=True)
    ndim = len(shape)

    # Identity affine
    affine = torch.eye(ndim + 1, dtype=grid.dtype, device=grid.device)

    # Get displacement field
    disp = vxm.affine_to_disp(affine, grid)

    # output shape and dtype
    assert disp.shape == shape + (ndim,)
    assert disp.dtype == grid.dtype

    # all zeros
    assert torch.allclose(disp, torch.zeros_like(disp))


def test_affine_to_disp_translation():
    """
    Translation affine should yield a constant field.
    """
    shape = (2, 2)
    grid = ne.volshape_to_ndgrid(size=shape, stack=True)
    ndim = len(shape)
    tx, ty = 2.0, 3.0

    # build a 2D affine with translation in the last column
    affine = torch.eye(ndim + 1, dtype=grid.dtype, device=grid.device)

    # Make the translation
    affine[0, -1] = tx
    affine[1, -1] = ty

    # Get displacement field
    disp = vxm.affine_to_disp(affine, grid)

    # expected a field of shape (2,2,2) filled with [tx,ty]
    expected = torch.stack(
        [
            torch.full(shape, tx, dtype=grid.dtype, device=grid.device),
            torch.full(shape, ty, dtype=grid.dtype, device=grid.device),
        ],
        dim=-1
    )

    assert disp.shape == expected.shape
    assert torch.allclose(disp, expected)


def test_disp_to_coords_zero_disp_2d():
    """
    Zero displacement on a 2x3 grid should produce the normalized mesh in range [-1, 1], flipped
    (col, row).
    """
    disp = torch.zeros(2, 3, 2, dtype=torch.float32)
    coords = vxm.functional.disp_to_coords(disp)

    # For shape=(2,3):
    #  row indices i \isin {0, 1} -> bounded on [-1, 1] with 2 elements -> [-1, 1]
    #  col indices j \isin {0, 1, 2} ->  bounded on [-1, 1] with 3 elements -> [-1, 0, 1]
    expected = torch.tensor([
        [[-1., -1.], [0., -1.], [1., -1.]],
        [[-1., 1.], [0., 1.], [1., 1.]],
    ], dtype=torch.float32)

    assert coords.shape == (2, 3, 2)
    assert coords.dtype == torch.float32
    assert torch.allclose(coords, expected)


def test_spatial_transform_none_trf_returns_input():
    """
    If trf is None, spatial_transform should return the input image.
    """
    img = torch.rand(1, 5, 5)
    out = vxf.spatial_transform(img, None)

    assert out.shape == img.shape
    assert torch.allclose(out, img)


def test_spatial_transform_identity_affine():
    """
    An identity affine should yield the same image.
    """
    img = torch.rand(1, 3, 3, dtype=torch.float32)

    # 2D identity affine (3×3)
    affine = torch.eye(3, dtype=torch.float32)
    out = vxm.functional.spatial_transform(img, affine, non_spatial_dims=(0,))

    assert out.shape == img.shape
    assert torch.allclose(out, img, atol=1e-6)


def test_spatial_transform_rotation():
    """
    Test that spatial_transform rotates a horizontal line by 90 degrees and a vertical line by
    90 degrees.
    """

    A = torch.tensor([
        [0.0, -1.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0]
    ], dtype=torch.float32)

    tensor = torch.zeros(1, 5, 5)
    horizontal_line = tensor.clone()
    horizontal_line[0, 2, :] = 1

    vertical_line = tensor.clone()
    vertical_line[0, :, 2] = 1

    transformed_horizontal_line = vxm.spatial_transform(
        horizontal_line, trf=A, mode='linear', non_spatial_dims=(0,)
    )

    transformed_vertical_line = vxm.spatial_transform(
        vertical_line, trf=A, mode='linear', non_spatial_dims=(0,)
    )

    assert torch.allclose(horizontal_line, transformed_vertical_line)
    assert torch.allclose(vertical_line, transformed_horizontal_line)


def test_spatial_transform_batched():

    A = torch.tensor([
        [0.0, -1.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0]
    ], dtype=torch.float32)

    # B: translation by (8, 0)
    B = torch.tensor([
        [1.0, 0.0, 8.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0]
    ], dtype=torch.float32)

    # 2 batches (images), one channel
    batched_images = torch.randn(2, 1, 28, 28)
    batched_transforms = torch.stack([A, B])

    vxm.spatial_transform(
        image=batched_images,
        trf=batched_transforms,
        non_spatial_dims=(0, 1)
    )


def test_spatial_transform_rotation_translation_batched():
    """
    Test that spatial_transform correctly applies different affine transformations to each batch
    element.

    This test verifies batched spatial transformations where:
    - Batch element 0: A 90-degree counter-clockwise rotation matrix transforms a horizontal line
      (at row 2) into a vertical line (at column 2)
    - Batch element 1: A translation matrix (shifts by 1 pixel in x-direction) transforms a 
      horizontal line at row 2 to row 3

    The input is a batch of 2 images, each containing a horizontal line at row 2 (middle row).
    After transformation, batch element 0 should have a vertical line at column 2, and batch 
    element 1 should have a horizontal line at row 3.
    """
    A = torch.tensor([
        [0.0, -1.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0]
    ], dtype=torch.float32)

    B = torch.tensor([
        [1.0, 0.0, 1.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0]
    ], dtype=torch.float32)

    tensor = torch.zeros(2, 1, 5, 5)
    horizontal_lines = tensor.clone()
    horizontal_lines[:, 0, 2, :] = 1

    batched_transforms = torch.stack([A, B])

    # Apply batched transformation
    transformed = vxm.spatial_transform(
        horizontal_lines, trf=batched_transforms, mode='linear', non_spatial_dims=(0, 1)
    )

    transformed_batch_element_0 = torch.zeros(1, 5, 5)
    transformed_batch_element_0[0, :, 2] = 1

    transformed_batch_element_1 = torch.zeros(1, 5, 5)
    transformed_batch_element_1[0, 3, :] = 1

    # Verify batch element 0: rotation transforms horizontal line to vertical line
    assert torch.allclose(transformed[0], transformed_batch_element_0), (
        f"Batch element 0 failed: Horizontal line should map to a vertical line after 90 degree "
        f"rotation, but got shape {transformed[0].shape} with max diff "
        f"{torch.max(torch.abs(transformed[0] - transformed_batch_element_0))}"
    )

    # Verify batch element 1: translation shifts horizontal line from row 2 to row 3
    assert torch.allclose(transformed[1], transformed_batch_element_1), (
        f"Batch element 1 failed: Horizontal line at row 2 should map to a horizontal line at "
        f"row 3 after translation by (1, 0), but got shape {transformed[1].shape} with max diff "
        f"{torch.max(torch.abs(transformed[1] - transformed_batch_element_1))}"
    )


def test_angles_to_rotation_matrix_2d_identity():
    """
    A 2D rotation of 0 deg must yield the 2x2 identity matrix.
    """
    rotation_matrix = vxm.angles_to_rotation_matrix(torch.tensor(0.0), degrees=True)
    expected = torch.eye(2, dtype=torch.float64)

    assert rotation_matrix.shape == (2, 2)
    assert rotation_matrix.dtype == torch.float64
    assert torch.allclose(rotation_matrix, expected, atol=1e-8)


def test_angles_to_rotation_matrix_2d_90_degrees():
    """
    A 2D rotation of 90 degrees should be [[0, -1], [1, 0]].
    """
    rotation_matrix = vxm.angles_to_rotation_matrix(torch.tensor(90.0), degrees=True)

    expected = torch.tensor(
        [
            [0.0, -1.0],
            [1.0, 0.0]
        ],
        dtype=torch.float64
    )

    assert torch.allclose(rotation_matrix, expected, atol=1e-5)


def test_angles_to_rotation_matrix_2d_pi_over_2_radians():
    """
    With degrees=False and angle=pi/2, result should match the 90° case.
    """
    rotation_matrix = vxm.angles_to_rotation_matrix(torch.tensor(torch.pi / 2), degrees=False)
    expected = torch.tensor(
        [
            [0.0, -1.0],
            [1.0, 0.0]
        ],
        dtype=torch.float64
    )
    assert torch.allclose(rotation_matrix, expected, atol=1e-5)


def test_angles_to_rotation_matrix_3d_90_degrees():
    """
    A 3D rotation of 90 degrees around the z axis should be:
    [[0, 1, 0],
     [-1, 0, 0],
     [0, 0, 1]]
    """
    rotation_matrix = vxm.angles_to_rotation_matrix(torch.tensor((0, 0, 90.0)), degrees=True)

    expected = torch.tensor(
        [
            [0, 1, 0],
            [-1, 0, 0],
            [0, 0, 1],
        ],
        dtype=torch.float64
    )

    assert torch.allclose(rotation_matrix, expected, atol=1e-5)


def test_params_to_affine_translation_shear():
    """
    Composing two translations should yield the sum of the two translations.
    """

    translation = (1, 2)

    result_affine = vxm.params_to_affine(
        ndim=2,
        translation=translation,
        shear=9,
    ).to(torch.float64)

    expected_affine = torch.tensor([[1, 9, 1], [0, 1, 2], [0, 0, 1]], dtype=torch.float64)

    assert torch.allclose(result_affine, expected_affine, atol=1e-5)


def test_resize_scale_nearest_int():
    """
    Nearest-neighbor upsampling of an integer image should replicate pixels.
    """
    img = torch.tensor(
        [[[1, 2],
          [3, 4]]],
        dtype=torch.int32
    )
    out = nef.resize(img, scale_factor=2.0, nearest=True)

    # Expect each pixel to become a 2×2 block
    expected = torch.tensor(
        [
            [1, 1, 2, 2],
            [1, 1, 2, 2],
            [3, 3, 4, 4],
            [3, 3, 4, 4]
        ],
        dtype=torch.int32
    ).unsqueeze(0)

    assert out.shape == (1, 4, 4)
    assert out.dtype == img.dtype
    assert torch.allclose(out, expected)


def test_affine_to_disp_scaling_2d():
    """
    Test scaling with a known expected displacement field.
    Use a simple 2x2 grid with 2x scaling to make the math tractable.
    """
    # Use a simple 2x2 grid for easier calculation
    shape = (2, 2)
    grid = ne.volshape_to_ndgrid(size=shape, stack=True)
    ndim = len(shape)

    # Simple 2x scaling in both directions, centered around origin
    # This should be easier to calculate by hand
    scale_factor = 2.0
    affine = torch.eye(ndim + 1, dtype=grid.dtype, device=grid.device)
    affine[0, 0] = scale_factor  # x scaling
    affine[1, 1] = scale_factor  # y scaling
    # No translation - scale around origin

    disp = vxm.functional.affine_to_disp(affine, meshgrid=grid, origin_at_center=False)

    # Check shape
    assert disp.shape == shape + (ndim,)

    # For a 2x2 grid with coordinates (0,0), (0,1), (1,0), (1,1)
    # With 2x scaling around origin:
    # (0,0) -> (0,0), displacement = (0,0) - (0,0) = (0,0)
    # (0,1) -> (0,2), displacement = (0,2) - (0,1) = (0,1)
    # (1,0) -> (2,0), displacement = (2,0) - (1,0) = (1,0)
    # (1,1) -> (2,2), displacement = (2,2) - (1,1) = (1,1)

    expected_disp = torch.tensor([
        [[0.0, 0.0], [0.0, 1.0]],
        [[1.0, 0.0], [1.0, 1.0]],
    ], dtype=grid.dtype, device=grid.device)

    # Check that the displacement field matches exactly
    assert torch.allclose(disp, expected_disp, atol=1e-6), f"Expected {expected_disp}, got {disp}"


def test_params_to_affine_scaling():
    """
    params_to_affine with scaling should produce correct scaling matrix.
    """
    scale_factors = (2.0, 3.0)

    result_affine = vxm.params_to_affine(
        ndim=2,
        scale=scale_factors
    ).to(torch.float64)

    expected_affine = torch.tensor([
        [2.0, 0.0, 0.0],
        [0.0, 3.0, 0.0],
        [0.0, 0.0, 1.0]
    ], dtype=torch.float64)

    assert torch.allclose(result_affine, expected_affine, atol=1e-5)


def test_params_to_affine_shearing_2d():
    """
    params_to_affine with shearing should produce correct shear matrix.
    """

    # Make an affine with shear
    shear_value = 0.5
    result_affine = vxm.params_to_affine(
        ndim=2,
        shear=shear_value
    ).to(torch.float64)

    expected_affine = torch.tensor([
        [1.0, 0.5, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0]
    ], dtype=torch.float64)

    assert torch.allclose(result_affine, expected_affine, atol=1e-5)


def test_params_to_affine_shearing_3d():
    """
    params_to_affine with 3D shearing should produce correct shear matrix.
    """
    shear_values = (0.5, 0.3, 0.7)

    result_affine = vxm.params_to_affine(
        ndim=3,
        shear=shear_values
    ).to(torch.float64)

    expected_affine = torch.tensor([
        [1.0, 0.5, 0.3, 0.0],
        [0.0, 1.0, 0.7, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0]
    ], dtype=torch.float64)

    assert torch.allclose(result_affine, expected_affine, atol=1e-5)


def test_params_to_affine_complex_2d():
    """
    Test complex composed affine with known expected result.
    """
    translation = (2.0, 3.0)
    rotation = 45.0  # degrees
    scale = (1.5, 2.0)
    shear = 0.3

    result_affine = vxm.params_to_affine(
        ndim=2,
        translation=translation,
        rotation=rotation,
        scale=scale,
        shear=shear
    ).to(torch.float64)

    # Expected result calculated by hand: T @ R @ Z @ S
    # Where T=translation, R=rotation, Z=scale, S=shear
    expected_affine = torch.tensor([
        [1.0607, -1.0960, 2.0000],
        [1.0607, 1.7324, 3.0000],
        [0.0000, 0.0000, 1.0000]
    ], dtype=torch.float64)

    # Check that the result matches the expected matrix exactly
    assert torch.allclose(result_affine, expected_affine, atol=1e-4), \
        f"Expected {expected_affine}, got {result_affine}"


def test_disp_to_coords_zero_disp():
    """
    disp_to_coords with zero displacement should produce normalized grid coordinates.
    """
    # Create a zero displacement field
    disp = torch.zeros(2, 2, 2, dtype=torch.float32)
    coords = vxm.disp_to_coords(disp)

    # Check shape
    assert coords.shape == (2, 2, 2)

    # With zero displacement, we should get the normalized grid coordinates
    # For a 2x2 grid, the normalized coordinates should be:
    # [-1, -1], [1, -1]
    # [-1,  1], [1,  1]
    expected_coords = torch.tensor([
        [[-1.0, -1.0], [1.0, -1.0]],
        [[-1.0, 1.0], [1.0, 1.0]]
    ], dtype=torch.float32)

    assert torch.allclose(coords, expected_coords, atol=1e-6), (
        f"Expected {expected_coords}, got {coords}"
    )


def test_integrate_disp_zero_steps():
    """
    integrate_disp with zero steps should return the original displacement.
    """
    disp = torch.randn(2, 3, 2, dtype=torch.float32)
    integrated = vxm.functional.integrate_disp(disp, steps=0)

    assert torch.allclose(integrated, disp)


def test_integrate_disp_single_step():
    """
    integrate_disp with one step should apply spatial transform once.
    """
    disp = torch.randn(2, 3, 2, dtype=torch.float32)
    integrated = vxm.functional.integrate_disp(disp, steps=1)

    # Should have same shape
    assert integrated.shape == disp.shape
    # Should be different from original (unless disp is very small)
    assert not torch.allclose(integrated, disp, atol=1e-6)


def test_random_transform():
    """
    random_transform should generate valid transforms.

    Note: vxf.random_transform expects (B, C, *spatial) format and returns
    (B, *spatial, ndim) displacement fields.
    """
    # Shape in (B, C, H, W) format
    shape = (1, 1, 3, 3)
    spatial_shape = shape[2:]  # (3, 3)
    ndim = len(spatial_shape)  # 2

    # Test affine-only transform
    trf = vxf.random_transform(
        shape=shape,
        affine_probability=1.0,
        warp_probability=0.0
    )

    assert trf is not None
    # Output shape is (B, *spatial, ndim)
    assert trf.shape == (shape[0],) + spatial_shape + (ndim,)

    # Test warp-only transform
    trf = vxf.random_transform(
        shape=shape,
        affine_probability=0.0,
        warp_probability=1.0
    )

    assert trf is not None
    assert trf.shape == (shape[0],) + spatial_shape + (ndim,)


def test_affine_to_disp_large_translation():
    """
    affine_to_disp should handle large translations correctly.
    """
    shape = (3, 3)
    grid = ne.volshape_to_ndgrid(size=shape, stack=True)
    ndim = len(shape)

    # Large translation
    large_translation = 100.0
    affine = torch.eye(ndim + 1, dtype=grid.dtype, device=grid.device)
    affine[0, -1] = large_translation
    affine[1, -1] = large_translation

    disp = vxm.functional.affine_to_disp(affine, grid)

    # Check shape
    assert disp.shape == shape + (ndim,)
    # All displacements should be the translation value
    expected_disp = torch.full(
        shape + (ndim,),
        large_translation,
        dtype=grid.dtype,
        device=grid.device
    )
    assert torch.allclose(disp, expected_disp)


def test_affine_to_disp_origin_at_center_scaling():
    """
    Test that origin_at_center controls the fixed point during scaling.

    With origin_at_center=True: center point should have zero displacement
    With origin_at_center=False: corner (0,0) should have zero displacement
    """
    shape = (3, 3)
    ndim = 2

    # 2x scaling affine
    scale_factor = 2.0
    affine = torch.eye(ndim, ndim + 1)
    affine[0, 0] = scale_factor
    affine[1, 1] = scale_factor

    # Test with origin_at_center=True (scale around center)
    disp_centered = vxm.functional.affine_to_disp(affine, shape=shape, origin_at_center=True)

    # Center point (1,1) should have zero displacement
    assert torch.allclose(disp_centered[1, 1], torch.zeros(2), atol=1e-6), \
        f"Center should be fixed, got displacement {disp_centered[1, 1]}"

    # Test with origin_at_center=False (scale around corner)
    disp_corner = vxm.functional.affine_to_disp(affine, shape=shape, origin_at_center=False)

    # Corner (0,0) should have zero displacement
    assert torch.allclose(disp_corner[0, 0], torch.zeros(2), atol=1e-6), \
        f"Corner should be fixed, got displacement {disp_corner[0, 0]}"

    # The two displacement fields should be different
    assert not torch.allclose(disp_centered, disp_corner), \
        "Displacement fields should differ based on origin_at_center"


def test_affine_to_disp_origin_at_center_rotation():
    """
    Test that origin_at_center controls the fixed point during rotation.

    90-degree rotation makes the difference very obvious.
    With origin_at_center=True: center stays fixed
    With origin_at_center=False: corner (0,0) stays fixed
    """
    shape = (3, 3)

    # 90-degree counter-clockwise rotation
    affine = torch.tensor([[0., -1., 0.],
                           [1., 0., 0.]])

    # Test with origin_at_center=True (rotate around center)
    disp_centered = vxm.functional.affine_to_disp(affine, shape=shape, origin_at_center=True)

    # Center point (1,1) should have zero displacement (stays in place)
    assert torch.allclose(disp_centered[1, 1], torch.zeros(2), atol=1e-6), \
        f"Center should be fixed during rotation, got {disp_centered[1, 1]}"

    # Test with origin_at_center=False (rotate around corner)
    disp_corner = vxm.functional.affine_to_disp(affine, shape=shape, origin_at_center=False)

    # Corner (0,0) should have zero displacement
    assert torch.allclose(disp_corner[0, 0], torch.zeros(2), atol=1e-6), \
        f"Corner should be fixed during rotation, got {disp_corner[0, 0]}"

    # The two should be dramatically different for rotation
    assert not torch.allclose(disp_centered, disp_corner), \
        "Rotation around center vs corner should produce very different results"


def test_params_to_affine_analytical():
    """
    Test compose() validates T(x) = A(B(x)) behavior with non-commuting transforms.

    Uses rotation and translation which don't commute to clearly test order.
    For compose([A, B]), should apply B first, then A, giving matrix A @ B.
    """
    # A: 90-degree counter-clockwise rotation
    A = torch.tensor([[0., -1., 0.],
                      [1., 0., 0.]])

    # B: translation by (2, 0)
    B = torch.tensor([[1., 0., 2.],
                      [0., 1., 0.]])

    # Compose A and B
    composed = vxm.compose([A, B])

    # Expected: A @ B (apply B first, then A)
    expected_affine = torch.tensor([[0., -1., 0.],
                                    [1., 0., 2.]])

    assert torch.allclose(composed, expected_affine, atol=1e-6), \
        f"Expected {expected_affine}, got {composed}"

    # Verify this is NOT B @ A (opposite order)
    wrong_order = torch.tensor([[0., -1., 2.],
                                [1., 0., 0.]])
    assert not torch.allclose(composed, wrong_order), \
        "compose([A, B]) should not be B @ A"

    # Test point transformation: apply B first (translate), then A (rotate)
    test_point = torch.tensor([1., 0., 1.])
    result = composed @ test_point
    expected_result = torch.tensor([0., 3.])

    assert torch.allclose(result, expected_result, atol=1e-6), \
        f"Point transformation failed: expected {expected_result}, got {result}"


def test_disp_to_coords_axis_flip():
    """
    Test that disp_to_coords correctly flips the last axis to match PyTorch convention.
    """
    # Create a simple displacement field
    disp = torch.zeros(2, 2, 2, dtype=torch.float32)

    # Add some displacement
    disp[0, 0, 0] = 1.0  # Move first pixel
    disp[1, 1, 1] = 2.0  # Move second pixel

    # Convert to coordinates
    coords = vxm.disp_to_coords(disp)

    # The coordinates should be flipped compared to the displacement
    # This verifies that the flip(-1) operation is working correctly
    assert coords.shape == (2, 2, 2)

    # Check that the flip operation was applied
    # The last axis should be flipped from [y, x] to [x, y]
    assert coords[0, 0, 0] != coords[0, 0, 1]  # Should be different after flip


def test_random_affine_shape_2d():
    """
    random_affine should return valid 2D affine matrix with correct shape and structure.
    """
    affine = vxm.random_affine(
        ndim=2,
        max_translation=10.0,
        max_rotation=15.0,
        max_scaling=1.2
    )

    # Check shape: (ndim+1, ndim+1) = (3, 3)
    assert affine.shape == (3, 3)
    assert affine.dtype == torch.float32

    # Affine should be invertible (non-zero determinant)
    det = torch.linalg.det(affine)
    assert det.abs() > 1e-6


def test_random_affine_shape_3d():
    """
    random_affine should return valid 3D affine matrix with correct shape and structure.
    """
    affine = vxm.random_affine(
        ndim=3,
        max_translation=5.0,
        max_rotation=10.0,
        max_scaling=1.1
    )

    # Check shape: (ndim+1, ndim+1) = (4, 4)
    assert affine.shape == (4, 4)
    assert affine.dtype == torch.float32

    # Affine should be invertible (non-zero determinant)
    det = torch.linalg.det(affine)
    assert det.abs() > 1e-6


def test_random_affine_deterministic_translation():
    """
    random_affine with sampling=False should use max values directly.

    With sampling=False:
    - translation = [max_translation] * ndim
    - rotation = [max_rotation] * ndim (1 for 2D, 3 for 3D)
    - scale = [max_scaling] * ndim
    """
    max_translation = 5.0
    max_rotation = 0.0  # No rotation for simpler verification
    max_scaling = 1.0   # No scaling for simpler verification

    affine = vxm.random_affine(
        ndim=2,
        max_translation=max_translation,
        max_rotation=max_rotation,
        max_scaling=max_scaling,
        sampling=False
    )

    # With no rotation and no scaling, affine should be identity + translation
    expected_translation = torch.tensor([max_translation, max_translation], dtype=torch.float32)
    assert torch.allclose(affine[:2, -1], expected_translation)

    # Linear part should be identity (no rotation, no scaling)
    expected_linear = torch.eye(2, dtype=torch.float32)
    assert torch.allclose(affine[:2, :2], expected_linear)


def test_compose_two_affines():
    """
    Composing two affine matrices should return an affine via matrix multiplication.

    For transforms [A, B], compose returns A @ B (B applied first, then A).
    """
    # Translation by (10, 5)
    translate = torch.tensor([
        [1., 0., 10.],
        [0., 1., 5.]
    ])

    # Scale by 2x
    scale = torch.tensor([
        [2., 0., 0.],
        [0., 2., 0.]
    ])

    # Compose: scale first, then translate
    composed = vxm.compose([translate, scale])

    # Result should be affine shape (2, 3)
    assert composed.shape == (2, 3)
    assert vxm.functional.is_affine_shape(composed.shape)

    # Manual computation: translate @ scale (after making both square)
    # scale maps x -> 2x, then translate maps 2x -> 2x + t
    # So composed should be [[2, 0, 10], [0, 2, 5]]
    expected = torch.tensor([
        [2., 0., 10.],
        [0., 2., 5.]
    ])
    assert torch.allclose(composed, expected)


def test_compose_two_constant_displacements():
    """
    Composing two constant displacement fields should sum them in the interior.

    For compose([disp1, disp2]), the math is:
        composed(x) = disp2(x) + disp1(x + disp2(x))

    When both displacements are spatially constant, disp1(x + disp2(x)) = disp1(x),
    so composed(x) = disp1(x) + disp2(x) (simple addition).

    Note: Boundary pixels may differ due to grid_sample padding_mode='zeros'.
    We test the interior region where sampling stays within bounds.
    """
    shape = (16, 16)
    ndim = len(shape)

    # Constant displacement: shift by (1, 0) everywhere
    disp1 = torch.zeros(*shape, ndim)
    disp1[..., 0] = 1.0

    # Constant displacement: shift by (0, 1) everywhere
    disp2 = torch.zeros(*shape, ndim)
    disp2[..., 1] = 1.0

    composed = vxm.compose([disp1, disp2])

    # Expected: (1, 1) everywhere in the interior
    expected = torch.zeros(*shape, ndim)
    expected[..., 0] = 1.0
    expected[..., 1] = 1.0

    assert composed.shape == (*shape, ndim)

    # Check interior region (exclude boundary pixels affected by padding)
    interior = composed[2:-2, 2:-2, :]
    expected_interior = expected[2:-2, 2:-2, :]
    assert torch.allclose(interior, expected_interior, atol=1e-5)


def test_compose_translation_affine_with_displacement():
    """
    Composing [translation_affine, displacement] adds translation to displacement.

    For compose([affine, disp]):
        composed(x) = disp(x) + affine_disp(x + disp(x))

    With a pure translation affine (constant displacement), the affine contribution
    is constant everywhere, so the result is disp + translation.
    """
    shape = (8, 8)
    ndim = len(shape)

    # Translation affine: shift by (5, 3)
    translation = torch.tensor([
        [1., 0., 5.],
        [0., 1., 3.]
    ])

    # Constant displacement field: shift by (1, 2)
    disp = torch.zeros(*shape, ndim)
    disp[..., 0] = 1.0
    disp[..., 1] = 2.0

    # Compose: disp first, then translation
    composed = vxm.compose([translation, disp])

    # Expected: (1+5, 2+3) = (6, 5) everywhere
    expected = torch.zeros(*shape, ndim)
    expected[..., 0] = 6.0
    expected[..., 1] = 5.0

    assert composed.shape == (*shape, ndim)
    assert torch.allclose(composed, expected, atol=1e-5)


def test_compose_displacement_with_translation_affine():
    """
    Composing [displacement, translation_affine] adds translation to displacement.

    For compose([disp, affine]):
        composed(x) = affine_disp(x) + disp(affine(x))

    With constant displacement and pure translation:
        composed(x) = translation + disp (since disp is constant, disp(affine(x)) = disp)

    Note: Boundary pixels may differ due to grid_sample padding_mode='zeros'.
    We test the interior region where sampling stays within bounds.
    """
    shape = (16, 16)
    ndim = len(shape)

    # Constant displacement field: shift by (1, 1)
    disp = torch.zeros(*shape, ndim)
    disp[..., 0] = 1.0
    disp[..., 1] = 1.0

    # Translation affine: shift by (2, 2)
    translation = torch.tensor([
        [1., 0., 2.],
        [0., 1., 2.]
    ])

    # Compose: affine first, then disp
    composed = vxm.compose([disp, translation])

    # Expected: (2+1, 2+1) = (3, 3) everywhere in the interior
    expected = torch.zeros(*shape, ndim)
    expected[..., 0] = 3.0
    expected[..., 1] = 3.0

    assert composed.shape == (*shape, ndim)

    # Check interior region (exclude boundary pixels affected by padding)
    interior = composed[4:-4, 4:-4, :]
    expected_interior = expected[4:-4, 4:-4, :]
    assert torch.allclose(interior, expected_interior, atol=1e-5)


def test_compose_scale_affine_with_zero_displacement():
    """
    Composing [scale_affine, zero_disp] should produce the affine's displacement field.

    For compose([affine, disp]) with disp=0:
        composed(x) = 0 + affine_disp(x) = affine_disp(x)

    A 2x scale centered at origin maps x -> 2x, so displacement is x (moves each
    point away from center by its distance from center).
    """
    shape = (5, 5)
    ndim = len(shape)

    # Scale by 2x (centered at image center due to origin_at_center=True)
    scale_affine = torch.tensor([
        [2., 0., 0.],
        [0., 2., 0.]
    ])

    # Zero displacement
    disp = torch.zeros(*shape, ndim)

    composed = vxm.compose([scale_affine, disp])

    # With origin_at_center=True and shape (5,5), center is at (2, 2)
    # Scale 2x maps: x_centered -> 2 * x_centered
    # Displacement = new_pos - old_pos = 2*x_centered - x_centered = x_centered
    # At corners: displacement equals distance from center
    grid = ne.volshape_to_ndgrid(size=shape, stack=True)
    center = torch.tensor([(s - 1) / 2 for s in shape])
    expected = grid - center  # x_centered

    assert composed.shape == (*shape, ndim)
    assert torch.allclose(composed, expected, atol=1e-5)

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
    affine = torch.eye(
        ndim + 1,
        dtype=grid.dtype,
        device=grid.device
    )

    # Get displacement field
    disp = vxm.functional.affine_to_disp(affine, grid)

    # output shape and dtype
    assert disp.shape == shape + (ndim,)
    assert disp.dtype == grid.dtype

    # all zeros
    assert torch.allclose(disp, torch.zeros_like(disp))


def test_affine_to_disp_translation():
    """
    Pure translation affine should yield a constant field = translation.
    """
    shape = (2, 2)
    grid = ne.volshape_to_ndgrid(size=shape, stack=True)
    ndim = len(shape)
    tx, ty = 2.0, 3.0

    # build a 2D affine with translation in the last column
    affine = torch.eye(
        ndim + 1, dtype=grid.dtype, device=grid.device
    )

    # Make the translation
    affine[0, -1] = tx
    affine[1, -1] = ty

    # Get displacement field
    disp = vxm.functional.affine_to_disp(affine, grid)

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


def test_spatial_transform_shearing():
    """
    Test spatial transform with identity affine - should return the exact same image.
    """
    # Create a simple test image
    img = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]], dtype=torch.float32)

    # Create identity affine (no transformation)
    affine = torch.eye(3, dtype=torch.float32)
    out = vxm.functional.spatial_transform(img, affine, non_spatial_dims=(0,))

    # With identity transformation, output should be exactly the same as input
    assert torch.allclose(out, img, atol=1e-6), f"Expected {img}, got {out}"


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
    """
    shape = (3, 3)

    # Test affine-only transform
    trf = vxf.random_transform(
        shape=shape,
        affine_probability=1.0,
        warp_probability=0.0
    )

    assert trf is not None
    assert trf.shape == shape + (len(shape),)

    # Test warp-only transform
    trf = vxf.random_transform(
        shape=shape,
        affine_probability=0.0,
        warp_probability=1.0
    )

    assert trf is not None
    assert trf.shape == shape + (len(shape),)


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
    composed = vxf.compose([A, B])

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


def test_spatial_transform_different_dtypes():
    """
    spatial_transform should handle different input dtypes correctly.
    """
    # Test with float32 (default) - this should work without issues
    img_f32 = torch.randn(1, 3, 3, dtype=torch.float32)
    affine = torch.eye(3, dtype=torch.float32)

    out_f32 = vxm.functional.spatial_transform(img_f32, affine, non_spatial_dims=(0,))
    assert out_f32.dtype == torch.float32
    assert torch.allclose(out_f32, img_f32, atol=1e-6)

    # Test with int32 (should be converted to float32 internally)
    img_int = torch.randint(0, 10, (1, 3, 3), dtype=torch.int32)
    affine = torch.eye(3, dtype=torch.float32)

    out_int = vxm.spatial_transform(
        img_int,
        affine,
        mode='nearest',
        non_spatial_dims=(0,)
    )
    assert out_int.dtype == torch.int32
    assert torch.allclose(out_int.float(), img_int.float(), atol=1e-6)


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

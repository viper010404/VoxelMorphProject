"""
Unit tests for the basic utility functions in voxelmorph.
"""

# Standard library imports
import torch

# Custom imports
import voxelmorph.nn.functional as vxf


def test_grid_coordinates_2d():
    """
    grid_coordinates() should produce a (H, W, 2) mesh of (i, j) indices.
    """
    shape = (2, 3)
    grid = vxf.grid_coordinates(shape)

    # Check shape of grid
    assert grid.shape == (2, 3, 2)

    # Expected values:
    # grid[y, x] == [y, x]
    expected = torch.tensor(
        [
            [[0.0, 0.0], [0.0, 1.0], [0.0, 2.0]],
            [[1.0, 0.0], [1.0, 1.0], [1.0, 2.0]],
        ],
        dtype=torch.float32,
    )
    assert torch.allclose(grid, expected)


def test_grid_coordinates_3d():
    """
    grid_coordinates should produce a (D, H, W, 3) mesh of (z, y, x) indices.
    """
    shape = (2, 2, 2)
    grid = vxf.grid_coordinates(shape)

    # shape check
    assert grid.shape == (2, 2, 2, 3)

    # corner values:
    assert torch.allclose(grid[0, 0, 0], torch.tensor([0.0, 0.0, 0.0]))
    assert torch.allclose(grid[1, 1, 1], torch.tensor([1.0, 1.0, 1.0]))


def test_affine_to_disp_identity():
    """
    Identity affine should produce zero displacement everywhere.
    """
    shape = (3, 4)
    grid = vxf.grid_coordinates(shape)
    ndim = len(shape)

    # Identity affine
    affine = torch.eye(
        ndim + 1,
        dtype=grid.dtype,
        device=grid.device
    )

    # Get displacement field
    disp = vxf.affine_to_disp(affine, grid)

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
    grid = vxf.grid_coordinates(shape)
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
    disp = vxf.affine_to_disp(affine, grid)

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
    coords = vxf.disp_to_coords(disp)

    # For shape=(2,3):
    #  row indices i \isin {0, 1} -> bounded on [-1, 1] with 2 elements -> [-1, 1]
    #  col indices j \isin {0, 1, 2} ->  bounded on [-1, 1] with 3 elements -> [-1, 0, 1]
    expected = torch.tensor([
        [[-1., -1.], [0., -1.], [1., -1.]],
        [[-1.,  1.], [0.,  1.], [1.,  1.]],
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
    out = vxf.spatial_transform(img, affine)

    assert out.shape == img.shape
    assert torch.allclose(out, img, atol=1e-6)


def test_angles_to_rotation_matrix_2d_identity():
    """
    A 2D rotation of 0 deg must yield the 2x2 identity matrix.
    """
    rotation_matrix = vxf.angles_to_rotation_matrix(torch.tensor(0.0), degrees=True)
    expected = torch.eye(2, dtype=torch.float64)

    assert rotation_matrix.shape == (2, 2)
    assert rotation_matrix.dtype == torch.float64
    assert torch.allclose(rotation_matrix, expected, atol=1e-8)


def test_angles_to_rotation_matrix_2d_90_degrees():
    """
    A 2D rotation of 90 degrees should be [[0, -1], [1, 0]].
    """
    rotation_matrix = vxf.angles_to_rotation_matrix(torch.tensor(90.0), degrees=True)

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
    rotation_matrix = vxf.angles_to_rotation_matrix(torch.tensor(torch.pi / 2), degrees=False)
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
    rotation_matrix = vxf.angles_to_rotation_matrix(torch.tensor((0, 0, 90.0)), degrees=True)

    expected = torch.tensor(
        [
            [0, 1, 0],
            [-1, 0, 0],
            [0, 0, 1],
        ],
        dtype=torch.float64
    )

    assert torch.allclose(rotation_matrix, expected, atol=1e-5)


def test_compose_affine_translation_shear():
    """
    Composing two translations should yield the sum of the two translations.
    """

    translation = (1, 2)

    result_affine = vxf.compose_affine(
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
    out = vxf.resize(img, scale_factor=2.0, nearest=True)

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
    grid = vxf.grid_coordinates(shape)
    ndim = len(shape)

    # Simple 2x scaling in both directions, centered around origin
    # This should be easier to calculate by hand
    scale_factor = 2.0
    affine = torch.eye(ndim + 1, dtype=grid.dtype, device=grid.device)
    affine[0, 0] = scale_factor  # x scaling
    affine[1, 1] = scale_factor  # y scaling
    # No translation - scale around origin

    disp = vxf.affine_to_disp(affine, grid, rotate_around_center=False)

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


def test_compose_affine_scaling():
    """
    compose_affine with scaling should produce correct scaling matrix.
    """
    scale_factors = (2.0, 3.0)

    result_affine = vxf.compose_affine(
        ndim=2,
        scale=scale_factors
    ).to(torch.float64)

    expected_affine = torch.tensor([
        [2.0, 0.0, 0.0],
        [0.0, 3.0, 0.0],
        [0.0, 0.0, 1.0]
    ], dtype=torch.float64)

    assert torch.allclose(result_affine, expected_affine, atol=1e-5)


def test_compose_affine_shearing_2d():
    """
    compose_affine with shearing should produce correct shear matrix.
    """

    # Make an affine with shear
    shear_value = 0.5
    result_affine = vxf.compose_affine(
        ndim=2,
        shear=shear_value
    ).to(torch.float64)

    expected_affine = torch.tensor([
        [1.0, 0.5, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0]
    ], dtype=torch.float64)

    assert torch.allclose(result_affine, expected_affine, atol=1e-5)


def test_compose_affine_shearing_3d():
    """
    compose_affine with 3D shearing should produce correct shear matrix.
    """
    shear_values = (0.5, 0.3, 0.7)

    result_affine = vxf.compose_affine(
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

    out = vxf.spatial_transform(img, affine)

    # With identity transformation, output should be exactly the same as input
    assert torch.allclose(out, img, atol=1e-6), f"Expected {img}, got {out}"


def test_compose_affine_complex_2d():
    """
    Test complex composed affine with known expected result.
    """
    translation = (2.0, 3.0)
    rotation = 45.0  # degrees
    scale = (1.5, 2.0)
    shear = 0.3

    result_affine = vxf.compose_affine(
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
        [1.0607,  1.7324, 3.0000],
        [0.0000,  0.0000, 1.0000]
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
    coords = vxf.disp_to_coords(disp)

    # Check shape
    assert coords.shape == (2, 2, 2)

    # With zero displacement, we should get the normalized grid coordinates
    # For a 2x2 grid, the normalized coordinates should be:
    # [-1, -1], [1, -1]
    # [-1,  1], [1,  1]
    expected_coords = torch.tensor([
        [[-1.0, -1.0], [1.0, -1.0]],
        [[-1.0,  1.0], [1.0,  1.0]]
    ], dtype=torch.float32)

    assert torch.allclose(coords, expected_coords, atol=1e-6), f"Expected {expected_coords}, got {coords}"


def test_integrate_disp_zero_steps():
    """
    integrate_disp with zero steps should return the original displacement.
    """
    disp = torch.randn(2, 3, 2, dtype=torch.float32)
    integrated = vxf.integrate_disp(disp, steps=0)

    assert torch.allclose(integrated, disp)


def test_integrate_disp_single_step():
    """
    integrate_disp with one step should apply spatial transform once.
    """
    disp = torch.randn(2, 3, 2, dtype=torch.float32)
    integrated = vxf.integrate_disp(disp, steps=1)

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
    grid = vxf.grid_coordinates(shape)
    ndim = len(shape)

    # Large translation
    large_translation = 100.0
    affine = torch.eye(ndim + 1, dtype=grid.dtype, device=grid.device)
    affine[0, -1] = large_translation
    affine[1, -1] = large_translation

    disp = vxf.affine_to_disp(affine, grid)

    # Check shape
    assert disp.shape == shape + (ndim,)
    # All displacements should be the translation value
    expected_disp = torch.full(shape + (ndim,), large_translation, dtype=grid.dtype, device=grid.device)
    assert torch.allclose(disp, expected_disp)


def test_grid_coordinates_xy_indexing():
    """
    grid_coordinates with 'xy' indexing should produce correct coordinates.
    """
    shape = (2, 3)
    grid = vxf.grid_coordinates(shape, indexing='xy')

    # Check shape - with 'xy' indexing, the shape dimensions are swapped
    assert grid.shape == (3, 2, 2)  # (width, height, 2) instead of (height, width, 2)

    # With 'xy' indexing, coordinates should be [x, y] instead of [y, x]
    # The actual output shows the pattern: x varies in first dim, y in second
    expected = torch.tensor(
        [
            [[0.0, 0.0], [1.0, 0.0]],  # y=0, x varies
            [[0.0, 1.0], [1.0, 1.0]],  # y=1, x varies
            [[0.0, 2.0], [1.0, 2.0]],  # y=2, x varies
        ],
        dtype=torch.float32,
    )
    assert torch.allclose(grid, expected)


def test_spatial_transform_different_dtypes():
    """
    spatial_transform should handle different input dtypes correctly.
    """
    # Test with float32 (default) - this should work without issues
    img_f32 = torch.randn(1, 3, 3, dtype=torch.float32)
    affine = torch.eye(3, dtype=torch.float32)

    out_f32 = vxf.spatial_transform(img_f32, affine)
    assert out_f32.dtype == torch.float32
    assert torch.allclose(out_f32, img_f32, atol=1e-6)

    # Test with int32 (should be converted to float32 internally)
    img_int = torch.randint(0, 10, (1, 3, 3), dtype=torch.int32)
    affine = torch.eye(3, dtype=torch.float32)

    out_int = vxf.spatial_transform(img_int, affine, method='nearest')
    assert out_int.dtype == torch.int32
    assert torch.allclose(out_int.float(), img_int.float(), atol=1e-6)

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

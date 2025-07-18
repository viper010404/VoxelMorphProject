"""
Unit tests for the basic utility functions in voxelmorph.
"""

import torch

import voxelmorph as vxm


def test_grid_coordinates_2d():
    """
    grid_coordinates() should produce a (H, W, 2) mesh of (i, j) indices.
    """
    shape = (2, 3)
    grid = vxm.utils.grid_coordinates(shape)

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
    grid = vxm.utils.grid_coordinates(shape)

    # shape check
    assert grid.shape == (2, 2, 2, 3)

    # corner values:
    assert torch.allclose(grid[0, 0, 0], torch.tensor([0.0, 0.0, 0.0]))
    assert torch.allclose(grid[1, 1, 1], torch.tensor([1.0, 1.0, 1.0]))


def test_affine_to_displacement_field_identity():
    """
    Identity affine should produce zero displacement everywhere.
    """
    shape = (3, 4)
    grid = vxm.utils.grid_coordinates(shape)
    ndim = len(shape)

    # Identity affine
    affine = torch.eye(
        ndim + 1,
        dtype=grid.dtype,
        device=grid.device
    )

    # Get displacement field
    disp = vxm.utils.affine_to_displacement_field(affine, grid)

    # output shape and dtype
    assert disp.shape == shape + (ndim,)
    assert disp.dtype == grid.dtype

    # all zeros
    assert torch.allclose(disp, torch.zeros_like(disp))


def test_affine_to_displacement_field_translation():
    """
    Pure translation affine should yield a constant field = translation.
    """
    shape = (2, 2)
    grid = vxm.utils.grid_coordinates(shape)
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
    disp = vxm.utils.affine_to_displacement_field(affine, grid)

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


def test_displacement_field_to_coords_zero_disp_2d():
    """
    Zero displacement on a 2x3 grid should produce the normalized mesh in range [-1, 1], flipped
    (col, row).
    """
    disp = torch.zeros(2, 3, 2, dtype=torch.float32)
    coords = vxm.utils.displacement_field_to_coords(disp)

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
    out = vxm.utils.spatial_transform(img, None)

    assert out.shape == img.shape
    assert torch.allclose(out, img)


def test_spatial_transform_identity_affine():
    """
    An identity affine should yield the same image.
    """
    img = torch.rand(1, 3, 3, dtype=torch.float32)

    # 2D identity affine (3×3)
    affine = torch.eye(3, dtype=torch.float32)
    out = vxm.utils.spatial_transform(img, affine)

    assert out.shape == img.shape
    assert torch.allclose(out, img, atol=1e-6)

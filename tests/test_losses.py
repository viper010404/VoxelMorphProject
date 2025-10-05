"""
Tests for VoxelMorph loss functions.

These tests define behavioral expectations for loss functions.
"""

import torch
import voxelmorph.nn.losses as vxm_losses


def test_grad_l1_penalty__uniform_field__zero_loss():
    """
    Test l1 penalty computation on uniform displacement field.

    Uniform fields have zero gradients everywhere, so gradient loss should be zero
    regardless of the uniform value. This tests the fundamental gradient computation.
    """
    torch.manual_seed(42)
    grad_loss = vxm_losses.Grad(penalty='l1')

    batch_size = 2
    uniform_field = torch.ones(batch_size, 2, 8, 8) * 0.5

    loss = grad_loss.loss(uniform_field)
    assert loss == 0.0, f"Expected Grad loss to be 0.0 for uniform field, got {loss}"


def test_grad_l2_penalty__linear_gradient__expected_magnitude():
    """
    Test l2 penalty computation on known linear gradient field.

    Linear gradients have constant, predictable finite differences.
    This validates the core gradient computation and l2 penalty application.
    """
    torch.manual_seed(42)
    grad_loss = vxm_losses.Grad(penalty='l2')

    # Create 2D field w/ linear grads
    batch_size = 1
    H, W = 4, 4
    field = torch.zeros(batch_size, 2, H, W)

    # Linearly increase: channel 0 increases with row, channel 1 increases with col
    for i in range(H):
        for j in range(W):
            field[0, 0, i, j] = float(i)
            field[0, 1, i, j] = float(j)

    loss = grad_loss.loss(field)
    expected_loss = torch.tensor(0.5)
    torch.testing.assert_close(loss, expected_loss, atol=1e-6, rtol=1e-6)


def test_grad_l1_vs_l2__same_field__different_penalties():
    """
    Test that l1 and l2 penalties produce different results on the same field.

    This verifies the penalty mode selection works correctly and produces
    the expected mathematical differences between l1 and l2 norms.
    """
    torch.manual_seed(42)

    # Create field with mixed positive/negative gradients
    batch_size = 1
    field = torch.zeros(batch_size, 2, 6, 6)
    field[0, 0, :, :] = torch.arange(6).float().unsqueeze(0) * 2
    field[0, 1, :, :] = torch.arange(6).float().unsqueeze(1) * (-1)

    grad_l1 = vxm_losses.Grad(penalty='l1')
    grad_l2 = vxm_losses.Grad(penalty='l2')

    loss_l1 = grad_l1.loss(field)
    loss_l2 = grad_l2.loss(field)

    # Height diffs: (0*30 + 1*30) / 60 = 0.5
    # Width diffs: (2*30 + 0*30) / 60 = 1.0
    # L1: (0.5 + 1.0) / 2 = 0.75
    # L2: (0.5 + 2.0) / 2 = 1.25
    expected_l1 = torch.tensor(0.75)
    expected_l2 = torch.tensor(1.25)

    torch.testing.assert_close(loss_l1, expected_l1, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(loss_l2, expected_l2, atol=1e-6, rtol=1e-6)

    assert not torch.allclose(loss_l1, loss_l2, atol=1e-6)


def test_grad_loss_mult__scaling_factor__multiplied_output():
    """
    Test loss_mult parameter correctly scales the final loss output.

    Validates that loss_mult acts as a simple multiplicative factor on the
    computed gradient loss, useful for downsampled displacement fields.
    """
    torch.manual_seed(42)

    batch_size = 1
    field = torch.randn(batch_size, 2, 6, 6) * 0.1

    grad_base = vxm_losses.Grad(penalty='l2')
    loss_base = grad_base.loss(field)

    scale_factor = 3.0
    grad_scaled = vxm_losses.Grad(penalty='l2', loss_mult=scale_factor)
    loss_scaled = grad_scaled.loss(field)

    expected_scaled = loss_base * scale_factor
    torch.testing.assert_close(loss_scaled, expected_scaled, atol=1e-6, rtol=1e-6)

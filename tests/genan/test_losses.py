"""CPU-only unit tests for roto.genan.losses."""

import os
import sys

import torch

_GENAN_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "genan")
sys.path.insert(0, _GENAN_DIR)

from losses import torque_loss  # noqa: E402


def test_torque_loss_zero_when_equal():
    pred = torch.randn(10, 16)
    loss = torque_loss(pred, pred.clone())
    assert torch.isclose(loss, torch.tensor(0.0), atol=1e-6)


def test_torque_loss_matches_manual_mse():
    pred = torch.randn(10, 16)
    label = torch.randn(10, 16)
    loss = torque_loss(pred, label)
    expected = ((pred - label) ** 2).mean()
    assert torch.isclose(loss, expected, atol=1e-6)


def test_torque_loss_broadcasts_label_across_ensemble_dim():
    pred = torch.randn(5, 10, 16)  # (ensemble_size, batch, num_joints)
    label = torch.randn(10, 16)  # (batch, num_joints), no ensemble dim
    loss = torque_loss(pred, label)
    expected = ((pred - label.unsqueeze(0)) ** 2).mean()
    assert torch.isclose(loss, expected, atol=1e-6)


def test_torque_loss_backpropagates():
    pred = torch.randn(10, 16, requires_grad=True)
    label = torch.randn(10, 16)
    loss = torque_loss(pred, label)
    loss.backward()
    assert pred.grad is not None
    assert torch.isfinite(pred.grad).all()

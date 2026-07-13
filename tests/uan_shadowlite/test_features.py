"""CPU-only unit tests for roto.tasks.uan_shadowlite.features."""

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "roto", "tasks", "uan_shadowlite"))

from features import DEFAULT_FEATURES, FeatureBuilder, FeatureContext  # noqa: E402

NUM_ENVS = 3
NUM_JOINTS = 16
NUM_CONTROL = 13


def _ctx():
    return FeatureContext(
        joint_pos=torch.linspace(-1, 1, NUM_ENVS * NUM_JOINTS).reshape(NUM_ENVS, NUM_JOINTS),
        joint_vel=torch.ones(NUM_ENVS, NUM_CONTROL) * 0.5,
        joint_pos_error=torch.ones(NUM_ENVS, NUM_JOINTS) * 0.1,
        action=torch.ones(NUM_ENVS, NUM_CONTROL) * 0.2,
        last_residual=torch.ones(NUM_ENVS, NUM_JOINTS) * 0.05,
        real_vel_t=torch.ones(NUM_ENVS, NUM_JOINTS) * 3.0,
        traj_phase=torch.tensor([0.0, 0.25, 0.5]),
    )


def test_default_features_are_58_dim_with_correct_per_feature_widths():
    assert DEFAULT_FEATURES == ["joint_pos", "joint_vel", "joint_pos_error", "action"]
    fb = FeatureBuilder(None, num_joints=NUM_JOINTS, num_control=NUM_CONTROL)
    assert fb.feature_list == DEFAULT_FEATURES
    # 16 (joint_pos) + 13 (joint_vel) + 16 (joint_pos_error) + 13 (action) = 58
    assert fb.output_dim == 58


def test_default_output_is_passthrough_no_rescaling():
    """joint_pos/joint_vel/joint_pos_error/action must be passed through
    UNCHANGED -- normalization is roto's own job (upstream), not FeatureBuilder's.
    """
    fb = FeatureBuilder(None, num_joints=NUM_JOINTS, num_control=NUM_CONTROL)
    ctx = _ctx()
    out = fb.build(ctx)
    assert out.shape == (NUM_ENVS, 58)
    assert torch.allclose(out[:, 0:16], ctx.joint_pos)
    assert torch.allclose(out[:, 16:29], ctx.joint_vel)
    assert torch.allclose(out[:, 29:45], ctx.joint_pos_error)
    assert torch.allclose(out[:, 45:58], ctx.action)


def test_joint_vel_and_action_are_13_dim_not_16():
    fb = FeatureBuilder(["joint_vel"], num_joints=NUM_JOINTS, num_control=NUM_CONTROL)
    assert fb.output_dim == NUM_CONTROL
    out = fb.build(_ctx())
    assert out.shape == (NUM_ENVS, NUM_CONTROL)

    fb2 = FeatureBuilder(["action"], num_joints=NUM_JOINTS, num_control=NUM_CONTROL)
    assert fb2.output_dim == NUM_CONTROL


def test_joint_pos_and_joint_pos_error_stay_16_dim():
    fb = FeatureBuilder(["joint_pos", "joint_pos_error"], num_joints=NUM_JOINTS, num_control=NUM_CONTROL)
    assert fb.output_dim == 2 * NUM_JOINTS


def test_build_returns_correct_shape_with_mixed_widths():
    features = ["joint_pos", "joint_vel", "joint_pos_error", "action", "last_residual", "real_vel", "sin_time", "cos_time"]
    fb = FeatureBuilder(features, num_joints=NUM_JOINTS, num_control=NUM_CONTROL)
    out = fb.build(_ctx())
    expected_dim = 16 + 13 + 16 + 13 + 16 + 16 + 1 + 1  # joint_pos+joint_vel+pos_error+action+last_residual+real_vel+sin+cos
    assert fb.output_dim == expected_dim
    assert out.shape == (NUM_ENVS, expected_dim)


def test_feature_order_determines_layout():
    fb_a = FeatureBuilder(["joint_pos", "joint_pos_error"], num_joints=NUM_JOINTS, num_control=NUM_CONTROL)
    fb_b = FeatureBuilder(["joint_pos_error", "joint_pos"], num_joints=NUM_JOINTS, num_control=NUM_CONTROL)
    ctx = _ctx()
    out_a = fb_a.build(ctx)
    out_b = fb_b.build(ctx)
    assert torch.allclose(out_a[:, :NUM_JOINTS], out_b[:, NUM_JOINTS:])
    assert torch.allclose(out_a[:, NUM_JOINTS:], out_b[:, :NUM_JOINTS])


def test_unknown_feature_raises():
    with pytest.raises(ValueError):
        FeatureBuilder(["not_a_real_feature"], num_joints=NUM_JOINTS, num_control=NUM_CONTROL)


def test_empty_feature_list_falls_back_to_default():
    fb = FeatureBuilder([], num_joints=NUM_JOINTS, num_control=NUM_CONTROL)
    assert fb.feature_list == DEFAULT_FEATURES


def test_scalar_features_are_one_dim():
    fb = FeatureBuilder(["sin_time", "cos_time"], num_joints=NUM_JOINTS, num_control=NUM_CONTROL)
    ctx = _ctx()
    out = fb.build(ctx)
    assert out.shape == (NUM_ENVS, 2)
    expected_sin = torch.sin(2 * torch.pi * ctx.traj_phase)
    expected_cos = torch.cos(2 * torch.pi * ctx.traj_phase)
    assert torch.allclose(out[:, 0], expected_sin, atol=1e-6)
    assert torch.allclose(out[:, 1], expected_cos, atol=1e-6)

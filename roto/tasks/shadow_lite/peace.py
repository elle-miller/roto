# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Static pose task environment for the Shadow Lite hand.
Rewards the hand for maintaining a fixed target joint configuration.
"""

from __future__ import annotations

import torch
from collections.abc import Sequence

from isaaclab.utils import configclass

from roto.tasks.shadow_lite.shadow_lite import ShadowLiteEnv, ShadowLiteEnvCfg


@configclass
class StaticPoseCfg(ShadowLiteEnvCfg):
    """Configuration for the Shadow Lite static pose task."""

    # How close joints must be to target (radians) to earn full reward
    pose_threshold: float = 0.05

    # Weight of the pose-matching reward
    pose_reward_weight: float = 1.0

    # Target joint positions (radians). None means use the robot's default pose.
    # Replace with a list of floats matching your robot's DOF count if desired,
    # e.g. target_joint_pos = [0.0, 0.5, 0.3, ...]
    target_joint_pos: list | None = None


class StaticPoseEnv(ShadowLiteEnv):
    """Hold the Shadow Lite hand in a static target pose."""

    cfg: StaticPoseCfg

    def __init__(self, cfg: StaticPoseCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # Will be set in _reset_idx once the robot asset is fully initialised
        self.target_joint_pos: torch.Tensor | None = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_target(self):
        """Build the target joint position tensor from config (or default pose)."""
        if self.cfg.target_joint_pos is not None:
            target = torch.tensor(
                self.cfg.target_joint_pos,
                dtype=self.dtype,
                device=self.device,
            )
            # Broadcast across all envs: shape (num_envs, num_joints)
            self.target_joint_pos = target.unsqueeze(0).expand(self.num_envs, -1)
        else:
            # Fall back to the robot's default joint positions
            self.target_joint_pos = self.robot.data.default_joint_pos.clone()

    # ------------------------------------------------------------------
    # Overridden environment methods
    # ------------------------------------------------------------------

    def _reset_idx(self, env_ids: Sequence[int] | None):
        """Reset selected environments and rebuild the target if needed."""
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES

        super()._reset_idx(env_ids)

        # Build target once after first reset (robot data is ready by then)
        if self.target_joint_pos is None:
            self._build_target()

    def _get_rewards(self) -> torch.Tensor:
        """Reward the hand for being close to the target pose."""
        total_reward, pose_reward, joint_errors = compute_rewards(
            joint_pos=self.robot.data.joint_pos,
            target_joint_pos=self.target_joint_pos,
            pose_threshold=self.cfg.pose_threshold,
            pose_reward_weight=self.cfg.pose_reward_weight,
        )

        self.extras["log"] = {
            "mean_joint_error": joint_errors.mean(dim=-1).float(),
            "pose_reward": pose_reward.float(),
        }

        return total_reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        """No early termination for a static pose task — just time out."""
        self._compute_intermediate_values()

        termination = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        time_out = self.episode_length_buf >= self.max_episode_length - 1

        return termination, time_out


# ------------------------------------------------------------------
# JIT-compiled reward function (mirrors the style of bounce.py)
# ------------------------------------------------------------------

@torch.jit.script
def compute_rewards(
    joint_pos: torch.Tensor,
    target_joint_pos: torch.Tensor,
    pose_threshold: float,
    pose_reward_weight: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Reward joints for being close to the target position.

    Args:
        joint_pos:         Current joint positions  (num_envs, num_joints).
        target_joint_pos:  Target joint positions   (num_envs, num_joints).
        pose_threshold:    Error below which a joint counts as 'on target' (rad).
        pose_reward_weight: Scalar weight applied to the reward.

    Returns:
        Tuple of (total_reward, pose_reward, joint_errors).
    """
    # Per-joint absolute error
    joint_errors = torch.abs(joint_pos - target_joint_pos)           # (num_envs, num_joints)

    # Fraction of joints within threshold → reward in [0, 1]
    on_target = (joint_errors < pose_threshold).float()              # (num_envs, num_joints)
    pose_reward = on_target.mean(dim=-1) * pose_reward_weight        # (num_envs,)

    total_reward = pose_reward
    return total_reward, pose_reward, joint_errors
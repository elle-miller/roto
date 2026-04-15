# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Peace-sign pose task for the ShadowLite hand (proprioception-only RL)."""

from __future__ import annotations

from collections.abc import Sequence

import torch

from isaaclab.utils import configclass

from roto.tasks.robots.shadowlite.shadowlite import ShadowLiteEnv, ShadowLiteEnvCfg


# ---------------------------------------------------------------------------
# Target joint positions for the peace sign
#
# ShadowLite joint order (verify against your URDF if unsure):
#   FF = index, MF = middle, RF = ring, LF = little, TH = thumb
#
#   Extended finger  → all joints near 0.0
#   Curled finger    → proximal ~1.4 rad, intermediate ~1.4 rad, distal ~1.2 rad
# ---------------------------------------------------------------------------

# fmt: off
_PEACE_SIGN_JOINT_POS = {
    # Index finger — EXTENDED
    "FFJ4": 0.0,   "FFJ3": 0.0,   "FFJ2": 0.0,   "FFJ1": 0.0,
    # Middle finger — EXTENDED
    "MFJ4": 0.0,   "MFJ3": 0.0,   "MFJ2": 0.0,   "MFJ1": 0.0,
    # Ring finger — CURLED
    "RFJ4": 0.0,   "RFJ3": 1.4,   "RFJ2": 1.4,   "RFJ1": 1.2,
    # Little finger — CURLED
    "LFJ5": 0.0,   "LFJ4": 0.0,   "LFJ3": 1.4,   "LFJ2": 1.4,   "LFJ1": 1.2,
    # Thumb — tucked
    "THJ5": -0.5,  "THJ4": 1.0,   "THJ3": 0.2,   "THJ2": 0.3,   "THJ1": 0.3,
}
# fmt: on


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@configclass
class PeaceSignCfg(ShadowLiteEnvCfg):
    """Configuration for the peace-sign pose task."""

    # Episode / task
    episode_length_s: float = 10.0

    # Reward shaping
    pose_reward_scale: float = 5.0    # scale on the exp(-error) shaped reward
    pose_error_sigma: float = 0.3     # controls how sharply reward peaks at target
    success_threshold: float = 0.05   # mean joint error (rad) counted as success


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------


class PeaceSignEnv(ShadowLiteEnv):
    """RL-only proprioceptive task: hold the ShadowLite in a peace-sign pose."""

    cfg: PeaceSignCfg

    def __init__(self, cfg: PeaceSignCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # Build target tensor in joint order from the robot articulation
        self._target_joint_pos = self._build_target_tensor()

    # ------------------------------------------------------------------
    # Scene: nothing extra — no ball, no HDR, no objects
    # ------------------------------------------------------------------

    def _setup_scene(self):
        super()._setup_scene()   # robot + ground only

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _build_target_tensor(self) -> torch.Tensor:
        """Map the named target dict to a tensor aligned with the robot's DOF order."""
        joint_names = self.robot.data.joint_names          # list[str]
        target = torch.zeros(len(joint_names), device=self.device)
        for i, name in enumerate(joint_names):
            if name in _PEACE_SIGN_JOINT_POS:
                target[i] = _PEACE_SIGN_JOINT_POS[name]
        # Expand to (num_envs, num_joints)
        return target.unsqueeze(0).expand(self.num_envs, -1)

    def _joint_error(self) -> torch.Tensor:
        """Mean absolute joint error across all DOFs, shape (num_envs,)."""
        current = self.robot.data.joint_pos          # (num_envs, num_joints)
        return torch.mean(torch.abs(current - self._target_joint_pos), dim=-1)

    # ------------------------------------------------------------------
    # RL interface
    # ------------------------------------------------------------------

    def _get_rewards(self) -> torch.Tensor:
        error = self._joint_error()                  # (num_envs,)

        # Shaped reward: 1 when perfect, decays with error
        pose_reward = torch.exp(-error / self.cfg.pose_error_sigma)
        total_reward = self.cfg.pose_reward_scale * pose_reward

        success = (error < self.cfg.success_threshold).float()

        self.extras["log"] = {
            "mean_joint_error_rad": error.clone(),
            "pose_reward": pose_reward.clone(),
            "success_rate": success.clone(),
        }
        return total_reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        # No termination conditions — only time-out
        termination = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        return termination, time_out

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        super()._reset_idx(env_ids)
        # Nothing extra to reset — no object state
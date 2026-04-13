# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Author: Elle Miller 2025

Orca-hand base environment utilities shared across RoTO tasks.
"""

from __future__ import annotations

import numpy as np
import torch
from collections.abc import Sequence

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.envs import ViewerCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensor, ContactSensorCfg
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils import configclass
from isaaclab.utils.math import quat_conjugate, quat_from_angle_axis, quat_mul
from isaaclab.utils.math import quat_apply

from roto.assets.orca import ORCA_HAND_CFG
from roto.tasks.roto_env import RotoEnv, RotoEnvCfg

from isaaclab.markers.config import FRAME_MARKER_CFG  # isort: skip

def quat_normalize(q: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return q / (torch.norm(q, dim=-1, keepdim=True) + eps)

def quat_from_two_vectors(v1: torch.Tensor, v2: torch.Tensor, eps = 1e-8) -> torch.Tensor:
    v1 = v1 / (torch.norm(v1, dim=-1, keepdim=True) + eps)
    v2 = v2 / (torch.norm(v2, dim=-1, keepdim=True) + eps)
    v = torch.cross(v1, v2, dim=-1)
    w = 1.0 + torch.sum(v1 * v2, dim=-1, keepdim = True)

    # If vectors are opposite, w ~ 0 -> choose an orthogonal axis
    mask = (w.squeeze(-1) < eps)
    if mask.any():
        # pick an axis not parallel to a
        axis = torch.zeros_like(v1)
        axis[..., 0] = 1.0
        alt = torch.zeros_like(v1)
        alt[..., 1] = 1.0
        # if a is too aligned with x, use y
        use_alt = (torch.abs(v1[..., 0]) > 0.9)
        axis[use_alt] = alt[use_alt]

        v2 = torch.cross(v1, axis, dim=-1)
        q2 = torch.cat([torch.zeros_like(w), v2], dim=-1)
        q = torch.cat([w, v], dim=-1)
        q[mask] = q2[mask]
    else:
        q = torch.cat([w, v], dim=-1)

    return quat_normalize(q)

@configclass
class OrcaEnvCfg(RotoEnvCfg):
    """Default configuration for the Orca hand."""

    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=4096, env_spacing=0.7, replicate_physics=True
    )

    eye = (1.1, 1.1, 0.6)
    lookat = (0.4, 0.4, 0.5)
    viewer: ViewerCfg = ViewerCfg(eye=eye, lookat=lookat, resolution=(1920, 1080))

    episode_length_s = 10.0
    num_actions = 16
    action_space = num_actions

    reset_joint_pos_noise = 0.2
    reset_joint_vel_noise = 0.0

    hand_height = 0.5
    robot_cfg: ArticulationCfg = ORCA_HAND_CFG.replace(
        prim_path="/World/envs/env_.*/Robot"
    ).replace(
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, hand_height),
            # rot=(0.7071, 0, 0.7071, 0),
            rot = (0.5, 0.5, 0.5, 0.5),
            # palm up: 0.270598, -0.653283, -0.270598, 0.65328
            #  palm side: (0.0, -0.9239, -0.3827, 0.0)
        )
    )

    # update to match the Orca hand
    actuated_joint_names = [
        "right_wrist",
        "right_index_abd", "right_index_mcp", "right_index_pip",
        "right_middle_abd", "right_middle_mcp", "right_middle_pip",
        "right_ring_abd", "right_ring_mcp", "right_ring_pip",
        "right_pinky_abd", "right_pinky_mcp", "right_pinky_pip",
        "right_thumb_abd", "right_thumb_mcp", "right_thumb_pip", "right_thumb_dip",
    ]


    # num actions has to correspond with the number of actuated joints
    num_actions = len(actuated_joint_names)

    marker_cfg = FRAME_MARKER_CFG.copy()
    marker_cfg.markers["frame"].scale = (0.05, 0.05, 0.05)
    marker_cfg.prim_path = "/Visuals/ContactCfg"

    # Update this to match the Orca hand
    # ['right_palm', 'right_index_mp', 'right_index_pp','right_index_ip', 
    # 'right_middle_mp', 'right_middle_pp', 'right_middle_ip', 'right_pinky_mp', 
    # 'right_pinky_pp','right_pinky_ip', 'right_ring_mp', 'right_ring_pp', 
    # 'right_ring_ip', 'right_thumb_mp', 'right_thumb_pp', 'right_thumb_ip', 'right_thumb_dp']
    robot_contact_sensor_cfg = ContactSensorCfg(
        # prim_path="/World/envs/env_.*/Robot",
        # prim_path="/World/envs/env_.*/Robot/(right_palm|.*_link_.*|.*_biotac_tip)",   
        prim_path="/World/envs/env_.*/Robot/(?!(.*jointbody))(right_(palm|index_.*|middle_.*|pinky_.*|ring_.*|thumb_.*))",
        update_period=0.0,
        history_length=1,
    )


class OrcaEnv(RotoEnv):
    """Orca-hand base env providing tactile + proprio pipelines."""

    cfg: OrcaEnvCfg

    def __init__(self, cfg: OrcaEnvCfg, render_mode: str | None = None, **kwargs):

        super().__init__(cfg, render_mode, **kwargs)

        print("NUM JOINTS:", len(self.robot.joint_names))
        print("JOINT NAMES:", self.robot.joint_names)
        print("BODY NAMES:", self.robot.body_names)
        print("TACTILE SENSORS:", self.robot_contact_sensor.body_names)
        print("TOTAL:", len(self.robot_contact_sensor.body_names))

    def _setup_scene(self):
        """Register the Orca hand, contact sensors, and lighting."""
        super()._setup_scene()

        self.robot = Articulation(self.cfg.robot_cfg)
        self.scene.clone_environments(copy_from_source=False)
        self.scene.articulations["robot"] = self.robot
        self.robot_contact_sensor = ContactSensor(self.cfg.robot_contact_sensor_cfg)
        self.scene.sensors["robot_contact_sensor"] = self.robot_contact_sensor


    def _reset_idx(self, env_ids: Sequence[int] | None):
        """Reset articulation state and optionally randomize joints.

        Args:
            env_ids: Environment indices to reset. If None, resets all environments.
        """
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES

        # Reset articulation and rigid body attributes
        super()._reset_idx(env_ids)

        # Reset hand with noise
        self._reset_robot(env_ids, joint_pos_noise=self.cfg.reset_joint_pos_noise)

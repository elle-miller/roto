# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Author: Elle Miller 2025

Shadow-hand base environment utilities shared across RoTO tasks.
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

from roto.assets.shadow_hand import SHADOW_HAND_CFG
from roto.tasks.roto_env import RotoEnv, RotoEnvCfg

from isaaclab.markers.config import FRAME_MARKER_CFG  # isort: skip


@configclass
class ShadowEnvCfg(RotoEnvCfg):
    """Default configuration for the Shadow hand."""

    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=4096, env_spacing=0.7, replicate_physics=True
    )

    eye = (4, -4, 2.1)
    lookat = (2, -2, 0.5)
    viewer: ViewerCfg = ViewerCfg(eye=eye, lookat=lookat, resolution=(1920, 1080))

    episode_length_s = 10.0
    num_actions = 20
    action_space = num_actions

    reset_joint_pos_noise = 0.2
    reset_joint_vel_noise = 0.0

    hand_height = 0.5
    robot_cfg: ArticulationCfg = SHADOW_HAND_CFG.replace(prim_path="/World/envs/env_.*/Robot").replace(
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, hand_height),
            rot=(1.0, 0.0, 0.0, 0.0),
            joint_pos={".*": 0.0},
        )
    )

    actuated_joint_names = [
        "robot0_WRJ1",
        "robot0_WRJ0",
        "robot0_FFJ3",
        "robot0_FFJ2",
        "robot0_FFJ1",
        "robot0_MFJ3",
        "robot0_MFJ2",
        "robot0_MFJ1",
        "robot0_RFJ3",
        "robot0_RFJ2",
        "robot0_RFJ1",
        "robot0_LFJ4",
        "robot0_LFJ3",
        "robot0_LFJ2",
        "robot0_LFJ1",
        "robot0_THJ4",
        "robot0_THJ3",
        "robot0_THJ2",
        "robot0_THJ1",
        "robot0_THJ0",
    ]
    fingertip_body_names = [
        "robot0_ffdistal",
        "robot0_mfdistal",
        "robot0_rfdistal",
        "robot0_lfdistal",
        "robot0_thdistal",
    ]

    marker_cfg = FRAME_MARKER_CFG.copy()
    marker_cfg.markers["frame"].scale = (0.05, 0.05, 0.05)
    marker_cfg.prim_path = "/Visuals/ContactCfg"

    robot_contact_sensor_cfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/robot0_(.*distal|.*middle|.*proximal|palm|lfmetacarpal)",
        update_period=0.0,
        history_length=1,
    )



class ShadowEnv(RotoEnv):
    """Shadow-hand base env providing tactile + proprio pipelines."""

    cfg: ShadowEnvCfg

    def __init__(self, cfg: ShadowEnvCfg, render_mode: str | None = None, **kwargs):

        super().__init__(cfg, render_mode, **kwargs)

        # Tactile buffers — sized to 0 initially, resized by _build_tactile_reindex()
        # on the first _get_tactile() call once the sensor is initialized.
        self.num_tactile_observations = 0
        self.tactile = torch.zeros((self.num_envs, 0), device=self.device)
        self.last_tactile = torch.zeros((self.num_envs, 0), device=self.device)

        self.extras["log"] = {
            "tactile_penalty": None,
            "success_reward": None,
            "action_penalty": None,
            "fall_penalty": None,
            "object_height": None,
            "object_z_linvel": None,
            "object_z_angvel": None,
            "sum_forces": None,
            "total_rotations": None,
            "cumulative_rotations": None,
            "ball_1_vel": None,
            "ball_2_vel": None,
            "ball_dist": None,
            "dist_penalty": None,
            "tactile_reward": None,
            "transition_reward": None,
            "bounce_reward": None,
            "air_reward": None,
        }

    def _setup_scene(self):
        """Register the Shadow hand, contact sensors, and lighting."""
        super()._setup_scene()

        self.robot = Articulation(self.cfg.robot_cfg)
        # spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())
        self.scene.clone_environments(copy_from_source=False)
        self.scene.articulations["robot"] = self.robot

        light_cfg = sim_utils.DomeLightCfg(intensity=200.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

        # Additional sphere lights (currently disabled)
        pink = (0.9882352941176471, 0.011764705882352941, 0.7098039215686275)
        aqua = (0.0, 1.0, 1.0)
        disco_intensity = 20000
        # light_cfg_1 = sim_utils.SphereLightCfg(intensity=disco_intensity, color=pink)
        # light_cfg_1.func("/World/ds", light_cfg_1, translation=(1, 1, 2))
        # light_cfg_2 = sim_utils.SphereLightCfg(intensity=disco_intensity, color=aqua)
        # light_cfg_2.func("/World/disk", light_cfg_2, translation=(-1, 1, 2))
        # light_cfg_3 = sim_utils.SphereLightCfg(intensity=disco_intensity, color=pink)
        # light_cfg_3.func("/World/ds1", light_cfg_3, translation=(-1, -1, 2))
        # light_cfg_4 = sim_utils.SphereLightCfg(intensity=disco_intensity, color=aqua)
        # light_cfg_4.func("/World/disk2", light_cfg_4, translation=(1, -1, 2))

        self.robot_contact_sensor = ContactSensor(self.cfg.robot_contact_sensor_cfg)
        self.scene.sensors["robot_contact_sensor"] = self.robot_contact_sensor


    def _get_tactile(self):
        """Return binary tactile activation per finger segment.

        Reindexes the single contact sensor to match the legacy ordering:
        [all distal, all proximal, all middle, palm, metacarpal].
        """

        forces = self.robot_contact_sensor.data.net_forces_w[:].clone()  # [N, B, 3]
        norm = torch.linalg.vector_norm(forces, dim=-1)  # [N, B]

        if self.tactile_cfg is not None and self.tactile_cfg.get("binary_tactile", True):
            norm = (norm > self.binary_threshold).float()

        self.last_tactile = self.tactile
        self.tactile = norm
        return norm

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
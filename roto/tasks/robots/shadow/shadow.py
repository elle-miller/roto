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

    actuated_joint_names = ["robot0_WRJ1", "robot0_WRJ0", "robot0_FFJ3", "robot0_FFJ2", "robot0_FFJ1", 
        "robot0_MFJ3", "robot0_MFJ2", "robot0_MFJ1", "robot0_RFJ3", "robot0_RFJ2", "robot0_RFJ1", 
        "robot0_LFJ4", "robot0_LFJ3", "robot0_LFJ2", "robot0_LFJ1", "robot0_THJ4", "robot0_THJ3", 
        "robot0_THJ2", "robot0_THJ1", "robot0_THJ0"]

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


    def _setup_scene(self):
        """Register the Shadow hand, contact sensors, and lighting."""
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
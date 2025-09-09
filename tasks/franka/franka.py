# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Author: Elle Miller 2025

Shared Franka parent environment for IsaacLab RL tasks.

This module provides a configurable RL environment for the Franka Panda robot,
including simulation setup, sensors, and reward utilities.
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np
import torch
from collections.abc import Sequence

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg, RigidObject, RigidObjectCfg
from isaaclab.sensors import (
    ContactSensor,
    ContactSensorCfg,
    FrameTransformer,
    FrameTransformerCfg,
    OffsetCfg,
)
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.math import (
    quat_conjugate,
    quat_from_angle_axis,
    quat_mul,
    sample_uniform,
)

from isaaclab.markers.config import FRAME_MARKER_CFG  # isort: skip

from assets.franka import FRANKA_PANDA_CFG  # isort: skip

from tasks.roto_env import RotoEnv, RotoEnvCfg


@configclass
class FrankaEnvCfg(RotoEnvCfg):
    """
    Configuration class for Franka RL environments.
    Defines simulation parameters, robot configs, sensors, and scene setup.
    """
    num_actions = 9       # Number of actions for Franka Panda
    action_space = num_actions

    # Robot configuration
    robot_cfg: ArticulationCfg = FRANKA_PANDA_CFG.replace(prim_path="/World/envs/env_.*/Robot")

    # Contact sensor marker configuration
    marker_cfg = FRAME_MARKER_CFG.copy()
    marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
    marker_cfg.prim_path = "/Visuals/ContactCfg"
    left_contact_cfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/left_contact_sensor",
        update_period=0.0,
        history_length=1,
        debug_vis=False,
        visualizer_cfg=marker_cfg,
        filter_prim_paths_expr=["/World/envs/env_.*/Object"],
    )
    right_contact_cfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/right_contact_sensor",
        update_period=0.0,
        history_length=1,
        debug_vis=False,
        visualizer_cfg=marker_cfg,
        filter_prim_paths_expr=["/World/envs/env_.*/Object"],
    )

    # Actuated joint names for Franka Panda
    actuated_joint_names = [
        "panda_joint1",
        "panda_joint2",
        "panda_joint3",
        "panda_joint4",
        "panda_joint5",
        "panda_joint6",
        "panda_joint7",
        "panda_finger_joint1",
        "panda_finger_joint2",
    ]

    # End-effector frame transformer configuration
    marker_cfg = FRAME_MARKER_CFG.copy()
    marker_cfg.markers["frame"].scale = (0.01, 0.01, 0.01)
    marker_cfg.prim_path = "/Visuals/EndEffectorFrameTransformer"
    ee_config: FrameTransformerCfg = FrameTransformerCfg(
        prim_path="/World/envs/env_.*/Robot/panda_link0",
        debug_vis=False,
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path="/World/envs/env_.*/Robot/panda_hand",
                name="end_effector",
                offset=OffsetCfg(
                    pos=[0.0, 0.0, 0.1034],
                ),
            ),
        ],
    )

    # Contact sensor frame transformer configuration
    left_sensor_cfg: FrameTransformerCfg = FrameTransformerCfg(
        prim_path="/World/envs/env_.*/Robot/panda_link0",
        debug_vis=False,
        visualizer_cfg=marker_cfg,
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path="/World/envs/env_.*/Robot/left_contact_sensor",
                name="left_sensor",
                offset=OffsetCfg(
                    pos=[0.0, 0.0, 0.0],
                ),
            ),
        ],
    )
    right_sensor_cfg: FrameTransformerCfg = FrameTransformerCfg(
        prim_path="/World/envs/env_.*/Robot/panda_link0",
        debug_vis=False,
        visualizer_cfg=marker_cfg,
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path="/World/envs/env_.*/Robot/right_contact_sensor",
                name="right_sensor",
                offset=OffsetCfg(
                    pos=[0.0, 0.0, 0.0],
                ),
            ),
        ],
    )


class FrankaEnv(RotoEnv):
    """
    RL environment for the Franka Panda robot.

    Handles simulation setup, action application, observation collection, and resets.
    """

    cfg: FrankaEnvCfg

    def __init__(self, cfg: FrankaEnvCfg, render_mode: str | None = None, **kwargs):
        """
        Initialize the Franka RL environment.

        Args:
            cfg (FrankaEnvCfg): Environment configuration.
            render_mode (str, optional): Rendering mode.
            **kwargs: Additional arguments.
        """
        super().__init__(cfg, render_mode, **kwargs)
       
        # End-effector and tactile state
        self.aperture = torch.zeros((self.num_envs,), device=self.device)
        self.tactile = torch.zeros((self.num_envs, 2), device=self.device)
        self.ee_pos = torch.zeros((self.num_envs, 3), device=self.device)
        self.ee_rot = torch.zeros((self.num_envs, 4), device=self.device)
      
        # Logging and counters for diagnostics
        self.extras["log"] = {
            "tactile": None,
            "aperture": None,
        }
        self.extras["counters"] = {}

        # Need these dimensions explicitly for tactile decoding
        self.num_prop_observations = self._get_proprioception().shape[1]
        self.num_tactile_observations = self._get_tactile().shape[1]

    def _setup_scene(self):
        """
        Set up the simulation scene, including robot, sensors, and lighting.
        """
        self.robot = Articulation(self.cfg.robot_cfg)

        # Frame transformers for end-effector and contact sensors
        self.ee_frame = FrameTransformer(self.cfg.ee_config)
        self.ee_frame.set_debug_vis(False)
        self.left_sensor_frame = FrameTransformer(self.cfg.left_sensor_cfg)
        self.right_sensor_frame = FrameTransformer(self.cfg.right_sensor_cfg)

        # Add ground plane
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg(size=(10000, 10000)))

        # Clone and replicate environments
        self.scene.clone_environments(copy_from_source=False)

        # Register components to scene
        self.scene.articulations["robot"] = self.robot
        self.scene.sensors["ee_frame"] = self.ee_frame
        self.scene.sensors["left_sensor_frame"] = self.left_sensor_frame
        self.scene.sensors["right_sensor_frame"] = self.right_sensor_frame

        # Add lighting
        yellow = (1.0, 0.96, 0.0)
        orange = (1.0, 0.5, 0.0)
        light_cfg = sim_utils.DomeLightCfg(intensity=1000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)
        light_cfg_1 = sim_utils.SphereLightCfg(intensity=10000.0, color=yellow)
        light_cfg_1.func("/World/ds", light_cfg_1, translation=(1, 0, 1))
        light_cfg_2 = sim_utils.SphereLightCfg(intensity=10000.0, color=orange)
        light_cfg_2.func("/World/disk", light_cfg_2, translation=(-1, 0, 1))

        # Add tactile contact sensors if required
        if "tactile" in self.cfg.obs_list:
            self.left_contact_sensor = ContactSensor(self.cfg.left_contact_cfg)
            self.scene.sensors["left_contact_sensor"] = self.left_contact_sensor

            self.right_contact_sensor = ContactSensor(self.cfg.right_contact_cfg)
            self.scene.sensors["right_contact_sensor"] = self.right_contact_sensor

    def _get_proprioception(self):
        """
        Get proprioceptive observations (joint positions, velocities, etc.).

        Returns:
            torch.Tensor: Proprioceptive observation vector.
        """
        control_errors = self.joint_pos_cmd - self.joint_pos
        prop = torch.cat(
            (
                self.normalised_joint_pos,
                self.normalised_joint_vel,
                self.aperture.unsqueeze(1),
                self.ee_pos,
                self.ee_rot,
                self.actions,
                control_errors
            ),
            dim=-1,
        )
        return prop

    def _get_tactile(self):
        """
        Get tactile sensor observations from contact sensors.

        Returns:
            torch.Tensor: Tactile observation vector.
        """
        forcesL_world = self.left_contact_sensor.data.net_forces_w[:].clone().reshape(self.num_envs, 3)
        forcesR_world = self.right_contact_sensor.data.net_forces_w[:].clone().reshape(self.num_envs, 3)

        forcesL_net = torch.linalg.vector_norm(forcesL_world, dim=1, keepdim=True)
        forcesR_net = torch.linalg.vector_norm(forcesR_world, dim=1, keepdim=True)

        if self.dtype == torch.float16:
            forcesL_norm = (forcesL_net > self.binary_threshold).half()
            forcesR_norm = (forcesR_net > self.binary_threshold).half()
        else:
            forcesL_norm = (forcesL_net > self.binary_threshold).float()
            forcesR_norm = (forcesR_net > self.binary_threshold).float()

        tactile = torch.cat(
            (
                forcesL_norm,
                forcesR_norm,
            ),
            dim=-1,
        )

        # try  bipolar encoding
        tactile[tactile == 0] = -1

        self.tactile = tactile
        return tactile

    def _compute_intermediate_values(self, env_ids: torch.Tensor | None = None):
        """
        Compute intermediate values for observations and rewards.

        Args:
            env_ids (torch.Tensor | None): Environment indices to update.
        """
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        super()._compute_intermediate_values(env_ids)
        
        # Update end-effector pose
        self.ee_pos[env_ids] = self.ee_frame.data.target_pos_source[..., 0, :][env_ids]
        self.ee_rot[env_ids] = self.ee_frame.data.target_quat_source[..., 0, :][env_ids]

        # Compute aperture (normalized gripper opening)
        max_aperture = 0.08
        self.aperture = (self.joint_pos[:, 7] + self.joint_pos[:, 8]) / max_aperture


@torch.jit.script
def randomize_rotation(rand0, rand1, x_unit_tensor, y_unit_tensor):
    """
    Generate a randomized rotation quaternion.

    Args:
        rand0 (Tensor): Random values for X rotation.
        rand1 (Tensor): Random values for Y rotation.
        x_unit_tensor (Tensor): X unit vector.
        y_unit_tensor (Tensor): Y unit vector.

    Returns:
        Tensor: Quaternion representing rotation.
    """
    return quat_mul(
        quat_from_angle_axis(rand0 * np.pi, x_unit_tensor), quat_from_angle_axis(rand1 * np.pi, y_unit_tensor)
    )




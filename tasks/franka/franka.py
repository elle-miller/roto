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

from tasks.roto_env import RotoEnv, RotoEnvCfg  # isort: skip


@configclass
class FrankaEnvCfg(RotoEnvCfg):
    """
    Configuration class for Franka RL environments.
    Defines simulation parameters, robot and object configs, sensors, and scene setup.
    """
    # Isaac 4.5 compatibility
    num_actions = 9       # Number of actions for Franka Panda
    action_space = num_actions

    # Object reset configuration
    default_object_pos = [0.5, 0, 0.03]
    reset_object_position_noise = 0.05

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

    # Object configuration
    object_cfg: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Object",
        init_state=RigidObjectCfg.InitialStateCfg(pos=default_object_pos, rot=[1, 0, 0, 0]),
        spawn=UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/DexCube/dex_cube_instanceable.usd",
            scale=(0.8, 0.8, 0.8),
            rigid_props=RigidBodyPropertiesCfg(
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=1,
                max_angular_velocity=1000.0,
                max_linear_velocity=1000.0,
                max_depenetration_velocity=5.0,
                disable_gravity=False,
            ),
        ),
    )

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
       
        # Tactile decoder dimensions
        self.num_prop_observations = 560
        self.num_tactile_observations = 16 * 2

      
        self.aperture = torch.zeros((self.num_envs,), device=self.device)
        self.object_pos = torch.zeros((self.num_envs, 3), device=self.device)
        self.tactile = torch.zeros((self.num_envs, 2), device=self.device)

        self.object_rot = torch.zeros((self.num_envs, 4), device=self.device)
        self.ee_pos = torch.zeros((self.num_envs, 3), device=self.device)
        self.ee_rot = torch.zeros((self.num_envs, 4), device=self.device)

        self.object_ee_distance = torch.zeros((self.num_envs, 3), device=self.device)
        self.object_ee_rotation = torch.zeros((self.num_envs, 4), device=self.device)
        self.object_ee_angular_distance = torch.zeros((self.num_envs,), device=self.device)
        self.object_ee_euclidean_distance = torch.zeros((self.num_envs,), device=self.device)

      
        # Logging and counters for diagnostics
        self.extras["log"] = {
            "reach_reward": None,
            "dist_reward": None,
            "object_ee_distance": None,
            "tactile": None,
        }

        self.extras["counters"] = {
            "timesteps_to_find_object_easy": None,
            "timesteps_to_find_object_med": None,
            "timesteps_to_find_object_hard": None,
            "object_found_easy": None,
            "object_found_med": None,
            "object_found_hard": None,
        }

    def _setup_scene(self):
        """
        Set up the simulation scene, including robot, object, sensors, and lighting.
        """
        self.robot = Articulation(self.cfg.robot_cfg)
        self.object = RigidObject(self.cfg.object_cfg)
        self.scene.rigid_objects["object"] = self.object

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
        control_errors = self.cur_targets - self.joint_pos
        prop = torch.cat(
            (
                self.normalised_joint_pos,
                self.normalised_joint_vel,
                self.aperture.unsqueeze(1),
                self.ee_pos,
                self.ee_rot,
                self.actions,
                self.cur_targets,
                self.prev_targets,
                control_errors
            ),
            dim=-1,
        )
        return prop

    def _get_gt(self):
        """
        Get ground-truth observations (object-EE distances, rotations, etc.).

        Returns:
            torch.Tensor: Ground-truth observation vector.
        """
        gt = torch.cat(
            (
                self.object_ee_distance,
                self.object_ee_rotation,
                self.object_ee_angular_distance.unsqueeze(1),
                self.object_ee_euclidean_distance.unsqueeze(1),
            ),
            dim=-1,
        )
        return gt

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
        self.tactile = tactile
        return tactile

    def _reset_idx(self, env_ids: Sequence[int] | None):
        """
        Reset the environment for the given indices.

        Args:
            env_ids (Sequence[int] | None): Environment indices to reset.
        """
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        super()._reset_idx(env_ids)

        self._reset_object_pose(env_ids)
        self._reset_robot(env_ids)
        self._compute_intermediate_values(reset=True, env_ids=env_ids)

    def _reset_object_pose(self, env_ids):
        """
        Reset the pose of the object in the environment.

        Args:
            env_ids (Sequence[int]): Environment indices to reset.
        """
        object_default_state = self.object.data.default_root_state.clone()[env_ids]
        pos_noise = sample_uniform(-1.0, 1.0, (len(env_ids), 3), device=self.device)

        object_default_state[:, 0:3] = (
            object_default_state[:, 0:3]
            + self.cfg.reset_object_position_noise * pos_noise
            + self.scene.env_origins[env_ids]
        )
        rot_noise = sample_uniform(-1.0, 1.0, (len(env_ids), 2), device=self.device)
        object_default_state[:, 3:7] = randomize_rotation(
            rot_noise[:, 0], rot_noise[:, 1], self.x_unit_tensor[env_ids], self.y_unit_tensor[env_ids]
        )
        object_default_state[:, 7:] = torch.zeros_like(self.object.data.default_root_state[env_ids, 7:])
        self.object.write_root_state_to_sim(object_default_state, env_ids)

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

        # Object pose and relative distances
        self.object_pos[env_ids] = self.object.data.root_pos_w[env_ids] - self.scene.env_origins[env_ids]
        self.object_rot[env_ids] = self.object.data.root_quat_w[env_ids]
        self.object_ee_distance[env_ids] = self.object_pos[env_ids] - self.ee_pos[env_ids]
        self.object_ee_euclidean_distance[env_ids] = torch.norm(self.object_ee_distance[env_ids], dim=1)
        self.object_ee_rotation[env_ids] = quat_mul(self.object_rot[env_ids], quat_conjugate(self.ee_rot[env_ids]))
        self.object_ee_angular_distance[env_ids] = rotation_distance(self.object_rot[env_ids], self.ee_rot[env_ids])

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Determine episode termination and timeout.

        Returns:
            tuple: (termination tensor, timeout tensor)
        """
        self._compute_intermediate_values()
        termination = torch.zeros((self.num_envs,)).to(device=self.device)
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        return termination, time_out

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


@torch.jit.script
def rotation_distance(object_rot, target_rot):
    """
    Compute angular distance between two quaternions.

    Args:
        object_rot (Tensor): Object rotation quaternion.
        target_rot (Tensor): Target rotation quaternion.

    Returns:
        Tensor: Angular distance in radians.
    """
    quat_diff = quat_mul(object_rot, quat_conjugate(target_rot))
    return 2.0 * torch.asin(torch.clamp(torch.norm(quat_diff[:, 1:4], p=2, dim=-1), max=1.0))


@torch.jit.script
def distance_reward(object_ee_distance, std: float = 0.1):
    """
    Reward function for reaching the object.

    Args:
        object_ee_distance (Tensor): Distance between object and end-effector.
        std (float): Standard deviation for scaling.

    Returns:
        Tensor: Reward value.
    """
    r_reach = 1 - torch.tanh(object_ee_distance / std)
    return r_reach



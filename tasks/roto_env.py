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

from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg, ViewerCfg
from isaaclab.scene import InteractiveSceneCfg

from isaaclab.sim import PhysxCfg, SimulationCfg
from isaaclab.sim.spawners.materials.physics_materials_cfg import RigidBodyMaterialCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import (
    quat_conjugate,
    quat_from_angle_axis,
    quat_mul,
    sample_uniform,
    saturate,
)


@configclass
class RotoEnvCfg(DirectRLEnvCfg):
    """
    Configuration class for RoTO RL environments.
    Defines simulation parameters, robot and object configs, sensors, and scene setup.
    """
    # Physics simulation parameters
    physics_dt = 1 / 120  # Simulation timestep (seconds)
    decimation = 2        # Number of physics steps per control step
    render_interval = 2   # Physics steps per rendering step

    # Isaac 4.5 compatibility
    observation_space = 0
    state_space = 0

    # Simulation configuration
    sim: SimulationCfg = SimulationCfg(
        dt=physics_dt,
        render_interval=decimation,
        physics_material=RigidBodyMaterialCfg(
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        physx=PhysxCfg(
            bounce_threshold_velocity=0.2,
        ),
    )

    # Scene configuration
    replicate_physics = True
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=4096, env_spacing=1.5, replicate_physics=replicate_physics
    )

    # Viewer configuration (not used directly)
    eye = (3, 3, 3)
    lookat = (0, 0, 0)
    viewer: ViewerCfg = ViewerCfg(eye=eye, lookat=lookat, resolution=(1920, 1080))


class RotoEnv(DirectRLEnv):
    """
    RL environment for the Roto benchmark.

    Handles simulation setup, action application, observation collection, and resets.
    """

    cfg: RotoEnvCfg

    def __init__(self, cfg: RotoEnvCfg, render_mode: str | None = None, **kwargs):
        """
        Initialize the Franka RL environment.

        Args:
            cfg (FrankaEnvCfg): Environment configuration.
            render_mode (str, optional): Rendering mode.
            **kwargs: Additional arguments.
        """
        super().__init__(cfg, render_mode, **kwargs)
        self.obs_stack = getattr(cfg, "obs_stack", 1)

        self.dtype = torch.float32
        self.binary_tactile = True
        self.binary_threshold = 0.01

        # Joint limits and targets
        self.robot_joint_pos_lower_limits = self.robot.data.soft_joint_pos_limits[0, :, 0].to(device=self.device)
        self.robot_joint_pos_upper_limits = self.robot.data.soft_joint_pos_limits[0, :, 1].to(device=self.device)
        self.robot_joint_vel_limits = self.robot.data.joint_vel_limits[0, :].to(device=self.device)

        self.cur_targets = torch.zeros((self.num_envs, self.robot.num_joints), device=self.device)
        self.prev_targets = torch.zeros((self.num_envs, self.robot.num_joints), device=self.device)

        # Indices of actuated joints
        self.actuated_dof_indices = [
            self.robot.joint_names.index(joint_name)
            for joint_name in cfg.actuated_joint_names
        ]
        self.actuated_dof_indices.sort()

        # Action and state tensors
        self.actions = torch.zeros((self.num_envs, self.cfg.num_actions), device=self.device)
        default_joint_pos = self.robot.data.default_joint_pos
        self.cur_targets[:, self.actuated_dof_indices] = default_joint_pos[:, self.actuated_dof_indices]
        self.prev_targets[:, self.actuated_dof_indices] = default_joint_pos[:, self.actuated_dof_indices]

        self.num_joints = self.robot.num_joints
        self.joint_pos = torch.zeros((self.num_envs, self.num_joints), device=self.device)
        self.joint_vel = torch.zeros((self.num_envs, self.num_joints), device=self.device)
        self.normalised_joint_pos = torch.zeros((self.num_envs, self.num_joints), device=self.device)
        self.normalised_joint_vel = torch.zeros((self.num_envs, self.num_joints), device=self.device)
        
        # Unit vectors for rotation calculations
        self.x_unit_tensor = torch.tensor([1, 0, 0], dtype=torch.float, device=self.device).repeat((self.num_envs, 1))
        self.y_unit_tensor = torch.tensor([0, 1, 0], dtype=torch.float, device=self.device).repeat((self.num_envs, 1))
        self.z_unit_tensor = torch.tensor([0, 0, 1], dtype=torch.float, device=self.device).repeat((self.num_envs, 1))

    def _configure_gym_env_spaces(self):
        """Configure Gymnasium observation and action spaces (placeholder)."""
        pass

    def set_spaces(self, single_obs, obs, single_action, action):
        """
        Set observation and action spaces for the environment.

        Args:
            single_obs: Single observation space.
            obs: Observation space.
            single_action: Single action space.
            action: Action space.
        """
        self.single_observation_space = single_obs
        self.observation_space = obs
        self.single_action_space = single_action
        self.action_space = action

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        """
        Store actions from policy in a class variable.

        Args:
            actions (torch.Tensor): Actions from the policy.
        """
        self.last_action = self.cur_targets[:, self.actuated_dof_indices]
        self.actions = actions.clone()

    def _apply_action(self) -> None:
        """
        Apply actions to the robot. Called multiple times per RL step for decimation.
        """
        self.cur_targets[:, self.actuated_dof_indices] = scale(
            self.actions,
            self.robot_joint_pos_lower_limits[self.actuated_dof_indices],
            self.robot_joint_pos_upper_limits[self.actuated_dof_indices],
        )
        self.cur_targets[:, self.actuated_dof_indices] = (
            self.cfg.act_moving_average * self.cur_targets[:, self.actuated_dof_indices]
            + (1.0 - self.cfg.act_moving_average) * self.prev_targets[:, self.actuated_dof_indices]
        )
        self.cur_targets[:, self.actuated_dof_indices] = saturate(
            self.cur_targets[:, self.actuated_dof_indices],
            self.robot_joint_pos_lower_limits[self.actuated_dof_indices],
            self.robot_joint_pos_upper_limits[self.actuated_dof_indices],
        )

        self.prev_targets[:, self.actuated_dof_indices] = self.cur_targets[:, self.actuated_dof_indices]

        self.robot.set_joint_position_target(
            self.cur_targets[:, self.actuated_dof_indices], joint_ids=self.actuated_dof_indices
        )

    def get_observations(self):
        """
        Public method to get observations for the current timestep.

        Returns:
            dict: Dictionary of observations.
        """
        return self._get_observations()

    def _get_observations(self) -> dict:
        """
        Collect observations based on the configured observation list.

        Returns:
            dict: Dictionary containing policy observations.
        """
        obs_dict = {}
        for k in self.cfg.obs_list:
            if k == "prop":
                obs_dict[k] = self._get_proprioception()
            elif k == "gt":
                obs_dict[k] = self._get_gt()
            elif k == "tactile":
                obs_dict[k] = self._get_tactile()
            else:
                print("Unknown observations type!")

        obs_dict = {"policy": obs_dict}
        return obs_dict
    
    def _reset_robot(self, env_ids, joint_pos_noise=0.125):
        """
        Reset the robot joint positions and velocities.

        Args:
            env_ids (Sequence[int]): Environment indices to reset.
        """
        joint_pos = self.robot.data.default_joint_pos[env_ids] + sample_uniform(
            -joint_pos_noise,
            joint_pos_noise,
            (len(env_ids), self.robot.num_joints),
            self.device,
        )
        joint_pos = torch.clamp(joint_pos, self.robot_joint_pos_lower_limits, self.robot_joint_pos_upper_limits)
        joint_vel = torch.zeros_like(joint_pos)
        self.robot.set_joint_position_target(joint_pos, env_ids=env_ids)
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)


    def _compute_intermediate_values(self, env_ids):
        # Get robot data
        self.joint_pos[env_ids] = self.robot.data.joint_pos[env_ids]
        self.joint_vel[env_ids] = self.robot.data.joint_vel[env_ids]
        # Normalize joint positions and velocities
        self.normalised_joint_pos[env_ids] = unscale(
            self.joint_pos[env_ids], self.robot_joint_pos_lower_limits, self.robot_joint_pos_upper_limits
        )
        self.normalised_joint_vel[env_ids] = unscale(
            self.joint_vel[env_ids], -self.robot_joint_vel_limits, self.robot_joint_vel_limits
        )


@torch.jit.script
def scale(x, lower, upper):
    """
    Scale input x from [-1, 1] to [lower, upper].

    Args:
        x (Tensor): Input tensor.
        lower (Tensor): Lower bounds.
        upper (Tensor): Upper bounds.

    Returns:
        Tensor: Scaled tensor.
    """
    return 0.5 * (x + 1.0) * (upper - lower) + lower


@torch.jit.script
def unscale(x, lower, upper):
    """
    Unscale input x from [lower, upper] to [-1, 1].

    Args:
        x (Tensor): Input tensor.
        lower (Tensor): Lower bounds.
        upper (Tensor): Upper bounds.

    Returns:
        Tensor: Unscaled tensor.
    """
    return (2.0 * x - upper - lower) / (upper - lower)
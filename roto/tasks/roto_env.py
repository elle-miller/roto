# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Author: Elle Miller

Generalisable robot parent environment for IsaacLab RL tasks. See README.md for more details.

This environment is used to define the basic robot control and observation logic for the RoTO tasks.

It is a child of `DirectRLEnv`, which is a base environment for Isaac Lab RL tasks.

"""

from __future__ import annotations

import torch

import isaaclab.sim as sim_utils
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg, ViewerCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import PhysxCfg, SimulationCfg
from isaaclab.sim.spawners.materials.physics_materials_cfg import RigidBodyMaterialCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import (
    sample_uniform,
    saturate,
)
from isaaclab.sensors import TiledCamera, TiledCameraCfg
from isaaclab.sensors import ContactSensor, ContactSensorCfg


@configclass
class RotoEnvCfg(DirectRLEnvCfg):
    """Simulation / scene defaults used by every RoTO task."""

    # Physics simulation parameters
    physics_dt = 1 / 120  # Simulation timestep (seconds)
    decimation = 2  # Number of physics steps per control step
    render_interval = 2  # Physics steps per rendering step

    # Isaac 4.5 compatibility
    observation_space = 0
    state_space = 0

    # Observation configuration (set from agent_cfg)
    obs_list: list[str] = []
    obs_stack: int = 1
    pixel_cfg: dict | None = None
    tactile_cfg: dict | None = None

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
    env_spacing = 1.5
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=4096, env_spacing=env_spacing, replicate_physics=replicate_physics
    )

    # Viewer configuration (not used directly)
    eye = (3, 3, 3)
    lookat = (0, 0, 0)
    viewer: ViewerCfg = ViewerCfg(eye=eye, lookat=lookat, resolution=(1920, 1080))

    # camera
    eye = (0.0, -0.6, 0.65)
    target = (0.0, -0.35, 0.5)

    render_cfg = sim_utils.RenderCfg(rendering_mode="quality")

    # tactile sensor configuration
    robot_contact_sensor_cfg = None


class RotoEnv(DirectRLEnv):
    """Shared RL base class handling control, observation, and reset logic."""

    cfg: RotoEnvCfg

    def __init__(self, cfg: RotoEnvCfg, render_mode: str | None = None, **kwargs):
        """Initialize tensors used by derived robot + task implementations."""
        
        # Observation configuration
        self.obs_stack = getattr(cfg, "obs_stack", 1)
        self.pixel_cfg = getattr(cfg, "pixel_cfg", None)
        self.tactile_cfg = getattr(cfg, "tactile_cfg", None)
        if self.tactile_cfg is not None:
            self.binary_threshold = self.tactile_cfg["binary_threshold"]
        self.dtype = torch.float32

        super().__init__(cfg, render_mode, **kwargs)

        # Joint limits and targets
        self.robot_joint_pos_lower_limits = self.robot.data.soft_joint_pos_limits[0, :, 0].to(device=self.device)
        self.robot_joint_pos_upper_limits = self.robot.data.soft_joint_pos_limits[0, :, 1].to(device=self.device)
        self.robot_joint_vel_limits = self.robot.data.joint_vel_limits[0, :].to(device=self.device)

        self.joint_pos_cmd = torch.zeros((self.num_envs, self.robot.num_joints), device=self.device)
        self.prev_joint_pos_cmd = torch.zeros((self.num_envs, self.robot.num_joints), device=self.device)

        # Indices of actuated joints
        self.actuated_dof_indices = [
            self.robot.joint_names.index(joint_name) for joint_name in cfg.actuated_joint_names
        ]
        self.actuated_dof_indices.sort()

        # Action and state tensors
        self.actions = torch.zeros((self.num_envs, self.cfg.num_actions), device=self.device)
        default_joint_pos = self.robot.data.default_joint_pos
        self.joint_pos_cmd[:, self.actuated_dof_indices] = default_joint_pos[:, self.actuated_dof_indices]
        self.prev_joint_pos_cmd[:, self.actuated_dof_indices] = default_joint_pos[:, self.actuated_dof_indices]

        self.num_joints = self.robot.num_joints
        self.joint_pos = torch.zeros((self.num_envs, self.num_joints), device=self.device)
        self.joint_vel = torch.zeros((self.num_envs, self.num_joints), device=self.device)
        self.joint_acc = torch.zeros((self.num_envs, self.num_joints), device=self.device)

        self.normalised_joint_pos = torch.zeros((self.num_envs, self.num_joints), device=self.device)
        self.normalised_joint_vel = torch.zeros((self.num_envs, self.num_joints), device=self.device)

        # Set up camera if pixels are in the observation list
        if "pixels" in self.cfg.obs_list:
            eyes = (
                torch.tensor(self.cfg.eye, dtype=torch.float, device=self.device).repeat((self.num_envs, 1))
                + self.scene.env_origins
            )
            targets = (
                torch.tensor(self.cfg.target, dtype=torch.float, device=self.device).repeat((self.num_envs, 1))
                + self.scene.env_origins
            )
            self._tiled_camera.set_world_poses_from_view(eyes=eyes, targets=targets)

    def _setup_scene(self):
        """Set up the simulation scene."""
        # Set up camera if pixels are in the observation list
        if "pixels" in self.cfg.obs_list and self.pixel_cfg is not None:
            print("Setting up camera for pixel observation with width: ", self.pixel_cfg["width"], "and height: ", self.pixel_cfg["height"])
            tiled_camera = TiledCameraCfg(
                prim_path="/World/envs/env_.*/Camera",
                offset=TiledCameraCfg.OffsetCfg(pos=(0.0, 0.0, 0.7), rot=(1.0, 0.0, 0.0, 0.0), convention="world"),
                data_types=["rgb", "depth"],
                spawn=sim_utils.PinholeCameraCfg(
                    focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, self.cfg.env_spacing)
                ),
                width=self.pixel_cfg["width"],
                height=self.pixel_cfg["height"],
            )
            self._tiled_camera = TiledCamera(tiled_camera)
            self.scene.sensors["tiled_camera"] = self._tiled_camera

        # Add tactile sensor if listed
        if "tactile" in self.cfg.obs_list:
            print("Adding contact sensor for tactile observation")
            self.robot_contact_sensor = ContactSensor(self.cfg.robot_contact_sensor_cfg)
            self.scene.sensors["robot_contact_sensor"] = self.robot_contact_sensor

    def _configure_gym_env_spaces(self):
        """Configure Gymnasium observation and action spaces (placeholder)."""

    def set_spaces(self, single_obs, obs, single_action, action):
        """Set Gymnasium observation + action spaces for downstream wrappers."""
        self.single_observation_space = single_obs
        self.observation_space = obs
        self.single_action_space = single_action
        self.action_space = action

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        """Store actions from policy before physics step.

        Args:
            actions: Actions from the policy.
        """
        self.last_action = self.joint_pos_cmd[:, self.actuated_dof_indices]
        self.actions = actions.clone()

    def _apply_action(self) -> None:
        """Apply actions to the robot.

        Called multiple times per RL step for decimation. Applies action smoothing
        and clamps actions to joint limits.
        """
        self.joint_pos_cmd[:, self.actuated_dof_indices] = scale(
            self.actions,
            self.robot_joint_pos_lower_limits[self.actuated_dof_indices],
            self.robot_joint_pos_upper_limits[self.actuated_dof_indices],
        )
        self.joint_pos_cmd[:, self.actuated_dof_indices] = (
            self.cfg.act_moving_average * self.joint_pos_cmd[:, self.actuated_dof_indices]
            + (1.0 - self.cfg.act_moving_average) * self.prev_joint_pos_cmd[:, self.actuated_dof_indices]
        )
        self.joint_pos_cmd[:, self.actuated_dof_indices] = saturate(
            self.joint_pos_cmd[:, self.actuated_dof_indices],
            self.robot_joint_pos_lower_limits[self.actuated_dof_indices],
            self.robot_joint_pos_upper_limits[self.actuated_dof_indices],
        )

        self.prev_joint_pos_cmd[:, self.actuated_dof_indices] = self.joint_pos_cmd[:, self.actuated_dof_indices]

        self.robot.set_joint_position_target(
            self.joint_pos_cmd[:, self.actuated_dof_indices], joint_ids=self.actuated_dof_indices
        )

    def get_observations(self):
        """Get observations for the current timestep.

        Returns:
            Dictionary of observations.
        """
        return self._get_observations()

    def _get_observations(self) -> dict:
        """Collect observations according to the requested cfg keys."""
        obs_dict = {}
        for k in self.cfg.obs_list:
            if k == "prop":
                obs_dict[k] = self._get_proprioception()
            elif k == "gt":
                obs_dict[k] = self._get_gt()
            elif k == "tactile":
                obs_dict[k] = self._get_tactile()
            elif k == "pixels":
                obs_dict[k] = self._get_pixels()
            else:
                raise ValueError(f"Unknown observation key '{k}'")

        obs_dict = {"policy": obs_dict}
        return obs_dict

    def _get_proprioception(self):
        """Return proprioceptive feature vector.

        Returns:
            Concatenated tensor containing normalized joint positions, normalized joint
            velocities, and actions.
        """
        prop = torch.cat(
            (
                self.normalised_joint_pos,
                self.normalised_joint_vel,
                self.actions,
            ),
            dim=-1,
        )

        return prop

    def _get_pixels(self) -> torch.Tensor:
        """Return rendered pixel observations.

        Processes RGB and depth camera data, handling edge cases like inf/NaN values
        in depth images and applying normalization to RGB images.

        Returns:
            Concatenated tensor of processed camera data.
        """
        if self.pixel_cfg is None:
            raise ValueError("pixel_cfg is not set. Make sure 'pixels' is in obs_list and pixel_cfg is provided in agent_cfg.")

        processed_data = []

        for data_type in self.pixel_cfg["types"]:
            # Clone the specific buffer
            data = self._tiled_camera.data.output[data_type].clone()

            if data_type == "depth":
                # Handle inf and NaN values which are common in depth sensors
                data[torch.isinf(data)] = 0.0
                data[torch.isnan(data)] = 0.0

            elif data_type == "rgb" and self.pixel_cfg["normalise_rgb"]:
                # Normalize RGB: convert to float, center by subtracting mean, then scale back
                data = data.float() / 255.0
                mean_tensor = torch.mean(data, dim=(1, 2), keepdim=True)
                data -= mean_tensor
                data = 255.0 * data  # Scale back to [0, 255]
                data = data.to(torch.uint8)

            processed_data.append(data)

        # Concatenate the processed tensors along the channel dimension
        camera_data = torch.cat(processed_data, dim=-1)

        return camera_data

    def _get_tactile(self):
        """Return tactile force.

        Computes contact forces from multiple sensors and converts them to binary
        activations based on a threshold.

        Returns:
            Concatenated tensor of tactile force.
        """

        # Forces is [num_envs, num_sensors, 3], take the norm to be [num_envs, num_sensors]
        forces = self.robot_contact_sensor.data.net_forces_w[:].clone()
        norm = torch.linalg.vector_norm(forces, dim=-1, keepdim=False)

        # Convert to binary activations based on threshold if binary_tactile is True
        if self.tactile_cfg is not None and self.tactile_cfg["binary_tactile"]:
            norm = (norm > self.binary_threshold).float()
            return norm
        else:
            return norm

    def _reset_robot(self, env_ids, joint_pos_noise=0.125):
        """Reset the robot joint positions and velocities.

        Args:
            env_ids: Environment indices to reset.
            joint_pos_noise: Standard deviation of noise added to joint positions.
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
        """Compute intermediate values for observations and rewards.

        Updates joint positions, velocities, accelerations, and their normalized versions.

        Args:
            env_ids: Environment indices to update.
        """
        # Get robot data
        self.joint_pos[env_ids] = self.robot.data.joint_pos[env_ids]
        self.joint_vel[env_ids] = self.robot.data.joint_vel[env_ids]
        self.joint_acc[env_ids] = self.robot.data.joint_acc[env_ids]

        # Normalize joint positions
        self.normalised_joint_pos[env_ids] = unscale(
            self.joint_pos[env_ids], self.robot_joint_pos_lower_limits, self.robot_joint_pos_upper_limits
        )
        # Normalize velocities by dividing by a fixed scale factor
        # Note: An alternative normalization using joint velocity limits is commented below
        self.normalised_joint_vel[env_ids] = self.joint_vel[env_ids] / 3.0
        # self.normalised_joint_vel[env_ids] = unscale(
        #     self.joint_vel[env_ids], -self.robot_joint_vel_limits, self.robot_joint_vel_limits
        # )


@torch.jit.script
def scale(x, lower, upper):
    """Scale input `x` from [-1, 1] to [lower, upper]."""
    return 0.5 * (x + 1.0) * (upper - lower) + lower


@torch.jit.script
def unscale(x, lower, upper):
    """Unscale input `x` from [lower, upper] to [-1, 1]."""
    return (2.0 * x - upper - lower) / (upper - lower)

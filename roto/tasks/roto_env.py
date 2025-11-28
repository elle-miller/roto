# Shared Franka parent environment for IsaacLab RL tasks.
#
# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Shared Franka parent environment for IsaacLab RL tasks.

This module provides a configurable RL environment for the Franka Panda robot,
including simulation setup, sensors, and reward utilities.
"""

from __future__ import annotations

import torch
import numpy as np
import gymnasium as gym

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

    Attributes:
        physics_dt: Simulation timestep in seconds.
        decimation: Number of physics steps per control step.
        render_interval: Physics steps per rendering step.
        observation_space: Placeholder for compatibility.
        state_space: Placeholder for compatibility.
        sim: SimulationCfg instance for physics and material settings.
        scene: InteractiveSceneCfg describing number of envs and spacing.
        viewer: ViewerCfg for camera defaults.
    """
    physics_dt = 1 / 120
    decimation = 2
    render_interval = 2

    # Isaac 4.5 compatibility placeholders
    observation_space = 0
    state_space = 0

    sim: SimulationCfg = SimulationCfg(
        dt=physics_dt,
        render_interval=decimation,
        physics_material=RigidBodyMaterialCfg(
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        physx=PhysxCfg(bounce_threshold_velocity=0.2),
    )

    replicate_physics = True
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=4096, env_spacing=1.5, replicate_physics=replicate_physics
    )

    eye = (3, 3, 3)
    lookat = (0, 0, 0)
    viewer: ViewerCfg = ViewerCfg(eye=eye, lookat=lookat, resolution=(1920, 1080))


class RotoEnv(DirectRLEnv):
    """
    RL environment for the Roto benchmark.

    Handles simulation setup, action application, observation collection, and resets.
    """

    cfg: RotoEnvCfg

    def __init__(self, cfg: RotoEnvCfg, render_mode: str | None = None, **kwargs) -> None:
        """
        Initialize the Franka RL environment.

        Args:
            cfg: RotoEnvCfg instance with simulation and scene parameters.
            render_mode: Optional rendering mode string.
            **kwargs: Additional arguments passed to the parent class.
        """
        super().__init__(cfg, render_mode, **kwargs)
        self.cfg = cfg
        self.obs_stack = getattr(cfg, "obs_stack", 1)
        self.dtype = torch.float32
        self.binary_tactile = True
        self.binary_threshold = 0.01

        # Joint limits and targets (kept as in original logic)
        # Note: self.robot and self.device are provided by DirectRLEnv parent.
        self.robot_joint_pos_lower_limits = self.robot.data.soft_joint_pos_limits[0, :, 0].to(
            device=self.device
        )

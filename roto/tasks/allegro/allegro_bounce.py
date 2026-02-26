# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Author: Elle Miller 2025

Allegro-hand bounce task environment.
"""

from __future__ import annotations

import torch
from collections.abc import Sequence

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObject, RigidObjectCfg
from isaaclab.sim.schemas.schemas_cfg import CollisionPropertiesCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import sample_uniform

from roto.tasks.allegro.allegro import AllegroEnv, AllegroEnvCfg


@configclass
class BounceCfg(AllegroEnvCfg):
    """Configuration parameters for the Allegro bounce task."""

    fall_height = 0.3
    reset_position_noise = 0.01
    object_y_pos = 0.0
    object_z_pos = 0.55
    default_object_pos = (0.1, object_y_pos, object_z_pos)
    out_of_bounds = 0.2
    min_timesteps_between_contact = 5

    ball_colour = (0.7294117647058823, 0.3176470588235294, 0.7137254901960784)

    radius_m = 0.035
    mass_g = 30
    mass_kg = mass_g / 1000
    object_cfg: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Object",
        init_state=RigidObjectCfg.InitialStateCfg(pos=default_object_pos, rot=(1.0, 0.0, 0.0, 0.0)),
        spawn=sim_utils.SphereCfg(
            radius=radius_m,
            physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=1.0, dynamic_friction=1.0, restitution=0.0),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=ball_colour, metallic=0.1),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=False,
                disable_gravity=False,
                enable_gyroscopic_forces=True,
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=0,
                sleep_threshold=0.005,
                stabilization_threshold=0.0025,
                max_depenetration_velocity=1000.0,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=mass_kg),
            collision_props=CollisionPropertiesCfg(collision_enabled=True),
        ),
    )


class BounceEnv(AllegroEnv):
    """Bounce a sphere using the Allegro hand and binary tactile signals."""

    cfg: BounceCfg

    def __init__(self, cfg: BounceCfg, render_mode: str | None = None, **kwargs):

        super().__init__(cfg, render_mode, **kwargs)

        # Object and tracking tensors
        self.object_pos = torch.zeros((self.num_envs, 3), device=self.device)
        self.object_rot = torch.zeros((self.num_envs, 4), device=self.device)
        self.object_linvel = torch.zeros((self.num_envs, 3), device=self.device)
        self.object_angvel = torch.zeros((self.num_envs, 3), device=self.device)

        self.num_transitions = torch.zeros((self.num_envs,), dtype=self.dtype, device=self.device)
        self.num_bounces = torch.zeros((self.num_envs,), dtype=self.dtype, device=self.device)
        self.waiting_for_contact = torch.zeros((self.num_envs,), dtype=torch.bool, device=self.device)
        self.new_bounces = torch.zeros((self.num_envs,), dtype=self.dtype, device=self.device)
        self.time_without_contact = torch.zeros((self.num_envs,), dtype=torch.int, device=self.device)
        self.tactile = super()._get_tactile()
        self.last_tactile = self.tactile.clone()

    def _get_gt(self):
        """Get ground-truth features for auxiliary heads/logging.

        Returns:
            Concatenated tensor containing object pose, velocities, and contact tracking information.
        """
        gt = torch.cat(
            (
                self.object_pos,
                self.object_rot,
                self.object_linvel,
                self.object_angvel,
                torch.norm(self.object_linvel, dim=1).unsqueeze(-1),
                torch.norm(self.object_angvel, dim=1).unsqueeze(-1),
                self.time_without_contact.unsqueeze(-1),
            ),
            dim=-1,
        )
        return gt

    def _setup_scene(self):
        """Attach the object to the shared scene."""
        super()._setup_scene()

        self.object = RigidObject(self.cfg.object_cfg)
        self.scene.rigid_objects["object"] = self.object

    def _get_tactile(self):
        """Return tactile force."""

        tactile = super()._get_tactile()
        self.last_tactile = self.tactile
        self.tactile = tactile

        return tactile

    def _compute_intermediate_values(self, env_ids=None):
        """Track object kinematics and bounce counts."""
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        super()._compute_intermediate_values(env_ids)

        # Object pose and relative distances
        self.object_pos[env_ids] = self.object.data.root_pos_w[env_ids] - self.scene.env_origins[env_ids]
        self.object_rot[env_ids] = self.object.data.root_quat_w[env_ids]
        self.object_linvel[env_ids] = self.object.data.root_lin_vel_w[env_ids]
        self.object_angvel[env_ids] = self.object.data.root_ang_vel_w[env_ids]

        prev_contact = (torch.sum(self.last_tactile, dim=1) > 0).int()
        curr_contact = (torch.sum(self.tactile, dim=1) > 0).int()

        lost_contact = (prev_contact == 1) & (curr_contact == 0)
        new_contact = (prev_contact == 0) & (curr_contact == 1)

        prev_time_without_contact = self.time_without_contact.clone()

        self.time_without_contact = torch.where(
            curr_contact == 0,
            self.time_without_contact + 1,
            torch.zeros_like(self.time_without_contact),
        )

        valid_new_contact = new_contact & (prev_time_without_contact >= self.cfg.min_timesteps_between_contact)

        self.new_bounces = (self.waiting_for_contact & valid_new_contact).float()
        self.num_bounces += self.new_bounces
        self.waiting_for_contact = self.waiting_for_contact | lost_contact
        self.waiting_for_contact = self.waiting_for_contact & ~valid_new_contact

        self.num_transitions += (lost_contact | valid_new_contact).float()

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Terminate when the ball falls or leaves the workspace."""
        self._compute_intermediate_values()

        fall = self.object_pos[:, 2] < self.cfg.fall_height
        out_of_bounds = torch.abs(self.object_pos[:, 1] - self.cfg.object_y_pos) > self.cfg.out_of_bounds
        termination = fall | out_of_bounds
        time_out = self.episode_length_buf >= self.max_episode_length - 1

        return termination, time_out

    def _reset_idx(self, env_ids: Sequence[int] | None):
        """Reset articulation and ball state for the selected envs."""

        if env_ids is None:
            env_ids = self.robot._ALL_INDICES

        super()._reset_idx(env_ids)

        self._reset_object(env_ids)

        self.num_transitions[env_ids] = 0
        self.num_bounces[env_ids] = 0
        self.new_bounces[env_ids] = 0
        self.waiting_for_contact[env_ids] = False
        self.time_without_contact[env_ids] = 0

    def _reset_object(self, env_ids):
        """Randomize the ball pose within a small neighbourhood.

        Args:
            env_ids: Environment indices to reset.
        """
        object_default_state = self.object.data.default_root_state.clone()[env_ids]
        pos_noise = sample_uniform(-1.0, 1.0, (len(env_ids), 3), device=self.device)
        object_default_state[:, 0:3] = (
            object_default_state[:, 0:3] + self.cfg.reset_position_noise * pos_noise + self.scene.env_origins[env_ids]
        )

        object_default_state[:, 7:] = torch.zeros_like(self.object.data.default_root_state[env_ids, 7:])
        self.object.write_root_pose_to_sim(object_default_state[:, :7], env_ids)
        self.object.write_root_velocity_to_sim(object_default_state[:, 7:], env_ids)

    def _get_rewards(self) -> torch.Tensor:
        """Reward bounces."""
        total_reward, bounce_reward = compute_rewards(self.new_bounces)

        self.extras["log"] = {
            "object_z_linvel": (self.object_linvel[:, 2].half()),
            "object_z_angvel": (self.object_angvel[:, 2].half()),
            "sum_forces": (torch.sum(self.tactile, dim=1)),
            "bounce_reward": (bounce_reward).float(),
        }

        self.extras["counters"] = {
            "num_bounces": (self.num_bounces).float(),
        }

        return total_reward


@torch.jit.script
def compute_rewards(new_bounces: torch.Tensor):
    """Compute rewards for ball bouncing task.

    Args:
        new_bounces: Tensor indicating new bounce events.

    Returns:
        Tuple of (total_reward, bounce_reward).
    """
    bounce_reward = new_bounces * 10

    total_reward = bounce_reward

    return total_reward, bounce_reward

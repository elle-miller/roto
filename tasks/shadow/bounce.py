# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


from __future__ import annotations


import torch
from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import DirectRLEnv
from isaaclab.markers import VisualizationMarkers
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils.math import quat_conjugate, quat_from_angle_axis, quat_mul, sample_uniform, saturate
import isaaclab.sim as sim_utils


from isaaclab.utils import configclass
from isaaclab.assets import ArticulationCfg, RigidObjectCfg
from isaaclab.sim.spawners.materials.physics_materials_cfg import RigidBodyMaterialCfg
from isaaclab.sim.schemas.schemas_cfg import CollisionPropertiesCfg, MassPropertiesCfg, RigidBodyPropertiesCfg

from tasks.shadow.shadow import ShadowEnvCfg, ShadowEnv

"""
Repose environment

every child env should implement own
- _get_rewards
- _get_dones
- compute_rewards


"""

@configclass
class BounceCfg(ShadowEnvCfg):
    episode_length_s = 10.0  # Episode length in seconds

    act_moving_average  = 1
    fall_height = 0.3
    reset_position_noise = 0.01  # range of position at reset
    object_y_pos = -0.39
    object_z_pos = 0.6
    default_object_pos = (0.0, object_y_pos, object_z_pos)
    out_of_bounds = 0.2
    min_timesteps_between_contact = 5

    brat = (0.5411764705882353, 0.807843137254902, 0)
    brat_pink = (0.7294117647058823, 0.3176470588235294, 0.7137254901960784)
    colour_2 = (0.741176, 0.878, 0.9960784)

    colour_1 = brat
    colour_2 = brat_pink
    # in-hand object
    # based off a stress ball, 70mm diameter and 30g weight
    radius_m = 0.035
    mass_g = 30
    mass_kg = mass_g / 1000 
    object_cfg: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Object",
        init_state=RigidObjectCfg.InitialStateCfg(pos=default_object_pos, rot=(1.0, 0.0, 0.0, 0.0)),
        spawn=sim_utils.SphereCfg(
            radius=radius_m,
            physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=1.0, dynamic_friction=1.0, restitution=0.0),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=colour_2, metallic=0.1),
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
            collision_props=CollisionPropertiesCfg(collision_enabled=True)
        ),
    )

    num_gt_observations = 14


class BounceEnv(ShadowEnv):
    cfg: BounceCfg

    def __init__(self, cfg: BounceCfg, render_mode: str | None = None, **kwargs):

        super().__init__(cfg, render_mode, **kwargs)

        # Object and tracking tensors
        self.object_pos = torch.zeros((self.num_envs, 3), device=self.device)
        self.object_rot = torch.zeros((self.num_envs, 4), device=self.device)
        self.object_linvel = torch.zeros((self.num_envs, 3), device=self.device)
        self.object_angvel = torch.zeros((self.num_envs, 3), device=self.device)

        self.num_transitions = torch.zeros((self.num_envs, ), dtype=self.dtype, device=self.device)
        self.num_bounces = torch.zeros((self.num_envs, ), dtype=self.dtype, device=self.device)
        self.waiting_for_contact = torch.zeros((self.num_envs, ), dtype=torch.bool, device=self.device)
        self.new_bounces = torch.zeros((self.num_envs, ), dtype=self.dtype, device=self.device)
        self.time_without_contact = torch.zeros((self.num_envs, ), dtype=torch.int, device=self.device)

    
    def _get_gt(self):

        gt = torch.cat(
            (
                self.object_pos,
                self.object_rot,
                self.object_linvel,
                self.object_angvel,
            ),
            dim=-1,
        )
        return gt
    
    def _setup_scene(self):
        super()._setup_scene()

        self.object = RigidObject(self.cfg.object_cfg)
        self.scene.rigid_objects["object"] = self.object

    def _compute_intermediate_values(self, env_ids=None):
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        super()._compute_intermediate_values(env_ids)

        # Object pose and relative distances
        self.object_pos[env_ids] = self.object.data.root_pos_w[env_ids] - self.scene.env_origins[env_ids]
        self.object_rot[env_ids] = self.object.data.root_quat_w[env_ids]
        self.object_linvel[env_ids] = self.object.data.root_lin_vel_w[env_ids]
        self.object_angvel[env_ids] = self.object.data.root_ang_vel_w[env_ids]

        # Check if there was any contact
        prev_contact = (torch.sum(self.last_tactile, dim=1) > 0).int()
        curr_contact = (torch.sum(self.tactile, dim=1) > 0).int()
    
        # Identify specific transitions
        lost_contact = (prev_contact == 1) & (curr_contact == 0)
        new_contact = (prev_contact == 0) & (curr_contact == 1)

        # Store the current time without contact before updating
        prev_time_without_contact = self.time_without_contact.clone()

        # Update time without contact
        self.time_without_contact = torch.where(
            curr_contact == 0,  # If no contact now
            self.time_without_contact + 1,  # Increment counter
            torch.zeros_like(self.time_without_contact)  # Reset counter if contact
        )

        # Valid transitions are those that happen after minimum time
        valid_new_contact = new_contact & (prev_time_without_contact >= self.cfg.min_timesteps_between_contact)
   
        # If we're waiting for contact and it happens, count a bounce
        self.new_bounces = (self.waiting_for_contact & valid_new_contact).float()
        self.num_bounces += self.new_bounces
        # 
        # If we just lost a contact, we start waiting for contact
        self.waiting_for_contact = self.waiting_for_contact | lost_contact
        # If we have just come into contact, we're no longer waiting for contact
        # make the new_contact 1s into 0s with ~, and disable the mask!
        self.waiting_for_contact = self.waiting_for_contact & ~valid_new_contact

        # Update transitions count if you still want to track this
        self.num_transitions += (lost_contact | valid_new_contact).float()


    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        self._compute_intermediate_values()

        # reset when cube has fallen
        fall = self.object_pos[:,2] < self.cfg.fall_height
        out_of_bounds = (torch.abs(self.object_pos[:,1] - self.cfg.object_y_pos) > self.cfg.out_of_bounds)
        termination = fall | out_of_bounds
        time_out = self.episode_length_buf >= self.max_episode_length - 1

        return termination, time_out
    

    def _reset_idx(self, env_ids: Sequence[int] | None):
        
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES

        # resets articulation and rigid body attributes
        super()._reset_idx(env_ids)

        # reset object
        self._reset_object(env_ids)

        self.num_transitions[env_ids] = 0
        self.num_bounces[env_ids] = 0
        self.new_bounces[env_ids] = 0
        self.waiting_for_contact[env_ids] = False
        self.time_without_contact[env_ids] = 0


    def _reset_object(self, env_ids):

        object_default_state = self.object.data.default_root_state.clone()[env_ids]
        pos_noise = sample_uniform(-1.0, 1.0, (len(env_ids), 3), device=self.device)
        object_default_state[:, 0:3] = (
            object_default_state[:, 0:3] + self.cfg.reset_position_noise * pos_noise + self.scene.env_origins[env_ids]
        )

        object_default_state[:, 7:] = torch.zeros_like(self.object.data.default_root_state[env_ids, 7:])
        self.object.write_root_pose_to_sim(object_default_state[:, :7], env_ids)
        self.object.write_root_velocity_to_sim(object_default_state[:, 7:], env_ids)
        

    def _get_rewards(self) -> torch.Tensor:

        (
            total_reward,
            bounce_reward,
            air_reward,
            
        ) = compute_rewards(
            self.new_bounces,
            self.time_without_contact
        )


        self.extras["log"] = {
            "object_z_linvel": (self.object_linvel[:,2].half()),
            "object_z_angvel": (self.object_angvel[:,2].half()),
            "sum_forces": (torch.sum(self.tactile, dim=1)),
            "bounce_reward": (bounce_reward).float(),
        }

        # self.extras["counters"] = {
        #     "num_transitions": (self.num_transitions).float(),
        #     "time_without_contact": (self.time_without_contact).float(),
        #     "num_bounces": (self.num_bounces).float(),
        # }
        
        return total_reward
    

        

@torch.jit.script
def compute_rewards(
    new_bounces: torch.Tensor,
    time_without_contact: torch.Tensor

):

    bounce_reward = new_bounces * 1
    air_reward = time_without_contact * 0

    total_reward = bounce_reward

    return total_reward, bounce_reward, air_reward,

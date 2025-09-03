# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


from __future__ import annotations


import torch

import isaaclab.sim as sim_utils


from isaaclab.utils import configclass
from isaaclab.assets import ArticulationCfg, RigidObjectCfg
from isaaclab.sim.spawners.materials.physics_materials_cfg import RigidBodyMaterialCfg
from isaaclab.sim.schemas.schemas_cfg import CollisionPropertiesCfg, MassPropertiesCfg, RigidBodyPropertiesCfg

from tasks.shadow.shadow_hand_env_cfg import ShadowHandEnvCfg
from tasks.shadow.inhand import InHandManipulationEnv, scale, unscale, randomize_rotation, rotation_distance

"""
Repose environment

every child env should implement own
- _get_rewards
- _get_dones
- compute_rewards


"""

@configclass
class BounceCfg(ShadowHandEnvCfg):

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
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, -0.39, 0.6), rot=(1.0, 0.0, 0.0, 0.0)),
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

    # reward scales
    action_penalty_scale = -0.0002
    reach_goal_bonus = 250
    fall_penalty = 0

    # fall_dist = 0.25
    fall_dist = 0.15

    # success_tolerance=0.10

    num_gt_observations = 14


class BounceEnv(InHandManipulationEnv):
    cfg: BounceCfg

    def __init__(self, cfg: BounceCfg, render_mode: str | None = None, **kwargs):

        self.obs_stack = 1
        super().__init__(cfg, render_mode, **kwargs)

        self.object_height_above_hand = torch.zeros((self.num_envs, ), dtype=self.dtype, device=self.device)
        self.num_transitions = torch.zeros((self.num_envs, ), dtype=self.dtype, device=self.device)
        self.num_bounces = torch.zeros((self.num_envs, ), dtype=self.dtype, device=self.device)
        self.waiting_for_contact = torch.zeros((self.num_envs, ), dtype=torch.bool, device=self.device)
        self.new_bounces = torch.zeros((self.num_envs, ), dtype=self.dtype, device=self.device)
        self.time_without_contact = torch.zeros((self.num_envs, ), dtype=torch.int, device=self.device)

        self.total_rew = 0

        self.bounce_rew = 0
        self.time = 0

    
    def _get_gt(self):

        gt = torch.cat(
            (
                # object
                self.object_pos,
                self.object_rot,
                self.object_linvel,
                self.cfg.vel_obs_scale * self.object_angvel,
                self.object_height_above_hand.unsqueeze(1),
            ),
            dim=-1,
        )
        # print("gt", gt.size())
        return gt
    

    def _compute_intermediate_values(self, reset=False, env_ids=None):
        self.hand_dof_pos = self.hand.data.joint_pos
        self.hand_dof_vel = self.hand.data.joint_vel

        # data for object
        self.object_pos = self.object.data.root_pos_w - self.scene.env_origins
        self.object_rot = self.object.data.root_quat_w
        self.object_velocities = self.object.data.root_vel_w
        self.object_linvel = self.object.data.root_lin_vel_w
        self.object_angvel = self.object.data.root_ang_vel_w

        # compute tactile
        self._get_tactile()

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

        # Minimum number of timesteps required between transitions
        min_timesteps = 10  # Adjust this value based on your simulation frequency

        # Valid transitions are those that happen after minimum time
        valid_new_contact = new_contact & (prev_time_without_contact >= min_timesteps)
   
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

        if reset == True:
            # print("reset, ", env_ids)
            self.num_transitions[env_ids] = 0
            self.num_bounces[env_ids] = 0
            self.new_bounces[env_ids] = 0
            self.waiting_for_contact[env_ids] = False
            self.time_without_contact[env_ids] = 0
        

    def _get_rewards(self) -> torch.Tensor:


        (
            total_reward,
            bounce_reward,
            air_reward,
            fall_penalty,
            self.successes[:],
            self.consecutive_successes[:]
        ) = compute_rewards(
            self.reset_buf,
            self.successes,
            self.consecutive_successes,
            self.tactile,
            self.object_pos,
            self.in_hand_pos,
            self.new_bounces,
            self.time_without_contact
        )

        # print(self.time, self.num_bounces[0])
        # self.time += 1
        # self.total_rew += total_reward[10]

        # self.bounce_rew += bounce_reward[10]
        # print(self.time, self.total_rew, self.bounce_rew, self.time_without_contact[10])
        # print(self.successes)
        # do not provide county things here!!!!
        # these things are all added up
        self.extras["log"] = {
            "fall_penalty": (fall_penalty).float(),
            "object_height": (self.object_height_above_hand.half()),
            "object_z_linvel": (self.object_linvel[:,2].half()),
            "object_z_angvel": (self.object_angvel[:,2].half()),
            "sum_forces": (torch.sum(self.tactile, dim=1)),
            "air_reward": (air_reward).float(),
            "bounce_reward": (bounce_reward).float(),
        }

        self.extras["counters"] = {
            "num_transitions": (self.num_transitions).float(),
            "time_without_contact": (self.time_without_contact).float(),
            "num_bounces": (self.num_bounces).float(),
            "consecutive_successes": (self.consecutive_successes),
            "successes": (self.successes),
        }
        
        return total_reward
    

        

@torch.jit.script
def compute_rewards(
    reset_buf: torch.Tensor,
    successes: torch.Tensor,
    consecutive_successes: torch.Tensor,
    tactile: torch.Tensor,
    object_pos: torch.Tensor,
    in_hand_pos: torch.Tensor,
    new_bounces: torch.Tensor,
    time_without_contact: torch.Tensor

):
    success_bonus = 1.0
    fall_penalty = -10.0
    fall_dist = 0.15
    av_factor = 0.1

    # sum of forces in each environment
    sum_forces = torch.sum(tactile, dim=1)
    no_contact = sum_forces == 0.0
    # success_reward = torch.where(no_contact, success_bonus, 0).float()
    # success_reward = time_without_contact #torch.where(no_contact, success_bonus, 0).float()
    bounce_reward = new_bounces * 10
    air_reward = time_without_contact * 0.01

    # penalties
    hand_object_dist =  torch.norm(object_pos - in_hand_pos, p=2, dim=-1)
    failed_envs = hand_object_dist > fall_dist
    fall_penalty = torch.where(failed_envs, fall_penalty, 0).float()
    resets = torch.where(failed_envs, torch.ones_like(reset_buf), reset_buf)
    num_resets = torch.sum(resets)

    # Find out which envs hit the goal and update successes count
    new_successes = torch.where(bounce_reward > 0, 1, 0)
    successes = successes + new_successes
    finished_cons_successes = torch.sum(successes * resets.float())
    cons_successes = torch.where(
        num_resets > 0,
        av_factor * finished_cons_successes / num_resets + (1.0 - av_factor) * consecutive_successes,
        consecutive_successes,
    )

    total_reward = bounce_reward + air_reward + fall_penalty

    return total_reward, bounce_reward, air_reward, fall_penalty, successes, cons_successes

# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.math import (
    quat_conjugate,
    quat_mul,
    sample_uniform,
)

from tasks.template.my_robot import MyRobotEnv, MyRobotEnvCfg, randomize_rotation, rotation_distance


@configclass
class MyTaskEnvCfg(MyRobotEnvCfg):

    # reset config
    reset_object_position_noise = 0.05
    reset_goal_position_noise = 0.01  # scale factor for -1 to 1 m
    default_goal_pos = [0.6, 0.0, 0.4]
    default_object_pos = [0.5, 0, 0.055]

    # lift stuff
    object_goal_tracking_scale = 16.0
    object_goal_tracking_finegrained_scale = 5.0

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
            # mass_props=sim_utils.MassPropertiesCfg(mass=1000.0)
        ),
    )

    # goal object
    goal_object_cfg: VisualizationMarkersCfg = VisualizationMarkersCfg(
        prim_path="/Visuals/goal_marker",
        markers={
            "goal": sim_utils.UsdFileCfg(
                usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/DexCube/dex_cube_instanceable.usd",
                scale=(1.0, 1.0, 1.0),
            )
        },
    )

    num_gt_observations = 18


class MyTaskEnv(MyRobotEnv):
    # pre-physics step calls
    #   |-- _pre_physics_step(action)
    #   |-- _apply_action()
    # post-physics step calls
    #   |-- _get_dones()
    #   |-- _get_rewards()
    #   |-- _reset_idx(env_ids)
    #   |-- _get_observations()

    cfg: MyTaskEnvCfg

    def __init__(self, cfg: MyTaskEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self.goal_pos = torch.zeros((self.num_envs, 3), dtype=torch.float, device=self.device)
        self.goal_pos[:, :] = torch.tensor(self.cfg.default_goal_pos, device=self.device)
        self.goal_rot = torch.zeros((self.num_envs, 4), dtype=torch.float, device=self.device)
        # self.goal_rot[:, :] = torch.tensor(self.cfg.default_goal_rot, device=self.device)

        self.object_goal_distance = torch.zeros((self.num_envs, 3), device=self.device)
        self.object_goal_euclidean_distance = torch.zeros((self.num_envs,), device=self.device)
        self.object_goal_rotation = torch.zeros((self.num_envs, 4), device=self.device)
        self.object_goal_angular_distance = torch.zeros((self.num_envs,), device=self.device)
        # save reward weights so they can be adjusted online
        self.object_goal_tracking_scale = cfg.object_goal_tracking_scale
        self.object_goal_tracking_finegrained_scale = cfg.object_goal_tracking_finegrained_scale

        # default goal positions
        self.default_goal_pos = torch.zeros((self.num_envs, 3), dtype=torch.float, device=self.device)
        self.default_goal_pos[:, :] = torch.tensor(self.cfg.default_goal_pos, device=self.device)

    def _setup_scene(self):
        super()._setup_scene()
        self.goal_markers = VisualizationMarkers(self.cfg.goal_object_cfg)

    def _get_gt(self):

        gt = torch.cat(
            (
                # xyz diffs (3,)
                self.object_ee_distance,
                self.object_goal_distance,
                # rotation quaternion (4,)
                self.object_ee_rotation,
                self.object_goal_rotation,
                # rotation quaternion (4,)
                self.object_ee_angular_distance.unsqueeze(1),
                self.object_goal_angular_distance.unsqueeze(1),
                # euclidean distances (1,) [transform from (num_envs,) to (num_envs,1)]
                self.object_goal_euclidean_distance.unsqueeze(1),
                self.object_ee_euclidean_distance.unsqueeze(1),
            ),
            dim=-1,
        )
        return gt

    def _get_rewards(self) -> torch.Tensor:

        (
            rewards,
            reaching_object,
            is_lifted,
            object_goal_tracking,
            joint_vel_penalty,
        ) = compute_rewards(
            self.reaching_object_scale,
            self.lift_object_scale,
            self.episode_length_buf,
            self.object_goal_tracking_scale,
            self.joint_vel_penalty_scale,
            self.object_pos,
            self.joint_vel,
            self.object_ee_euclidean_distance,
            self.object_goal_euclidean_distance,
            self.cfg.minimal_height,
        )

        self.extras["log"] = {
            "reach_reward": (reaching_object),
            "lift_reward": (is_lifted),
            "object_goal_tracking": (object_goal_tracking),
            "joint_vel_penalty": (joint_vel_penalty),
        }

        if "tactile" in self.cfg.obs_list:
            tactile_dict = {
                "normalised_forces_left_x": (self.normalised_forces[:, 0]),
                "normalised_forces_right_x": (self.normalised_forces[:, 1]),
            }
            self.extras["log"].update(tactile_dict)

        self.extras["counters"] = {}

        return rewards

    def _reset_target_pose(self, env_ids):
        # reset goal rotation
        rand_floats = sample_uniform(-1.0, 1.0, (len(env_ids), 2), device=self.device)
        new_rot = randomize_rotation(
            rand_floats[:, 0], rand_floats[:, 1], self.x_unit_tensor[env_ids], self.y_unit_tensor[env_ids]
        )
        self.goal_rot[env_ids] = new_rot

        # reset goal position
        goal_pos_noise = sample_uniform(-1.0, 1.0, (len(env_ids), 3), device=self.device)
        self.goal_pos[env_ids] = self.default_goal_pos[env_ids] + goal_pos_noise * self.cfg.reset_goal_position_noise

        # visualise goals (need to express in world frame)
        goal_pos_world = self.goal_pos + self.scene.env_origins
        self.goal_markers.visualize(goal_pos_world, self.goal_rot)

    def _compute_intermediate_values(self, reset=False, env_ids: torch.Tensor | None = None):
        super()._compute_intermediate_values()
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES

        self.object_goal_distance[env_ids] = self.object_pos[env_ids] - self.goal_pos[env_ids]
        self.object_goal_euclidean_distance[env_ids] = torch.norm(self.object_goal_distance[env_ids], dim=1)
        self.object_goal_rotation[env_ids] = quat_mul(self.object_rot[env_ids], quat_conjugate(self.goal_rot[env_ids]))
        self.object_goal_angular_distance[env_ids] = rotation_distance(self.object_rot[env_ids], self.goal_rot[env_ids])


from tasks.franka.franka import distance_reward, joint_vel_penalty, lift_reward, object_goal_reward


@torch.jit.script
def compute_rewards(
    reaching_object_scale: float,
    lift_object_scale: float,
    episode_timestep_counter: torch.Tensor,
    object_goal_tracking_scale: float,
    joint_vel_penalty_scale: float,
    object_pos: torch.Tensor,
    robot_joint_vel: torch.Tensor,
    object_ee_distance: torch.Tensor,
    object_goal_distance: torch.Tensor,
    minimal_height: float,
):
    # reaching objects
    r_ee_object = distance_reward(object_ee_distance, std=0.1) * reaching_object_scale
    r_lift = lift_reward(object_pos, minimal_height, episode_timestep_counter) * lift_object_scale
    r_object_goal = object_goal_reward(object_goal_distance, r_lift, std=0.3) * object_goal_tracking_scale
    r_joint_vel = joint_vel_penalty(robot_joint_vel) * joint_vel_penalty_scale

    rewards = r_ee_object + r_lift + r_object_goal + r_joint_vel

    return (rewards, r_ee_object, r_lift, r_object_goal, r_joint_vel)

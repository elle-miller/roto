# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Author: Elle Miller 2025

Shadow-hand baoding task environment.
"""

from __future__ import annotations

import torch
from collections.abc import Sequence

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObject, RigidObjectCfg
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.sim.schemas.schemas_cfg import CollisionPropertiesCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import sample_uniform
from isaaclab.utils.math import quat_apply

from roto.tasks.orca.orca import OrcaEnv, OrcaEnvCfg


@configclass
class BaodingCfg(OrcaEnvCfg):
    """Configuration for the baoding environment running on the Shadow hand."""

    act_moving_average = 1

    # in-hand ball
    ball_mass_g = 55
    ball_mass_kg = 0.001 * ball_mass_g
    ball_diameter_inches = 1.5
    ball_radius_m = (ball_diameter_inches / 2) * 2.54 / 100
    ball_reset_height = 0.55
    ball_diameter_m = ball_radius_m * 2
    target_offset = ball_diameter_m / 1.73205080757 + 0.001

    ball_dist_terminate = 0.15
    success_tolerance = 0.01

    palm_target_x = -0.03
    palm_target_y = -0.38
    palm_target_z = 0.46
    diagonal_target_x = palm_target_x + target_offset
    diagonal_target_y = palm_target_y + target_offset
    diagonal_target_z = palm_target_z + target_offset

    colour_1 = (0.4, 0.9882352941176471, 0.011764705882352941)
    colour_2 = (0.0, 1.0, 1.0)

    ball_1_cfg: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/ball1",
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.01, -0.37, ball_reset_height), rot=(1.0, 0.0, 0.0, 0.0)),
        spawn=sim_utils.SphereCfg(
            radius=ball_radius_m,
            physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=1.0, dynamic_friction=1.0, restitution=0.0),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=colour_1, metallic=0.5),
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
            mass_props=sim_utils.MassPropertiesCfg(mass=ball_mass_kg),
            collision_props=CollisionPropertiesCfg(collision_enabled=True),
        ),
    )
    ball_2_cfg: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/ball2",
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.01, -0.41, ball_reset_height), rot=(1.0, 0.0, 0.0, 0.0)),
        spawn=sim_utils.SphereCfg(
            radius=ball_radius_m,
            physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=1.0, dynamic_friction=1.0, restitution=0.0),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=colour_2, metallic=0.5),
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
            mass_props=sim_utils.MassPropertiesCfg(mass=ball_mass_kg),
            collision_props=CollisionPropertiesCfg(collision_enabled=True),
        ),
    )
    target1_cfg: VisualizationMarkersCfg = VisualizationMarkersCfg(
        prim_path="/Visuals/target_1",
        markers={
            "target_1": sim_utils.SphereCfg(
                radius=ball_radius_m * 0.6 * 0.001, visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=colour_1)
            ),
        },
    )
    target2_cfg: VisualizationMarkersCfg = VisualizationMarkersCfg(
        prim_path="/Visuals/target_2",
        markers={
            "target_2": sim_utils.SphereCfg(
                radius=ball_radius_m * 0.6 * 0.001, visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=colour_2)
            ),
        },
    )


class BaodingEnv(OrcaEnv):
    cfg: BaodingCfg

    def __init__(self, cfg: BaodingCfg, render_mode: str | None = None, **kwargs):
        """Shadow-hand baoding task with two tracked goal locations."""
        super().__init__(cfg, render_mode, **kwargs)

        self.palm_idx = self.robot.body_names.index("palm_link")
        # these buffers are populated in the reward computation with 1 if the goal has been reached
        self.reset_goal_1_buf = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.reset_goal_2_buf = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        # euclidean distance
        self.ball_1_pos = torch.zeros((self.num_envs, 3), dtype=torch.float, device=self.device)
        self.ball_2_pos = torch.zeros((self.num_envs, 3), dtype=torch.float, device=self.device)

        self.ball_1_goal_dist = torch.ones((self.num_envs,), dtype=torch.float, device=self.device)
        self.ball_2_goal_dist = torch.ones((self.num_envs,), dtype=torch.float, device=self.device)
        self.ball_1_goal_dist3 = torch.ones((self.num_envs, 3), dtype=torch.float, device=self.device)
        self.ball_2_goal_dist3 = torch.ones((self.num_envs, 3), dtype=torch.float, device=self.device)

        # tracking ball properties
        self.ball_height_above_hand = torch.zeros((self.num_envs,), dtype=self.dtype, device=self.device)
        self.balls_center_vector = torch.zeros((self.num_envs, 3), dtype=self.dtype, device=self.device)
        self.ball_dist = torch.zeros((self.num_envs,), dtype=self.dtype, device=self.device)
        self.ball_1_linvel = torch.zeros((self.num_envs,), dtype=self.dtype, device=self.device)
        self.ball_2_linvel = torch.zeros((self.num_envs,), dtype=self.dtype, device=self.device)

        # dense reward stuff
        self.current_angle = torch.zeros((self.num_envs,), dtype=self.dtype, device=self.device)
        self.prev_angle = torch.zeros((self.num_envs,), dtype=self.dtype, device=self.device)
        self.angle_change = torch.zeros((self.num_envs,), dtype=self.dtype, device=self.device)
        self.cumulative_rotations = torch.zeros((self.num_envs,), dtype=self.dtype, device=self.device)
        self.total_rotations = torch.zeros((self.num_envs,), dtype=self.dtype, device=self.device)
        self.num_rotations = torch.zeros((self.num_envs,), dtype=torch.int, device=self.device)

        target_1 = torch.tensor(
            (self.cfg.palm_target_x, self.cfg.palm_target_y, self.cfg.palm_target_z),
            dtype=torch.float,
            device=self.device,
        )
        target_2 = torch.tensor(
            (self.cfg.diagonal_target_x, self.cfg.diagonal_target_y, self.cfg.diagonal_target_z),
            dtype=torch.float,
            device=self.device,
        )
        self.goal_pos1 = target_1.repeat(self.num_envs, 1)
        self.goal_pos2 = target_2.repeat(self.num_envs, 1)
        self.goal_rot = torch.zeros((self.num_envs, 4), dtype=torch.float, device=self.device)
        self.target1.visualize(self.goal_pos1 + self.scene.env_origins, self.goal_rot)
        self.target2.visualize(self.goal_pos2 + self.scene.env_origins, self.goal_rot)

        self.ball_goal_idx = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.update_goal_pos()


    def _setup_scene(self):
        """Register balls and visualization markers."""
        super()._setup_scene()
        # add hand, in-hand ball, and goal ball
        self.ball_1 = RigidObject(self.cfg.ball_1_cfg)
        self.ball_2 = RigidObject(self.cfg.ball_2_cfg)
        self.scene.rigid_objects["ball_1"] = self.ball_1
        self.scene.rigid_objects["ball_2"] = self.ball_2

        self.target1 = VisualizationMarkers(self.cfg.target1_cfg)
        self.target2 = VisualizationMarkers(self.cfg.target2_cfg)
            

    def _get_gt(self) -> torch.Tensor:
        """Return a concatenated tensor of ball poses and velocities.

        Returns:
            Ground-truth observation tensor containing positions, linear velocities,
            angular velocities, and inter-ball distance for both balls.
        """
        gt = torch.cat(
            (
                self.ball_1_pos,
                self.ball_2_pos,
                self.ball_1.data.root_lin_vel_w,
                self.ball_2.data.root_lin_vel_w,
                self.ball_1.data.root_ang_vel_w,
                self.ball_2.data.root_ang_vel_w,
                self.ball_dist.unsqueeze(1),
            ),
            dim=-1,
        )
        return gt


    def _compute_intermediate_values(self, env_ids=None):
        """Update ball distances, velocities and helper buffers."""
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        super()._compute_intermediate_values(env_ids)

        # data for ball
        self.ball_1_pos = self.ball_1.data.root_pos_w - self.scene.env_origins
        self.ball_2_pos = self.ball_2.data.root_pos_w - self.scene.env_origins
        self.balls_center_vector = self.ball_1_pos - self.ball_2_pos

        self.ball_1_goal_dist3 = self.ball_1_pos - self.ball_1_goal_pos
        self.ball_2_goal_dist3 = self.ball_2_pos - self.ball_2_goal_pos
        self.ball_1_goal_dist = torch.norm(self.ball_1_goal_dist3, dim=1)
        self.ball_2_goal_dist = torch.norm(self.ball_2_goal_dist3, dim=1)

        self.ball_1_linvel = torch.norm(self.ball_1.data.root_lin_vel_w, dim=1)
        self.ball_2_linvel = torch.norm(self.ball_2.data.root_lin_vel_w, dim=1)
        self.ball_dist = torch.norm(self.balls_center_vector, dim=1)

    def _get_rewards(self) -> torch.Tensor:
        """Dense distance reward plus sparse success bonus."""
        self.reset_goal_1_buf[self.ball_1_goal_dist <= self.cfg.success_tolerance] = True
        self.reset_goal_2_buf[self.ball_2_goal_dist <= self.cfg.success_tolerance] = True
        goal_reached = (self.reset_goal_1_buf & self.reset_goal_2_buf).float()
        goal_reached_ids = goal_reached.nonzero(as_tuple=False).squeeze(-1)

        (
            total_reward,
            reach_goal_reward,
        ) = compute_rewards(
            goal_reached,
            self.ball_1_goal_dist,
            self.ball_2_goal_dist,
        )

        self.extras["log"] = {
            "success_reward": (reach_goal_reward),
            "ball_1_vel": (self.ball_1_linvel),
            "ball_2_vel": (self.ball_2_linvel),
            "ball_dist": (self.ball_dist),
        }

        self.extras["counters"] = {"num_rotations": (self.num_rotations).float()}

        if len(goal_reached_ids) > 0:
            self._reset_target_pose(goal_reached_ids)

        return total_reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Reset when balls drift apart or the episode times out."""
        self._compute_intermediate_values()

        out_of_reach = self.ball_dist >= self.cfg.ball_dist_terminate

        time_out = self.episode_length_buf >= self.max_episode_length - 1

        return out_of_reach, time_out

    def _reset_idx(self, env_ids: Sequence[int] | None):
        """Reset articulation state plus both baoding balls."""

        if env_ids is None:
            env_ids = self.robot._ALL_INDICES

        super()._reset_idx(env_ids)

        self._reset_object(env_ids)

        self.num_rotations[env_ids] = 0

    def _reset_object(self, env_ids: Sequence[int]) -> None:
        """Reset both balls close to their default pose with a small position noise."""
        palm_pos_w = self.robot.data.body_pos_w[env_ids, self.palm_idx, :]    # (N,3)
        palm_quat_w = self.robot.data.body_quat_w[env_ids, self.palm_idx, :]  # (N,4) wxyz

        # base offsets in palm LOCAL frame (tune)
        z_up = self.cfg.ball_radius_m + 0.02
        x_fwd = 0.04   # 3 cm toward fingers (tune: 0.01–0.05)
        y_side = 0.04  # sideways shift if you want
        off1_local = torch.tensor([x_fwd,  y_side, z_up], device=self.device).repeat(len(env_ids), 1)
        off2_local = torch.tensor([x_fwd, -y_side, z_up], device=self.device).repeat(len(env_ids), 1)

        off1_w = quat_apply(palm_quat_w, off1_local)
        off2_w = quat_apply(palm_quat_w, off2_local)

        # noise (same as OG): ±5mm
        pos_noise = sample_uniform(-0.005, 0.005, (len(env_ids), 3), device=self.device)

        ball1_pos_w = palm_pos_w + off1_w + pos_noise
        ball2_pos_w = palm_pos_w + off2_w + pos_noise

        quat_w = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device).repeat(len(env_ids), 1)
        ball1_pose = torch.cat([ball1_pos_w, quat_w], dim=-1)
        ball2_pose = torch.cat([ball2_pos_w, quat_w], dim=-1)

        self.ball_1.write_root_pose_to_sim(ball1_pose, env_ids)
        self.ball_2.write_root_pose_to_sim(ball2_pose, env_ids)

        zeros6 = torch.zeros((len(env_ids), 6), device=self.device)
        self.ball_1.write_root_velocity_to_sim(zeros6, env_ids)
        self.ball_2.write_root_velocity_to_sim(zeros6, env_ids)

    def _reset_target_pose(self, reached_goal_ids):
        """Swap the active target for envs that completed a rotation."""

        self.ball_goal_idx[reached_goal_ids] = ~self.ball_goal_idx[reached_goal_ids]
        self.update_goal_pos()

        self.num_rotations[reached_goal_ids] += 1

        # update goal pose and markers
        self.target1.visualize(self.ball_1_goal_pos + self.scene.env_origins, self.goal_rot)
        self.target2.visualize(self.ball_2_goal_pos + self.scene.env_origins, self.goal_rot)

        self.reset_goal_1_buf[reached_goal_ids] = 0
        self.reset_goal_2_buf[reached_goal_ids] = 0

    def update_goal_pos(self):
        """Update goal positions for each ball based on `ball_goal_idx`.

        Swaps the target goals between the two balls when the index toggles.
        """
        self.ball_1_goal_pos = torch.where(
            self.ball_goal_idx.unsqueeze(-1),
            self.goal_pos2,  # When True
            self.goal_pos1,  # When False
        )

        self.ball_2_goal_pos = torch.where(
            self.ball_goal_idx.unsqueeze(-1),
            self.goal_pos1,  # When True
            self.goal_pos2,  # When False
        )


@torch.jit.script
def distance_reward(object_ee_distance, std: float = 0.1):
    """Compute smooth distance-based reward using tanh shaping.

    Args:
        object_ee_distance: Distance between object and end-effector.
        std: Standard deviation for scaling the distance.

    Returns:
        Distance reward value.
    """
    r_reach = 1 - torch.tanh(object_ee_distance / std)
    return r_reach


@torch.jit.script
def compute_rewards(
    goal_reached: torch.Tensor,
    ball_1_goal_dist: torch.Tensor,
    ball_2_goal_dist: torch.Tensor,
):
    """Compute rewards for baoding task.

    Combines dense distance shaping with a sparse success bonus.

    Args:
        goal_reached: Binary tensor indicating if both goals were reached.
        ball_1_goal_dist: Distance from ball 1 to its goal.
        ball_2_goal_dist: Distance from ball 2 to its goal.

    Returns:
        Tuple of (total_reward, reach_goal_reward).
    """
    dense_dist_reward = (distance_reward(ball_1_goal_dist) + distance_reward(ball_2_goal_dist)) * 0.1

    reach_goal_reward = torch.where(goal_reached == 1, 10, 0).float()

    total_reward = reach_goal_reward + dense_dist_reward

    return total_reward, reach_goal_reward

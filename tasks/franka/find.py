# SPDX-License-Identifier: BSD-3-Clause

"""
FrankaFind RL Task Environment

This module defines the FindEnv environment for the Franka Panda robot,
where the goal is to find and reach a target object. It provides
object pose randomization, workspace visualization, and reward tracking
for multiple difficulty thresholds.
"""

import torch
from collections.abc import Sequence
import os

from tasks.franka.franka import FrankaEnv, FrankaEnvCfg, randomize_rotation
import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg, RigidObject
from isaaclab.sim.schemas.schemas_cfg import CollisionPropertiesCfg, RigidBodyPropertiesCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import (
    quat_conjugate,
    quat_mul,
    sample_uniform,
)
from isaaclab.markers import VisualizationMarkersCfg, VisualizationMarkers

@configclass
class FindEnvCfg(FrankaEnvCfg):
    """
    Configuration for the Franka 'Find' RL task.
    Sets object and workspace properties, including randomization and visualization.
    """
    episode_length_s = 5.0  # Episode length in seconds
    act_moving_average = 0.1  # Action smoothing factor
    default_object_pos = [0.5, 0, 0.03]
    reset_object_position_noise = 0.1
    object_height_success_threshold = 0.05  # Height threshold for success

    brat = (0.541, 0.808, 0)
    brat_pink = (0.329, 0.318, 0.914)

    object_cfg: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Object",
        init_state=RigidObjectCfg.InitialStateCfg(pos=default_object_pos, rot=[1, 0, 0, 0]),
        spawn=sim_utils.CuboidCfg(
            size=[0.03, 0.03, 0.03],
            physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=1.0, dynamic_friction=0.8, restitution=0.8),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=brat_pink),
            rigid_props=RigidBodyPropertiesCfg(kinematic_enabled=False),
            mass_props=sim_utils.MassPropertiesCfg(mass=3),
            collision_props=CollisionPropertiesCfg(collision_enabled=True)
        ),
    )

    workspace_pos = [0.5, 0, 0.0]
    workspace_cfg: VisualizationMarkersCfg = VisualizationMarkersCfg(
        prim_path="/Visuals/workspace",
        markers={
            "workspace": sim_utils.CuboidCfg(
                size=(2*reset_object_position_noise, 2*reset_object_position_noise, 0.01),
                visual_material=sim_utils.PreviewSurfaceCfg(opacity=0.1, diffuse_color=brat)
            )
        },
    )

class FindEnv(FrankaEnv):
    """
    RL environment for the Franka Panda robot to find and reach an object.
    Tracks time to find the object at multiple difficulty thresholds.
    """
    cfg: FindEnvCfg

    def __init__(self, cfg: FindEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        self.cfg = cfg

        # Object and tracking tensors
        self.default_object_pos = torch.zeros((self.num_envs, 3), device=self.device)
        self.default_object_pos[:, :] = torch.tensor(self.cfg.default_object_pos)
        self.object_pos = torch.zeros((self.num_envs, 3), device=self.device)
        self.object_rot = torch.zeros((self.num_envs, 4), device=self.device)
        self.object_ee_distance = torch.zeros((self.num_envs, 3), device=self.device)
        self.object_ee_rotation = torch.zeros((self.num_envs, 4), device=self.device)
        self.object_ee_angular_distance = torch.zeros((self.num_envs,), device=self.device)
        self.object_ee_euclidean_distance = torch.zeros((self.num_envs,), device=self.device)

        self.timesteps_to_find_object_easy = torch.zeros((self.num_envs,), dtype=torch.float, device=self.device)
        self.timesteps_to_find_object_med = torch.zeros((self.num_envs,), dtype=torch.float, device=self.device)
        self.timesteps_to_find_object_hard = torch.zeros((self.num_envs,), dtype=torch.float, device=self.device)

        self.object_found_easy = torch.zeros((self.num_envs,), dtype=torch.float, device=self.device)
        self.object_found_med = torch.zeros((self.num_envs,), dtype=torch.float, device=self.device)
        self.object_found_hard = torch.zeros((self.num_envs,), dtype=torch.float, device=self.device)
        
        # Logging and counters for diagnostics
        self.extras["log"].update({
            "dist_reward": None,
            "object_ee_distance": None,
            "contact_reward": None,
            "height_bonus": None,
        })
        self.extras["counters"].update({
            "timesteps_to_find_object_easy": None,
            "timesteps_to_find_object_med": None,
            "timesteps_to_find_object_hard": None,
            "object_found_easy": None,
            "object_found_med": None,
            "object_found_hard": None,
            "success": None,
            "failure": None,
        })

    def _setup_scene(self):
        """
        Set up the simulation scene, including object and workspace visualization.
        """
        super()._setup_scene()
        self.object = RigidObject(self.cfg.object_cfg)
        self.scene.rigid_objects["object"] = self.object

        self.workspace = VisualizationMarkers(self.cfg.workspace_cfg)
        self.workspace_pos = torch.zeros((self.num_envs, 3), dtype=torch.float, device=self.device)
        self.workspace_pos[:, :] = torch.tensor(self.cfg.workspace_pos, device=self.device)
        self.workspace_rot = torch.zeros((self.num_envs, 4), dtype=torch.float, device=self.device)
        self.workspace.visualize(self.workspace_pos + self.scene.env_origins, self.workspace_rot)

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

    def _compute_intermediate_values(self, env_ids=None):
        """
        Compute object pose, EE distances, and update find counters.

        Args:
            env_ids (Sequence[int] | None): Environment indices to update.
        """
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        super()._compute_intermediate_values()

        # Object pose and relative distances
        self.object_pos[env_ids] = self.object.data.root_pos_w[env_ids] - self.scene.env_origins[env_ids]
        self.object_rot[env_ids] = self.object.data.root_quat_w[env_ids]
        self.object_ee_distance[env_ids] = self.object_pos[env_ids] - self.ee_pos[env_ids]
        self.object_ee_euclidean_distance[env_ids] = torch.norm(self.object_ee_distance[env_ids], dim=1)
        self.object_ee_rotation[env_ids] = quat_mul(self.object_rot[env_ids], quat_conjugate(self.ee_rot[env_ids]))
        self.object_ee_angular_distance[env_ids] = rotation_distance(self.object_rot[env_ids], self.ee_rot[env_ids])

        # Difficulty thresholds
        easy_threshold = 0.03
        med_threshold = 0.01
        hard_threshold = 0.005

        # Update found flags and counters
        # this is triggered once
        self.object_found_easy = torch.logical_or(self.object_ee_euclidean_distance < easy_threshold, self.object_found_easy)
        self.object_found_med = torch.logical_or(self.object_ee_euclidean_distance < med_threshold, self.object_found_med)
        self.object_found_hard = torch.logical_or(self.object_ee_euclidean_distance < hard_threshold, self.object_found_hard)

        self.timesteps_to_find_object_easy = torch.where(
            self.object_found_easy,
            self.timesteps_to_find_object_easy,
            self.timesteps_to_find_object_easy + 1
        )
        self.timesteps_to_find_object_med = torch.where(
            self.object_found_med,
            self.timesteps_to_find_object_med,
            self.timesteps_to_find_object_med + 1
        )
        self.timesteps_to_find_object_hard = torch.where(
            self.object_found_hard,
            self.timesteps_to_find_object_hard,
            self.timesteps_to_find_object_hard + 1
        )        

        self.success = self.object_pos[:, 2] > self.cfg.object_height_success_threshold
        self.failure = torch.norm(self.object_pos - self.default_object_pos, dim=1) > 0.3


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

        # Reset counters and flags
        self.object_found_easy[env_ids] = 0
        self.timesteps_to_find_object_easy[env_ids] = 0
        self.object_found_med[env_ids] = 0
        self.timesteps_to_find_object_med[env_ids] = 0
        self.object_found_hard[env_ids] = 0
        self.timesteps_to_find_object_hard[env_ids] = 0

    def _reset_object_pose(self, env_ids):
        """
        Reset the pose of the object in the environment.

        Args:
            env_ids (Sequence[int]): Environment indices to reset.
        """
        object_default_state = self.object.data.default_root_state.clone()[env_ids]
        pos_noise = sample_uniform(-1.0, 1.0, (len(env_ids), 3), device=self.device)
        pos_noise[:, 2] = 0  # No vertical noise
        object_default_state[:, :3] = (
            object_default_state[:, :3]
            + self.cfg.reset_object_position_noise * pos_noise
            + self.scene.env_origins[env_ids]
        )
        object_default_state[:, 7:] = torch.zeros_like(self.object.data.default_root_state[env_ids, 7:])
        self.object.write_root_state_to_sim(object_default_state, env_ids)

    def _get_rewards(self) -> torch.Tensor:
        """
        Compute and log rewards for the current step.

        Returns:
            torch.Tensor: Reward values.
        """
        # print(self.object_ee_euclidean_distance)
        # print(self.tactile)
        # print(self.object_pos[:, 2])
        # print("*****")

        rewards, dist_reward, height_bonus = compute_rewards(self.episode_length_buf, self.object_pos, self.cfg.object_height_success_threshold, self.object_ee_euclidean_distance, self.tactile, self.aperture)
        self.extras["log"] = {
            "aperture": self.aperture,
            "dist_reward": dist_reward,
            "object_ee_distance": self.object_ee_euclidean_distance,
            "height_bonus": height_bonus,
        }
        self.extras["counters"] = {
            # "timesteps_to_find_object_easy": self.timesteps_to_find_object_easy.float(),
            # "timesteps_to_find_object_med": self.timesteps_to_find_object_med.float(),
            # "timesteps_to_find_object_hard": self.timesteps_to_find_object_hard.float(),
            # "object_found_easy": self.object_found_easy.float(),
            # "object_found_med": self.object_found_med.float(),
            # "object_found_hard": self.object_found_hard.float(),
            "success": self.success.float(),
            "failure": self.failure.float(),
        }
        if "tactile" in self.cfg.obs_list:
            tactile_dict = {
                "tactile": torch.sum(self.tactile, dim=1),
            }
            self.extras["log"].update(tactile_dict)
        return rewards
    
    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Determine episode termination and timeout.

        Returns:
            tuple: (termination tensor, timeout tensor)
        """
        self._compute_intermediate_values()
        termination = self.success | self.failure
        time_out = self.episode_length_buf >= self.max_episode_length - 1

        return termination, time_out

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


@torch.jit.script
def compute_rewards(episode_length_buf: torch.Tensor, object_pos: torch.Tensor, object_height_success_threshold: float, object_ee_distance: torch.Tensor, tactile: torch.Tensor, aperture: torch.Tensor):
    """
    Compute distance-based rewards.

    Args:
        object_ee_distance (Tensor): Distance between object and end-effector.

    Returns:
        Tuple[Tensor, Tensor]: (reward, distance reward)
    """
    std = 0.03
    r_dist = distance_reward(object_ee_distance, std=std) * 1
    
    #I don't like this because this triggers if the object is flipped
    object_height = object_pos[:, 2]
    dist_mask = (object_ee_distance < 0.03).float()
    success = (object_height > object_height_success_threshold).float() * dist_mask 
    height_bonus = success * (300 - episode_length_buf) * 10
    
    rewards = r_dist + height_bonus
    return rewards, r_dist, height_bonus

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

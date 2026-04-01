# SPDX-License-Identifier: BSD-3-Clause

"""
Author: Elle Miller 2025

Franka Pick and Place RL Task Environment

This module defines the PickAndPlaceEnv environment for the Franka Panda robot,
where the goal is to pick up and place a target object. 
"""

import torch
from collections.abc import Sequence

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObject, RigidObjectCfg
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.sim.schemas.schemas_cfg import CollisionPropertiesCfg, RigidBodyPropertiesCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import (
    quat_conjugate,
    quat_mul,
    sample_uniform,
)

from roto.tasks.robots.franka.franka import FrankaEnv, FrankaEnvCfg


@configclass
class PickAndPlaceEnvCfg(FrankaEnvCfg):
    """
    Configuration for the Franka 'Find' RL task.
    Sets object and workspace properties, including randomization and visualization.
    """

    episode_length_s = 5.0  # Episode length in seconds
    act_moving_average = 1.0  # Action smoothing factor. 1.0 means no smoothing
    default_object_pos = [0.5, 0, 0.03]
    reset_object_position_noise = 0.3

    object_cfg: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Object",
        init_state=RigidObjectCfg.InitialStateCfg(pos=default_object_pos, rot=[1, 0, 0, 0]),
        spawn=sim_utils.SphereCfg(
            radius=0.03,
            physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=1.0, dynamic_friction=0.8, restitution=0.8),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.541, 0.808, 0)),
            rigid_props=RigidBodyPropertiesCfg(kinematic_enabled=True),
            mass_props=sim_utils.MassPropertiesCfg(mass=1),
            collision_props=CollisionPropertiesCfg(collision_enabled=True),
        ),
    )

    workspace_pos = [0.5, 0, 0.0]
    workspace_cfg: VisualizationMarkersCfg = VisualizationMarkersCfg(
        prim_path="/Visuals/workspace",
        markers={
            "workspace": sim_utils.CuboidCfg(
                size=(2 * reset_object_position_noise, 2 * reset_object_position_noise, 0.01),
                visual_material=sim_utils.PreviewSurfaceCfg(opacity=0.1, diffuse_color=(0.541, 0.808, 0)),
            )
        },
    )
    
    # camera
    eye = (1.0, 0, 0.45)
    target = (0.4, 0, 0.2)


class PickAndPlaceEnv(FrankaEnv):
    """
    RL environment for the Franka Panda robot to pick up and place an object.
    """

    cfg: PickAndPlaceEnvCfg

    def __init__(self, cfg: PickAndPlaceEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        # Object and tracking tensors
        self.default_object_pos = torch.zeros((self.num_envs, 3), device=self.device)
        self.default_object_pos[:, :] = torch.tensor(self.cfg.default_object_pos, device=self.device)
        self.object_pos = torch.zeros((self.num_envs, 3), device=self.device)
        self.object_rot = torch.zeros((self.num_envs, 4), device=self.device)
        self.object_ee_distance = torch.zeros((self.num_envs, 3), device=self.device)
        self.object_ee_rotation = torch.zeros((self.num_envs, 4), device=self.device)
        self.object_ee_angular_distance = torch.zeros((self.num_envs,), device=self.device)
        self.object_ee_euclidean_distance = torch.zeros((self.num_envs,), device=self.device)


        # Logging and counters for diagnostics
        self.extras["log"].update(
            {
                "dist_reward": None,
                "object_ee_distance": None,
                "contact_reward": None,
                "height_bonus": None,
            }
        )
        self.extras["counters"].update(
            {
                "object_found_easy": None,
                "object_found_med": None,
                "object_found_hard": None,
                "success": None,
                "failure": None,
            }
        )

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
                self.object_pos,
                self.object_rot,
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
        super()._compute_intermediate_values(env_ids)

        # Object pose and relative distances
        self.object_pos[env_ids] = self.object.data.root_pos_w[env_ids] - self.scene.env_origins[env_ids]
        self.object_rot[env_ids] = self.object.data.root_quat_w[env_ids]
        self.object_ee_distance[env_ids] = self.object_pos[env_ids] - self.ee_pos[env_ids]
        self.object_ee_euclidean_distance[env_ids] = torch.norm(self.object_ee_distance[env_ids], dim=1)
        self.object_ee_rotation[env_ids] = quat_mul(self.object_rot[env_ids], quat_conjugate(self.ee_rot[env_ids]))
        self.object_ee_angular_distance[env_ids] = rotation_distance(self.object_rot[env_ids], self.ee_rot[env_ids])


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

        rewards, r_dist, r_lift = compute_rewards(self.object_ee_euclidean_distance)
        self.extras["log"] = {
            "aperture": self.aperture,
            "object_ee_distance": self.object_ee_euclidean_distance,
            "r_lift": r_lift,
            "r_dist": r_dist,
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
        termination = self.episode_length_buf > self.max_episode_length
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
def compute_rewards(object_ee_distance: torch.Tensor):
    """
    Compute distance-based rewards.

    Args:
        object_ee_distance (Tensor): Distance between object and end-effector.

    Returns:
            torch.Tensor: Distance reward.
    """
    std = 0.1
    r_dist = distance_reward(object_ee_distance, std=std)
    r_lift = (object_ee_distance[:, 2] > 0.04).float() * 10.0

    total = r_dist + r_lift

    return total, r_dist, r_lift


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

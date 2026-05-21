# SPDX-License-Identifier: BSD-3-Clause

"""Find task: reach a target object (Franka Panda only for now)."""

from __future__ import annotations

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
class FindTaskCfg:
    """Shared Find task parameters."""

    episode_length_s = 5.0
    act_moving_average = 1.0
    default_object_pos = [0.5, 0, 0.03]
    reset_object_position_noise = 0.1

    object_cfg: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Object",
        init_state=RigidObjectCfg.InitialStateCfg(pos=default_object_pos, rot=[1, 0, 0, 0]),
        spawn=sim_utils.SphereCfg(
            radius=0.03,
            physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=1.0, dynamic_friction=0.8, restitution=0.8),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.541, 0.808, 0)),
            rigid_props=RigidBodyPropertiesCfg(kinematic_enabled=False),
            mass_props=sim_utils.MassPropertiesCfg(mass=1000000),
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

    eye = (1.0, 0, 0.45)
    target = (0.4, 0, 0.2)


@configclass
class FindCfg(FindTaskCfg, FrankaEnvCfg):
    """Find task on the Franka Panda (registered env ``Find``).

    ``FindTaskCfg`` is first so ``eye`` / ``target`` override :class:`RotoEnvCfg`.
    """


class FindEnv(FrankaEnv):
    """Reach an object; tracks time-to-find at several distance thresholds."""

    cfg: FindCfg

    def __init__(self, cfg: FindCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        self.default_object_pos = torch.zeros((self.num_envs, 3), device=self.device)
        self.default_object_pos[:, :] = torch.tensor(self.cfg.default_object_pos, device=self.device)
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
                "timesteps_to_find_object_easy": None,
                "timesteps_to_find_object_med": None,
                "timesteps_to_find_object_hard": None,
                "object_found_easy": None,
                "object_found_med": None,
                "object_found_hard": None,
                "success": None,
                "failure": None,
            }
        )

    def _setup_scene(self):
        super()._setup_scene()
        self.object = RigidObject(self.cfg.object_cfg)
        self.scene.rigid_objects["object"] = self.object

        self.workspace = VisualizationMarkers(self.cfg.workspace_cfg)
        self.workspace_pos = torch.zeros((self.num_envs, 3), dtype=torch.float, device=self.device)
        self.workspace_pos[:, :] = torch.tensor(self.cfg.workspace_pos, device=self.device)
        self.workspace_rot = torch.zeros((self.num_envs, 4), dtype=torch.float, device=self.device)
        self.workspace.visualize(self.workspace_pos + self.scene.env_origins, self.workspace_rot)

    def _get_gt(self):
        return torch.cat(
            (
                self.object_ee_distance,
                self.object_ee_rotation,
                self.object_ee_angular_distance.unsqueeze(1),
                self.object_ee_euclidean_distance.unsqueeze(1),
            ),
            dim=-1,
        )

    def _compute_intermediate_values(self, env_ids=None):
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        super()._compute_intermediate_values(env_ids)

        self.object_pos[env_ids] = self.object.data.root_pos_w[env_ids] - self.scene.env_origins[env_ids]
        self.object_rot[env_ids] = self.object.data.root_quat_w[env_ids]
        self.object_ee_distance[env_ids] = self.object_pos[env_ids] - self.ee_pos[env_ids]
        self.object_ee_euclidean_distance[env_ids] = torch.norm(self.object_ee_distance[env_ids], dim=1)
        self.object_ee_rotation[env_ids] = quat_mul(self.object_rot[env_ids], quat_conjugate(self.ee_rot[env_ids]))
        self.object_ee_angular_distance[env_ids] = rotation_distance(self.object_rot[env_ids], self.ee_rot[env_ids])

        easy_threshold = 0.03
        med_threshold = 0.01
        hard_threshold = 0.005

        self.object_found_easy = torch.logical_or(
            self.object_ee_euclidean_distance < easy_threshold, self.object_found_easy
        )
        self.object_found_med = torch.logical_or(
            self.object_ee_euclidean_distance < med_threshold, self.object_found_med
        )
        self.object_found_hard = torch.logical_or(
            self.object_ee_euclidean_distance < hard_threshold, self.object_found_hard
        )

        self.timesteps_to_find_object_easy = torch.where(
            self.object_found_easy, self.timesteps_to_find_object_easy, self.timesteps_to_find_object_easy + 1
        )
        self.timesteps_to_find_object_med = torch.where(
            self.object_found_med, self.timesteps_to_find_object_med, self.timesteps_to_find_object_med + 1
        )
        self.timesteps_to_find_object_hard = torch.where(
            self.object_found_hard, self.timesteps_to_find_object_hard, self.timesteps_to_find_object_hard + 1
        )

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        super()._reset_idx(env_ids)

        self._reset_object_pose(env_ids)
        self._reset_robot(env_ids)

        self.object_found_easy[env_ids] = 0
        self.timesteps_to_find_object_easy[env_ids] = 0
        self.object_found_med[env_ids] = 0
        self.timesteps_to_find_object_med[env_ids] = 0
        self.object_found_hard[env_ids] = 0
        self.timesteps_to_find_object_hard[env_ids] = 0

    def _reset_object_pose(self, env_ids):
        object_default_state = self.object.data.default_root_state.clone()[env_ids]
        pos_noise = sample_uniform(-1.0, 1.0, (len(env_ids), 3), device=self.device)
        pos_noise[:, 2] = 0
        object_default_state[:, :3] = (
            object_default_state[:, :3]
            + self.cfg.reset_object_position_noise * pos_noise
            + self.scene.env_origins[env_ids]
        )
        object_default_state[:, 7:] = torch.zeros_like(self.object.data.default_root_state[env_ids, 7:])
        self.object.write_root_state_to_sim(object_default_state, env_ids)

    def _get_rewards(self) -> torch.Tensor:
        rewards = compute_rewards(self.object_ee_euclidean_distance)
        self.extras["log"] = {
            "aperture": self.aperture,
            "object_ee_distance": self.object_ee_euclidean_distance,
        }
        self.extras["counters"] = {
            "timesteps_to_find_object_easy": self.timesteps_to_find_object_easy.float(),
            "timesteps_to_find_object_med": self.timesteps_to_find_object_med.float(),
            "timesteps_to_find_object_hard": self.timesteps_to_find_object_hard.float(),
            "object_found_easy": self.object_found_easy.float(),
            "object_found_med": self.object_found_med.float(),
            "object_found_hard": self.object_found_hard.float(),
        }
        if "tactile" in self.cfg.obs_list:
            self.extras["log"]["tactile"] = torch.sum(self.tactile, dim=1)
        return rewards

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        self._compute_intermediate_values()
        termination = self.episode_length_buf > self.max_episode_length
        time_out = self.episode_length_buf >= self.max_episode_length - 1

        return termination, time_out


@torch.jit.script
def distance_reward(object_ee_distance, std: float = 0.1):
    r_reach = 1 - torch.tanh(object_ee_distance / std)
    return r_reach


@torch.jit.script
def compute_rewards(object_ee_distance: torch.Tensor):
    std = 0.1
    r_dist = distance_reward(object_ee_distance, std=std)
    return r_dist


@torch.jit.script
def rotation_distance(object_rot, target_rot):
    quat_diff = quat_mul(object_rot, quat_conjugate(target_rot))
    return 2.0 * torch.asin(torch.clamp(torch.norm(quat_diff[:, 1:4], p=2, dim=-1), max=1.0))

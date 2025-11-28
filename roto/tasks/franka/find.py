# FrankaFind RL Task Environment
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Find environment for the Franka Panda robot.

The environment randomizes a target object position and provides distance-based
rewards and diagnostics for multiple difficulty thresholds.
"""

from collections.abc import Sequence

import torch

from isaaclab.assets import RigidObject, RigidObjectCfg
import isaaclab.sim as sim_utils
from isaaclab.sim.schemas.schemas_cfg import CollisionPropertiesCfg, RigidBodyPropertiesCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import quat_conjugate, quat_mul, sample_uniform
from isaaclab.markers import VisualizationMarkersCfg, VisualizationMarkers

from roto.tasks.franka.franka import FrankaEnv, FrankaEnvCfg


@configclass
class FindEnvCfg(FrankaEnvCfg):
    """
    Configuration for the Franka 'Find' RL task.
    """
    episode_length_s = 5.0
    act_moving_average = 0.0
    default_object_pos = [0.5, 0, 0.03]
    reset_object_position_noise = 0.1

    object_cfg: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Object",
        init_state=RigidObjectCfg.InitialStateCfg(pos=default_object_pos, rot=[1, 0, 0, 0]),
        spawn=sim_utils.SphereCfg(
            radius=0.03,
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=1.0, dynamic_friction=0.8, restitution=0.8
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.541, 0.808, 0)),
            rigid_props=RigidBodyPropertiesCfg(kinematic_enabled=True),
            mass_props=sim_utils.MassPropertiesCfg(mass=100),
            collision_props=CollisionPropertiesCfg(collision_enabled=True),
        ),
    )

    workspace_pos = [0.5, 0, 0.0]
    workspace_cfg: VisualizationMarkersCfg = VisualizationMarkersCfg(
        prim_path="/Visuals/workspace",
        markers={
            "workspace": sim_utils.CuboidCfg(
                size=(2 * reset_object_position_noise, 2 * reset_object_position_noise, 0.01),
                visual_material=sim_utils.PreviewSurfaceCfg(
                    opacity=0.1, diffuse_color=(0.541, 0.808, 0)
                ),
            )
        },
    )


class FindEnv(FrankaEnv):
    """
    RL environment where the robot must find and reach a target object.

    The environment tracks whether the object has been found at easy/med/hard
    thresholds and records timesteps-to-find for diagnostics.
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

        self.timesteps_to_find_object_easy = torch.zeros(
            (self.num_envs,), dtype=torch.float, device=self.device
        )
        self.timesteps_to_find_object_med = torch.zeros(
            (self.num_envs,), dtype=torch.float, device=self.device
        )
        self.timesteps_to_find_object_hard = torch.zeros(
            (self.num_envs,), dtype=torch.float, device=self.device
        )

        self.object_found_easy = torch.zeros((self.num_envs,), dtype=torch.float, device=self.device)
        self.object_found_med = torch.zeros((self.num_envs,), dtype=torch.float, device=self.device)
        self.object_found_hard = torch.zeros((self.num_envs,), dtype=torch.float, device=self.device)

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
        """
        Set up object and workspace visualization in the scene.
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
        Get ground-truth observation vector concatenating distance and rotation info.

        Returns:
            torch.Tensor: concatenated ground-truth vector.
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
        Update object pose, EE distances, and find counters.
        """
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        super()._compute_intermediate_values()

        # Object pose and relative distances
        self.object_pos[env_ids] = self.object.data.root_pos_w[env_ids] - self.scene.env_origins[env_ids]
        self.object_rot[env_ids] = self.object.data.root_quat_w[env_ids]
        self.object_ee_distance[env_ids] = self.object_pos[env_ids] - self.ee_pos[env_ids]
        self.object_ee_euclidean_distance[env_ids] = torch.norm(self.object_ee_distance[env_ids], dim=1)
        self.object_ee_rotation[env_ids] = quat_mul(
            self.object_rot[env_ids], quat_conjugate(self.ee_rot[env_ids])
        )
        self.object_ee_angular_distance[env_ids] = rotation_distance(
            self.object_rot[env_ids], self.ee_rot[env_ids]
        )

        # Difficulty thresholds
        easy_threshold = 0.03
        med_threshold = 0.01
        hard_threshold = 0.005

        # Update found flags and counters
        self.object_found_easy = torch.logical_or(self.object_ee_euclidean_distance < easy_threshold, self.object_found_easy)
        self.object_found_med = torch.logical_or(self.object_ee_euclidean_distance < med_threshold, self.object_found_med)
        self.object_found_hard = torch.logical_or(self.object_ee_euclidean_distance < hard_threshold, self.object_found_hard)

        self.timesteps_to_find_object_easy = torch.where(
            self.object_found_easy, self.timesteps_to_find_object_easy, self.timesteps_to_find_object_easy + 1
        )
        self.timesteps_to_find_object_med = torch.where(
            self.object_found_med, self.timesteps_to_find_object_med, self.timesteps_to_find_object_med + 1
        )
        self.timesteps_to_find_object_hard = torch.where(
            self.object_found_hard, self.timesteps_to_find_object_hard, self.timesteps_to_find_object_hard + 1
        )

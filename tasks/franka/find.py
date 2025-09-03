# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
import torch

from tasks.franka.franka import FrankaEnv, FrankaEnvCfg, randomize_rotation
import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg

from isaaclab.utils import configclass

from isaaclab.sim.schemas.schemas_cfg import CollisionPropertiesCfg, MassPropertiesCfg, RigidBodyPropertiesCfg

from isaaclab.utils.math import (
    quat_conjugate,
    quat_from_angle_axis,
    quat_mul,
    sample_uniform,
    saturate,
)
from isaaclab.sim.schemas.schemas_cfg import CollisionPropertiesCfg, MassPropertiesCfg, RigidBodyPropertiesCfg
from isaaclab.assets import RigidObjectCfg
from isaaclab.utils import configclass

from isaaclab.markers import VisualizationMarkersCfg, VisualizationMarkers

import os

# e.g. '/home/elle/code/external/IsaacLab/roto'
parent_dir = os.getcwd()
user = os.getlogin()


"""

FrankaFind

"""


@configclass
class FindEnvCfg(FrankaEnvCfg):

    act_moving_average = 0.5

    default_object_pos = [0.5, 0, 0.03]
    reset_object_position_noise = 0.1

    brat = (0.5411764705882353, 0.807843137254902, 0)
    brat_pink = (0.3294117647058823, 0.3176470588235294, 0.9137254901960784)

    object_cfg: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Object",
        init_state=RigidObjectCfg.InitialStateCfg(pos=default_object_pos, rot=[1, 0, 0, 0]),
        spawn=sim_utils.SphereCfg(
            radius=0.03,
            physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=1.0, dynamic_friction=1.0, restitution=0.0),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=brat_pink),
            rigid_props=RigidBodyPropertiesCfg(kinematic_enabled=False),
            mass_props=sim_utils.MassPropertiesCfg(mass=1000000),
            collision_props=CollisionPropertiesCfg(collision_enabled=True)
        ),
    )

    workspace_pos = [0.5, 0, 0.0]
    workspace_cfg: VisualizationMarkersCfg = VisualizationMarkersCfg(
        prim_path="/Visuals/workspace",
        markers={
            "workspace": sim_utils.CuboidCfg(
            size=(0.2, 0.2, 0.01),
            visual_material=sim_utils.PreviewSurfaceCfg(opacity=0.1, diffuse_color=brat)) 
        },
    )



class FindEnv(FrankaEnv):
    cfg: FindEnvCfg

    def __init__(self, cfg: FindEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        self.cfg = cfg
        self.timesteps_to_find_object_easy = torch.zeros((self.num_envs,), dtype=torch.float, device=self.device)
        self.timesteps_to_find_object_med = torch.zeros((self.num_envs,), dtype=torch.float, device=self.device)
        self.timesteps_to_find_object_hard = torch.zeros((self.num_envs,), dtype=torch.float, device=self.device)

        self.object_found_easy = torch.zeros((self.num_envs,), dtype=torch.float, device=self.device)
        self.object_found_med = torch.zeros((self.num_envs,), dtype=torch.float, device=self.device)
        self.object_found_hard = torch.zeros((self.num_envs,), dtype=torch.float, device=self.device)

    def _setup_scene(self):
        super()._setup_scene()

        self.workspace = VisualizationMarkers(self.cfg.workspace_cfg)
        self.workspace_pos = torch.zeros((self.num_envs, 3), dtype=torch.float, device=self.device)
        self.workspace_pos[:, :] = torch.tensor(self.cfg.workspace_pos, device=self.device)
        self.workspace_rot = torch.zeros((self.num_envs, 4), dtype=torch.float, device=self.device)
        self.workspace.visualize(self.workspace_pos + self.scene.env_origins, self.workspace_rot)


    def _compute_intermediate_values(self, reset=False, env_ids=None):
        super()._compute_intermediate_values()

        easy_threshold = 0.03
        med_threshold = 0.01
        hard_threshold = 0.005

        # Only increment for environments where object hasn't been found yet
        self.object_found_easy = torch.logical_or(self.object_ee_euclidean_distance < easy_threshold, self.object_found_easy)
        self.object_found_med = torch.logical_or(self.object_ee_euclidean_distance < med_threshold, self.object_found_med)
        self.object_found_hard = torch.logical_or(self.object_ee_euclidean_distance < hard_threshold, self.object_found_hard)

        self.timesteps_to_find_object_easy = torch.where(
            self.object_found_easy,  # Object is found now OR was found before
            self.timesteps_to_find_object_easy,  # Keep the value
            self.timesteps_to_find_object_easy + 1  # Increment only if not found
        )
        self.timesteps_to_find_object_med = torch.where(
            self.object_found_med,  # Object is found now OR was found before
            self.timesteps_to_find_object_med,  # Keep the value
            self.timesteps_to_find_object_med + 1  # Increment only if not found
        )
        self.timesteps_to_find_object_hard = torch.where(
            self.object_found_hard,  # Object is found now OR was found before
            self.timesteps_to_find_object_hard,  # Keep the value
            self.timesteps_to_find_object_hard + 1  # Increment only if not found
        )

        if reset:
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
        # global object positions (for writing to sim)
        object_default_state[:, :3] = (
            object_default_state[:, :3]
            + self.cfg.reset_object_position_noise * pos_noise
            + self.scene.env_origins[env_ids]
        )
        object_default_state[:, 7:] = torch.zeros_like(self.object.data.default_root_state[env_ids, 7:])
        self.object.write_root_state_to_sim(object_default_state, env_ids)

    def _get_rewards(self) -> torch.Tensor:

        (
            rewards,
            dist_reward,
        ) = compute_rewards(
            self.object_ee_euclidean_distance
        )
        self.extras["log"] = {
            "dist_reward": (dist_reward),
            "object_ee_distance": (self.object_ee_euclidean_distance)
        }
        self.extras["counters"] = {
            "timesteps_to_find_object_easy": (self.timesteps_to_find_object_easy).float(),
            "timesteps_to_find_object_med": (self.timesteps_to_find_object_med).float(),
            "timesteps_to_find_object_hard": (self.timesteps_to_find_object_hard).float(),
            "object_found_easy": (self.object_found_easy).float(),
            "object_found_med": (self.object_found_med).float(),
            "object_found_hard": (self.object_found_hard).float(),
        }

        if "tactile" in self.cfg.obs_list:
            tactile_dict = {
                "tactile":  (torch.sum(self.tactile, dim=1)),
            }
            self.extras["log"].update(tactile_dict)

        return rewards


from tasks.franka.franka import distance_reward

@torch.jit.script
def compute_rewards(
    object_ee_distance: torch.Tensor,
):

    # reaching objects
    r_dist = distance_reward(object_ee_distance)
    
    rewards = r_dist
    
    return (
        rewards,
        r_dist,
    )

 
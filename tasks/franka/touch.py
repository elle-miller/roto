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

Why do we need a different file for each object?

Each object has its own
- initial position and rotation

- reset randomisation conditions

e.g. dont want to randomise bowl rotation

"""


@configclass
class TouchEnvCfg(FrankaEnvCfg):

    default_object_pos = [0.5, 0, 0.03]
    reset_object_position_noise = 0.1

    brat_pink = (0.9882352941176471, 0.011764705882352941, 0.7098039215686275)

    object_cfg: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Object",
        init_state=RigidObjectCfg.InitialStateCfg(pos=default_object_pos, rot=[1, 0, 0, 0]),
        spawn=sim_utils.SphereCfg(
            radius=0.03,
            physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=1.0, dynamic_friction=1.0, restitution=0.0),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=brat_pink),
            rigid_props=RigidBodyPropertiesCfg(kinematic_enabled=False),
            # rigid_props=RigidBodyPropertiesCfg(
            #     solver_position_iteration_count=16,
            #     solver_velocity_iteration_count=1,
            #     max_angular_velocity=1000.0,
            #     max_linear_velocity=1000.0,
            #     max_depenetration_velocity=5.0,
            #     disable_gravity=False,
            # ),
            mass_props=sim_utils.MassPropertiesCfg(mass=1000000),
            collision_props=CollisionPropertiesCfg(collision_enabled=True)
        ),
    )

    colour_1 = (0.4, 0.9882352941176471, 0.011764705882352941)
    colour_2 = (0.0, 1.0, 1.0)

    brat = (0.5411764705882353, 0.807843137254902, 0)
    brat_pink = (0.7294117647058823, 0.3176470588235294, 0.7137254901960784)

    colour_1 = (0.80392, 0.7058, 0.858823) 
    colour_2 = (0.741176, 0.878, 0.9960784)
    brat_pink = (0.3294117647058823, 0.3176470588235294, 0.9137254901960784)

    colour_1 = (0.4, 0.9882352941176471, 0.011764705882352941)
    colour_2 = (0.0, 1.0, 1.0)

    workspace_pos = [0.5, 0, 0.0]
    workspace_cfg: VisualizationMarkersCfg = VisualizationMarkersCfg(
        prim_path="/Visuals/workspace",
        markers={
            "workspace": sim_utils.CuboidCfg(
            size=(0.2, 0.2, 0.01),
            visual_material=sim_utils.PreviewSurfaceCfg(opacity=0.1, diffuse_color=brat))
            ,
            "box": sim_utils.CuboidCfg(
            size=(0.2, 0.2, 0.01),
            visual_material=sim_utils.PreviewSurfaceCfg(opacity=0.1, diffuse_color=(1.0, 1.0, 0.0)))    
        },
    )



class TouchEnv(FrankaEnv):
    cfg: TouchEnvCfg

    def __init__(self, cfg: TouchEnvCfg, render_mode: str | None = None, **kwargs):
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
        self.goal_pos = torch.zeros((self.num_envs, 3), dtype=torch.float, device=self.device)
        self.goal_pos[:, :] = torch.tensor(self.cfg.workspace_pos, device=self.device)
        self.goal_rot = torch.zeros((self.num_envs, 4), dtype=torch.float, device=self.device)
        goal_pos_world = self.goal_pos + self.scene.env_origins
        self.workspace.visualize(goal_pos_world, self.goal_rot)




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
            self.reaching_object_scale,
            self.contact_reward_scale,
            self.joint_vel_penalty_scale,
            self.joint_vel,
            self.aperture,
            self.binary_tactile,
            self.tactile,
            self.ee_pos,
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

            if not self.binary_tactile:
                tactile_dict = {
                    "unnormalised_forces_left_x": (self.unnormalised_forces[:, 0]),
                    "unnormalised_forces_right_x": (self.unnormalised_forces[:, 1]),
                    "normalised_forces_left_x": (self.normalised_forces[:, 0]),
                    "normalised_forces_right_x": (self.normalised_forces[:, 1]),
                }
                self.extras["log"].update(tactile_dict)


        return rewards


from tasks.franka.franka import distance_reward

@torch.jit.script
def compute_rewards(
    reaching_scale: float,
    contact_reward_scale: float,
    joint_vel_penalty_scale: float,
    robot_joint_vel: torch.Tensor,
    aperture: torch.Tensor,
    binary_tactile: bool,
    tactile: torch.Tensor,
    ee_pos: torch.Tensor,
    object_ee_distance: torch.Tensor,
):

    # reaching objects
    r_dist = distance_reward(object_ee_distance)
    
    rewards = r_dist
    
    return (
        rewards,
        r_dist,
    )

 
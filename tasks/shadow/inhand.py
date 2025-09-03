# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


from __future__ import annotations

import gymnasium as gym
import numpy as np
import os
import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING
import matplotlib.pyplot as plt

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import DirectRLEnv
from isaaclab.markers import VisualizationMarkers
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils.math import quat_conjugate, quat_from_angle_axis, quat_mul, sample_uniform, saturate
from isaaclab.sensors import (
    FrameTransformer,
    FrameTransformerCfg,
    OffsetCfg,
    TiledCamera,
    TiledCameraCfg,
    ContactSensor,
    ContactSensorCfg
)

from tasks.shadow.shadow_hand_env_cfg import ShadowHandEnvCfg



class InHandManipulationEnv(DirectRLEnv):
    cfg: ShadowHandEnvCfg

    def __init__(self, cfg: ShadowHandEnvCfg, render_mode: str | None = None, **kwargs):

        self.obs_stack = cfg.obs_stack
        super().__init__(cfg, render_mode, **kwargs)

        self.num_hand_dofs = self.hand.num_joints

        self.dtype = torch.float32
        self.binary_tactile = cfg.binary_tactile
        print("binary tactile:", self.binary_tactile)

        # buffers for position targets
        self.hand_dof_targets = torch.zeros((self.num_envs, self.num_hand_dofs), dtype=self.dtype, device=self.device)
        self.prev_targets = torch.zeros((self.num_envs, self.num_hand_dofs), dtype=self.dtype, device=self.device)
        self.cur_targets = torch.zeros((self.num_envs, self.num_hand_dofs), dtype=self.dtype, device=self.device)

        self.num_actions = 20
        self.actions = torch.zeros((self.num_envs, self.num_actions), device=self.device)
        self.tactile = torch.zeros((self.num_envs, self.cfg.num_tactile_observations), device=self.device)
        self.last_tactile = torch.zeros((self.num_envs, self.cfg.num_tactile_observations), device=self.device)

        self.num_prop_observations = 272
        self.num_tactile_observations = 68

        # list of actuated joints
        self.actuated_dof_indices = list()
        for joint_name in cfg.actuated_joint_names:
            self.actuated_dof_indices.append(self.hand.joint_names.index(joint_name))
        self.actuated_dof_indices.sort()

        # finger bodies
        self.finger_bodies = list()
        for body_name in self.cfg.fingertip_body_names:
            self.finger_bodies.append(self.hand.body_names.index(body_name))
        self.finger_bodies.sort()
        self.num_fingertips = len(self.finger_bodies)

        self.palm_body = self.hand.body_names.index("robot0_palm")

        # joint limits
        joint_pos_limits = self.hand.root_physx_view.get_dof_limits().to(self.device)
        self.hand_dof_lower_limits = joint_pos_limits[..., 0]
        self.hand_dof_upper_limits = joint_pos_limits[..., 1]

        # used to compare object position
        default = torch.tensor([0.0, -0.39, 0.6], dtype=self.dtype, device=self.device)
        self.in_hand_pos = torch.zeros((self.num_envs, 3), dtype=self.dtype, device=self.device)
        self.in_hand_pos[:] = default
        self.in_hand_pos[:, 2] -= 0.04

        # track successes
        self.successes = torch.zeros(self.num_envs, dtype=self.dtype, device=self.device)
        self.consecutive_successes = torch.zeros(1, dtype=self.dtype, device=self.device)

        # unit tensors
        self.x_unit_tensor = torch.tensor([1, 0, 0], dtype=self.dtype, device=self.device).repeat((self.num_envs, 1))
        self.y_unit_tensor = torch.tensor([0, 1, 0], dtype=self.dtype, device=self.device).repeat((self.num_envs, 1))
        self.z_unit_tensor = torch.tensor([0, 0, 1], dtype=self.dtype, device=self.device).repeat((self.num_envs, 1))

        # camera stuff
        self.count = 0
        self.init = False

        if "log" not in self.extras:
            self.extras["log"] = dict()

        self.extras["log"] = {
            "tactile_penalty": None,
            "success_reward": None,
            "action_penalty": None,
            "fall_penalty": None,
            "object_height": None,
            "object_z_linvel": None,
            "object_z_angvel": None,
            "sum_forces": None,
            "total_rotations": None,
            "cumulative_rotations": None,
            "ball_1_vel": None,
            "ball_2_vel": None,
            "ball_dist": None,
            "dist_penalty": None,
            "tactile_reward": None,
            "transition_reward": None,
            "bounce_reward": None,
            "air_reward": None
        }

        self.extras["counters"] = {
            "num_transitions": None,
            "num_bounces": None,
            "consecutive_successes": None,
            "success_reward": None,
            "time_without_contact": None,
            "num_rotations": None
        }

    def _setup_scene(self):
        # add hand, in-hand object, and goal object
        self.hand = Articulation(self.cfg.robot_cfg)
        self.object = RigidObject(self.cfg.object_cfg)
        # add ground plane
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())
        # clone and replicate (no need to filter for this environment)
        self.scene.clone_environments(copy_from_source=False)
        # add articulation to scene - we must register to scene to randomize with EventManager
        self.scene.articulations["robot"] = self.hand
        self.scene.rigid_objects["object"] = self.object
        # # add lights

        colour_1 = (0.4, 0.9882352941176471, 0.011764705882352941)
        brat_pink = (0.9882352941176471, 0.011764705882352941, 0.7098039215686275)
        colour_2 = (0.0, 1.0, 1.0)
        light_cfg = sim_utils.DomeLightCfg(intensity=100.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)
        light_cfg_1 = sim_utils.SphereLightCfg(intensity=10000.0, color=colour_1)
        light_cfg_1.func("/World/ds", light_cfg_1, translation=(1, 0, 1))
        light_cfg_2 = sim_utils.SphereLightCfg(intensity=10000.0, color=colour_2)
        light_cfg_2.func("/World/disk", light_cfg_2, translation=(-1, 0, 1))

        enable_cameras = False
        if enable_cameras == 1:
            self._tiled_camera = TiledCamera(self.cfg.tiled_camera)
            self.scene.sensors["tiled_camera"] = self._tiled_camera

        # if "tactile" in self.cfg.obs_list:
        self.distal_sensor = ContactSensor(self.cfg.distal_contact_cfg)
        self.proximal_sensor = ContactSensor(self.cfg.proximal_contact_cfg)
        self.middle_sensor = ContactSensor(self.cfg.middle_contact_cfg)
        self.palm_sensor = ContactSensor(self.cfg.palm_contact_cfg)
        self.metacarpal_sensor = ContactSensor(self.cfg.metacarpal_contact_cfg)

        self.scene.sensors["distal_sensor"] = self.distal_sensor
        self.scene.sensors["proximal_sensor"] = self.proximal_sensor
        self.scene.sensors["middle_sensor"] = self.middle_sensor
        self.scene.sensors["palm_sensor"] = self.palm_sensor
        self.scene.sensors["metacarpal_sensor"] = self.metacarpal_sensor

    
    def _configure_gym_env_spaces(self):
        pass
    
    def set_spaces(self, single_obs, obs, single_action, action):
        self.single_observation_space = single_obs
        self.observation_space = obs
        self.single_action_space = single_action
        self.action_space = action

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.actions = actions.clone()

    def _apply_action(self) -> None:
        self.cur_targets[:, self.actuated_dof_indices] = scale(
            self.actions,
            self.hand_dof_lower_limits[:, self.actuated_dof_indices],
            self.hand_dof_upper_limits[:, self.actuated_dof_indices],
        )
        self.cur_targets[:, self.actuated_dof_indices] = (
            self.cfg.act_moving_average * self.cur_targets[:, self.actuated_dof_indices]
            + (1.0 - self.cfg.act_moving_average) * self.prev_targets[:, self.actuated_dof_indices]
        )
        self.cur_targets[:, self.actuated_dof_indices] = saturate(
            self.cur_targets[:, self.actuated_dof_indices],
            self.hand_dof_lower_limits[:, self.actuated_dof_indices],
            self.hand_dof_upper_limits[:, self.actuated_dof_indices],
        )

        self.prev_targets[:, self.actuated_dof_indices] = self.cur_targets[:, self.actuated_dof_indices]

        self.hand.set_joint_position_target(
            self.cur_targets[:, self.actuated_dof_indices], joint_ids=self.actuated_dof_indices
        )

    def get_observations(self):
        # public method
        return self._get_observations()

    def _get_observations(self) -> dict:

        obs_dict = {}
        for k in self.cfg.obs_list:
            if k == "prop":
                obs_dict[k] = self._get_proprioception()
            elif k == "pixels":
                obs_dict[k] = self._get_images()
            elif k == "gt":
                obs_dict[k] = self._get_gt()
            elif k == "tactile":
                obs_dict[k] = self._get_tactile()
            else:
                print("Unknown observations type!")

        obs_dict = {"policy": obs_dict}

        return obs_dict
    
    def _get_proprioception(self):
        prop = torch.cat(
            (
                # hand (48)
                unscale(self.hand_dof_pos, self.hand_dof_lower_limits, self.hand_dof_upper_limits),
                self.cfg.vel_obs_scale * self.hand_dof_vel,
                # actions (20 = 68)
                self.actions,
                # fingertips
                # self.fingertip_pos
                # fingertips
                # 
                # self.fingertip_pos.view(self.num_envs, self.num_fingertips * 3),
                # self.fingertip_rot.view(self.num_envs, self.num_fingertips * 4),
                # self.fingertip_velocities.view(self.num_envs, self.num_fingertips * 6),
            ),
            dim=-1,
        )

        # tmp to do inverse scaling
        # print("prop", prop.size())
        return prop

    
    def _get_tactile(self):
        # tactile = torch.cat(
        #     (
        #     self.cfg.force_torque_obs_scale
        #         * self.fingertip_force_sensors.view(self.num_envs, self.num_fingertips * 6),
        #     ),
        #     dim=-1
        # )
        # return tactile

        # distal_friction_forces = self.distal_sensor.contact_physx_view.get_friction_data(1/120)
        # print(distal_friction_forces)

        distal_forces = self.distal_sensor.data.net_forces_w[:].clone() #.reshape(self.num_envs, 3 * 5)
        proximal_forces = self.proximal_sensor.data.net_forces_w[:].clone()
        middle_forces = self.middle_sensor.data.net_forces_w[:].clone()
        palm_forces = self.palm_sensor.data.net_forces_w[:].clone()
        metacarpal_forces = self.metacarpal_sensor.data.net_forces_w[:].clone()

        distal_norm = torch.norm(distal_forces, dim=-1)
        proximal_norm = torch.norm(proximal_forces, dim=-1)
        middle_norm = torch.norm(middle_forces, dim=-1)
        palm_norm = torch.norm(palm_forces, dim=-1)
        metacarpal_norm = torch.norm(metacarpal_forces, dim=-1)

        
        if self.binary_tactile:
            if self.dtype == torch.float16:
                distal_norm = (distal_norm > 0).half()
                proximal_norm = (proximal_norm > 0).half()
                middle_norm = (middle_norm > 0).half()
                palm_norm = (palm_norm > 0).half()
                metacarpal_norm = (metacarpal_norm > 0).half()
            else:
                distal_norm = (distal_norm > 0).float()
                proximal_norm = (proximal_norm > 0).float()
                middle_norm = (middle_norm > 0).float()
                palm_norm = (palm_norm > 0).float()
                metacarpal_norm = (metacarpal_norm > 0).float()

        tactile = torch.cat((
            distal_norm,
            proximal_norm,
            middle_norm,
            palm_norm,
            metacarpal_norm
            ), 
            dim=-1
        )

        if not self.binary_tactile:
            # Clip the tensor values and normalise 0 to 1
            clamped_tactile = torch.clamp(tactile, min=self.cfg.tactile_min_val, max=self.cfg.tactile_max_val)
            tactile = (
                ((clamped_tactile - self.cfg.tactile_min_val) / (self.cfg.tactile_max_val - self.cfg.tactile_min_val))
            )

        self.last_tactile = self.tactile
        self.tactile = tactile
        return tactile
    
    def tactile_pretty(self, tactile):
        # print(tactile.shape)
        assert len(tactile.shape) == 1 
        assert tactile.shape[0] == self.cfg.num_tactile_observations
        distal = [f"{val:.1f}" for val in tactile[:5].tolist()]
        proximal = [f"{val:.1f}" for val in tactile[5:10].tolist()]
        middle = [f"{val:.1f}" for val in tactile[10:15].tolist()]
        palm = tactile[15]
        meta = tactile[16]
        print(f"dist: {distal} prox: {proximal} mid: {middle} palm: {palm:.1f} meta: {meta:.1f}")


    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        self._compute_intermediate_values()

        # reset when cube has fallen
        goal_dist = torch.norm(self.object_pos - self.in_hand_pos, p=2, dim=-1)
        out_of_reach = goal_dist >= self.cfg.fall_dist

        time_out = self.episode_length_buf >= self.max_episode_length - 1

        return out_of_reach, time_out

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self.hand._ALL_INDICES
        # resets articulation and rigid body attributes
        super()._reset_idx(env_ids)

        # reset hand
        self._reset_hand(env_ids)

        # reset object
        self._reset_object(env_ids)

        self.successes[env_ids] = 0
        self._compute_intermediate_values(reset=True, env_ids=env_ids)

    def _reset_hand(self, env_ids):
        # reset hand
        delta_max = self.hand_dof_upper_limits[env_ids] - self.hand.data.default_joint_pos[env_ids]
        delta_min = self.hand_dof_lower_limits[env_ids] - self.hand.data.default_joint_pos[env_ids]

        dof_pos_noise = sample_uniform(-1.0, 1.0, (len(env_ids), self.num_hand_dofs), device=self.device)
        rand_delta = delta_min + (delta_max - delta_min) * 0.5 * dof_pos_noise
        dof_pos = self.hand.data.default_joint_pos[env_ids] + self.cfg.reset_dof_pos_noise * rand_delta

        dof_vel_noise = sample_uniform(-1.0, 1.0, (len(env_ids), self.num_hand_dofs), device=self.device)
        dof_vel = self.hand.data.default_joint_vel[env_ids] + self.cfg.reset_dof_vel_noise * dof_vel_noise

        self.prev_targets[env_ids] = dof_pos
        self.cur_targets[env_ids] = dof_pos
        self.hand_dof_targets[env_ids] = dof_pos

        self.hand.set_joint_position_target(dof_pos, env_ids=env_ids)
        self.hand.write_joint_state_to_sim(dof_pos, dof_vel, env_ids=env_ids)

    def _reset_object(self, env_ids):
        # reset object
        object_default_state = self.object.data.default_root_state.clone()[env_ids]
        pos_noise = sample_uniform(-1.0, 1.0, (len(env_ids), 3), device=self.device)
        # global object positions
        object_default_state[:, 0:3] = (
            object_default_state[:, 0:3] + self.cfg.reset_position_noise * pos_noise + self.scene.env_origins[env_ids]
        )

        rot_noise = sample_uniform(-1.0, 1.0, (len(env_ids), 2), device=self.device)  # noise for X and Y rotation
        object_default_state[:, 3:7] = randomize_rotation(
            rot_noise[:, 0], rot_noise[:, 1], self.x_unit_tensor[env_ids], self.y_unit_tensor[env_ids]
        )

        object_default_state[:, 7:] = torch.zeros_like(self.object.data.default_root_state[env_ids, 7:])
        self.object.write_root_pose_to_sim(object_default_state[:, :7], env_ids)
        self.object.write_root_velocity_to_sim(object_default_state[:, 7:], env_ids)


    def _compute_intermediate_values(self):
        # self.in_hand_pos = self.hand.data.body_pos_w[:, self.palm_body] - self.scene.env_origins

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


    def _get_images(self):
        if not self.init:
            eyes = (
                torch.tensor(self.cfg.eye, dtype=self.dtype, device=self.device).repeat((self.num_envs, 1))
                + self.scene.env_origins
            )
            targets = (
                torch.tensor(self.cfg.target, dtype=self.dtype, device=self.device).repeat((self.num_envs, 1))
                + self.scene.env_origins
            )
            self._tiled_camera.set_world_poses_from_view(eyes=eyes, targets=targets)
            self.init = True
        data_type = "rgb" if "rgb" in self.cfg.tiled_camera.data_types else "depth"
        img_batch = self._tiled_camera.data.output[data_type].clone()
        batch_size = img_batch.size()[0]
        flattened_images = img_batch.view(batch_size, -1)

        # if self.cfg.write_image_to_file:
        #     name = self.count
        #     name = "shadow"
        #     # img_dir = "/workspace/isaaclab/IsaacLabExtension/images/franka"
        #     img_dir = "/home/elle/code/external/IsaacLab/roto"
        #     file_path = os.path.join(img_dir, f"{name}.png")
        #     fig = plt.figure(figsize=(4, 4))
        #     plt.imshow(img_batch[0].cpu())
        #     plt.show()
            # save_images_to_file(img_batch, file_path)
            # self.count += 1

        return flattened_images


@torch.jit.script
def scale(x, lower, upper):
    return 0.5 * (x + 1.0) * (upper - lower) + lower


@torch.jit.script
def unscale(x, lower, upper):
    return (2.0 * x - upper - lower) / (upper - lower)


@torch.jit.script
def randomize_rotation(rand0, rand1, x_unit_tensor, y_unit_tensor):
    return quat_mul(
        quat_from_angle_axis(rand0 * np.pi, x_unit_tensor), quat_from_angle_axis(rand1 * np.pi, y_unit_tensor)
    )


@torch.jit.script
def rotation_distance(object_rot, target_rot):
    # Orientation alignment for the cube in hand and goal cube
    quat_diff = quat_mul(object_rot, quat_conjugate(target_rot))
    return 2.0 * torch.asin(torch.clamp(torch.norm(quat_diff[:, 1:4], p=2, dim=-1), max=1.0))  # changed quat convention


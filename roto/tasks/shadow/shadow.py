# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


"""Shadow-hand base environment utilities shared across RoTO tasks."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.envs import ViewerCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensor, ContactSensorCfg
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils import configclass
from isaaclab.utils.math import quat_conjugate, quat_from_angle_axis, quat_mul

from roto.assets.shadow_hand import SHADOW_HAND_CFG
from roto.tasks.roto_env import RotoEnv, RotoEnvCfg

from isaaclab.markers.config import FRAME_MARKER_CFG  # isort: skip



@configclass
class ShadowEnvCfg(RotoEnvCfg):
    """Default configuration for the Shadow hand."""

    eye = (4, -4, 2.1)
    lookat = (2, -2, 0.5)
    viewer: ViewerCfg = ViewerCfg(eye=eye, lookat=lookat, resolution=(1920, 1080))

    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=4096, env_spacing=2, replicate_physics=True)

    episode_length_s = 10.0
    num_actions = 20
    action_space = num_actions

    reset_joint_pos_noise = 0.2
    reset_joint_vel_noise = 0.0

    hand_height = 0.5
    robot_cfg: ArticulationCfg = SHADOW_HAND_CFG.replace(prim_path="/World/envs/env_.*/Robot").replace(
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, hand_height),
            rot=(1.0, 0.0, 0.0, 0.0),
            joint_pos={".*": 0.0},
        )
    )

    actuated_joint_names = [
        "robot0_WRJ1",
        "robot0_WRJ0",
        "robot0_FFJ3",
        "robot0_FFJ2",
        "robot0_FFJ1",
        "robot0_MFJ3",
        "robot0_MFJ2",
        "robot0_MFJ1",
        "robot0_RFJ3",
        "robot0_RFJ2",
        "robot0_RFJ1",
        "robot0_LFJ4",
        "robot0_LFJ3",
        "robot0_LFJ2",
        "robot0_LFJ1",
        "robot0_THJ4",
        "robot0_THJ3",
        "robot0_THJ2",
        "robot0_THJ1",
        "robot0_THJ0",
    ]
    fingertip_body_names = [
        "robot0_ffdistal",
        "robot0_mfdistal",
        "robot0_rfdistal",
        "robot0_lfdistal",
        "robot0_thdistal",
    ]

    marker_cfg = FRAME_MARKER_CFG.copy()
    marker_cfg.markers["frame"].scale = (0.05, 0.05, 0.05)
    marker_cfg.prim_path = "/Visuals/ContactCfg"
    distal_contact_cfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/robot0_.*distal",
        update_period=0.0,
        history_length=1,
    )
    middle_contact_cfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/robot0_.*middle",
        update_period=0.0,
        history_length=1,
    )
    proximal_contact_cfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/robot0_.*proximal",
        update_period=0.0,
        history_length=1,
    )
    palm_contact_cfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/robot0_palm",
        update_period=0.0,
        history_length=1,
    )
    metacarpal_contact_cfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/robot0_lfmetacarpal",
        update_period=0.0,
        history_length=1,
    )


class ShadowEnv(RotoEnv):
    """Shadow-hand base env providing tactile + proprio pipelines."""
    cfg: ShadowEnvCfg

    def __init__(self, cfg: ShadowEnvCfg, render_mode: str | None = None, **kwargs):

        super().__init__(cfg, render_mode, **kwargs)

        self.num_prop_observations = 272
        self.num_tactile_observations = 68

        self.tactile = torch.zeros((self.num_envs, self.num_tactile_observations), device=self.device)
        self.last_tactile = torch.zeros((self.num_envs, self.num_tactile_observations), device=self.device)

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
            "air_reward": None,
        }

    def _setup_scene(self):
        """Register the Shadow hand, contact sensors, and lighting."""
        self.robot = Articulation(self.cfg.robot_cfg)
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())
        self.scene.clone_environments(copy_from_source=False)
        self.scene.articulations["robot"] = self.robot
        
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

        # cool disco lights
        pink = (0.9882352941176471, 0.011764705882352941, 0.7098039215686275)
        aqua = (0.0, 1.0, 1.0)
        disco_intensity = 0
        light_cfg_1 = sim_utils.SphereLightCfg(intensity=disco_intensity, color=pink)
        light_cfg_1.func("/World/ds", light_cfg_1, translation=(1, 1, 2))
        light_cfg_2 = sim_utils.SphereLightCfg(intensity=disco_intensity, color=aqua)
        light_cfg_2.func("/World/disk", light_cfg_2, translation=(-1, 1, 2))
        light_cfg_3 = sim_utils.SphereLightCfg(intensity=disco_intensity, color=pink)
        light_cfg_3.func("/World/ds1", light_cfg_3, translation=(-1, -1, 2))
        light_cfg_4 = sim_utils.SphereLightCfg(intensity=disco_intensity, color=aqua)
        light_cfg_4.func("/World/disk2", light_cfg_4, translation=(1, -1, 2))

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

    def _get_proprioception(self):
        """Return proprioceptive feature vector (positions, velocities, actions)."""
        prop = torch.cat(
            (
                self.normalised_joint_pos,
                self.normalised_joint_vel,
                self.actions,
            ),
            dim=-1,
        )

        return prop

    def _get_tactile(self):
        """Return binary tactile activation per finger segment."""
        distal_forces = self.distal_sensor.data.net_forces_w[:].clone()
        proximal_forces = self.proximal_sensor.data.net_forces_w[:].clone()
        middle_forces = self.middle_sensor.data.net_forces_w[:].clone()
        palm_forces = self.palm_sensor.data.net_forces_w[:].clone()
        metacarpal_forces = self.metacarpal_sensor.data.net_forces_w[:].clone()

        distal_norm = torch.norm(distal_forces, dim=-1)
        proximal_norm = torch.norm(proximal_forces, dim=-1)
        middle_norm = torch.norm(middle_forces, dim=-1)
        palm_norm = torch.norm(palm_forces, dim=-1)
        metacarpal_norm = torch.norm(metacarpal_forces, dim=-1)

        if self.dtype == torch.float16:
            distal_norm = (distal_norm > self.binary_threshold).half()
            proximal_norm = (proximal_norm > self.binary_threshold).half()
            middle_norm = (middle_norm > self.binary_threshold).half()
            palm_norm = (palm_norm > self.binary_threshold).half()
            metacarpal_norm = (metacarpal_norm > self.binary_threshold).half()
        else:
            distal_norm = (distal_norm > self.binary_threshold).float()
            proximal_norm = (proximal_norm > self.binary_threshold).float()
            middle_norm = (middle_norm > self.binary_threshold).float()
            palm_norm = (palm_norm > self.binary_threshold).float()
            metacarpal_norm = (metacarpal_norm > self.binary_threshold).float()

        tactile = torch.cat((distal_norm, proximal_norm, middle_norm, palm_norm, metacarpal_norm), dim=-1)

        self.last_tactile = self.tactile
        self.tactile = tactile
        return tactile

    def _reset_idx(self, env_ids: Sequence[int] | None):
        """Reset articulation state and optionally randomize joints."""
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES

        # resets articulation and rigid body attributes
        super()._reset_idx(env_ids)

        # reset hand
        self._reset_robot(env_ids, joint_pos_noise=self.cfg.reset_joint_pos_noise)


@torch.jit.script
def randomize_rotation(rand0, rand1, x_unit_tensor, y_unit_tensor):
    """Return a quaternion composed of rotations around the X and Y axes."""
    return quat_mul(
        quat_from_angle_axis(rand0 * np.pi, x_unit_tensor), quat_from_angle_axis(rand1 * np.pi, y_unit_tensor)
    )


@torch.jit.script
def rotation_distance(object_rot, target_rot):
    """Orientation alignment helper between the cube in hand and goal cube."""
    quat_diff = quat_mul(object_rot, quat_conjugate(target_rot))
    return 2.0 * torch.asin(torch.clamp(torch.norm(quat_diff[:, 1:4], p=2, dim=-1), max=1.0))

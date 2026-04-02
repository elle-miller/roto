# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Author: Elle Miller 2025

Allegro-hand base environment utilities shared across RoTO tasks.
"""

from __future__ import annotations

import numpy as np
import torch
from collections.abc import Sequence

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.envs import ViewerCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensor, ContactSensorCfg
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils import configclass
from isaaclab.utils.math import quat_conjugate, quat_from_angle_axis, quat_mul
from isaaclab.utils.math import quat_apply

from roto.assets.allegro import ALLEGRO_HAND_CFG
from roto.tasks.roto_env import RotoEnv, RotoEnvCfg

from isaaclab.markers.config import FRAME_MARKER_CFG  # isort: skip

def quat_normalize(q: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return q / (torch.norm(q, dim=-1, keepdim=True) + eps)

def quat_from_two_vectors(v1: torch.Tensor, v2: torch.Tensor, eps = 1e-8) -> torch.Tensor:
    v1 = v1 / (torch.norm(v1, dim=-1, keepdim=True) + eps)
    v2 = v2 / (torch.norm(v2, dim=-1, keepdim=True) + eps)
    v = torch.cross(v1, v2, dim=-1)
    w = 1.0 + torch.sum(v1 * v2, dim=-1, keepdim = True)

    # If vectors are opposite, w ~ 0 -> choose an orthogonal axis
    mask = (w.squeeze(-1) < eps)
    if mask.any():
        # pick an axis not parallel to a
        axis = torch.zeros_like(v1)
        axis[..., 0] = 1.0
        alt = torch.zeros_like(v1)
        alt[..., 1] = 1.0
        # if a is too aligned with x, use y
        use_alt = (torch.abs(v1[..., 0]) > 0.9)
        axis[use_alt] = alt[use_alt]

        v2 = torch.cross(v1, axis, dim=-1)
        q2 = torch.cat([torch.zeros_like(w), v2], dim=-1)
        q = torch.cat([w, v], dim=-1)
        q[mask] = q2[mask]
    else:
        q = torch.cat([w, v], dim=-1)

    return quat_normalize(q)

@configclass
class AllegroEnvCfg(RotoEnvCfg):
    """Default configuration for the Allegro hand."""

    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=4096, env_spacing=0.7, replicate_physics=True
    )

    eye = (4, -4, 2.1)
    lookat = (2, -2, 0.5)
    viewer: ViewerCfg = ViewerCfg(eye=eye, lookat=lookat, resolution=(1920, 1080))

    episode_length_s = 10.0
    num_actions = 16
    action_space = num_actions

    reset_joint_pos_noise = 0.2
    reset_joint_vel_noise = 0.0

    hand_height = 0.5
    robot_cfg: ArticulationCfg = ALLEGRO_HAND_CFG.replace(
        prim_path="/World/envs/env_.*/Robot"
    ).replace(
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, hand_height),
            # palm up: 0.270598, -0.653283, -0.270598, 0.65328
            #  palm side: (0.0, -0.9239, -0.3827, 0.0)
            rot = (0.0, 0.3827, -0.9239, 0.0), # (0.63742, -0.33667, 0.59201, -0.36038) ,   # -90 about Y
            # joint_pos=ALLEGRO_HAND_CFG.init_state.joint_pos,  # keep thumb_joint_0=0.28 etc.
            # start with curled pos
            joint_pos={
                "index_joint_1": 0.6, "index_joint_2": 0.6, "index_joint_3": 0.6,
                "middle_joint_1": 0.6, "middle_joint_2": 0.6, "middle_joint_3": 0.6,
                "ring_joint_1": 0.6, "ring_joint_2": 0.6, "ring_joint_3": 0.6,
                "thumb_joint_0": 0.5, "thumb_joint_1": 0.6, "thumb_joint_2": 0.6, "thumb_joint_3": 0.6,
                }
        )
    )

    actuated_joint_names = [
        "index_joint_0", "middle_joint_0", "ring_joint_0", "thumb_joint_0",
        "index_joint_1", "middle_joint_1", "ring_joint_1", "thumb_joint_1",
        "index_joint_2", "middle_joint_2", "ring_joint_2", "thumb_joint_2",
        "index_joint_3", "middle_joint_3", "ring_joint_3", "thumb_joint_3",
    ]

    # num actions has to correspond with the number of actuated joints
    num_actions = len(actuated_joint_names)

    marker_cfg = FRAME_MARKER_CFG.copy()
    marker_cfg.markers["frame"].scale = (0.05, 0.05, 0.05)
    marker_cfg.prim_path = "/Visuals/ContactCfg"

    # Updated to match the Allegro hand
    robot_contact_sensor_cfg = ContactSensorCfg(
        # Captures palm_link, any finger link_0-3, and any biotac_tip
        prim_path="/World/envs/env_.*/Robot/(palm_link|.*_link_.*|.*_biotac_tip)",
        update_period=0.0,
        history_length=1,
        debug_vis=True, 
    )


class AllegroEnv(RotoEnv):
    """Allegro-hand base env providing tactile + proprio pipelines."""

    cfg: AllegroEnvCfg

    def __init__(self, cfg: AllegroEnvCfg, render_mode: str | None = None, **kwargs):

        super().__init__(cfg, render_mode, **kwargs)

        print("NUM JOINTS:", len(self.robot.joint_names))
        print("JOINT NAMES:", self.robot.joint_names)
        print("BODY NAMES:", self.robot.body_names)
        print("NUM TACTILE SENSORS:", len(self.robot_contact_sensor.body_names))


    def _level_palm_to_world_up(self, env_ids = None): 
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES

        palm_idx = self.robot.body_names.index("palm_link")

        # pick a candidate local normal axis of the palm link (try +Z first)
        palm_local_normal = torch.tensor([0.0, 0.0, 1.0], device=self.device).repeat(len(env_ids), 1)
        world_up = torch.tensor([0.0, 0.0, 1.0], device=self.device).repeat(len(env_ids), 1)

        # env 0: palm orientation in world
        palm_quat_w = self.robot.data.body_quat_w[env_ids, palm_idx, :]  # (w,x,y,z)
        palm_normal_w = quat_apply(palm_quat_w, palm_local_normal)   # (N, 3) --> normal direction of the palm
        # print("before palm_normal_w[0]:", palm_normal_w[0].tolist())
        
        q_delta = quat_from_two_vectors(palm_normal_w, world_up)          # (N,4)

        # Current root pose
        root_state = self.robot.data.root_state_w[env_ids].clone()        # (N,13): pos(3), quat(4), linvel(3), angvel(3)
        root_pos = root_state[:, 0:3]
        root_quat = root_state[:, 3:7]

        # Apply delta to root orientation: q_new = q_delta ⊗ q_root
        root_quat_new = quat_mul(q_delta, root_quat)

        # Write back (zero velocities so it doesn't kick)
        self.robot.write_root_pose_to_sim(torch.cat([root_pos, root_quat_new], dim=-1), env_ids)
        self.robot.write_root_velocity_to_sim(torch.zeros((len(env_ids), 6), device=self.device), env_ids)

        # step once (or otherwise refresh robot.data) before re-reading body_quat_w
        self.sim.step(render=False)

        palm_quat_w2 = self.robot.data.body_quat_w[env_ids, palm_idx, :]
        palm_normal_w2 = quat_apply(palm_quat_w2, palm_local_normal)
        # print("after  palm_normal_w[0]:", palm_normal_w2[0].tolist())

    def _setup_scene(self):
        """Register the Allegro hand, contact sensors, and lighting."""
        super()._setup_scene()

        self.robot = Articulation(self.cfg.robot_cfg)
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())
        self.scene.clone_environments(copy_from_source=False)
        self.scene.articulations["robot"] = self.robot

        light_cfg = sim_utils.DomeLightCfg(intensity=200.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

        self.robot_contact_sensor = ContactSensor(self.cfg.robot_contact_sensor_cfg)
        self.scene.sensors["robot_contact_sensor"] = self.robot_contact_sensor


    def _reset_idx(self, env_ids: Sequence[int] | None):
        """Reset articulation state and optionally randomize joints.

        Args:
            env_ids: Environment indices to reset. If None, resets all environments.
        """
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES

        # Reset articulation and rigid body attributes
        super()._reset_idx(env_ids)

        # make palm horizontal
        # self._level_palm_to_world_up(env_ids)

        # Reset hand with noise
        self._reset_robot(env_ids, joint_pos_noise=self.cfg.reset_joint_pos_noise)

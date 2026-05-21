# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Central bounce task: robot-specific configs and envs share one implementation."""

from __future__ import annotations

from pathlib import Path

import torch
from collections.abc import Sequence

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, RigidObject, RigidObjectCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import sample_uniform

from roto.tasks.robots.allegro.allegro import (
    ALLEGRO_BOUNCE_ROOT_ROT_WXYZ,
    ALLEGRO_DEFAULT_JOINT_POS,
    ALLEGRO_HAND_HEIGHT_M,
    AllegroEnv,
    AllegroEnvCfg,
    build_allegro_robot_cfg,
)
from roto.tasks.robots.orca.orca import OrcaEnv, OrcaEnvCfg
from roto.tasks.physics import (
    bounce_ball_material,
    bounce_ball_mass_props,
    bounce_ball_rigid_props,
    bounce_ball_radius_m,
    contact_props,
)
from roto.tasks.robots.shadow.shadow import ShadowEnv, ShadowEnvCfg
from roto.tasks.robots.shadowlite.shadowlite import ShadowLiteEnv, ShadowLiteEnvCfg

_BOUNCE_HDR = Path(__file__).resolve().parent.parent.parent / "assets/rooms/stierberg_sunrise_4k.hdr"
# _BOUNCE_HDR = Path(__file__).resolve().parent.parent.parent / "assets/rooms/qwantani_dusk_2_4k.hdr"

_BOUNCE_BALL_COLOUR = (0.7294117647058823, 0.3176470588235294, 0.7137254901960784)
_SIMPLE_SPHERE_RADIUS_M = 0.035
_SIMPLE_SPHERE_MASS_KG = 30 / 1000.0


def _make_bouncy_ball_cfg(default_object_pos: tuple[float, float, float]) -> RigidObjectCfg:
    return RigidObjectCfg(
        prim_path="/World/envs/env_.*/Object",
        init_state=RigidObjectCfg.InitialStateCfg(pos=default_object_pos, rot=(1.0, 0.0, 0.0, 0.0)),
        spawn=sim_utils.SphereCfg(
            radius=bounce_ball_radius_m,
            physics_material=bounce_ball_material,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=_BOUNCE_BALL_COLOUR, metallic=0.1),
            rigid_props=bounce_ball_rigid_props,
            mass_props=bounce_ball_mass_props,
            collision_props=contact_props,
        ),
    )



# --- Configs -----------------------------------------------------------------


@configclass
class BounceTaskCfg:
    """Shared bounce task parameters (mixed into each robot env cfg)."""

    episode_length_s = 10.0
    fall_height = 0.4
    reset_position_noise = 0.01
    object_y_pos = -0.39
    object_z_pos = 0.6
    out_of_bounds = 0.2
    min_timesteps_between_contact = 5
    ball_colour = _BOUNCE_BALL_COLOUR


@configclass
class BounceShadowLiteCfg(BounceTaskCfg, ShadowLiteEnvCfg):
    
    fall_height = 0.3          
    object_y_pos = -0.28    
    object_z_pos = 0.6
    default_object_pos = (0., -0.265, 0.6)  # is this affecting the ball position at all? cuz this is not changing anything in the viewer
    object_cfg: RigidObjectCfg = _make_bouncy_ball_cfg((0., -0.265, 0.6)  )


@configclass
class BounceCfg(BounceTaskCfg, ShadowEnvCfg):
    """Bounce task on the Shadow hand (default registered env ``Bounce``)."""

    default_object_pos = (0.0, -0.39, 0.6)
    object_cfg: RigidObjectCfg = _make_bouncy_ball_cfg((0.0, -0.39, 0.6))


@configclass
class BounceOrcaCfg(BounceTaskCfg, OrcaEnvCfg):
    """Orca hand: override only what differs from :class:`BounceTaskCfg`."""

    fall_height = 0.3
    object_y_pos = 0.1
    object_z_pos = 0.6
    default_object_pos = (0.23, 0.1, 0.6)
    object_cfg: RigidObjectCfg = _make_bouncy_ball_cfg((0.23, 0.1, 0.6))


@configclass
class BounceAllegroCfg(BounceTaskCfg, AllegroEnvCfg):
    """Allegro hand: override only what differs from :class:`BounceTaskCfg`."""

    initial_root_rot = ALLEGRO_BOUNCE_ROOT_ROT_WXYZ
    robot_cfg: ArticulationCfg = build_allegro_robot_cfg(
        initial_root_rot=initial_root_rot,
        hand_height=ALLEGRO_HAND_HEIGHT_M,
        default_joint_pos=ALLEGRO_DEFAULT_JOINT_POS,
    )

    fall_height = 0.3
    object_y_pos = 0.0
    object_z_pos = 0.55
    default_object_pos = (0.1, 0.0, 0.55)
    object_cfg: RigidObjectCfg = _make_bouncy_ball_cfg((0.1, 0.0, 0.55))


# --- Shared logic ------------------------------------------------------------

_BAODING_HDR = Path(__file__).resolve().parent.parent.parent / "assets/rooms/kloppenheim_02_puresky_4k.hdr"

class BounceMixin:
    """Object tracking, bounce detection, and rewards shared by all robots."""

    cfg: BounceTaskCfg

    def _bounce_spawn_object_and_hdr(self) -> None:
        """Attach the ball and dome HDR (same for every robot)."""
        self.object = RigidObject(self.cfg.object_cfg)
        self.scene.rigid_objects["object"] = self.object
        light = sim_utils.DomeLightCfg(
            color=(1.0, 1.0, 1.0),
            intensity=1000.0,
            texture_file=str(_BOUNCE_HDR),
            texture_format="latlong",
        )
        light.func("/World/bglight", light)
        # light = sim_utils.DomeLightCfg(
        #     color=(0.81,0.86,1.28),
        #     intensity=1000.0,
        #     texture_file=str(_BAODING_HDR),
        #     texture_format="latlong",
        # )
        # light.func("/World/bglight", light)

        # light = sim_utils.SphereLightCfg(
        #     intensity=1000.0,
        #     color=(1.0, 1.0, 1.0),
        # )
        # light.func("/World/spotlight_1", light, translation=(0.4, -0.4, 1.1))

    def _get_gt(self):
        """Ground-truth features for auxiliary heads / logging (shared across robots)."""
        return torch.cat(
            (
                self.object_pos,
                self.object_linvel,
                torch.norm(self.object_linvel, dim=1).unsqueeze(-1),
            ),
            dim=-1,
        )

    def _init_bounce_tracking(self) -> None:
        self.object_pos = torch.zeros((self.num_envs, 3), device=self.device)
        self.object_rot = torch.zeros((self.num_envs, 4), device=self.device)
        self.object_linvel = torch.zeros((self.num_envs, 3), device=self.device)
        self.object_angvel = torch.zeros((self.num_envs, 3), device=self.device)

        self.num_transitions = torch.zeros((self.num_envs,), dtype=self.dtype, device=self.device)
        self.num_bounces = torch.zeros((self.num_envs,), dtype=self.dtype, device=self.device)
        self.waiting_for_contact = torch.zeros((self.num_envs,), dtype=torch.bool, device=self.device)
        self.new_bounces = torch.zeros((self.num_envs,), dtype=self.dtype, device=self.device)
        self.time_without_contact = torch.zeros((self.num_envs,), dtype=torch.int, device=self.device)
        self._get_tactile()
        self.last_tactile = self.tactile.clone()

    def _compute_intermediate_values(self, env_ids=None):
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        super()._compute_intermediate_values(env_ids)

        self.object_pos[env_ids] = self.object.data.root_pos_w[env_ids] - self.scene.env_origins[env_ids]
        self.object_rot[env_ids] = self.object.data.root_quat_w[env_ids]
        self.object_linvel[env_ids] = self.object.data.root_lin_vel_w[env_ids]
        self.object_angvel[env_ids] = self.object.data.root_ang_vel_w[env_ids]

        prev_contact = (torch.sum(self.last_tactile, dim=1) > 0).int()
        curr_contact = (torch.sum(self.tactile, dim=1) > 0).int()

        lost_contact = (prev_contact == 1) & (curr_contact == 0)
        new_contact = (prev_contact == 0) & (curr_contact == 1)

        prev_time_without_contact = self.time_without_contact.clone()

        self.time_without_contact = torch.where(
            curr_contact == 0,
            self.time_without_contact + 1,
            torch.zeros_like(self.time_without_contact),
        )

        valid_new_contact = new_contact & (prev_time_without_contact >= self.cfg.min_timesteps_between_contact)

        self.new_bounces = (self.waiting_for_contact & valid_new_contact).float()
        self.num_bounces += self.new_bounces
        self.waiting_for_contact = self.waiting_for_contact | lost_contact
        self.waiting_for_contact = self.waiting_for_contact & ~valid_new_contact

        self.num_transitions += (lost_contact | valid_new_contact).float()

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        self._compute_intermediate_values()

        fall = self.object_pos[:, 2] < self.cfg.fall_height
        out_of_bounds = torch.abs(self.object_pos[:, 1] - self.cfg.object_y_pos) > self.cfg.out_of_bounds
        termination = fall | out_of_bounds
        time_out = self.episode_length_buf >= self.max_episode_length - 1

        return termination, time_out

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES

        super()._reset_idx(env_ids)

        self._reset_object(env_ids)

        self.num_transitions[env_ids] = 0
        self.num_bounces[env_ids] = 0
        self.new_bounces[env_ids] = 0
        self.waiting_for_contact[env_ids] = False
        self.time_without_contact[env_ids] = 0

    def _reset_object(self, env_ids):
        object_default_state = self.object.data.default_root_state.clone()[env_ids]
        pos_noise = sample_uniform(-1.0, 1.0, (len(env_ids), 3), device=self.device)
        object_default_state[:, 0:3] = (
            object_default_state[:, 0:3] + self.cfg.reset_position_noise * pos_noise + self.scene.env_origins[env_ids]
        )

        object_default_state[:, 7:] = torch.zeros_like(self.object.data.default_root_state[env_ids, 7:])
        self.object.write_root_pose_to_sim(object_default_state[:, :7], env_ids)
        self.object.write_root_velocity_to_sim(object_default_state[:, 7:], env_ids)

    def _get_rewards(self) -> torch.Tensor:

        total_reward, bounce_reward = compute_rewards(self.new_bounces)

        self.extras["log"] = {
            "object_z_linvel": torch.abs(self.object_linvel[:, 2]).clone(),
            "sum_forces": torch.sum(self.tactile, dim=1).clone(),
            "bounce_reward": bounce_reward.float().clone(),
        }

        self.extras["counters"] = {
            "num_bounces": self.num_bounces.float().clone(),
        }

        return total_reward


# --- Robot-specific envs -----------------------------------------------------


class BounceShadowEnv(BounceMixin, ShadowEnv):
    """Bounce a sphere using the Shadow hand and binary tactile signals."""

    cfg: BounceCfg

    def __init__(self, cfg: BounceCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        self._init_bounce_tracking()

    def _setup_scene(self):
        super()._setup_scene()
        self._bounce_spawn_object_and_hdr()


class BounceShadowLiteEnv(BounceMixin, ShadowLiteEnv):
    """Bounce a sphere using the Shadow Lite hand and binary tactile signals."""

    cfg: BounceShadowLiteCfg

    def __init__(self, cfg: BounceShadowLiteCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        self._init_bounce_tracking()

    def _setup_scene(self):
        super()._setup_scene()
        self._bounce_spawn_object_and_hdr()


class BounceOrcaEnv(BounceMixin, OrcaEnv):
    """Bounce a sphere using the Orca hand and binary tactile signals."""

    cfg: BounceOrcaCfg

    def __init__(self, cfg: BounceOrcaCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        self._init_bounce_tracking()

    def _setup_scene(self):
        super()._setup_scene()
        self._bounce_spawn_object_and_hdr()

    # the ORCA hand would cheat by trapping it at the wrist, so have added
    # a check to make sure the ball has some distance from the wrist
    def _get_rewards(self) -> torch.Tensor:

        valid_bounces = self.new_bounces.clone()
        ball_x_pos = self.object_pos[:, 0]
        valid_bounces[ball_x_pos < 0.17] = 0.0
        
        total_reward, bounce_reward = compute_rewards_orca(valid_bounces)

        self.extras["log"] = {
            "object_z_linvel": torch.abs(self.object_linvel[:, 2]).clone(),
            "sum_forces": torch.sum(self.tactile, dim=1).clone(),
            "bounce_reward": bounce_reward.float().clone(),
        }

        self.extras["counters"] = {
            "num_bounces": self.num_bounces.float().clone(),
        }

        return total_reward


class BounceAllegroEnv(BounceMixin, AllegroEnv):
    """Bounce a sphere using the Allegro hand and binary tactile signals."""

    cfg: BounceAllegroCfg

    def __init__(self, cfg: BounceAllegroCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        self._init_bounce_tracking()

    def _setup_scene(self):
        super()._setup_scene()
        self._bounce_spawn_object_and_hdr()


@torch.jit.script
def compute_rewards(new_bounces: torch.Tensor):
    bounce_reward = new_bounces * 10
    total_reward = bounce_reward
    return total_reward, bounce_reward

@torch.jit.script
def compute_rewards_orca(new_bounces: torch.Tensor):
    bounce_reward = new_bounces * 10
    total_reward = bounce_reward
    return total_reward, bounce_reward

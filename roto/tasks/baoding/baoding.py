# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Central baoding task: shared logic and robot-specific reset behavior."""

from __future__ import annotations

import torch
from collections.abc import Sequence

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, RigidObject, RigidObjectCfg
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.sim.schemas.schemas_cfg import CollisionPropertiesCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import quat_apply, sample_uniform

from roto.tasks.robots.allegro.allegro import (
    ALLEGRO_BAODING_ROOT_ROT_WXYZ,
    ALLEGRO_DEFAULT_JOINT_POS,
    ALLEGRO_HAND_HEIGHT_M,
    AllegroEnv,
    AllegroEnvCfg,
    build_allegro_robot_cfg,
)
from roto.tasks.robots.orca.orca import OrcaEnv, OrcaEnvCfg
from roto.tasks.robots.shadow.shadow import ShadowEnv, ShadowEnvCfg
from pathlib import Path
_BAODING_HDR = Path(__file__).resolve().parent.parent.parent / "assets/rooms/stierberg_sunrise_4k.hdr"
# _BAODING_HDR = Path(__file__).resolve().parent.parent.parent / "assets/rooms/qwantani_dusk_2_4k.hdr"

def make_baoding_object_cfgs(
    *,
    ball_mass_kg: float,
    ball_radius_m: float,
    ball_1_pos: tuple[float, float, float],
    ball_2_pos: tuple[float, float, float],
    colour_1: tuple[float, float, float],
    colour_2: tuple[float, float, float],
) -> dict[str, RigidObjectCfg | VisualizationMarkersCfg]:
    """Build ball rigid bodies and goal markers from scalar task parameters."""
    ball_1_cfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/ball1",
        init_state=RigidObjectCfg.InitialStateCfg(pos=ball_1_pos, rot=(1.0, 0.0, 0.0, 0.0)),
        spawn=sim_utils.SphereCfg(
            radius=ball_radius_m,
            physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=1.0, dynamic_friction=1.0, restitution=0.0),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=colour_1, metallic=0.5),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=False,
                disable_gravity=False,
                enable_gyroscopic_forces=True,
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=0,
                sleep_threshold=0.005,
                stabilization_threshold=0.0025,
                max_depenetration_velocity=1000.0,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=ball_mass_kg),
            collision_props=CollisionPropertiesCfg(collision_enabled=True),
        ),
    )
    ball_2_cfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/ball2",
        init_state=RigidObjectCfg.InitialStateCfg(pos=ball_2_pos, rot=(1.0, 0.0, 0.0, 0.0)),
        spawn=sim_utils.SphereCfg(
            radius=ball_radius_m,
            physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=1.0, dynamic_friction=1.0, restitution=0.0),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=colour_2, metallic=0.5),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=False,
                disable_gravity=False,
                enable_gyroscopic_forces=True,
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=0,
                sleep_threshold=0.005,
                stabilization_threshold=0.0025,
                max_depenetration_velocity=1000.0,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=ball_mass_kg),
            collision_props=CollisionPropertiesCfg(collision_enabled=True),
        ),
    )
    target1_cfg = VisualizationMarkersCfg(
        prim_path="/Visuals/target_1",
        markers={
            "target_1": sim_utils.SphereCfg(
                radius=ball_radius_m * 0.98,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=colour_1),
            ),
        },
    )
    target2_cfg = VisualizationMarkersCfg(
        prim_path="/Visuals/target_2",
        markers={
            "target_2": sim_utils.SphereCfg(
                radius=ball_radius_m * 0.98,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=colour_2),
            ),
        },
    )
    return {
        "ball_1_cfg": ball_1_cfg,
        "ball_2_cfg": ball_2_cfg,
        "target1_cfg": target1_cfg,
        "target2_cfg": target2_cfg,
    }


# Bootstrap for :class:`BaodingTaskCfg` defaults (must match that class's scalar fields).
_OBJ = make_baoding_object_cfgs(
    ball_mass_kg=0.001 * 55,
    ball_radius_m=(1.5 / 2) * 2.54 / 100,
    ball_1_pos=(0.01, -0.37, 0.55),
    ball_2_pos=(0.01, -0.41, 0.55),
    colour_1=(0.4, 0.9882352941176471, 0.011764705882352941),
    colour_2=(0.0, 1.0, 1.0),
)


# --- Configs -----------------------------------------------------------------


@configclass
class BaodingTaskCfg:
    """Shared baoding parameters (same for every robot)."""

    episode_length_s = 10.0
    act_moving_average = 1
    ball_mass_g = 55
    ball_mass_kg = 0.001 * ball_mass_g
    ball_diameter_inches = 1.5
    ball_radius_m = (ball_diameter_inches / 2) * 2.54 / 100
    ball_reset_height = 0.55
    ball_diameter_m = ball_radius_m * 2
    target_offset = ball_diameter_m / 1.73205080757 + 0.001
    ball_dist_terminate = 0.15
    success_tolerance = 0.01 # 1cm by default
    palm_target_x = -0.03
    palm_target_y = -0.38
    palm_target_z = 0.46
    diagonal_target_x = palm_target_x + target_offset
    diagonal_target_y = palm_target_y + target_offset
    diagonal_target_z = palm_target_z + target_offset
    colour_1 = (0.4, 0.9882352941176471, 0.011764705882352941)
    colour_2 = (0.0, 1.0, 1.0)
    ball_1_init_x = 0.01
    ball_1_init_y = -0.37
    ball_2_init_x = 0.01
    ball_2_init_y = -0.41
    ball_1_cfg: RigidObjectCfg = _OBJ["ball_1_cfg"]  # type: ignore[assignment]
    ball_2_cfg: RigidObjectCfg = _OBJ["ball_2_cfg"]  # type: ignore[assignment]
    target1_cfg: VisualizationMarkersCfg = _OBJ["target1_cfg"]  # type: ignore[assignment]
    target2_cfg: VisualizationMarkersCfg = _OBJ["target2_cfg"]  # type: ignore[assignment]


def apply_baoding_object_cfgs_from_scalars(cfg: BaodingTaskCfg) -> None:
    """Mutate ``cfg.ball_*_cfg`` / ``target*_cfg`` so they match mass, size, colours, and spawn x,y,z."""
    z = cfg.ball_reset_height
    ball_1_pos = (cfg.ball_1_init_x, cfg.ball_1_init_y, z)
    ball_2_pos = (cfg.ball_2_init_x, cfg.ball_2_init_y, z)
    obj = make_baoding_object_cfgs(
        ball_mass_kg=cfg.ball_mass_kg,
        ball_radius_m=cfg.ball_radius_m,
        ball_1_pos=ball_1_pos,
        ball_2_pos=ball_2_pos,
        colour_1=cfg.colour_1,
        colour_2=cfg.colour_2,
    )
    cfg.ball_1_cfg = obj["ball_1_cfg"]
    cfg.ball_2_cfg = obj["ball_2_cfg"]
    cfg.target1_cfg = obj["target1_cfg"]
    cfg.target2_cfg = obj["target2_cfg"]


@configclass
class BaodingCfg(BaodingTaskCfg, ShadowEnvCfg):
    """Baoding on the Shadow hand (registered env ``Baoding``)."""


@configclass
class BaodingOrcaCfg(BaodingTaskCfg, OrcaEnvCfg):
    """Baoding on the Orca hand."""


@configclass
class BaodingAllegroCfg(BaodingTaskCfg, AllegroEnvCfg):
    """Baoding on the Allegro hand."""

    initial_root_rot = ALLEGRO_BAODING_ROOT_ROT_WXYZ
    robot_cfg: ArticulationCfg = build_allegro_robot_cfg(
        initial_root_rot=initial_root_rot,
        hand_height=ALLEGRO_HAND_HEIGHT_M,
        default_joint_pos=ALLEGRO_DEFAULT_JOINT_POS,
    )

    ball_reset_height = 0.48
    ball_dist_terminate = 0.15

    ball_diameter_inches = 2
    ball_radius_m = (ball_diameter_inches / 2) * 2.54 / 100
    ball_diameter_m = ball_radius_m * 2

    target_offset = ball_diameter_m + 0.005
    success_tolerance = 0.015 # 1cm by default
    palm_target_x = 0.12
    palm_target_y = -0.03
    palm_target_z = 0.45
    diagonal_target_x = palm_target_x # + target_offset
    diagonal_target_y = palm_target_y + target_offset
    diagonal_target_z = palm_target_z # + target_offset


# --- Shared env logic --------------------------------------------------------


class BaodingMixin:
    """Two-ball baoding task logic; subclasses implement :meth:`_baoding_reset_balls`."""

    cfg: BaodingTaskCfg

    def _init_baoding_state(self) -> None:
        self.reset_goal_1_buf = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.reset_goal_2_buf = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        self.ball_1_pos = torch.zeros((self.num_envs, 3), dtype=torch.float, device=self.device)
        self.ball_2_pos = torch.zeros((self.num_envs, 3), dtype=torch.float, device=self.device)

        self.ball_1_goal_dist = torch.ones((self.num_envs,), dtype=torch.float, device=self.device)
        self.ball_2_goal_dist = torch.ones((self.num_envs,), dtype=torch.float, device=self.device)
        self.ball_1_goal_dist3 = torch.ones((self.num_envs, 3), dtype=torch.float, device=self.device)
        self.ball_2_goal_dist3 = torch.ones((self.num_envs, 3), dtype=torch.float, device=self.device)

        self.ball_height_above_hand = torch.zeros((self.num_envs,), dtype=self.dtype, device=self.device)
        self.balls_center_vector = torch.zeros((self.num_envs, 3), dtype=self.dtype, device=self.device)
        self.ball_dist = torch.zeros((self.num_envs,), dtype=self.dtype, device=self.device)
        self.ball_1_linvel = torch.zeros((self.num_envs,), dtype=self.dtype, device=self.device)
        self.ball_2_linvel = torch.zeros((self.num_envs,), dtype=self.dtype, device=self.device)

        self.current_angle = torch.zeros((self.num_envs,), dtype=self.dtype, device=self.device)
        self.prev_angle = torch.zeros((self.num_envs,), dtype=self.dtype, device=self.device)
        self.angle_change = torch.zeros((self.num_envs,), dtype=self.dtype, device=self.device)
        self.cumulative_rotations = torch.zeros((self.num_envs,), dtype=self.dtype, device=self.device)
        self.total_rotations = torch.zeros((self.num_envs,), dtype=self.dtype, device=self.device)
        self.num_rotations = torch.zeros((self.num_envs,), dtype=torch.int, device=self.device)

        target_1 = torch.tensor(
            (self.cfg.palm_target_x, self.cfg.palm_target_y, self.cfg.palm_target_z),
            dtype=torch.float,
            device=self.device,
        )
        target_2 = torch.tensor(
            (self.cfg.diagonal_target_x, self.cfg.diagonal_target_y, self.cfg.diagonal_target_z),
            dtype=torch.float,
            device=self.device,
        )
        self.goal_pos1 = target_1.repeat(self.num_envs, 1)
        self.goal_pos2 = target_2.repeat(self.num_envs, 1)
        self.goal_rot = torch.zeros((self.num_envs, 4), dtype=torch.float, device=self.device)
        self.target1.visualize(self.goal_pos1 + self.scene.env_origins, self.goal_rot)
        self.target2.visualize(self.goal_pos2 + self.scene.env_origins, self.goal_rot)

        self.ball_goal_idx = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.update_goal_pos()

    def _setup_scene(self) -> None:
        super()._setup_scene()
        self.ball_1 = RigidObject(self.cfg.ball_1_cfg)
        self.ball_2 = RigidObject(self.cfg.ball_2_cfg)
        self.scene.rigid_objects["ball_1"] = self.ball_1
        self.scene.rigid_objects["ball_2"] = self.ball_2

        light = sim_utils.DomeLightCfg(
            color=(1.0, 1.0, 1.0),
            intensity=1000.0,
            texture_file=str(_BAODING_HDR),
            texture_format="latlong",
        )
        light.func("/World/bglight", light)

        self.target1 = VisualizationMarkers(self.cfg.target1_cfg)
        self.target2 = VisualizationMarkers(self.cfg.target2_cfg)

    def _get_gt(self) -> torch.Tensor:
        return torch.cat(
            (
                self.ball_1_pos,
                self.ball_2_pos,
                self.ball_1.data.root_lin_vel_w,
                self.ball_2.data.root_lin_vel_w,
                self.ball_dist.unsqueeze(1),
            ),
            dim=-1,
        )

    def _compute_intermediate_values(self, env_ids=None):
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        super()._compute_intermediate_values(env_ids)

        self.ball_1_pos = self.ball_1.data.root_pos_w - self.scene.env_origins
        self.ball_2_pos = self.ball_2.data.root_pos_w - self.scene.env_origins
        self.balls_center_vector = self.ball_1_pos - self.ball_2_pos

        self.ball_1_goal_dist3 = self.ball_1_pos - self.ball_1_goal_pos
        self.ball_2_goal_dist3 = self.ball_2_pos - self.ball_2_goal_pos
        self.ball_1_goal_dist = torch.norm(self.ball_1_goal_dist3, dim=1)
        self.ball_2_goal_dist = torch.norm(self.ball_2_goal_dist3, dim=1)

        self.ball_1_linvel = torch.norm(self.ball_1.data.root_lin_vel_w, dim=1)
        self.ball_2_linvel = torch.norm(self.ball_2.data.root_lin_vel_w, dim=1)
        self.ball_dist = torch.norm(self.balls_center_vector, dim=1)

    def _get_rewards(self) -> torch.Tensor:
        self.reset_goal_1_buf[self.ball_1_goal_dist <= self.cfg.success_tolerance] = True
        self.reset_goal_2_buf[self.ball_2_goal_dist <= self.cfg.success_tolerance] = True
        goal_reached = (self.reset_goal_1_buf & self.reset_goal_2_buf).float()
        goal_reached_ids = goal_reached.nonzero(as_tuple=False).squeeze(-1)

        total_reward, reach_goal_reward = compute_rewards(
            goal_reached,
            self.ball_1_goal_dist,
            self.ball_2_goal_dist,
        )

        self.extras["log"] = {
            "success_reward": (reach_goal_reward),
            "ball_1_vel": (self.ball_1_linvel),
            "ball_2_vel": (self.ball_2_linvel),
            "ball_dist": (self.ball_dist),
        }

        self.extras["counters"] = {"num_rotations": (self.num_rotations).float()}

        if len(goal_reached_ids) > 0:
            self._reset_target_pose(goal_reached_ids)

        return total_reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        self._compute_intermediate_values()

        out_of_reach = self.ball_dist >= self.cfg.ball_dist_terminate
        time_out = self.episode_length_buf >= self.max_episode_length - 1

        return out_of_reach, time_out

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES

        super()._reset_idx(env_ids)

        self._reset_object(env_ids)

        self.num_rotations[env_ids] = 0

    def _reset_object(self, env_ids: Sequence[int]) -> None:
        self._baoding_reset_balls(env_ids)

    def _reset_target_pose(self, reached_goal_ids):
        self.ball_goal_idx[reached_goal_ids] = ~self.ball_goal_idx[reached_goal_ids]
        self.update_goal_pos()

        self.num_rotations[reached_goal_ids] += 1

        self.target1.visualize(self.ball_1_goal_pos + self.scene.env_origins, self.goal_rot)
        self.target2.visualize(self.ball_2_goal_pos + self.scene.env_origins, self.goal_rot)

        self.reset_goal_1_buf[reached_goal_ids] = 0
        self.reset_goal_2_buf[reached_goal_ids] = 0

    def update_goal_pos(self):
        self.ball_1_goal_pos = torch.where(
            self.ball_goal_idx.unsqueeze(-1),
            self.goal_pos2,
            self.goal_pos1,
        )

        self.ball_2_goal_pos = torch.where(
            self.ball_goal_idx.unsqueeze(-1),
            self.goal_pos1,
            self.goal_pos2,
        )


class BaodingShadowEnv(BaodingMixin, ShadowEnv):
    """Baoding on the Shadow hand; balls reset from default spawn with noise."""

    cfg: BaodingCfg

    def __init__(self, cfg: BaodingCfg, render_mode: str | None = None, **kwargs):
        apply_baoding_object_cfgs_from_scalars(cfg)
        super().__init__(cfg, render_mode, **kwargs)
        self._init_baoding_state()

    def _baoding_reset_balls(self, env_ids: Sequence[int]) -> None:
        ball_1_default_state = self.ball_1.data.default_root_state.clone()[env_ids]
        ball_2_default_state = self.ball_2.data.default_root_state.clone()[env_ids]
        pos_noise = sample_uniform(-0.005, 0.005, (len(env_ids), 3), device=self.device)

        ball_1_default_state[:, 0:3] = ball_1_default_state[:, 0:3] + pos_noise + self.scene.env_origins[env_ids]
        ball_1_default_state[:, 7:] = torch.zeros_like(self.ball_1.data.default_root_state[env_ids, 7:])

        ball_2_default_state[:, 0:3] = ball_2_default_state[:, 0:3] + pos_noise + self.scene.env_origins[env_ids]
        ball_2_default_state[:, 7:] = torch.zeros_like(self.ball_2.data.default_root_state[env_ids, 7:])

        self.ball_1.write_root_pose_to_sim(ball_1_default_state[:, :7], env_ids)
        self.ball_1.write_root_velocity_to_sim(ball_1_default_state[:, 7:], env_ids)
        self.ball_2.write_root_pose_to_sim(ball_2_default_state[:, :7], env_ids)
        self.ball_2.write_root_velocity_to_sim(ball_2_default_state[:, 7:], env_ids)


class BaodingPalmResetMixin:
    """Palm-relative ball placement (Orca / Allegro)."""

    palm_idx: int

    def _baoding_reset_balls(self, env_ids: Sequence[int]) -> None:
        palm_pos_w = self.robot.data.body_pos_w[env_ids, self.palm_idx, :]
        palm_quat_w = self.robot.data.body_quat_w[env_ids, self.palm_idx, :]

        x_fwd = 0.04
        y_side = 0.04
        off1_local = torch.tensor([x_fwd, y_side, 0.0], device=self.device).repeat(len(env_ids), 1)
        off2_local = torch.tensor([x_fwd, -y_side, 0.0], device=self.device).repeat(len(env_ids), 1)

        off1_w = quat_apply(palm_quat_w, off1_local)
        off2_w = quat_apply(palm_quat_w, off2_local)

        pos_noise = sample_uniform(-0.005, 0.005, (len(env_ids), 3), device=self.device)

        ball1_pos_w = palm_pos_w + off1_w + pos_noise
        ball2_pos_w = palm_pos_w + off2_w + pos_noise

        env_o = self.scene.env_origins[env_ids]
        ball1_env = ball1_pos_w - env_o
        ball2_env = ball2_pos_w - env_o
        ball1_env[:, 2] = self.cfg.ball_reset_height
        ball2_env[:, 2] = self.cfg.ball_reset_height
        ball1_pos_w = ball1_env + env_o
        ball2_pos_w = ball2_env + env_o

        quat_w = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device).repeat(len(env_ids), 1)
        ball1_pose = torch.cat([ball1_pos_w, quat_w], dim=-1)
        ball2_pose = torch.cat([ball2_pos_w, quat_w], dim=-1)

        self.ball_1.write_root_pose_to_sim(ball1_pose, env_ids)
        self.ball_2.write_root_pose_to_sim(ball2_pose, env_ids)

        zeros6 = torch.zeros((len(env_ids), 6), device=self.device)
        self.ball_1.write_root_velocity_to_sim(zeros6, env_ids)
        self.ball_2.write_root_velocity_to_sim(zeros6, env_ids)


class BaodingOrcaEnv(BaodingMixin, BaodingPalmResetMixin, OrcaEnv):
    """Baoding on the Orca hand."""

    cfg: BaodingOrcaCfg

    def __init__(self, cfg: BaodingOrcaCfg, render_mode: str | None = None, **kwargs):
        apply_baoding_object_cfgs_from_scalars(cfg)
        super().__init__(cfg, render_mode, **kwargs)
        self.palm_idx = self.robot.body_names.index("palm_link")
        self._init_baoding_state()


class BaodingAllegroEnv(BaodingMixin, BaodingPalmResetMixin, AllegroEnv):
    """Baoding on the Allegro hand."""

    cfg: BaodingAllegroCfg

    def __init__(self, cfg: BaodingAllegroCfg, render_mode: str | None = None, **kwargs):
        apply_baoding_object_cfgs_from_scalars(cfg)
        super().__init__(cfg, render_mode, **kwargs)
        self.palm_idx = self.robot.body_names.index("palm_link")
        self._init_baoding_state()


@torch.jit.script
def distance_reward(object_ee_distance, std: float = 0.1):
    r_reach = 1 - torch.tanh(object_ee_distance / std)
    return r_reach


@torch.jit.script
def compute_rewards(
    goal_reached: torch.Tensor,
    ball_1_goal_dist: torch.Tensor,
    ball_2_goal_dist: torch.Tensor,
):
    dense_dist_reward = (distance_reward(ball_1_goal_dist) + distance_reward(ball_2_goal_dist)) * 0.1
    reach_goal_reward = torch.where(goal_reached == 1, 10, 0).float()
    total_reward = reach_goal_reward + dense_dist_reward
    return total_reward, reach_goal_reward

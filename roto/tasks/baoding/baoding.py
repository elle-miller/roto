# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Central baoding task: shared logic."""

from __future__ import annotations

import inspect
from collections.abc import Sequence
from pathlib import Path

import torch

import isaaclab.envs.mdp as mdp
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, RigidObject, RigidObjectCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.sim.schemas.schemas_cfg import CollisionPropertiesCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import sample_uniform

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
from roto.tasks.robots.shadowlite.shadowlite import ShadowLiteEnv, ShadowLiteEnvCfg

_BAODING_HDR = Path(__file__).resolve().parent.parent.parent / "assets/rooms/stierberg_sunrise_4k.hdr"
_BAODING_HDR = Path(__file__).resolve().parent.parent.parent / "assets/rooms/qwantani_dusk_2_4k.hdr"
# _BAODING_HDR = Path(__file__).resolve().parent.parent.parent / "assets/rooms/kloppenheim_02_puresky_4k.hdr"

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
                radius=ball_radius_m * 0.01,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=colour_1),
            ),
        },
    )
    target2_cfg = VisualizationMarkersCfg(
        prim_path="/Visuals/target_2",
        markers={
            "target_2": sim_utils.SphereCfg(
                radius=ball_radius_m * 0.01,
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
class ShadowLiteFrictionEventCfg:
    """Per-segment friction domain randomization, re-sampled every episode reset.

    One event term per finger-segment type (distal / middle / proximal / knuckle);
    each targets a disjoint set of bodies via a body-name regex and gets its own
    friction range. ``num_buckets`` > 1 is required so that the 4096 parallel envs
    see varied friction (with the default of 1, every env would get identical
    friction). ``make_consistent`` enforces sampled dynamic_friction <=
    static_friction.

    Note: palm/forearm are merged into the fixed base during URDF->USD conversion,
    so they are not separate articulation bodies; palm friction is randomized via
    the base body 'world' (which carries the palm collision shapes).
    """

    distal_friction = EventTerm(  # fingertips that grip the balls
        func=mdp.randomize_rigid_body_material,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="rh_.*distal"),
            "static_friction_range": (0.8, 1.5),
            "dynamic_friction_range": (0.7, 1.3),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 250,
            "make_consistent": True,
        },
    )
    middle_friction = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="rh_.*middle"),
            "static_friction_range": (0.7, 1.2),
            "dynamic_friction_range": (0.6, 1.0),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 250,
            "make_consistent": True,
        },
    )
    proximal_friction = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="rh_.*proximal"),
            "static_friction_range": (0.7, 1.2),
            "dynamic_friction_range": (0.6, 1.0),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 250,
            "make_consistent": True,
        },
    )
    knuckle_friction = EventTerm(  # ff/mf/rf knuckles + thumb base
        func=mdp.randomize_rigid_body_material,
        mode="reset",
        params={
            # Note: the palm/forearm links are merged into the fixed base during
            # URDF->USD conversion, so they are not separate articulation bodies
            # and cannot be targeted here. The knuckles + thumb base are the`1`
            # proximal-most randomizable bodies.
            "asset_cfg": SceneEntityCfg("robot", body_names="rh_(.*knuckle|thbase)"),
            "static_friction_range": (0.5, 1.0),
            "dynamic_friction_range": (0.4, 0.9),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 250,
            "make_consistent": True,
        },
    )
    palm_friction = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="reset",
        params={
            # palm/forearm collision is merged onto the fixed base body 'world',
            # so targeting 'world' randomizes the palm contact surface.
            "asset_cfg": SceneEntityCfg("robot", body_names="world"),
            "static_friction_range": (0.3, 0.8),
            "dynamic_friction_range": (0.3, 0.8),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 250,
            "make_consistent": True,
        },
    )


@configclass
class BaodingShadowLiteCfg(BaodingTaskCfg, ShadowLiteEnvCfg):
    """Baoding on the ShadowLite hand."""

    events: ShadowLiteFrictionEventCfg = ShadowLiteFrictionEventCfg()

    # Ball friction DR: one value sampled per env each reset, applied to BOTH balls.
    ball_friction_range: tuple[float, float] = (0.2, 0.6)

    ball_reset_height = 0.46

    ball_mass_g = 55
    success_tolerance = 0.013

    # ball size
    ball_diameter_inches = 1.5
    ball_radius_m = (ball_diameter_inches / 2) * 2.54 / 100
    ball_diameter_m = ball_radius_m * 2

    # initial ball positions
    ball_1_init_x = -0.03
    ball_1_init_y = -.225
    ball_2_init_x = 0.01
    ball_2_init_y = -0.255

    # target positions
    palm_target_x = 0
    palm_target_y = -0.25
    palm_target_z = 0.41

    # for 40 degree tilt forward
    # ball_1_init_x = -0.03
    # ball_1_init_y = -.17
    # ball_2_init_x = -0.01
    # ball_2_init_y = -0.19

    # # target positions
    # palm_target_x = 0
    # palm_target_y = -0.19
    # palm_target_z = 0.28

 


    target_offset = ball_diameter_m / 1.73205080757 + 0.001
    diagonal_target_x = palm_target_x - target_offset
    diagonal_target_y = palm_target_y + target_offset
    diagonal_target_z = palm_target_z + target_offset


@configclass
class BaodingOrcaCfg(BaodingTaskCfg, OrcaEnvCfg):
    """Baoding on the Orca hand."""

    ball_reset_height = 0.6

    # ball size
    ball_diameter_inches = 1.5
    ball_radius_m = (ball_diameter_inches / 2) * 2.54 / 100
    ball_diameter_m = ball_radius_m * 2

    # initial ball positions
    ball_1_init_x = 0.21
    ball_1_init_y = 0.08
    ball_2_init_x = 0.26
    ball_2_init_y = 0.08

    # target positions 
    palm_target_x = 0.25
    palm_target_y = 0.07
    palm_target_z = 0.5

    target_offset = ball_diameter_m / 1.73205080757 + 0.001
    diagonal_target_x = palm_target_x - target_offset
    diagonal_target_y = palm_target_y + target_offset
    diagonal_target_z = palm_target_z + target_offset


@configclass
class BaodingAllegroCfg(BaodingTaskCfg, AllegroEnvCfg):
    """Baoding on the Allegro hand."""

    initial_root_rot = ALLEGRO_BAODING_ROOT_ROT_WXYZ
    robot_cfg: ArticulationCfg = build_allegro_robot_cfg(
        initial_root_rot=initial_root_rot,
        hand_height=ALLEGRO_HAND_HEIGHT_M,
        default_joint_pos=ALLEGRO_DEFAULT_JOINT_POS,
    )

    # initial ball positions
    # ball_1_init_x = 0.14
    # ball_1_init_y = 0.0
    # ball_2_init_x = 0.19
    # ball_2_init_y = 0.0

    ball_1_init_x = 0.14
    ball_1_init_y = -0.03
    ball_2_init_x = 0.14
    ball_2_init_y = 0.03
    ball_reset_height = 0.50
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
    """Two-ball baoding task logic."""

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
            color=(0.81,0.86,1.28),
            intensity=1000.0,
            texture_file=str(_BAODING_HDR),
            texture_format="latlong",
        )
        light.func("/World/bglight", light)

        # light = sim_utils.SphereLightCfg(
        #     intensity=1000.0,
        #     color=(1.0, 1.0, 1.0),
        # )
        # light.func("/World/spotlight_1", light, translation=(0.4, -0.4, 1.1))

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
        ball_1_fall = self.ball_1_pos[:, 2] < 0.2
        ball_2_fall = self.ball_2_pos[:, 2] < 0.2 # changed from 0.3 to 0.2 for shadowlite 40 degree since the reset height is lower and we don't want episodes to terminate immediately after reset 
        termination = out_of_reach | ball_1_fall | ball_2_fall
        time_out = self.episode_length_buf >= self.max_episode_length - 1

        return termination, time_out

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES

        super()._reset_idx(env_ids)

        self._reset_object(env_ids)

        self.num_rotations[env_ids] = 0

    def _reset_object(self, env_ids: Sequence[int]) -> None:
        self._baoding_reset_balls(env_ids)
        if getattr(self.cfg, "ball_friction_range", None) is not None:
            self._randomize_ball_friction(env_ids)

    def _randomize_ball_friction(self, env_ids: Sequence[int]) -> None:
        """Sample one friction value per env and apply it to BOTH balls (same each reset).

        randomize_rigid_body_material can't express "same value across two separate
        RigidObjects", so this writes the ball material buffers directly, mirroring
        that function's buffer handling.
        """
        lo, hi = self.cfg.ball_friction_range
        eids = env_ids.cpu()
        mu = sample_uniform(lo, hi, (len(eids), 1, 1), device="cpu")  # [n_env, 1 shape, 1]
        for ball in (self.ball_1, self.ball_2):
            materials = ball.root_physx_view.get_material_properties()  # [N, 1, 3] on CPU
            materials[eids, :, 0:1] = mu   # static friction
            materials[eids, :, 1:2] = mu   # dynamic friction (= static)
            materials[eids, :, 2:3] = 0.0  # restitution
            ball.root_physx_view.set_material_properties(materials, eids)

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

    def _baoding_reset_balls(self, env_ids: Sequence[int]) -> None:
        ball_1_default_state = self.ball_1.data.default_root_state.clone()[env_ids]
        ball_2_default_state = self.ball_2.data.default_root_state.clone()[env_ids]
        pos_noise = sample_uniform(-0.005, 0.005, (len(env_ids), 3), device=self.device)# - > added noise 
        #pos_noise = sample_uniform(0.0, 0.0, (len(env_ids), 3), device=self.device)

        ball_1_default_state[:, 0:3] = ball_1_default_state[:, 0:3] + pos_noise + self.scene.env_origins[env_ids]
        ball_1_default_state[:, 7:] = torch.zeros_like(self.ball_1.data.default_root_state[env_ids, 7:])

        ball_2_default_state[:, 0:3] = ball_2_default_state[:, 0:3] + pos_noise + self.scene.env_origins[env_ids]
        ball_2_default_state[:, 7:] = torch.zeros_like(self.ball_2.data.default_root_state[env_ids, 7:])

        self.ball_1.write_root_pose_to_sim(ball_1_default_state[:, :7], env_ids)
        self.ball_1.write_root_velocity_to_sim(ball_1_default_state[:, 7:], env_ids)
        self.ball_2.write_root_pose_to_sim(ball_2_default_state[:, :7], env_ids)
        self.ball_2.write_root_velocity_to_sim(ball_2_default_state[:, 7:], env_ids)


class BaodingShadowEnv(BaodingMixin, ShadowEnv):
    """Baoding on the Shadow hand."""

    cfg: BaodingCfg

    def __init__(self, cfg: BaodingCfg, render_mode: str | None = None, **kwargs):
        apply_baoding_object_cfgs_from_scalars(cfg)
        super().__init__(cfg, render_mode, **kwargs)
        self._init_baoding_state()


class BaodingShadowLiteEnv(BaodingMixin, ShadowLiteEnv):
    """Baoding on the Shadow Lite hand."""

    cfg: BaodingShadowLiteCfg

    def __init__(self, cfg: BaodingShadowLiteCfg, render_mode: str | None = None, **kwargs):
        apply_baoding_object_cfgs_from_scalars(cfg)
        super().__init__(cfg, render_mode, **kwargs)

        # Friction DR uses class-based event terms (randomize_rigid_body_material,
        # a ManagerTermBase). Isaac Lab normally instantiates those classes lazily
        # via a timeline PLAY-event callback, but that callback does not fire in
        # this DirectRLEnv construction path, leaving the terms as raw classes — so
        # the reset-mode apply() ends up calling the class __init__ and raises
        # "unexpected keyword argument 'asset_cfg'". Instantiate them now that the
        # sim is playing. We only resolve a SceneEntityCfg if it is still
        # unresolved: the framework already resolves body_names -> body_ids (but
        # leaves body_names as the regex), so a second resolve would raise a
        # spurious "body_names and body_ids inconsistent" error.
        em = getattr(self, "event_manager", None)
        if em is not None and self.sim.is_playing():
            for term_cfgs in em._mode_term_cfgs.values():
                for term_cfg in term_cfgs:
                    for value in term_cfg.params.values():
                        if (
                            isinstance(value, SceneEntityCfg)
                            and value.body_names is not None
                            and value.body_ids == slice(None)
                        ):
                            value.resolve(self.scene)
                    if inspect.isclass(term_cfg.func):
                        term_cfg.func = term_cfg.func(cfg=term_cfg, env=self)
            # Mark resolved so a later timeline PLAY callback does not re-resolve
            # the (now resolved) SceneEntityCfgs and raise the inconsistency error.
            em._is_scene_entities_resolved = True

        self._init_baoding_state()


class BaodingOrcaEnv(BaodingMixin, OrcaEnv):
    """Baoding on the Orca hand."""

    cfg: BaodingOrcaCfg

    def __init__(self, cfg: BaodingOrcaCfg, render_mode: str | None = None, **kwargs):
        apply_baoding_object_cfgs_from_scalars(cfg)
        super().__init__(cfg, render_mode, **kwargs)
        self._init_baoding_state()


class BaodingAllegroEnv(BaodingMixin, AllegroEnv):
    """Baoding on the Allegro hand."""

    cfg: BaodingAllegroCfg

    def __init__(self, cfg: BaodingAllegroCfg, render_mode: str | None = None, **kwargs):
        apply_baoding_object_cfgs_from_scalars(cfg)
        super().__init__(cfg, render_mode, **kwargs)
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

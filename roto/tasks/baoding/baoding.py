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
from roto.tasks.robots.shadowlite.shadowlite import (
    ShadowLiteEnv,
    ShadowLiteEnvCfg,
    ShadowLitePadTacEnv,
    ShadowLitePadTacEnvCfg,
    ShadowLitePadTacBTEnv,
    ShadowLitePadTacBTEnvCfg,
)

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
    # Steps at episode start where the hand holds its cradle pose so balls can
    # settle from their drop height into the palm before the policy takes control.
    # 0 = disabled (backward-compatible). ~15 steps ≈ 0.25 s at 60 Hz.
    settle_steps: int = 15
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

    # Ball mass DR (kg): one value sampled per env each reset, applied to BOTH balls.
    ball_mass_range: tuple[float, float] = (0.045, 0.100)

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
class BaodingShadowLitePadTacCfg(BaodingTaskCfg, ShadowLitePadTacEnvCfg):
    """Baoding on Shadow Lite with FSR pad tactile (TouchLab + shadow_padtac.usd)."""

    events: ShadowLiteFrictionEventCfg = ShadowLiteFrictionEventCfg()
    ball_friction_range: tuple[float, float] = (0.2, 0.6)
    ball_mass_range: tuple[float, float] = (0.045, 0.100)

    ball_reset_height = 0.46
    ball_mass_g = 55
    success_tolerance = 0.013

    ball_diameter_inches = 1.5
    ball_radius_m = (ball_diameter_inches / 2) * 2.54 / 100
    ball_diameter_m = ball_radius_m * 2

    ball_1_init_x = -0.03
    ball_1_init_y = -.225
    ball_2_init_x = 0.01
    ball_2_init_y = -0.255

    palm_target_x = 0
    palm_target_y = -0.25
    palm_target_z = 0.41

    target_offset = ball_diameter_m / 1.73205080757 + 0.001
    diagonal_target_x = palm_target_x - target_offset
    diagonal_target_y = palm_target_y + target_offset
    diagonal_target_z = palm_target_z + target_offset

    # trial15/27 (the checkpoints this hand is evaluated against) were trained
    # under this coupling law; the base ShadowLiteEnvCfg keeps the sysid-derived
    # 0.875 for the plain "shadowlite" hand.
    coupling_theta: float = 0.785


@configclass
class BaodingShadowLitePadTacBTCfg(BaodingShadowLitePadTacCfg):
    """Baoding on Shadow Lite with 12 FSR pads + 4 BioTac fingertips.

    Identical task/ball setup to BaodingShadowLitePadTacCfg; only the robot asset
    (shadow_padtac_biotac.usd) and the contact-sensor prim path change so the 4
    BioTac distal tips are sensed alongside the 12 pads.

    Command slew (HW SPEED_FRAC) is opt-in via ShadowLiteEnvCfg:
      cmd_speed_frac=0.5                    # fixed
      cmd_speed_frac_range=(0.3, 1.0)        # DR (leave cmd_speed_frac=None)
    Default both None = classic Trial-15/27 plant (no slew).

    FSR taxel DR (opt-in via tactile_fsr_corrupt_max):
      each episode select k~U{0..max} of the 12 FSR channels and give each a
      forced value of 0 or 1 (mixed); BioTac distal channels untouched.
      tactile_flip_prob_* then dithers ONLY those k channels, so a broken taxel
      is intermittent (~90/10) rather than locked for the episode. Every
      unselected FSR pad and all 4 BioTac channels stay exact. None = off.
    """

    robot_cfg: ArticulationCfg = (
        ShadowLitePadTacBTEnvCfg.__dataclass_fields__["robot_cfg"].default_factory()
    )
    robot_contact_sensor_cfg = (
        ShadowLitePadTacBTEnvCfg.__dataclass_fields__["robot_contact_sensor_cfg"].default_factory()
    )

    # Robust overnight scratch stack (toggle off individually for classic runs):
    cmd_speed_frac_range = None #(0.3, 1.0)
    tactile_fsr_corrupt_max = 8
    # Per-step dither applied ONLY to the k FSR channels selected above. A
    # channel forced to 1 reads 1 on ~90% of steps and drops to 0 on ~10%; a
    # channel forced to 0 reads 0 on ~90% and blips to 1 on ~10%. Unselected FSR
    # pads and all 4 BioTac tips are never noised and stay exact.
    tactile_flip_prob_off_to_on = 0.1
    tactile_flip_prob_on_to_off = 0.1


@configclass
class BaodingShadowLitePadTacBTSparseCfg(BaodingShadowLitePadTacBTCfg):
    """Sparser corrupt draw (k <= 6) with a two-rate per-step flip over all 12 FSR
    pads: the stuck subset dithers lightly around its forced value while the
    remaining pads carry heavier noise on the true contact signal. See
    ShadowLiteEnvCfg.tactile_flip_scope for the mechanism difference from
    BaodingShadowLitePadTacBTCfg's default single-rate DR.
    """

    tactile_fsr_corrupt_max = 6
    # Two-rate FSR dither (tactile_flip_scope="both"): every FSR pad is eligible,
    # but the k stuck channels and the remaining (12 - k) get independent rates.
    #   stuck k     -> 0.02 flip, i.e. a broken taxel holds its forced value on
    #                  ~98% of control steps and blips to the opposite on ~2%.
    #   other 12-k  -> 0.15 flip both directions on the true contact signal.
    # BioTac distal channels are never touched under either rate.
    tactile_flip_prob_off_to_on = 0.02
    tactile_flip_prob_on_to_off = 0.02
    tactile_flip_prob_unsel_off_to_on = 0.15
    tactile_flip_prob_unsel_on_to_off = 0.15
    tactile_flip_scope = "both"
    # Explicitly off, matching BaodingShadowLitePadTacBTCfg's current default.
    cmd_speed_frac_range = None


@configclass
class BaodingShadowLitePadTacBTStuck8Cfg(BaodingShadowLitePadTacBTCfg):
    """Stuck-taxel-only FSR DR: a wider corrupt draw (k <= 8) whose stuck channels
    dither lightly around their forced value, with every other FSR pad passing the
    true contact signal through untouched.

    Contrast with BaodingShadowLitePadTacBTSparseCfg, which dithers all 12 pads at
    two rates. Here the (12 - k) unselected pads are exact by construction.
    """

    # Wider draw than the sparse profile: k ~ Uniform{0..8} of the 12 FSR pads.
    tactile_fsr_corrupt_max = 8

    # Only the k stuck channels dither: 0.05 flip means a broken taxel holds its
    # forced value on ~95% of control steps and blips to the opposite on ~5%.
    tactile_flip_prob_off_to_on = 0.05
    tactile_flip_prob_on_to_off = 0.05
    # scope="corrupted" confines the dither to exactly those k channels, so the
    # other (12 - k) FSR pads and all 4 BioTac channels carry the exact contact
    # signal. The unsel rates below are inert under this scope; pinned at 0.0 so
    # the intent survives a later scope change.
    tactile_flip_prob_unsel_off_to_on = 0.0
    tactile_flip_prob_unsel_on_to_off = 0.0
    tactile_flip_scope = "corrupted"

    # Coupled J1 joints follow the coupling law rather than being pinned at 0.
    # Stated explicitly so this profile is reproducible regardless of the base
    # ShadowLiteEnvCfg default.
    lock_coupled_dependent_at_zero = False

    # Explicitly off, matching BaodingShadowLitePadTacBTCfg's current default.
    cmd_speed_frac_range = None


@configclass
class BaodingShadowLitePadTacBTLegacyCfg(BaodingShadowLitePadTacBTCfg):
    """The hardware-validated profile (~55 rotations on the real hand).

    Ball friction DR and ball mass DR both at the BaodingShadowLitePadTacCfg
    defaults, command slew DR on over (0.3, 1.0), and a pure stuck-taxel FSR
    corruption: each episode k ~ U{0..6} of the 12 pads are forced to 0 or 1 and
    HOLD that value for the whole episode, while the other (12 - k) pads and all
    4 BioTac channels pass the exact contact signal.

    Differs from BaodingShadowLitePadTacBTCfg in the corrupt draw (6 rather than
    8), in dropping the per-step dither entirely, and in enabling slew DR.
    """

    # Narrower corrupt draw than the k<=8 base: k ~ Uniform{0..6} of the 12 pads.
    tactile_fsr_corrupt_max = 6
    # No per-step dither: a stuck channel holds its forced 0/1 for the entire
    # episode rather than flickering (the k<=8 base uses 0.1 both directions).
    # With both rates at 0 the scope below is inert, but it is pinned so the
    # intent survives a later default change.
    tactile_flip_prob_off_to_on = 0.0
    tactile_flip_prob_on_to_off = 0.0
    tactile_flip_prob_unsel_off_to_on = 0.0
    tactile_flip_prob_unsel_on_to_off = 0.0
    tactile_flip_scope = "corrupted"

    # HW SPEED_FRAC slew, randomized per episode (leave cmd_speed_frac None).
    cmd_speed_frac = None
    cmd_speed_frac_range = (0.3, 1.0)

    # Coupled J1 joints follow the coupling law rather than being pinned at 0.
    # Stated explicitly so this profile is reproducible regardless of the base
    # ShadowLiteEnvCfg default.
    lock_coupled_dependent_at_zero = False


@configclass
class BaodingShadowLitePadTacBTLegacyNoSlewCfg(BaodingShadowLitePadTacBTLegacyCfg):
    """Legacy profile with the command-rate limiter removed (no slew DR).

    Ball friction DR, ball mass DR and the k<=6 FSR corruption are unchanged.
    """

    cmd_speed_frac = None
    cmd_speed_frac_range = None


@configclass
class BaodingShadowLitePadTacBTLegacyNoMassDRCfg(BaodingShadowLitePadTacBTLegacyCfg):
    """Legacy profile with ball-mass DR removed (fixed ball mass).

    ``ball_mass_range = None`` gates off _randomize_ball_mass entirely, so both
    balls keep the asset's own mass every episode. Friction DR, slew DR over
    (0.3, 1.0) and the k<=6 FSR corruption are unchanged.
    """

    ball_mass_range = None


@configclass
class BaodingShadowLitePadTacBTLegacyNoTacCorruptCfg(BaodingShadowLitePadTacBTLegacyCfg):
    """Legacy profile with FSR corruption removed entirely (perfect tactile).

    ``tactile_fsr_corrupt_max = None`` skips the stuck-taxel draw altogether
    (see _init_tactile_fsr_corrupt), so every one of the 12 FSR pads passes the
    exact contact signal every step -- no channel is ever forced to a constant
    0/1. BioTac channels were never touched by corruption anyway. Friction DR,
    ball-mass DR and slew DR over (0.3, 1.0) are unchanged from the legacy base.
    """

    tactile_fsr_corrupt_max = None


@configclass
class BaodingShadowLitePadTacBTLegacyFrictionMassOnlyCfg(BaodingShadowLitePadTacBTLegacyCfg):
    """Legacy profile with ONLY friction DR and mass DR active.

    Combines BaodingShadowLitePadTacBTLegacyNoSlewCfg (``cmd_speed_frac_range =
    None``) and BaodingShadowLitePadTacBTLegacyNoTacCorruptCfg
    (``tactile_fsr_corrupt_max = None``) into one profile: no command-rate
    slew DR, no FSR taxel corruption (perfect tactile), no per-step flip (the
    legacy base already pins ``tactile_flip_prob_*`` at 0.0). Only ball
    friction DR and ball mass DR, both at the BaodingShadowLitePadTacCfg
    defaults inherited unchanged, remain active.

    Replaces the legacy_notac run (which still had slew DR on) -- see
    checkpoint_snapshots/legacy_notac_snapshot_*.pt for that run's last
    checkpoint before it was stopped.
    """

    cmd_speed_frac_range = None
    tactile_fsr_corrupt_max = None


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

        # Per-env countdown used by roto_env._pre_physics_step to hold the catch
        # pose while balls settle. Set to cfg.settle_steps on every reset.
        self.settle_counter = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

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
        physics_termination = out_of_reach | ball_1_fall | ball_2_fall
        # Suppress termination while the settle countdown is active so a ball that
        # hasn't landed yet can't trigger an immediate episode reset.
        settling = getattr(self, "settle_counter", None)
        if settling is not None:
            physics_termination = physics_termination & (settling == 0)
        termination = physics_termination
        time_out = self.episode_length_buf >= self.max_episode_length - 1

        return termination, time_out

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES

        super()._reset_idx(env_ids)

        self._reset_object(env_ids)

        self.num_rotations[env_ids] = 0

        # Restart settle countdown so the hand holds its pose while balls drop.
        self.settle_counter[env_ids] = getattr(self.cfg, "settle_steps", 0)

    def _reset_object(self, env_ids: Sequence[int]) -> None:
        self._baoding_reset_balls(env_ids)
        if getattr(self.cfg, "ball_friction_range", None) is not None:
            self._randomize_ball_friction(env_ids)
        if getattr(self.cfg, "ball_mass_range", None) is not None:
            self._randomize_ball_mass(env_ids)

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

    def _randomize_ball_mass(self, env_ids: Sequence[int]) -> None:
        """Sample one mass (kg) per env and apply it to BOTH balls (same each reset).

        Mirrors _randomize_ball_friction: mdp.randomize_rigid_body_mass samples
        independently per asset_cfg call, so it can't express "same value across two
        separate RigidObjects" either. Writes the mass buffer directly and rescales
        inertia by the mass ratio (uniform-density sphere, radius unchanged), matching
        the recompute done inside mdp.randomize_rigid_body_mass.
        """
        lo, hi = self.cfg.ball_mass_range
        eids = env_ids.cpu()
        mass = sample_uniform(lo, hi, (len(eids), 1), device="cpu")  # [n_env, 1 body]
        for ball in (self.ball_1, self.ball_2):
            masses = ball.root_physx_view.get_masses()  # [N, 1] on CPU
            ratios = mass / masses[eids]
            masses[eids] = mass
            ball.root_physx_view.set_masses(masses, eids)

            inertias = ball.root_physx_view.get_inertias()  # [N, 9] on CPU
            inertias[eids] = inertias[eids] * ratios
            ball.root_physx_view.set_inertias(inertias, eids)

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


class BaodingShadowLitePadTacEnv(BaodingMixin, ShadowLitePadTacEnv):
    """Baoding on Shadow Lite with FSR pad tactile."""

    cfg: BaodingShadowLitePadTacCfg

    def __init__(self, cfg: BaodingShadowLitePadTacCfg, render_mode: str | None = None, **kwargs):
        apply_baoding_object_cfgs_from_scalars(cfg)
        super().__init__(cfg, render_mode, **kwargs)

        # Same friction-event fix as BaodingShadowLiteEnv
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
            em._is_scene_entities_resolved = True

        self._init_baoding_state()


class BaodingShadowLitePadTacBTEnv(BaodingMixin, ShadowLitePadTacBTEnv):
    """Baoding on Shadow Lite with 12 FSR pads + 4 BioTac fingertips."""

    cfg: BaodingShadowLitePadTacBTCfg

    def __init__(self, cfg: BaodingShadowLitePadTacBTCfg, render_mode: str | None = None, **kwargs):
        apply_baoding_object_cfgs_from_scalars(cfg)
        super().__init__(cfg, render_mode, **kwargs)

        # Same friction-event fix as BaodingShadowLitePadTacEnv
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

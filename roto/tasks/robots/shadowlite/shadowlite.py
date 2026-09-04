# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Author: Elle Miller 2025

Shadow-hand base environment utilities shared across RoTO tasks.
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
from isaaclab.utils.math import quat_conjugate, quat_from_angle_axis, quat_mul, sample_uniform

from roto.assets.shadow_hand_lite import (
    SHADOW_HAND_LITE_CFG,
    SHADOW_HAND_LITE_PADTAC_CFG,
    SHADOW_HAND_LITE_PADTAC_BT_CFG,
)
from roto.tasks.roto_env import RotoEnv, RotoEnvCfg

from isaaclab.markers.config import FRAME_MARKER_CFG  # isort: skip

NUM_TACTILE_CHANNELS = 24

PAD_LINK_TO_CHANNEL: dict[str, int] = {
    "rh_fsr_pad_C00": 10,  # thprox
    "rh_fsr_pad_C01": 7,   # ffprox
    "rh_fsr_pad_C02": 4,   # mfknuckle
    "rh_fsr_pad_C03": 9,   # rfprox
    "rh_fsr_pad_C04": 5,   # rfknuckle
    "rh_fsr_pad_C05": 2,   # palm
    "rh_fsr_pad_C06": 11,  # ffmid
    "rh_fsr_pad_C07": 3,   # ffknuckle
    "rh_fsr_pad_C08": 8,   # mfprox
    "rh_fsr_pad_C09": 18,  # thmiddle
    "rh_fsr_pad_C10": 12,  # mfmid
    "rh_fsr_pad_C11": 13,  # rfmid
}

# 16-sensor map: the 12 FSR pads above + the 4 BioTac fingertips. The BioTac tips
# are sensed on the distal links themselves (their tip mesh is the BioTac SP), and
# scatter into the same distal channels the 24-link policy used
# (15/16/17/22 == ffdist/mfdist/rfdist/thdist; see deploy_policy 24-channel order).
PAD_BT_LINK_TO_CHANNEL: dict[str, int] = {
    **PAD_LINK_TO_CHANNEL,
    "rh_ffdistal": 15,
    "rh_mfdistal": 16,
    "rh_rfdistal": 17,
    "rh_thdistal": 22,
}


@configclass
class ShadowLiteEnvCfg(RotoEnvCfg):
    """Default configuration for the Shadow hand."""

    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=4096, env_spacing=0.7, replicate_physics=True
    )

    # eye = (0, 0, 2)
    # lookat = (0, 0.3, 0.5)
    eye = (-1, -1.8, 0.7)
    lookat = (0.5, 0.4, 0.7)
    viewer: ViewerCfg = ViewerCfg(eye=eye, lookat=lookat, resolution=(1920, 1080))

    episode_length_s = 10.0

    reset_joint_pos_noise = 0.1
    reset_joint_vel_noise = 0.0

    # --- Policy I/O noise (sim-to-real). Per-step i.i.d. Gaussian; std is in the
    # native units of each signal. 0.0 disables (old cfgs stay noise-free). ---
    # Actuator noise: added to the raw [-1,1] action before scaling to joint cmds,
    # then stored into self.actions so the policy also sees it in its obs history.
    action_noise_std: float = 0.0
    # Proprioception (sensor) noise, per prop sub-vector:
    obs_noise_std_joint_pos: float = 0.0        # on normalised_joint_pos (~[-1,1])
    obs_noise_std_joint_vel: float = 0.0        # on normalised_joint_vel (~[-1,1])
    obs_noise_std_joint_pos_error: float = 0.0  # on joint_pos_error (rad, ~0.6 deg)

    tacsl_contact_expr: str | None = "{ENV_REGEX_NS}/ball1"
    """Prim path expression for the TacSL contact object.
    Set to None to disable TacSL and fall back to ContactSensor (e.g. --no_ball mode).
    """

    hand_height = 0.5
    # Stock Shadow Lite (PST caps) from shadow_hand_lite.py.
    robot_cfg: ArticulationCfg = SHADOW_HAND_LITE_CFG.replace(prim_path="/World/envs/env_.*/Robot").replace(
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, hand_height),
            #rot=(0.0, 0.0, -0.7071, 0.7071),
            #rot=(-0.7071, 0, 0.0, 0.7071), #upright pos 
            #rot=(0.0, 0.0, -0.7373, 0.6756),
            rot=(0.0, 0.0, -0.7933, 0.6087), #15 degree tilt forward facing up
            joint_pos={
                # ── Knuckle abduction (J4) — fan FF/RF away from the middle finger.
                #    FFJ4 axis is (0,-1,0) but RFJ4 axis is (0,1,0) (mirrored), so the
                #    SAME numeric sign rotates them in opposite world directions = spread.
                #    Limit is ±0.349 rad (±20°); flip both signs if they converge instead.
                "rh_FFJ4":  -0.349,   # index fans out (toward thumb side)
                "rh_RFJ4":  -0.349,   # ring fans out (toward little-finger side)
                # ── Finger curl — open a bit for a larger ball ───────────────────
                "rh_FFJ3":  0.65,    # MCP ~37°
                "rh_FFJ2":  0.87,    # PIP ~50°
                "rh_MFJ3":  0.65,    # MCP ~37°
                "rh_MFJ2":  0.87,    # PIP ~50°
                "rh_RFJ3":  0.65,    # MCP ~37°
                "rh_RFJ2":  0.87,    # PIP ~50°
                # ── Thumb (TH) — back off so it sits a bit more open, not tucked ─
                "rh_THJ5":  0.4,     # rotate thumb inward ~23°
                "rh_THJ4":  0.5,     # abduct across palm ~29°
                "rh_THJ2":  0.35,    # flex ~20°
                # ── Locked coupled-dependent joints — pinned at 0, see
                #    lock_coupled_dependent_at_zero below ─────────────────────────
                "rh_FFJ1":  0.0,
                "rh_MFJ1":  0.0,
                "rh_RFJ1":  0.0,
            },

        #     joint_pos = {
        #     # ── Index finger (FF) — EXTENDED and spread outward ──────────────────
        #     "rh_FFJ4": -0.33,   # abduct index away from middle (toward thumb side)
        #     "rh_FFJ3":  0.0,    # MCP straight
        #     "rh_FFJ2":  0.0,    # PIP straight
        #     "rh_FFJ1":  0.0,    # DIP straight (coupled)

        #     # ── Middle finger (MF) — EXTENDED and spread outward ─────────────────
        #     "rh_MFJ4":  0.33,   # abduct middle away from index
        #     "rh_MFJ3":  0.0,
        #     "rh_MFJ2":  0.0,
        #     "rh_MFJ1":  0.0,

        #     # ── Ring finger (RF) — CURLED ─────────────────────────────────────────
        #     "rh_RFJ4":  0.0,    # no abduction
        #     "rh_RFJ3":  1.55,    # MCP curl
        #     "rh_RFJ2":  1.55,    # PIP curl
        #     "rh_RFJ1":  1.2,    # DIP curl (coupled, will follow J2)

        #     # ── Thumb (TH) — tucked toward palm center ────────────────────────────
        #     "rh_THJ5": 0.8,    # rotate thumb inward
        #     "rh_THJ4":  1.22,    # abduct thumb across palm
        #     "rh_THJ2":  0.,    # slight flex
        #     "rh_THJ1":  0.,    # distal curl
        # }
        )
    )

    #+++++++++++++++++++++++++++++++++++++++++++++++++++++Baoding-specific overrides+++++++++++++++++++++++++++++++++++++++++++++++++++++
    # tilting (-15 degree) forward. # finalized angle for hand rot=(0.0, 0.0, -0.7933, 0.6087)
    # ball_mass_g = 20
    # ball_reset_height = 0.46

    # # ball size
    # ball_diameter_inches = 1.1
    # ball_radius_m = (ball_diameter_inches / 2) * 2.54 / 100
    # ball_diameter_m = ball_radius_m * 2

    # # initial ball positions
    # ball_1_init_x = -0.03
    # ball_1_init_y = -.2
    # ball_2_init_x = 0.01
    # ball_2_init_y = -0.22

    # # target positions
    # palm_target_x = 0
    # palm_target_y = -0.25
    # palm_target_z = 0.39

    # target_offset = ball_diameter_m / 1.73205080757 + 0.001
    # diagonal_target_x = palm_target_x - target_offset
    # diagonal_target_y = palm_target_y + target_offset
    # diagonal_target_z = palm_target_z + target_offset
    #=========================================BOUNCE SHADOWLITE =====================================================
    # tilting (-15 degree) forward. # finalized angle for hand rot=(0.0, 0.0, -0.7933, 0.6087)
    # fall_height = 0.3          
    # object_y_pos = -0.28    
    # object_z_pos = 0.6
    # default_object_pos = (0., -0.265, 0.6)  # is this affecting the ball position at all? cuz this is not changing anything in the viewer
    # object_cfg: RigidObjectCfg = _make_bouncy_ball_cfg((0., -0.265, 0.6)  )

    control_joint_names = [
    "rh_FFJ4", "rh_MFJ4", "rh_RFJ4", "rh_THJ5",   # 0,1,2,3
    "rh_FFJ3", "rh_MFJ3", "rh_RFJ3", "rh_THJ4",   # 4,5,6,7
    "rh_FFJ2", "rh_MFJ2", "rh_RFJ2",              # 8,9,10  ← the J2 drivers
    "rh_THJ2", "rh_THJ1",                          # 11,12
]

    coupled_joint_map = {
        "rh_FFJ1": "rh_FFJ2",
        "rh_MFJ1": "rh_MFJ2",
        "rh_RFJ1": "rh_RFJ2",
    }

    # Hard-lock the coupled dependent joints (FF/MF/RF J1) at 0 rad: their commanded
    # position is always zero regardless of the J2-derived coupling law below. J2's
    # own command/state-machine bookkeeping is untouched, so J2 dynamics stay
    # identical — only J1's actual motion is disabled.
    lock_coupled_dependent_at_zero: bool = False

    # J2 must reach this angle (rad) before J1 starts moving.
    # 0.785 rad = 45°: first half of J2's range drives J2, second half drives J1.
    coupling_theta: float = 0.875

    # Route-2 sequencing: gate the J1 mimic on MEASURED J2 so J1 can't lead its
    # driver. J1's commanded curl is scaled by how close measured J2 is to its
    # limit, ramping over [opens_at, J2_max] where opens_at = J2_max - band.
    # frac=1.0 (strict): J1 only fires when J2 is within couple_gate_j2_tol of its limit.
    # frac<1.0: gate opens earlier, at frac * J2_max (legacy behaviour).
    couple_gate_j1_on_measured: bool = True
    couple_gate_lo_frac: float = 1.0    # strict: J1 only once J2 reaches its limit
    couple_gate_j2_tol: float = 0.035   # rad (~2°) tolerance band at the J2 limit

    # Stateful backlash coupling (supersedes the measured-J2 gate when True). On
    # uncurl J2 unlocks early at a FIXED per-finger angle R (combined ffj0 frame,
    # degrees), J1 unwinds to 0 at 100°, and reversing inside (100°,R) freezes J1
    # until the motor returns to R. See RotoEnv._asymmetric_backlash.
    couple_asymmetric_backward: bool = False
    # R is a constant mechanical property per finger (no per-episode randomization,
    # to match real hardware). Scalar = same R for all 3 fingers, or a per-finger
    # (FF, MF, RF) tuple. R=100 -> no backlash; larger R -> more slop. Set to the
    # measured hardware backlash.
    couple_release_deg: tuple[float, float, float] = (140.0, 125.0, 100.0)
    couple_dir_deadband: float = 0.002   # rad; |Δm| below this latches direction

    # Hand mounting tilt. (lo, hi) equal -> fixed mount (no DR), which is the default:
    # the hand sits at the fixed 15° forward tilt from init_state. Widen to e.g.
    # (0.0, 15.0) to domain-randomize the tilt per episode.
    hand_tilt_range_deg: tuple[float, float] = (15.0, 15.0)

    # Command rate limit (matches HW deploy SPEED_FRAC). Opt-in; default OFF so
    # classic padtac_bt scratch (Trial-15/27) is unchanged.
    #   Fixed:  cmd_speed_frac=0.5, cmd_speed_frac_range=None
    #   DR:     cmd_speed_frac=None, cmd_speed_frac_range=(0.3, 1.0)
    #   Off:    both None (default)
    cmd_speed_frac: float | None = None
    cmd_speed_frac_range: tuple[float, float] | None = None

    # FSR taxel DR (binary 24-d obs). Opt-in; default OFF.
    # Each episode: select k ~ Uniform{0..max} of the 12 FSR channels only
    # (PAD_LINK_TO_CHANNEL); each chosen channel gets a forced value of 0 or 1.
    # The forced value is a baseline, not a hold -- tactile_flip_prob_* below
    # dithers these channels so a broken taxel is intermittent.
    # BioTac distal channels (15/16/17/22) are never touched. None = off.
    tactile_fsr_corrupt_max: int | None = None

    # Per-step dither on the SELECTED FSR channels (binary 24-d obs). Opt-in;
    # default OFF. Requires tactile_fsr_corrupt_max > 0.
    # A selected channel is intermittent, not locked: it reads its forced value
    # on ~(1 - p) of control steps and the opposite on ~p, resampled each step.
    #   forced 1 -> reads 1 with prob (1 - tactile_flip_prob_on_to_off)
    #   forced 0 -> reads 1 with prob tactile_flip_prob_off_to_on
    # Scope is exactly the k selected channels. The other (12 - k) FSR pads, all
    # 4 BioTac channels, and the 8 structurally-empty slots are never touched and
    # carry the exact contact signal. 0.0 = off.
    tactile_flip_prob_off_to_on: float = 0.0
    tactile_flip_prob_on_to_off: float = 0.0

    # Per-step dither on the UNSELECTED FSR channels -- the (12 - k) pads this
    # episode's corrupt draw did NOT stick. Only read when tactile_flip_scope is
    # "both", where the selected k dither at tactile_flip_prob_* above and these
    # unselected pads dither at their own, independent rates. 0.0 = off.
    tactile_flip_prob_unsel_off_to_on: float = 0.0
    tactile_flip_prob_unsel_on_to_off: float = 0.0

    # Which channels the per-step flip DR is eligible on:
    #   "corrupted" (default) -- only the channels this episode's tactile_fsr_corrupt_max
    #     draw selected as stuck (current behaviour, unchanged).
    #   "all_fsr" -- the 12 FSR channels MINUS whichever ones this episode's corrupt draw
    #     selected as stuck: a channel that is stuck this episode is structurally
    #     excluded from the flip mask, so it is never touched by the flip regardless of
    #     step order. Reproduces the original FSR taxel DR (commit 7f9500f) before it was
    #     rescoped to "corrupted".
    #   "both" -- every FSR pad is eligible, but at two independent rates: the k stuck
    #     channels dither at tactile_flip_prob_*, the remaining (12 - k) at
    #     tactile_flip_prob_unsel_*. BioTac channels still pass through exact.
    tactile_flip_scope: str = "corrupted"

    # GRDF coupling (experimental): derive the coupled J1/J2 commands from the
    # phase couplings declared in the GRDF robot file instead of the
    # coupling_theta split above. Same law today, but the coupling lives in the
    # robot description (single source of truth, upgradeable from hardware
    # sweeps without touching env code). Keep False until the Baoding A/B
    # untangling runs finish.
    

    actuated_joint_names = ['rh_FFJ4', 'rh_MFJ4', 'rh_RFJ4', 'rh_THJ5', 'rh_FFJ3', 'rh_MFJ3', 'rh_RFJ3', 'rh_THJ4', 'rh_FFJ2', 'rh_MFJ2', 'rh_RFJ2', 'rh_FFJ1', 'rh_MFJ1', 'rh_RFJ1', 'rh_THJ2', 'rh_THJ1']


    num_actions = len(control_joint_names)

    action_space = num_actions

    marker_cfg = FRAME_MARKER_CFG.copy()
    marker_cfg.markers["frame"].scale = (0.05, 0.05, 0.05)
    marker_cfg.prim_path = "/Visuals/ContactCfg"

    robot_contact_sensor_cfg = ContactSensorCfg(
    prim_path="/World/envs/env_.*/Robot/rh_(ffdistal|mfdistal|rfdistal|thdistal)",
    #prim_path="/World/envs/env_.*/Robot/.*",
    update_period=0.0,
    history_length=1,
)


@configclass
class ShadowLitePadTacEnvCfg(ShadowLiteEnvCfg):
    """Shadow Lite + 12 FSR pad contact sensors only (scatter -> 24-d tactile)."""

    robot_cfg: ArticulationCfg = SHADOW_HAND_LITE_PADTAC_CFG.replace(
        prim_path="/World/envs/env_.*/Robot"
    ).replace(
        init_state=ShadowLiteEnvCfg.__dataclass_fields__["robot_cfg"].default_factory().init_state,
    )

    robot_contact_sensor_cfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/rh_fsr_pad_.*",
        update_period=0.0,
        history_length=1,
    )


@configclass
class ShadowLitePadTacBTEnvCfg(ShadowLiteEnvCfg):
    """Shadow Lite + 12 FSR pads + 4 BioTac fingertips (16 -> scatter to 24-d tactile)."""

    robot_cfg: ArticulationCfg = SHADOW_HAND_LITE_PADTAC_BT_CFG.replace(
        prim_path="/World/envs/env_.*/Robot"
    ).replace(
        init_state=ShadowLiteEnvCfg.__dataclass_fields__["robot_cfg"].default_factory().init_state,
    )

    # 12 pad links + the 4 distal links (whose tip mesh is the BioTac SP).
    robot_contact_sensor_cfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/rh_(fsr_pad_C.*|ffdistal|mfdistal|rfdistal|thdistal)",
        update_period=0.0,
        history_length=1,
    )


class ShadowLiteEnv(RotoEnv):
    """Shadow-hand base env providing tactile + proprio pipelines."""

    cfg: ShadowLiteEnvCfg

    def __init__(self, cfg: ShadowLiteEnvCfg, render_mode: str | None = None, **kwargs):

        super().__init__(cfg, render_mode, **kwargs)

        # Hardware has only 13 actuators; the 3 coupled J1 mimics (FFJ1/MFJ1/RFJ1)
        # are not independently observable. Build proprioception over the 13
        # policy-controlled joints (control_joint_names order) to match deployment.
        self.prop_dof_indices = self.control_dof_indices

        print("NUM TACTILE BODIES:", self.robot_contact_sensor.data.net_forces_w.shape)
        self.num_tactile_observations = 0
        self.tactile = torch.zeros((self.num_envs, 0), device=self.device)
        self.last_tactile = torch.zeros((self.num_envs, 0), device=self.device)



    def _setup_scene(self):
        """Register the Shadow hand, contact sensors, and lighting."""
        super()._setup_scene()

        self.robot = Articulation(self.cfg.robot_cfg)
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())
        self.scene.clone_environments(copy_from_source=False)
        self.scene.articulations["robot"] = self.robot
        self.robot_contact_sensor = ContactSensor(self.cfg.robot_contact_sensor_cfg)
        self.scene.sensors["robot_contact_sensor"] = self.robot_contact_sensor


    def _get_tactile(self):
        """Return binary tactile activation per finger segment.

        Reindexes the single contact sensor to match the legacy ordering:
        [all distal, all proximal, all middle, palm, metacarpal].
        """

        forces = self.robot_contact_sensor.data.net_forces_w[:].clone()  # [N, B, 3]
        norm = torch.linalg.vector_norm(forces, dim=-1)  # [N, B]

        if self.tactile_cfg is not None and self.tactile_cfg.get("binary_tactile", True):
            norm = (norm > self.binary_threshold).float()

        self.last_tactile = self.tactile
        self.tactile = norm
        return norm

    def _reset_idx(self, env_ids: Sequence[int] | None):
        """Reset articulation state and optionally randomize joints.

        Args:
            env_ids: Environment indices to reset. If None, resets all environments.
        """
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES

        # Reset articulation and rigid body attributes
        super()._reset_idx(env_ids)

        # Reset hand with noise
        self._reset_robot(env_ids, joint_pos_noise=self.cfg.reset_joint_pos_noise)

        # Per-episode coupling DR: latch a new backlash unlock angle and clear state.
        if getattr(self, "couple_asymmetric_backward", False):
            self._sample_coupling_params(env_ids)

        # Per-episode hand mounting-tilt DR.
        self._randomize_hand_tilt(env_ids)

        # Per-episode command-rate DR (HW SPEED_FRAC); no-op if slew off / fixed.
        if getattr(self, "use_cmd_slew", False):
            self._sample_cmd_speed_frac(env_ids)

    # 0° "facing up" and 15° forward-tilt root quaternions (w, x, y, z); the tilt DR
    # interpolates between them.
    _Q_TILT_0  = (0.0, 0.0, -0.7071, 0.7071)
    _Q_TILT_15 = (0.0, 0.0, -0.7933, 0.6087)

    def _randomize_hand_tilt(self, env_ids):
        """Write a per-episode randomized forward tilt to the (fixed-base) hand root.

        Samples tilt in cfg.hand_tilt_range_deg and nlerps between the 0° and 15°
        root quaternions (the two are ~15° apart, so a normalized lerp matches slerp
        to <0.1° over this arc). When lo == hi (default) there is no DR, so we skip
        the root-pose write entirely and leave the hand at its fixed init_state tilt.
        """
        lo, hi = self.cfg.hand_tilt_range_deg
        if lo == hi:
            return                              # no DR: keep the fixed init mount
        n = len(env_ids)
        q0  = torch.tensor(self._Q_TILT_0,  device=self.device)
        q15 = torch.tensor(self._Q_TILT_15, device=self.device)
        # endpoints are 15° apart -> t = tilt_deg / 15 maps tilt to the interpolation
        t = sample_uniform(lo / 15.0, hi / 15.0, (n, 1), self.device)
        q = (1.0 - t) * q0 + t * q15
        q = q / torch.linalg.vector_norm(q, dim=-1, keepdim=True)

        root_pose = self.robot.data.default_root_state[env_ids, :7].clone()
        root_pose[:, 0:3] = root_pose[:, 0:3] + self.scene.env_origins[env_ids]
        root_pose[:, 3:7] = q
        self.robot.write_root_pose_to_sim(root_pose, env_ids)

class ShadowLitePadTacEnv(ShadowLiteEnv):
    """Tactile from FSR pad links only -> 24-d deploy vector."""

    cfg: ShadowLitePadTacEnvCfg

    # Which contact-sensor bodies feed which of the 24 tactile channels. Subclasses
    # (e.g. the BioTac variant) override this to add fingertip channels.
    LINK_TO_CHANNEL: dict[str, int] = PAD_LINK_TO_CHANNEL

    def __init__(self, cfg: ShadowLitePadTacEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        body_names = list(self.robot_contact_sensor.body_names)
        pad_body_indices: list[int] = []
        pad_channels: list[int] = []
        for link_name, channel in self.LINK_TO_CHANNEL.items():
            if link_name not in body_names:
                raise RuntimeError(
                    f"Pad link {link_name!r} not in contact sensor bodies. "
                    f"Found: {body_names}"
                )
            pad_body_indices.append(body_names.index(link_name))
            pad_channels.append(channel)

        self._pad_body_indices = torch.tensor(pad_body_indices, device=self.device, dtype=torch.long)
        self._pad_channels = torch.tensor(pad_channels, device=self.device, dtype=torch.long)
        # Always the 12 physical FSR channel indices (never BioTac distals).
        self._fsr_channels = torch.tensor(
            list(PAD_LINK_TO_CHANNEL.values()), device=self.device, dtype=torch.long
        )

        print("PAD TAC bodies:", [body_names[i] for i in pad_body_indices])
        print("PAD TAC channels:", pad_channels)
        print("Tactile out dim:", NUM_TACTILE_CHANNELS)

        self._init_tactile_fsr_corrupt()
        self._init_tactile_flip()
        self._init_tactile_smoothing()

    def _init_tactile_flip(self) -> None:
        """Allocate the per-step taxel flip DR mask; default OFF."""
        p_off_on = float(getattr(self.cfg, "tactile_flip_prob_off_to_on", 0.0) or 0.0)
        p_on_off = float(getattr(self.cfg, "tactile_flip_prob_on_to_off", 0.0) or 0.0)
        q_off_on = float(getattr(self.cfg, "tactile_flip_prob_unsel_off_to_on", 0.0) or 0.0)
        q_on_off = float(getattr(self.cfg, "tactile_flip_prob_unsel_on_to_off", 0.0) or 0.0)
        for name, p in (
            ("off_to_on", p_off_on),
            ("on_to_off", p_on_off),
            ("unsel_off_to_on", q_off_on),
            ("unsel_on_to_off", q_on_off),
        ):
            if not 0.0 <= p <= 1.0:
                raise ValueError(f"tactile_flip_prob_{name} must be in [0, 1], got {p}")

        scope = getattr(self.cfg, "tactile_flip_scope", "corrupted")
        if scope not in ("corrupted", "all_fsr", "both"):
            raise ValueError(
                f"tactile_flip_scope must be 'corrupted', 'all_fsr' or 'both', got {scope!r}"
            )
        self._tac_flip_scope = scope

        if scope == "both":
            self.use_tactile_flip = max(p_off_on, p_on_off, q_off_on, q_on_off) > 0.0
        else:
            self.use_tactile_flip = p_off_on > 0.0 or p_on_off > 0.0
        self._tac_p_off_on = p_off_on
        self._tac_p_on_off = p_on_off
        self._tac_q_off_on = q_off_on
        self._tac_q_on_off = q_on_off
        if not self.use_tactile_flip:
            print("[taxel_flip] off")
            return

        if scope == "corrupted":
            # Scope: the k FSR channels picked by the episode-constant corrupt draw,
            # and nothing else. The mask is _tac_fsr_mask, so it is per-env and is
            # resampled at every reset. Without a corrupt draw nothing is selected
            # and the flip would be a silent no-op, so refuse that combination.
            if not getattr(self, "use_tactile_fsr_corrupt", False):
                raise ValueError(
                    "tactile_flip_prob_* requires tactile_fsr_corrupt_max > 0: the "
                    "per-step dither is scoped to the FSR channels selected by the "
                    "corrupt draw, so with no draw it would do nothing."
                )

            print(
                f"[taxel_flip] per-step dither on selected FSR channels only: "
                f"p(0->1)={p_off_on} p(1->0)={p_on_off}; unselected FSR pads and all "
                f"BioTac channels stay exact"
            )
        elif scope == "all_fsr":
            # Candidate FSR channels; the actual flip mask excludes whichever of
            # these this episode's corrupt draw selected as stuck (computed live in
            # _apply_tactile_flip, since that set changes every reset). Does not
            # require tactile_fsr_corrupt_max: works standalone with k=0 corrupted.
            eligible = torch.zeros((1, NUM_TACTILE_CHANNELS), device=self.device, dtype=torch.bool)
            eligible[0, self._fsr_channels] = True
            self._tac_fsr_eligible = eligible

            print(
                f"[taxel_flip] per-step flip on all 12 FSR channels MINUS this "
                f"episode's stuck subset: p(0->1)={p_off_on} p(1->0)={p_on_off}; "
                f"BioTac channels stay exact"
            )
        else:  # "both"
            # Every FSR pad is eligible, at two rates split by the corrupt draw.
            # That stuck subset changes every reset, so the split itself is
            # computed live in _apply_tactile_flip; only the FSR candidate set is
            # precomputed here. The selected-channel rates are meaningless without
            # a corrupt draw, so refuse that combination the way "corrupted" does.
            if (p_off_on > 0.0 or p_on_off > 0.0) and not getattr(
                self, "use_tactile_fsr_corrupt", False
            ):
                raise ValueError(
                    "tactile_flip_scope='both' with a non-zero tactile_flip_prob_* "
                    "requires tactile_fsr_corrupt_max > 0: those rates are scoped to "
                    "the channels the corrupt draw sticks, so with no draw they would "
                    "do nothing. Use tactile_flip_prob_unsel_* to dither all 12 pads."
                )
            eligible = torch.zeros((1, NUM_TACTILE_CHANNELS), device=self.device, dtype=torch.bool)
            eligible[0, self._fsr_channels] = True
            self._tac_fsr_eligible = eligible

            print(
                f"[taxel_flip] per-step flip on all 12 FSR channels at two rates: "
                f"stuck subset p(0->1)={p_off_on} p(1->0)={p_on_off}; the other "
                f"(12-k) pads p(0->1)={q_off_on} p(1->0)={q_on_off}; BioTac channels "
                f"stay exact"
            )

    def _apply_tactile_flip(self, tactile: torch.Tensor) -> torch.Tensor:
        """Dither the eligible taxels only. Returns strict 0.0/1.0.

        Scope depends on cfg.tactile_flip_scope:
          "corrupted" -- only the channels this episode's corrupt draw stuck.
          "all_fsr"   -- the 12 FSR channels minus that same stuck subset, so a
                         stuck channel is structurally excluded from the flip.
          "both"      -- all 12 FSR channels, the stuck subset dithering at
                         tactile_flip_prob_* and the rest at tactile_flip_prob_unsel_*.
        Every BioTac channel passes through exactly under all three scopes; under
        "corrupted" and "all_fsr" so do the out-of-scope FSR pads.

        Output must stay binary: DynamicsMemory stores ``tactile`` as ``torch.bool``
        when ``binary_tactile`` is set (multimodal_rl/ssl/physics_memory.py:59),
        which the forward-dynamics configs turn on.
        """
        # tactile is exactly 0.0/1.0, so these expressions select the on->off rate
        # where it is on and the off->on rate where it is off, without allocating
        # an index tensor.
        if self._tac_flip_scope == "both":
            # Per-channel rate: stuck pads take p, the rest take q. The eligible
            # mask keeps BioTac and the structurally-empty slots exact.
            p_sel = self._tac_p_off_on + (self._tac_p_on_off - self._tac_p_off_on) * tactile
            p_uns = self._tac_q_off_on + (self._tac_q_on_off - self._tac_q_off_on) * tactile
            p = torch.where(self._tac_fsr_mask, p_sel, p_uns)
            flip = (torch.rand_like(tactile) < p) & self._tac_fsr_eligible
            return torch.where(flip, 1.0 - tactile, tactile)

        if self._tac_flip_scope == "corrupted":
            mask = self._tac_fsr_mask
        else:  # "all_fsr"
            mask = self._tac_fsr_eligible & ~self._tac_fsr_mask
        p = self._tac_p_off_on + (self._tac_p_on_off - self._tac_p_off_on) * tactile
        flip = (torch.rand_like(tactile) < p) & mask
        return torch.where(flip, 1.0 - tactile, tactile)

    def _init_tactile_smoothing(self) -> None:
        """Allocate the temporal hold-filter state; default OFF.

        ``tactile_cfg.smoothing`` is an optional ``{k_on, k_off}`` block. A taxel
        must read ON for ``k_on`` consecutive control steps before the policy
        sees a 1, and OFF for ``k_off`` consecutive steps before it returns to 0.
        ``k_on == k_off == 1`` (and an absent block) reproduce the raw signal.
        """
        cfg = (self.tactile_cfg or {}).get("smoothing")
        self.use_tactile_smoothing = bool(cfg)
        if not self.use_tactile_smoothing:
            print("[taxel_smooth] off")
            return

        self._k_on = int(cfg.get("k_on", 1))
        self._k_off = int(cfg.get("k_off", 1))
        if self._k_on < 1 or self._k_off < 1:
            raise ValueError(
                f"tactile smoothing needs k_on >= 1 and k_off >= 1, got "
                f"k_on={self._k_on}, k_off={self._k_off}"
            )

        shape = (self.num_envs, NUM_TACTILE_CHANNELS)
        # Counters saturate at max(k_on, k_off) so a long hold cannot overflow.
        self._tac_ct_max = max(self._k_on, self._k_off)
        self._tac_on_ct = torch.zeros(shape, device=self.device, dtype=torch.int16)
        self._tac_off_ct = torch.zeros(shape, device=self.device, dtype=torch.int16)
        self._tac_hold_state = torch.zeros(shape, device=self.device, dtype=torch.float32)

        print(
            f"[taxel_smooth] hold filter on: k_on={self._k_on} k_off={self._k_off} "
            f"(~{self._k_on * self.step_dt * 1e3:.0f} ms onset, "
            f"~{self._k_off * self.step_dt * 1e3:.0f} ms release)"
        )

    def _apply_tactile_smoothing(self, tactile: torch.Tensor) -> torch.Tensor:
        """Debounce the binary tactile vector in time. Returns strict 0.0/1.0.

        Output must stay binary: DynamicsMemory stores ``tactile`` as ``torch.bool``
        when ``binary_tactile`` is set (multimodal_rl/ssl/physics_memory.py:59),
        so fractional values would be silently truncated on the SSL path.
        """
        raw = tactile > 0.5

        # Consecutive-run counters; any interruption resets the opposing counter.
        self._tac_on_ct = torch.where(raw, self._tac_on_ct + 1, torch.zeros_like(self._tac_on_ct))
        self._tac_off_ct = torch.where(raw, torch.zeros_like(self._tac_off_ct), self._tac_off_ct + 1)
        self._tac_on_ct.clamp_(max=self._tac_ct_max)
        self._tac_off_ct.clamp_(max=self._tac_ct_max)

        latched = self._tac_hold_state > 0.5
        turn_on = ~latched & (self._tac_on_ct >= self._k_on)
        turn_off = latched & (self._tac_off_ct >= self._k_off)

        self._tac_hold_state = torch.where(
            turn_on,
            torch.ones_like(self._tac_hold_state),
            torch.where(turn_off, torch.zeros_like(self._tac_hold_state), self._tac_hold_state),
        )
        return self._tac_hold_state.clone()

    def _init_tactile_fsr_corrupt(self) -> None:
        """Allocate episode-constant FSR corrupt buffers; default OFF."""
        max_k = getattr(self.cfg, "tactile_fsr_corrupt_max", None)
        self.use_tactile_fsr_corrupt = max_k is not None and int(max_k) > 0
        self._tac_fsr_mask = torch.zeros(
            (self.num_envs, NUM_TACTILE_CHANNELS), device=self.device, dtype=torch.bool
        )
        self._tac_fsr_val = torch.zeros(
            (self.num_envs, NUM_TACTILE_CHANNELS), device=self.device, dtype=torch.float32
        )
        if not self.use_tactile_fsr_corrupt:
            print("[taxel_dr] off")
            return
        self._tactile_fsr_corrupt_max = int(max_k)
        if self._tactile_fsr_corrupt_max > len(self._fsr_channels):
            raise ValueError(
                f"tactile_fsr_corrupt_max={self._tactile_fsr_corrupt_max} exceeds "
                f"n_fsr={len(self._fsr_channels)}"
            )
        print(
            f"[taxel_dr] FSR corrupt k in [0, {self._tactile_fsr_corrupt_max}] "
            f"(12 FSR only; mixed forced 0/1; BioTac untouched)"
        )
        self._sample_tactile_fsr_corrupt(None)

    def _sample_tactile_fsr_corrupt(self, env_ids: Sequence[int] | None) -> None:
        """Resample per-env FSR corrupt mask (episode-constant until next reset)."""
        if not getattr(self, "use_tactile_fsr_corrupt", False):
            return
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long).view(-1)
        n = env_ids.numel()
        n_fsr = int(self._fsr_channels.numel())
        max_k = self._tactile_fsr_corrupt_max

        # Clear previous overrides on these envs.
        self._tac_fsr_mask[env_ids] = False
        self._tac_fsr_val[env_ids] = 0.0

        # k ~ Uniform{0..max_k}; pick k FSR slots via random ranks; each gets 0 or 1.
        k = torch.randint(0, max_k + 1, (n,), device=self.device)
        scores = torch.rand(n, n_fsr, device=self.device)
        ranks = scores.argsort(dim=-1).argsort(dim=-1)
        active = ranks < k.unsqueeze(1)  # (n, 12)
        values = torch.randint(0, 2, (n, n_fsr), device=self.device, dtype=torch.float32)

        self._tac_fsr_mask[env_ids[:, None], self._fsr_channels[None, :]] = active
        self._tac_fsr_val[env_ids[:, None], self._fsr_channels[None, :]] = values

    def _reset_idx(self, env_ids: Sequence[int] | None):
        super()._reset_idx(env_ids)
        if getattr(self, "use_tactile_fsr_corrupt", False):
            self._sample_tactile_fsr_corrupt(env_ids)
        # Clear hold-filter state so contacts cannot bleed across episodes:
        # _reset_idx runs before _get_observations in the step loop.
        if getattr(self, "use_tactile_smoothing", False):
            ids = self.robot._ALL_INDICES if env_ids is None else env_ids
            ids = torch.as_tensor(ids, device=self.device, dtype=torch.long).view(-1)
            self._tac_on_ct[ids] = 0
            self._tac_off_ct[ids] = 0
            self._tac_hold_state[ids] = 0.0

    def _get_tactile(self):
        forces = self.robot_contact_sensor.data.net_forces_w[:].clone()
        norm = torch.linalg.vector_norm(forces, dim=-1)

        if self.tactile_cfg is not None and self.tactile_cfg.get("binary_tactile", True):
            norm = (norm > self.binary_threshold).float()

        tactile = torch.zeros((self.num_envs, NUM_TACTILE_CHANNELS), device=self.device)
        tactile[:, self._pad_channels] = norm[:, self._pad_body_indices]

        # Temporal debounce on the raw sensor signal, upstream of the corrupt DR
        # and flip so smoothing acts on genuine contact, not stuck/flipped values.
        if getattr(self, "use_tactile_smoothing", False):
            tactile = self._apply_tactile_smoothing(tactile)

        # Corrupt DR before flip: the forced value is the baseline the dither
        # acts on, so a selected channel sits at its value ~(1 - p) of the time
        # and takes the opposite ~p. The flip is masked to these same channels.
        if getattr(self, "use_tactile_fsr_corrupt", False):
            tactile = torch.where(self._tac_fsr_mask, self._tac_fsr_val, tactile)

        # Per-step flip DR. Before zero_tactile so the prop-only ablation stays
        # all-zero.
        if getattr(self, "use_tactile_flip", False):
            tactile = self._apply_tactile_flip(tactile)

        if self.tactile_cfg is not None and self.tactile_cfg.get("zero_tactile", False):
            tactile.zero_()

        self.last_tactile = self.tactile
        self.tactile = tactile
        return tactile


class ShadowLitePadTacBTEnv(ShadowLitePadTacEnv):
    """Tactile from 12 FSR pads + 4 BioTac fingertips -> 24-d deploy vector.

    Identical pipeline to ShadowLitePadTacEnv; only the body->channel map grows to
    include the 4 distal (BioTac) links, so all 16 hardware channels exist in sim.
    """

    cfg: ShadowLitePadTacBTEnvCfg
    LINK_TO_CHANNEL: dict[str, int] = PAD_BT_LINK_TO_CHANNEL
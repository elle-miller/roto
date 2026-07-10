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

from roto.assets.shadow_hand_lite import SHADOW_HAND_LITE_CFG
from roto.tasks.roto_env import RotoEnv, RotoEnvCfg

from isaaclab.markers.config import FRAME_MARKER_CFG  # isort: skip


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

    # J2 must reach this angle (rad) before J1 starts moving.
    # 0.8727 rad = 50° (corrected from 45° per user), matched to the real coupling split.
    coupling_theta: float = 0.8727

    # Route-2 sequencing: gate the J1 mimic on MEASURED J2 so J1 can't lead its
    # driver. J1's commanded curl is scaled by how close measured J2 is to its
    # limit, ramping over [opens_at, J2_max] where opens_at = J2_max - band.
    # frac=1.0 (strict): J1 only fires when J2 is within couple_gate_j2_tol of its limit.
    # frac<1.0: gate opens earlier, at frac * J2_max (legacy behaviour).
    couple_gate_j1_on_measured: bool = True
    couple_gate_lo_frac: float = 1.0    # strict: J1 only once J2 reaches its limit
    couple_gate_j2_tol: float = 0.035   # rad (~2°) tolerance band at the J2 limit

    # Stateful backlash coupling (supersedes the measured-J2 gate when True). On
    # uncurl J2 unlocks early at a per-episode random angle R (combined ffj0 frame,
    # degrees), J1 unwinds to 0 at 100°, and reversing inside (100°,R) freezes J1
    # until the motor returns to R. See RotoEnv._asymmetric_backlash.
    # Disabled per user decision -- falls back to the couple_gate_j1_on_measured
    # path above (simpler, no per-episode latched state).
    couple_asymmetric_backward: bool = False
    couple_release_range_deg: tuple[float, float] = (100.0, 140.0)
    couple_dir_deadband: float = 0.002   # rad; |Δm| below this latches direction

    # Hand mounting tilt. (lo, hi) equal -> fixed mount (no DR), which is the default:
    # the hand sits at the fixed 15° forward tilt from init_state. Widen to e.g.
    # (0.0, 15.0) to domain-randomize the tilt per episode.
    hand_tilt_range_deg: tuple[float, float] = (15.0, 15.0)

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
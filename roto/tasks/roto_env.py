# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Author: Elle Miller

Generalisable robot parent environment for IsaacLab RL tasks. See README.md for more details.

This environment is used to define the basic robot control and observation logic for the RoTO tasks.

It is a child of `DirectRLEnv`, which is a base environment for Isaac Lab RL tasks.

"""

from __future__ import annotations

import math

import torch

import isaaclab.sim as sim_utils
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg, ViewerCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import PhysxCfg, SimulationCfg
from isaaclab.sim.spawners.materials.physics_materials_cfg import RigidBodyMaterialCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import (
    sample_uniform,
    saturate,
)
from isaaclab.sensors import TiledCamera, TiledCameraCfg
from isaaclab.sensors import ContactSensor, ContactSensorCfg
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg

from roto.tasks.physics import roto_sim_cfg, DECIMATION, PHYSICS_DT

@configclass
class RotoEnvCfg(DirectRLEnvCfg):
    """Simulation / scene defaults used by every RoTO task."""

    physics_dt = PHYSICS_DT
    decimation = DECIMATION
    # Must assign default; `sim: roto_sim_cfg` alone only annotates and leaves DirectRLEnvCfg's SimulationCfg().
    sim: SimulationCfg = roto_sim_cfg

    # Isaac 4.5 compatibility
    observation_space = 0
    state_space = 0

    # Observation configuration (set from agent_cfg)
    obs_list: list[str] = []
    obs_stack: int = 1
    pixel_cfg: dict | None = None
    tactile_cfg: dict | None = None

    # Scene configuration
    replicate_physics = True
    env_spacing = 1.5
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=4096, env_spacing=env_spacing, replicate_physics=replicate_physics
    )

    # Viewer configuration (not used directly)
    eye = (3, 3, 3)
    lookat = (0, 0, 0)
    viewer: ViewerCfg = ViewerCfg(eye=eye, lookat=lookat, resolution=(1920, 1080))

    # camera
    eye = (0.0, -0.6, 0.65)
    target = (0.0, -0.35, 0.5)

    render_cfg = sim_utils.RenderCfg() #rendering_mode="quality")

    # tactile sensor configuration
    robot_contact_sensor_cfg = None
    
    # Evaluation environment visualization
    num_eval_envs: int = 1  # Number of evaluation environments (for visual markers)
    eval_marker_cfg: VisualizationMarkersCfg = VisualizationMarkersCfg(
        prim_path="/Visuals/eval_env_markers",
        markers={
            "eval_box": sim_utils.CuboidCfg(
                size=(1.0, 1.0, 0.01),  # 1m x 1m x 1m box
                visual_material=sim_utils.PreviewSurfaceCfg(
                    opacity=1,  # Fully opaque
                    diffuse_color=(1.0, 0.2, 0.2),  # Pink color (RGB)
                ),
            )
        },
    )


class RotoEnv(DirectRLEnv):
    """Shared RL base class handling control, observation, and reset logic."""

    cfg: RotoEnvCfg

    def __init__(self, cfg: RotoEnvCfg, render_mode: str | None = None, **kwargs):
        """Initialize tensors used by derived robot + task implementations."""
        
        # Observation configuration
        self.obs_stack = getattr(cfg, "obs_stack", 1)
        self.pixel_cfg = getattr(cfg, "pixel_cfg", None)
        self.tactile_cfg = getattr(cfg, "tactile_cfg", None)
        if self.tactile_cfg is not None:
            self.binary_threshold = self.tactile_cfg["binary_threshold"]
            self.binary_tactile = self.tactile_cfg['binary_tactile']
        self.dtype = torch.float32

        super().__init__(cfg, render_mode, **kwargs)

        print("--------------------------------")
        print("RL frequency: ", 1 / (self.cfg.sim.dt * self.cfg.decimation))
        print("Physics frequency: ", 1 / self.cfg.sim.dt)
        print("Episode length: ", self.cfg.episode_length_s)
        print("Episode length in steps: ", self.cfg.episode_length_s / (self.cfg.sim.dt * self.cfg.decimation))
        print("--------------------------------")

        # Joint limits and targets
        self.robot_joint_pos_lower_limits = self.robot.data.soft_joint_pos_limits[0, :, 0].to(device=self.device)
        self.robot_joint_pos_upper_limits = self.robot.data.soft_joint_pos_limits[0, :, 1].to(device=self.device)
        self.robot_joint_vel_limits = self.robot.data.joint_vel_limits[0, :].to(device=self.device)

        self.joint_pos_cmd = torch.zeros((self.num_envs, self.robot.num_joints), device=self.device)
        self.prev_joint_pos_cmd = torch.zeros((self.num_envs, self.robot.num_joints), device=self.device)
        self.joint_pos_error = torch.zeros((self.num_envs, self.robot.num_joints), device=self.device)

        # Indices of actuated joints
        self.actuated_dof_indices = [
            self.robot.joint_names.index(joint_name) for joint_name in cfg.actuated_joint_names
        ]
        self.actuated_dof_indices.sort()

        # Policy-controlled joints, in action-vector order
        self.control_dof_indices = [self.robot.joint_names.index(n) for n in cfg.control_joint_names]

        # Joints used for the proprioception vector. Defaults to all actuated joints
        # (current behavior); robots whose obs should exclude coupled mimic joints
        # override this (see ShadowLiteEnv).
        self.prop_dof_indices = self.actuated_dof_indices

        # Coupled joints: dependent J1 <- driver J2 (same order as the dict)
        self.coupled_dependent_indices = [self.robot.joint_names.index(d) for d in cfg.coupled_joint_map.keys()]
        self.coupled_driver_indices    = [self.robot.joint_names.index(d) for d in cfg.coupled_joint_map.values()]
        # θ (rad): J2 must exceed this before J1 starts moving
        self.coupling_theta = getattr(cfg, "coupling_theta", 0.0)
        # Optional closed-loop sequencing: gate J1's command on MEASURED J2 so the
        # mimic can't out-run its driver (see _handle_coupled_joints).
        self.couple_gate_j1_on_measured = getattr(cfg, "couple_gate_j1_on_measured", False)
        self.couple_gate_lo_frac        = getattr(cfg, "couple_gate_lo_frac", 0.9)
        # Angular band width at the top of J2's range over which the gate ramps 0→1.
        # At frac=1 gate_lo==j2_upper so the band collapses to tol, meaning J1 fires
        # only when measured J2 is within tol of its limit (not never, which was the bug).
        self.couple_gate_j2_tol         = getattr(cfg, "couple_gate_j2_tol", 0.035)  # ~2°

        # Stateful backlash coupling (asymmetric forward/backward). When on, the
        # J1<-J2 coupling carries J1 as state with a per-episode latched "unlock"
        # angle R (combined ffj0 frame). On uncurl J2 unlocks early at R while J1
        # unwinds to 0 at 100°; reversing inside (100°, R) freezes J1 until the
        # motor returns to R. See _handle_coupled_joints. Disabled -> old gate path.
        self.couple_asymmetric_backward = getattr(cfg, "couple_asymmetric_backward", False)
        _rel_lo, _rel_hi = getattr(cfg, "couple_release_range_deg", (100.0, 140.0))
        self.couple_release_lo = _rel_lo * math.pi / 180.0   # combined-frame rad
        self.couple_release_hi = _rel_hi * math.pi / 180.0
        self.couple_dir_deadband = getattr(cfg, "couple_dir_deadband", 0.002)  # rad
        n_coupled = len(self.coupled_driver_indices)
        # combined-frame split point (rad): J2 owns [0, j2_upper], J1 owns the rest.
        self._couple_j2_top = self.robot_joint_pos_upper_limits[self.coupled_driver_indices].clone()    # (3,) ~100°
        self._couple_j1_span = self.robot_joint_pos_upper_limits[self.coupled_dependent_indices].clone() # (3,) ~80°
        self._couple_m_top = self._couple_j2_top + self._couple_j1_span                                  # (3,) ~180°
        self.couple_release = torch.full((self.num_envs, n_coupled), self.couple_release_hi, device=self.device)  # latched/episode
        self.prev_m         = torch.zeros((self.num_envs, n_coupled), device=self.device)
        self.couple_dir     = torch.ones((self.num_envs, n_coupled), device=self.device)   # +1 curl / -1 uncurl
        self.j1_state       = torch.zeros((self.num_envs, n_coupled), device=self.device)
        # Backlash freeze state: set when the finger reverses curl->uncurl inside the
        # (100°, R) zone; J1 holds at couple_frozen_val until the motor returns to R.
        self.couple_frozen_flag = torch.zeros((self.num_envs, n_coupled), dtype=torch.bool, device=self.device)
        self.couple_frozen_val  = torch.zeros((self.num_envs, n_coupled), device=self.device)


        # Action and state tensors
        self.actions = torch.zeros((self.num_envs, self.cfg.num_actions), device=self.device)
        default_joint_pos = self.robot.data.default_joint_pos
        self.joint_pos_cmd[:, self.actuated_dof_indices] = default_joint_pos[:, self.actuated_dof_indices]
        self.prev_joint_pos_cmd[:, self.actuated_dof_indices] = default_joint_pos[:, self.actuated_dof_indices]

        self.num_joints = self.robot.num_joints
        self.joint_pos = torch.zeros((self.num_envs, self.num_joints), device=self.device)
        self.joint_vel = torch.zeros((self.num_envs, self.num_joints), device=self.device)
        self.joint_acc = torch.zeros((self.num_envs, self.num_joints), device=self.device)

        self.normalised_joint_pos = torch.zeros((self.num_envs, self.num_joints), device=self.device)
        self.normalised_joint_vel = torch.zeros((self.num_envs, self.num_joints), device=self.device)

        # Set up camera if rgb or depth are in the observation list
        if "rgb" in self.cfg.obs_list or "depth" in self.cfg.obs_list:
            eyes = (
                torch.tensor(self.cfg.eye, dtype=torch.float, device=self.device).repeat((self.num_envs, 1))
                + self.scene.env_origins
            )
            targets = (
                torch.tensor(self.cfg.target, dtype=torch.float, device=self.device).repeat((self.num_envs, 1))
                + self.scene.env_origins
            )
            self._tiled_camera.set_world_poses_from_view(eyes=eyes, targets=targets)
        
        # Current / previous-step tactile (contact) readings for tasks (bounce, logging, etc.).
        self.tactile = torch.zeros((self.num_envs, 0), device=self.device)
        self.last_tactile = torch.zeros((self.num_envs, 0), device=self.device)

        # Visualize evaluation environment markers (pink boxes)
        if self.cfg.num_eval_envs > 0 and hasattr(self, "eval_markers"):
            # Position markers at evaluation environment origins (first num_eval_envs)
            # Offset by 0.5m in Z to center the 1m tall box on the ground
            eval_positions = self.scene.env_origins[: self.cfg.num_eval_envs].clone()
            eval_positions[:, 2] += 0.0  # Raise by half box height to center on ground
            eval_rotations = torch.zeros((self.cfg.num_eval_envs, 4), dtype=torch.float, device=self.device)
            eval_rotations[:, 0] = 1.0  # Identity quaternion
            self.eval_markers.visualize(eval_positions, eval_rotations)

    def _setup_scene(self):
        """Set up the simulation scene."""
        # Set up camera if rgb or depth are in the observation list
        if ("rgb" in self.cfg.obs_list or "depth" in self.cfg.obs_list) and self.pixel_cfg is not None:
            print("Setting up camera for pixel observation with width: ", self.pixel_cfg["width"], "and height: ", self.pixel_cfg["height"])
            tiled_camera = TiledCameraCfg(
                prim_path="/World/envs/env_.*/Camera",
                offset=TiledCameraCfg.OffsetCfg(pos=(0.0, 0.0, 0.7), rot=(1.0, 0.0, 0.0, 0.0), convention="world"),
                data_types=["rgb", "depth"],
                spawn=sim_utils.PinholeCameraCfg(
                    focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, self.cfg.env_spacing)
                ),
                width=self.pixel_cfg["width"],
                height=self.pixel_cfg["height"],
            )
            self._tiled_camera = TiledCamera(tiled_camera)
            self.scene.sensors["tiled_camera"] = self._tiled_camera

        # Add tactile sensor if listed
        if "tactile" in self.cfg.obs_list:
            print("Adding contact sensor for tactile observation")
            self.robot_contact_sensor = ContactSensor(self.cfg.robot_contact_sensor_cfg)
            self.scene.sensors["robot_contact_sensor"] = self.robot_contact_sensor
        
        # Create visual markers for evaluation environments (pink boxes)
        if self.cfg.num_eval_envs > 0:
            self.eval_markers = VisualizationMarkers(self.cfg.eval_marker_cfg)

    def _configure_gym_env_spaces(self):
        """Configure Gymnasium observation and action spaces (placeholder)."""

    def set_spaces(self, single_obs, obs, single_action, action):
        """Set Gymnasium observation + action spaces for downstream wrappers."""
        self.single_observation_space = single_obs
        self.observation_space = obs
        self.single_action_space = single_action
        self.action_space = action

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.prev_joint_pos_cmd[:] = self.joint_pos_cmd
        self.actions = actions.clone()  # (num_envs, 13)

        # 13 actions -> 13 directly controlled joints
        self.joint_pos_cmd[:, self.control_dof_indices] = scale(
            self.actions,
            self.robot_joint_pos_lower_limits[self.control_dof_indices],
            self.robot_joint_pos_upper_limits[self.control_dof_indices],
        )

        # Snapshot the stateful-coupling buffers so the settle phase can freeze them:
        # while the hand is held in its catch pose, the policy's curl proxy must NOT
        # advance the backlash state (otherwise J1/J2 drift away from the open hand and
        # hand over a corrupt state when the policy takes over).
        sc = getattr(self, "settle_counter", None)
        settling = (sc > 0) if sc is not None else None
        snap = None
        if settling is not None and self.couple_asymmetric_backward and settling.any():
            snap = (self.prev_m.clone(), self.couple_dir.clone(), self.j1_state.clone(),
                    self.couple_frozen_flag.clone(), self.couple_frozen_val.clone())

        # fill the 3 coupled J1 commands from the J2 drivers
        self._handle_coupled_joints()

        # Settle phase: hold the default catch pose while balls drop into the palm.
        # BaodingMixin sets self.settle_counter on each reset; we check generically so
        # other envs (no balls) are unaffected.
        if settling is not None and settling.any():
            self.joint_pos_cmd[settling] = self.robot.data.default_joint_pos[settling]
            if snap is not None:
                # restore the frozen coupling state for the settling envs only
                self.prev_m[settling]             = snap[0][settling]
                self.couple_dir[settling]         = snap[1][settling]
                self.j1_state[settling]           = snap[2][settling]
                self.couple_frozen_flag[settling] = snap[3][settling]
                self.couple_frozen_val[settling]  = snap[4][settling]
            self.settle_counter[settling] -= 1


    def _handle_coupled_joints(self) -> None:
        """Split a single 'finger curl' proxy action into J2 and J1 commands.

        The policy action for J2 is scaled to [0, J2_upper] by _pre_physics_step and
        treated here as a combined curl proxy.  coupling_theta (rad) is the split point:

          proxy ∈ [0,     theta]:  J2 ramps 0 → J2_max,  J1 = 0
          proxy ∈ [theta, J2_max]: J2 = J2_max (held),   J1 ramps 0 → J1_max

        With coupling_theta = 0.8727 rad (50°) and J2_max = J1_max = pi/2 (90°):
          proxy = 0.8727 (50°) → J2 = 90°, J1 =  0°
          proxy = 1.2217 (70°) → J2 = 90°, J1 = 45°
          proxy = 1.5708 (90°) → J2 = 90°, J1 = 90°
        """
        # proxy = what _pre_physics_step wrote for J2, in [0, J2_upper]
        proxy   = self.joint_pos_cmd[:, self.coupled_driver_indices]          # (N, 3)
        j2_upper = self.robot_joint_pos_upper_limits[self.coupled_driver_indices]    # (3,)
        j1_upper = self.robot_joint_pos_upper_limits[self.coupled_dependent_indices] # (3,)
        theta = self.coupling_theta  # scalar (rad)

        zeros = torch.zeros_like(j2_upper)  # tensor min — torch.clamp requires min/max same type

        # J2: proxy in [0, theta] maps linearly to [0, J2_max]; clamp above theta
        j2_cmd = torch.clamp(proxy * (j2_upper / theta), zeros, j2_upper)

        # J1: proxy in [theta, J2_max] maps linearly to [0, J1_max]; zero below theta
        j1_cmd = torch.clamp(
            (proxy - theta) / (j2_upper - theta) * j1_upper,
            zeros, j1_upper,
        )

        if self.couple_asymmetric_backward:
            # Stateful backlash coupling supersedes the measured-J2 gate.
            j2_cmd, j1_cmd = self._asymmetric_backlash(j2_cmd, j1_cmd, j2_upper, j1_upper)
        elif self.couple_gate_j1_on_measured:
            # Closed-loop sequencing: scale J1's command by how far the MEASURED J2
            # has reached its limit, so J1 cannot lead its driver no matter what the
            # proxy asks. gate ramps 0->1 over [couple_gate_lo_frac * J2_max, J2_max].
            meas_j2  = self.robot.data.joint_pos[:, self.coupled_driver_indices]   # (N, 3)
            gate_lo  = self.couple_gate_lo_frac * j2_upper                          # (3,)
            # Guarantee a finite ramp band even when frac=1 (gate_lo == j2_upper).
            # At frac=1: band==tol, gate opens over [j2_upper-tol, j2_upper].
            # At frac<1: band==j2_upper-gate_lo (same as before, tol has no effect).
            tol      = self.couple_gate_j2_tol
            band     = torch.clamp(j2_upper - gate_lo, min=tol)
            opens_at = j2_upper - band
            gate = torch.clamp(
                (meas_j2 - opens_at) / band,
                zeros, torch.ones_like(j2_upper),
            )
            j1_cmd = j1_cmd * gate

        if getattr(self.cfg, "lock_coupled_dependent_at_zero", False):
            j1_cmd = torch.zeros_like(j1_cmd)

        self.joint_pos_cmd[:, self.coupled_driver_indices]    = j2_cmd
        self.joint_pos_cmd[:, self.coupled_dependent_indices] = j1_cmd

    def _asymmetric_backlash(self, j2_fwd, j1_fwd, j2_upper, j1_upper):
        """Stateful backlash J1<-J2 coupling in the combined ffj0 frame.

        The combined motor target m = j2_fwd + j1_fwd (rad, in [0, j2_upper+j1_upper] ~
        [0,180°]) is the single input; j2_upper~100° is the J2/J1 split. Per finger we
        carry j1 as state plus a per-episode latched unlock angle R = couple_release.

          • curl (m rising, fresh): j2 = min(m, 100°), j1 = clamp(m-100°, 0, 80°) — J1
            only moves once J2 saturates at 100°.
          • uncurl (m falling): j1 = clamp(m-100°, 0, 80°) (hits 0 at m=100°); J2 unlocks
            EARLY at R: j2 = (m/R)·100° for m<R, so over [100°,R] both J2 and J1 drop.
          • reversal (uncurl, stop in (100°,R), curl back): j1 FREEZES at its value until
            m climbs back to R, then resumes from the frozen value up to 80° at m=180°.

        Returns (j2_cmd, j1_cmd), both (N,3), and advances the per-finger state.
        """
        j2_top  = self._couple_j2_top      # (3,) ~100°
        j1_span = self._couple_j1_span     # (3,) ~80°
        m_top   = self._couple_m_top       # (3,) ~180°
        R       = self.couple_release      # (N,3) latched per episode
        db      = self.couple_dir_deadband
        eps     = 1e-4

        m = j2_fwd + j1_fwd                                       # combined motor target
        zeros = torch.zeros_like(m)

        delta   = m - self.prev_m
        rising  = delta >  db
        falling = delta < -db
        # latch direction when steady (|delta| <= db)
        new_dir = torch.where(rising, torch.ones_like(m),
                              torch.where(falling, -torch.ones_like(m), self.couple_dir))

        l = torch.clamp(m - j2_top, zeros, j1_span)              # demand curve (engage @100°)

        frozen = self.couple_frozen_flag
        fval   = self.couple_frozen_val
        # enter freeze: direction flips uncurl->curl while below full curl and below R
        flip_up = (self.couple_dir < 0) & (new_dir > 0)
        enter   = flip_up & (self.j1_state < j1_span - eps) & (m < R)
        frozen  = frozen | enter
        fval    = torch.where(enter, self.j1_state, fval)
        # uncurling clears the freeze (J1 tracks the demand back down). Use the LATCHED
        # direction, not instantaneous `falling`, so a steady frame (slider not moving,
        # |delta|<deadband) keeps the uncurl state instead of reverting.
        uncurling = new_dir < 0
        frozen  = frozen & ~uncurling

        # resume ramp: frozen_val -> j1_span over [R, m_top]; hold below R
        denom   = torch.clamp(m_top - R, min=eps)
        resume  = fval + (m - R) / denom * (j1_span - fval)
        resume  = torch.clamp(resume, fval, j1_span)
        j1_frozen_branch = torch.where(m >= R, resume, fval)

        j1 = torch.where(frozen, j1_frozen_branch, l)
        # freeze done once the resume ramp has caught the demand (top region)
        frozen = frozen & ~(j1 >= l - eps)

        # J2: unlock-at-R curve while uncurling or frozen; fresh saturate-at-100° while
        # curling. Keyed on the LATCHED direction so steady frames during an uncurl don't
        # snap J2 back to the j2_fresh (=100°) branch — that was the "J2 bounces back to
        # the limit and won't uncurl" bug, since most slider frames are steady.
        j2_down  = torch.clamp(m / R * j2_top, zeros, j2_top)    # unlock at R
        j2_fresh = torch.clamp(m, zeros, j2_top)                 # engage at 100° fresh
        j2 = torch.where(uncurling | frozen, j2_down, j2_fresh)

        # advance state
        self.couple_dir         = new_dir
        self.prev_m             = m
        self.j1_state           = j1
        self.couple_frozen_flag = frozen
        self.couple_frozen_val  = fval
        return j2, j1

    def _sample_coupling_params(self, env_ids):
        """Draw the per-episode latched backlash unlock R and clear coupling state.

        Called on reset (from ShadowLiteEnv._reset_idx). Each finger gets a fresh
        R ~ U(release_lo, release_hi); direction, motor history, J1 state and the
        freeze latch are reset so the new episode starts from a clean open hand.
        """
        n = len(env_ids)
        k = len(self.coupled_driver_indices)
        self.couple_release[env_ids] = sample_uniform(
            self.couple_release_lo, self.couple_release_hi, (n, k), self.device
        )
        self.prev_m[env_ids]             = 0.0
        self.couple_dir[env_ids]         = 1.0
        self.j1_state[env_ids]           = 0.0
        self.couple_frozen_flag[env_ids] = False
        self.couple_frozen_val[env_ids]  = 0.0


    def _apply_action(self) -> None:
        """Apply actions to the robot.

        Called multiple times per RL step for decimation. 
        """
        self.robot.set_joint_position_target(
            self.joint_pos_cmd[:, self.actuated_dof_indices], joint_ids=self.actuated_dof_indices
        )

    def get_observations(self):
        """Get observations for the current timestep.

        Returns:
            Dictionary of observations.
        """
        return self._get_observations()

    def _get_observations(self) -> dict:
        """Collect observations according to the requested cfg keys."""
        obs_dict = {}
        for k in self.cfg.obs_list:
            if k == "prop":
                obs_dict[k] = self._get_proprioception()
            elif k == "gt":
                obs_dict[k] = self._get_gt()
            elif k == "tactile":
                obs_dict[k] = self._get_tactile()
            elif k == "rgb":
                obs_dict[k] = self._get_rgb()
            elif k == "depth":
                obs_dict[k] = self._get_depth()
            else:
                raise ValueError(f"Unknown observation key '{k}'")

        obs_dict = {"policy": obs_dict}
        return obs_dict

    def _get_proprioception(self):
        """Return proprioceptive feature vector.

        Returns:
            Concatenated tensor containing normalized joint positions, normalized joint
            velocities, and actions.
        """
        prop = torch.cat(
            (
                self.normalised_joint_pos[:, self.prop_dof_indices],
                self.normalised_joint_vel[:, self.prop_dof_indices],
                self.joint_pos_error[:, self.prop_dof_indices],
                self.actions,
            ),
            dim=-1,
        )

        return prop

    def _get_rgb(self) -> torch.Tensor:
        """Return rendered RGB observations.

        Processes RGB camera data, optionally applying normalization.

        Returns:
            RGB tensor with shape [num_envs, height, width, 3] and dtype uint8.
        """
        if self.pixel_cfg is None:
            raise ValueError("pixel_cfg is not set. Make sure 'rgb' is in obs_list and pixel_cfg is provided in agent_cfg.")

        # Clone the RGB buffer
        data = self._tiled_camera.data.output["rgb"].clone()

        if self.pixel_cfg.get("normalise_rgb", False):
            # Normalize RGB: convert to float, center by subtracting mean, then scale back
            data = data.float() / 255.0
            mean_tensor = torch.mean(data, dim=(1, 2), keepdim=True)
            data -= mean_tensor
            data = 255.0 * data  # Scale back to [0, 255]
            data = data.to(torch.uint8)

        return data

    def _get_depth(self) -> torch.Tensor:
        """Return rendered depth observations.

        Processes depth camera data, handling edge cases like inf/NaN values.

        Returns:
            Depth tensor with shape [num_envs, height, width, 1] and dtype float32.
        """
        if self.pixel_cfg is None:
            raise ValueError("pixel_cfg is not set. Make sure 'depth' is in obs_list and pixel_cfg is provided in agent_cfg.")

        # Clone the depth buffer
        data = self._tiled_camera.data.output["depth"].clone()

        min_depth = 0.0
        max_depth = self.pixel_cfg["max_depth"]

        # 1. Handle inf/NaN: Set to max_depth instead of 0
        data[torch.isinf(data) | torch.isnan(data)] = max_depth
        
        # 2. Clip: Ensure no values are outside [min_depth, max_depth]
        data = torch.clamp(data, min_depth, max_depth)
        
        # 3. Normalize: Scale to [0.0, 1.0]
        # Form: (value - min) / (max - min)
        data = (data - min_depth) / (max_depth - min_depth)
        
        # Optional: Invert if you want closer objects to be "brighter"
        data = 1.0 - data

        # Ensure depth has a channel dimension: [num_envs, height, width, 1]
        if data.dim() == 3:
            data = data.unsqueeze(-1)

        return data

    def _get_tactile(self):
        """Return tactile force.

        Computes contact forces from multiple sensors and converts them to binary
        activations based on a threshold.

        Returns:
            Concatenated tensor of tactile force.
        """

        # Forces is [num_envs, num_sensors, 3], take the norm to be [num_envs, num_sensors]
        forces = self.robot_contact_sensor.data.net_forces_w[:].clone()
        norm = torch.linalg.vector_norm(forces, dim=-1, keepdim=False)

        # Convert to binary activations based on threshold if binary_tactile is True
        if self.tactile_cfg is not None and self.tactile_cfg["binary_tactile"]:
            norm = (norm > self.binary_threshold).float()
        self.last_tactile = self.tactile
        self.tactile = norm
        return norm

    def _reset_robot(self, env_ids, joint_pos_noise=0.125):
        """Reset the robot joint positions and velocities.

        Args:
            env_ids: Environment indices to reset.
            joint_pos_noise: Standard deviation of noise added to joint positions.
        """
        noise = sample_uniform(
            -joint_pos_noise,
            joint_pos_noise,
            (len(env_ids), self.robot.num_joints),
            self.device,
        )
        if getattr(self.cfg, "lock_coupled_dependent_at_zero", False):
            noise[:, self.coupled_dependent_indices] = 0.0
        joint_pos = self.robot.data.default_joint_pos[env_ids] + noise
        joint_pos = torch.clamp(joint_pos, self.robot_joint_pos_lower_limits, self.robot_joint_pos_upper_limits)
        joint_vel = torch.zeros_like(joint_pos)
        self.robot.set_joint_position_target(joint_pos, env_ids=env_ids)
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)

    def _compute_intermediate_values(self, env_ids=None):
        """Compute intermediate values for observations and rewards.

        Updates joint positions, velocities, accelerations, and their normalized versions.

        Args:
            env_ids: Environment indices to update.
        """
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        
        # Get robot data
        self.joint_pos[env_ids] = self.robot.data.joint_pos[env_ids]
        self.joint_vel[env_ids] = self.robot.data.joint_vel[env_ids]
        self.joint_acc[env_ids] = self.robot.data.joint_acc[env_ids]
        self.joint_pos_error[env_ids] = self.joint_pos_cmd[env_ids] - self.joint_pos[env_ids]

        # Normalize joint positions
        self.normalised_joint_pos[env_ids] = unscale(
            self.joint_pos[env_ids], self.robot_joint_pos_lower_limits, self.robot_joint_pos_upper_limits
        )
        # Normalize velocities by dividing by a fixed scale factor
        self.normalised_joint_vel[env_ids] = self.joint_vel[env_ids] / 3.0
        self.normalised_joint_vel[env_ids] = unscale(
            self.joint_vel[env_ids], -self.robot_joint_vel_limits, self.robot_joint_vel_limits
        )


@torch.jit.script
def scale(x, lower, upper):
    """Scale input `x` from [-1, 1] to [lower, upper]."""
    return 0.5 * (x + 1.0) * (upper - lower) + lower


@torch.jit.script
def unscale(x, lower, upper):
    """Unscale input `x` from [lower, upper] to [-1, 1]."""
    return (2.0 * x - upper - lower) / (upper - lower)
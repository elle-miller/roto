"""UAN (Unsupervised Actuator Network) task for the ShadowLite hand.

Registers the gym environment id "UAN_Shadowlite". `UANShadowLiteEnv`
subclasses roto's `ShadowLiteEnv` unchanged (same robot USD, same
`SHADOW_HAND_LITE_CFG` implicit-PD actuator config, i.e. identical KP/KD) and
adds nothing to the scene beyond what `ShadowLiteEnv._setup_scene` already
spawns (hand + ground plane) -- no ball/object, nothing mobile.

What changes relative to a normal roto task:

  * The policy's 16 actions ARE torque commands, but they play TWO DIFFERENT
    roles depending on the joint (see `_pre_physics_step`):
      - 10 directly-driven joints: a small RESIDUAL correction added on top
        of PhysX's implicit PD, which itself chases the real commanded
        setpoint for these joints (`action_scale`/`residual_clip`, small).
      - 6 mechanically-coupled DOFs (FFJ1/FFJ2, MFJ1/MFJ2, RFJ1/RFJ2): the
        network's output IS the (near-)entire commanded torque, not a small
        correction. There is no reliable independent setpoint for PD to
        chase for these (D5 -- their PD-target-if-any is just the measured
        position, not a real intention), so instead of correcting an
        arguably-meaningless PD baseline, the network is given a much
        larger torque budget (`full_torque_scale`/`full_torque_clip`) and
        does the (near-)full job itself. PhysX's PD is neutralized for these
        6 (see `pd_drive_target` below), not removed -- its small velocity-
        damping term still runs alongside the network's own torque.
  * `_pre_physics_step` replays the recorded target (`dataset.q_cmd[t]`)
    into `joint_pos_cmd` for all 16 actuated joints -- unchanged, still used
    for reward/features/the "action" feature regardless of joint group. A
    SEPARATE `pd_drive_target` buffer is what's actually sent to PhysX's
    position-target register: identical to `joint_pos_cmd` for the 10 direct
    joints, but tracks the CURRENT simulated position for the 6 coupled
    joints (so PD's position-error term is ~0 for them, letting the
    network's own torque dominate instead of fighting a PD baseline chasing
    a non-causal proxy target).
  * `_apply_action` keeps roto's `set_joint_position_target` call verbatim
    (PhysX's implicit PD -- KP/KD -- is untouched either way) and ADDS one
    line, `set_joint_effort_target(residual)`, which PhysX sums with the PD
    torque -- a small correction for 10 joints, the (near-)full torque for 6.
  * The reward is UAN's multi-sharpness exponential position-tracking
    reward, comparing simulated joint position against the real measured
    trajectory, plus an optional calibration-free torque-sign-agreement term
    (see `compute_uan_reward`).
  * `_get_proprioception` returns a 58-dim default layout close to roto's own
    proprioception (`joint_pos`(16) + `joint_vel`(13) + `joint_pos_error`(16)
    + `action`(13)), all from roto's own inherited, already-normalized
    buffers, sliced to whichever index set is physically meaningful per
    feature (16 kinematic DOFs for position; 13 independent motors for
    velocity/action -- see features.py for the full reasoning).
"""

from __future__ import annotations

import gymnasium as gym
import torch

from isaaclab.utils import configclass

from roto.tasks.robots.shadowlite.shadowlite import ShadowLiteEnv, ShadowLiteEnvCfg
from roto.tasks.roto_env import unscale
from roto.tasks.uan_shadowlite.dataset import COUPLED_JOINT_PAIRS, AlignedTrajectoryDataset, DatasetKeys, TrajectoryDataset
from roto.tasks.uan_shadowlite.features import DEFAULT_FEATURES, FeatureBuilder, FeatureContext
from roto.tasks.uan_shadowlite.reward import compute_uan_reward, soft_limit_avoidance

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

_DEFAULT_REWARD_CFG = {
    "survival": 0.0,
    "l1": -1.5,
    "exp_l2_loose": 4.0,
    "coef_loose": 100.0,
    "exp_l2": 4.0,
    "coef_l2": 300.0,
    "exp_l2_strict": 5.0,
    "coef_strict": 1000.0,
    "exp_action_rate": 0.5,
    "coef_action_rate": 0.5,
    # Optional, calibration-free (sign-agreement, not magnitude) torque term.
    # 0.0 by default -- inert unless explicitly enabled in yaml. See
    # compute_uan_reward and DESIGN.md for why sign-agreement, not magnitude.
    "torque_sign": 0.0,
}

_DEFAULT_DATASET_CFG = {
    "format": "aligned",  # "aligned" (current recordings) or "legacy" (old joint_pos_cmd/joint_pos npz)
    "paths": ["PLACEHOLDER_SET_dataset.paths_IN_YAML"],
    "min_horizon": 8,
    # legacy-format-only:
    "keys": {"cmd": "joint_pos_cmd", "meas": "joint_pos", "names": "actuated_names", "ends": "episode_ends", "dt": "rl_dt"},
}

_DEFAULT_UAN_CFG = {
    "features": DEFAULT_FEATURES,
    # Applies to the 10 directly-driven joints: a small correction added on top of
    # PhysX's own PD (which chases the real commanded setpoint for these joints).
    "action_scale": 0.05,
    "residual_clip": 0.3,
    # Applies to the 6 mechanically-coupled DOFs (FFJ1/FFJ2, MFJ1/MFJ2, RFJ1/RFJ2): the
    # network's output IS the (near-)entire commanded torque for these, not a small
    # correction -- see UANShadowLiteEnv's module docstring / _pre_physics_step for why.
    # Needs a much larger range than action_scale/residual_clip since it has to span a
    # meaningful torque, not a small delta. Defaults are a conservative starting point
    # (well under the 30 N*m sim effort ceiling) -- tune once real training is underway.
    "full_torque_scale": 2.0,
    "full_torque_clip": 10.0,
    # Soft joint-limit avoidance for the 6 coupled joints (radians): outward torque
    # (pushing further past whichever limit is close) is smoothly scaled to 0 as the
    # joint gets within this margin of its lower/upper bound. Only applies to the
    # coupled joints -- the 10 direct joints still have PD tracking a real setpoint
    # within limits, which already keeps them off their bounds. See _pre_physics_step.
    "joint_limit_margin": 0.1,
    "reset_to_random": True,
    "early_terminate": False,
    "max_joint_error": 0.35,
    "tracked_joint_names": None,
    "reward": _DEFAULT_REWARD_CFG,
}


@configclass
class UANShadowLiteEnvCfg(ShadowLiteEnvCfg):
    """Config for the UAN residual-torque task.

    Inherits the full ShadowLite robot setup from `ShadowLiteEnvCfg`
    unchanged: `robot_cfg` (USD, implicit-PD actuator with KP=1.0/KD=0.1),
    `actuated_joint_names` (16), joint limits, contact sensor cfg. Nothing
    about the robot/actuator/PD is modified.
    """

    episode_length_s = 20.0
    num_eval_envs = 1

    num_actions = 16
    action_space = 16

    obs_list: list[str] = ["prop"]
    obs_stack: int = 1

    dataset: dict = _DEFAULT_DATASET_CFG
    uan: dict = _DEFAULT_UAN_CFG


# ---------------------------------------------------------------------------
# Env
# ---------------------------------------------------------------------------


class UANShadowLiteEnv(ShadowLiteEnv):
    """ShadowLite env that replays a recorded real trajectory and learns a
    residual torque to close the sim-to-real gap (see module docstring).
    """

    cfg: UANShadowLiteEnvCfg

    def __init__(self, cfg: UANShadowLiteEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        n_act = len(self.actuated_dof_indices)
        # Dataset must be name-aligned to the SORTED articulation-index order
        # (RotoEnv sorts actuated_dof_indices after building it), not
        # cfg.actuated_joint_names's literal list order.
        joint_names = [self.robot.joint_names[i] for i in self.actuated_dof_indices]

        ds_cfg = dict(_DEFAULT_DATASET_CFG, **cfg.dataset)
        fmt = ds_cfg.get("format", "aligned")
        if fmt == "aligned":
            self.dataset = AlignedTrajectoryDataset(
                paths=ds_cfg["paths"],
                joint_names=joint_names,
                device=self.device,
                min_horizon=ds_cfg.get("min_horizon", 1),
            )
        elif fmt == "legacy":
            keys = DatasetKeys(**ds_cfg.get("keys", {}))
            self.dataset = TrajectoryDataset(
                paths=ds_cfg["paths"],
                joint_names=joint_names,
                device=self.device,
                keys=keys,
                min_horizon=ds_cfg.get("min_horizon", 1),
            )
        else:
            raise ValueError(f"Unknown dataset.format '{fmt}'; expected 'aligned' or 'legacy'.")

        # 13 independently-controlled motors -- used to slice roto's own inherited
        # buffers for the "joint_vel"/"action" features (see features.py module
        # docstring for why velocity/action are motor-level, not per-kinematic-DOF).
        n_ctrl = len(self.control_dof_indices)
        self.control_pos_lower = self.robot_joint_pos_lower_limits[self.control_dof_indices]
        self.control_pos_upper = self.robot_joint_pos_upper_limits[self.control_dof_indices]

        uan_cfg = dict(_DEFAULT_UAN_CFG, **cfg.uan)
        self.feature_builder = FeatureBuilder(uan_cfg.get("features"), num_joints=n_act, num_control=n_ctrl)

        # Local indices (0..15, into the actuated_dof_indices/joint_names axis) splitting
        # the 16 joints into the 6 mechanically-coupled DOFs (network outputs the entire
        # torque) and the 10 directly-driven ones (network outputs a small PD correction).
        # Same COUPLED_JOINT_PAIRS dataset.py uses to decide their PD-target source --
        # reused here, not duplicated, so the two decisions can never drift out of sync.
        coupled_names = {n for pair in COUPLED_JOINT_PAIRS.values() for n in pair}
        name_to_local_all = {n: i for i, n in enumerate(joint_names)}
        self.coupled_local_idx = torch.tensor(
            sorted(name_to_local_all[n] for n in coupled_names), dtype=torch.long, device=self.device
        )
        self.direct_local_idx = torch.tensor(
            sorted(i for n, i in name_to_local_all.items() if n not in coupled_names), dtype=torch.long, device=self.device
        )

        # action_scale/residual_clip only ever apply at the 10 direct-joint indices;
        # full_torque_scale/full_torque_clip only ever apply at the 6 coupled indices --
        # see _pre_physics_step for where these get combined into one 16-dim output.
        self.action_scale = self._broadcast_per_joint(uan_cfg.get("action_scale", 0.05), len(self.direct_local_idx))
        self.residual_clip = self._broadcast_per_joint(uan_cfg.get("residual_clip", 0.3), len(self.direct_local_idx))
        self.full_torque_scale = self._broadcast_per_joint(
            uan_cfg.get("full_torque_scale", 2.0), len(self.coupled_local_idx)
        )
        self.full_torque_clip = self._broadcast_per_joint(
            uan_cfg.get("full_torque_clip", 10.0), len(self.coupled_local_idx)
        )
        # Cached joint limits at the 6 coupled indices (soft_joint_pos_limit_factor=1.0
        # in SHADOW_HAND_LITE_CFG, so these are the joint's full physical range) -- used
        # by the joint-limit safety envelope in _pre_physics_step. Necessary because PD's
        # own restoring force is neutralized for these joints (pd_drive_target tracks
        # current position), so nothing else pulls them back from a limit if the
        # network's torque pushes outward -- PhysX's own limit enforcement is a soft
        # constraint under continuous torque, not a rigid wall, and is not sufficient
        # on its own once PD is no longer helping.
        self.coupled_pos_lower = self.robot_joint_pos_lower_limits[self.actuated_dof_indices][self.coupled_local_idx]
        self.coupled_pos_upper = self.robot_joint_pos_upper_limits[self.actuated_dof_indices][self.coupled_local_idx]
        self.joint_limit_margin = float(uan_cfg.get("joint_limit_margin", 0.1))
        self.reset_to_random = bool(uan_cfg.get("reset_to_random", True))
        self.early_terminate = bool(uan_cfg.get("early_terminate", False))
        self.max_joint_error = float(uan_cfg.get("max_joint_error", 0.35))

        tracked_names = uan_cfg.get("tracked_joint_names", None)
        if tracked_names is None:
            self.tracked_idx = torch.arange(n_act, device=self.device)
        else:
            name_to_local = {n: i for i, n in enumerate(joint_names)}
            missing = [n for n in tracked_names if n not in name_to_local]
            if missing:
                raise KeyError(f"uan.tracked_joint_names contains unknown joint(s): {missing}")
            self.tracked_idx = torch.tensor([name_to_local[n] for n in tracked_names], dtype=torch.long, device=self.device)

        rew = dict(_DEFAULT_REWARD_CFG, **uan_cfg.get("reward", {}))
        self.rew_survival = float(rew["survival"])
        self.rew_l1 = float(rew["l1"])
        self.rew_exp_l2_loose = float(rew["exp_l2_loose"])
        self.coef_loose = float(rew["coef_loose"])
        self.rew_exp_l2 = float(rew["exp_l2"])
        self.coef_l2 = float(rew["coef_l2"])
        self.rew_exp_l2_strict = float(rew["exp_l2_strict"])
        self.coef_strict = float(rew["coef_strict"])
        self.rew_exp_action_rate = float(rew["exp_action_rate"])
        self.coef_action_rate = float(rew["coef_action_rate"])
        self.rew_torque_sign = float(rew.get("torque_sign", 0.0))

        self.traj_t = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.residual = torch.zeros(self.num_envs, n_act, device=self.device)
        self.last_residual = torch.zeros(self.num_envs, n_act, device=self.device)
        self.last_actions = torch.zeros(self.num_envs, self.cfg.num_actions, device=self.device)
        # What PhysX's implicit PD actually chases -- differs from joint_pos_cmd (which
        # stays the real/measured target used for reward & features, unchanged) only at
        # the 6 coupled indices, where it tracks current position instead (see
        # _pre_physics_step) so the network's own torque output dominates there.
        self.pd_drive_target = torch.zeros(self.num_envs, n_act, device=self.device)

        self.traj_t[:] = self.dataset.sample_start_indices(self.num_envs)

    def _broadcast_per_joint(self, value, n: int) -> torch.Tensor:
        if isinstance(value, (list, tuple)):
            if len(value) != n:
                raise ValueError(f"Expected {n} per-joint values, got {len(value)}: {value}")
            return torch.tensor(value, dtype=torch.float32, device=self.device)
        return torch.full((n,), float(value), dtype=torch.float32, device=self.device)

    # ------------------------------------------------------------------
    # Control
    # ------------------------------------------------------------------

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.prev_joint_pos_cmd[:] = self.joint_pos_cmd
        self.actions = actions.clone()
        self.last_residual[:] = self.residual

        t = self.dataset.clamp(self.traj_t)
        # joint_pos_cmd stays the real/measured target for ALL 16 joints, unchanged --
        # this is what reward/joint_pos_error/the "action" feature all read. It is NOT
        # necessarily what PhysX's PD drive chases (see pd_drive_target below).
        self.joint_pos_cmd[:, self.actuated_dof_indices] = self.dataset.q_cmd[t]

        # Two different roles for the network's 16 outputs, split by joint group:
        #  - 10 direct joints: a small correction added on top of PD-toward-real-setpoint
        #    (unchanged from before).
        #  - 6 coupled joints (FFJ1/FFJ2, MFJ1/MFJ2, RFJ1/RFJ2): the network's own output
        #    IS the (near-)entire commanded torque -- there is no reliable independent
        #    setpoint for these to task PD with chasing (D5), so instead of a small
        #    residual on top of a PD-toward-measured-position baseline, the network gets
        #    a much larger torque budget and does the (near-)full job itself.
        torque = torch.zeros(self.num_envs, len(self.actuated_dof_indices), device=self.device)
        torque[:, self.direct_local_idx] = torch.clamp(
            self.actions[:, self.direct_local_idx] * self.action_scale, -self.residual_clip, self.residual_clip
        )
        torque[:, self.coupled_local_idx] = torch.clamp(
            self.actions[:, self.coupled_local_idx] * self.full_torque_scale,
            -self.full_torque_clip,
            self.full_torque_clip,
        )

        # Soft joint-limit avoidance, coupled joints only: PD's restoring force is
        # neutralized for these (pd_drive_target below), so nothing else pulls them back
        # from a limit if the network's torque pushes outward, and PhysX's own limit
        # enforcement is a soft constraint under continuous torque, not guaranteed to
        # hold on its own. See soft_limit_avoidance()'s own docstring for the math.
        coupled_pos = self.joint_pos[:, self.actuated_dof_indices][:, self.coupled_local_idx]
        torque[:, self.coupled_local_idx] = soft_limit_avoidance(
            torque[:, self.coupled_local_idx],
            coupled_pos,
            self.coupled_pos_lower,
            self.coupled_pos_upper,
            self.joint_limit_margin,
        )

        self.residual = torque

        # pd_drive_target: what actually gets sent to PhysX's position-target buffer.
        #  - 10 direct joints: the real setpoint (= joint_pos_cmd), same as before.
        #  - 6 coupled joints: CURRENT simulated position, captured once here (before
        #    this step's physics runs, consistent with joint_pos_cmd/pd_drive_target both
        #    being fixed for the whole decimation loop, not re-measured per substep).
        #    This makes PD's position-error term (KP*(target-pos)) ~0 for these joints,
        #    leaving only its velocity-damping term (-KD*vel) still active alongside the
        #    network's own torque -- not perfectly zero PD contribution, but close, and
        #    the residual damping is a reasonable (arguably helpful, stabilizing) leftover
        #    rather than something that needs eliminating via a roto/actuator-cfg edit.
        self.pd_drive_target = self.joint_pos_cmd[:, self.actuated_dof_indices].clone()
        self.pd_drive_target[:, self.coupled_local_idx] = self.joint_pos[:, self.actuated_dof_indices][
            :, self.coupled_local_idx
        ]

    def _apply_action(self) -> None:
        # Same set_joint_position_target call as roto's own RotoEnv._apply_action -- only
        # the VALUE being sent differs now (pd_drive_target, not joint_pos_cmd directly)
        # for the 6 coupled joints; PhysX's implicit PD (KP=1.0, KD=0.1) itself is
        # completely untouched either way.
        self.robot.set_joint_position_target(self.pd_drive_target, joint_ids=self.actuated_dof_indices)
        # The only new physics call: PhysX additively sums this into the actuation force
        # alongside the implicit PD torque -- a small correction for the 10 direct
        # joints, the (near-)entire torque for the 6 coupled ones (see _pre_physics_step).
        self.robot.set_joint_effort_target(self.residual, joint_ids=self.actuated_dof_indices)

    # ------------------------------------------------------------------
    # Episode lifecycle
    # ------------------------------------------------------------------

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        self._compute_intermediate_values()

        t = self.dataset.clamp(self.traj_t)
        truncated = self.dataset.is_at_boundary(t) | (self.episode_length_buf >= self.max_episode_length - 1)

        if self.early_terminate:
            q_sim = self.joint_pos[:, self.actuated_dof_indices]
            q_cmd_t = self.dataset.q_cmd[t]
            err = (q_cmd_t - q_sim).abs()
            terminated = (err > self.max_joint_error).any(dim=1)
        else:
            terminated = torch.zeros_like(truncated)

        return terminated, truncated

    def _get_rewards(self) -> torch.Tensor:
        t = self.dataset.clamp(self.traj_t)
        q_sim = self.joint_pos[:, self.actuated_dof_indices][:, self.tracked_idx]
        q_real = self.dataset.q_meas[t][:, self.tracked_idx]

        # Total applied torque in sim = PD (roto's own bookkeeping) + our
        # manually-injected residual (D1: injected via a separate effort-
        # target write that bypasses the actuator model, so it does NOT show
        # up in robot.data.applied_torque on its own).
        total_torque_sim = self.robot.data.applied_torque[:, self.actuated_dof_indices] + self.residual
        torque_real = self.dataset.q_torque[t]

        reward, se_sum, ae_sum = compute_uan_reward(
            q_real,
            q_sim,
            self.actions,
            self.last_actions,
            total_torque_sim,
            torque_real,
            self.rew_survival,
            self.rew_l1,
            self.rew_exp_l2_loose,
            self.coef_loose,
            self.rew_exp_l2,
            self.coef_l2,
            self.rew_exp_l2_strict,
            self.coef_strict,
            self.rew_exp_action_rate,
            self.coef_action_rate,
            self.rew_torque_sign,
        )

        # Split, not averaged together: direct-joint correction (~0.3 max) and coupled-
        # joint full torque (~10.0 max) are on very different scales, so one combined
        # mean would be uninterpretable.
        self.extras["log"] = {
            "pos_l1": ae_sum.clone(),
            "pos_rmse": torch.sqrt(se_sum / self.tracked_idx.numel()).clone(),
            "mean_abs_residual_direct": self.residual[:, self.direct_local_idx].abs().mean(dim=1).clone(),
            "mean_abs_torque_coupled": self.residual[:, self.coupled_local_idx].abs().mean(dim=1).clone(),
        }

        self.last_actions[:] = self.actions
        self.traj_t += 1
        return reward

    def _reset_idx(self, env_ids) -> None:
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        super()._reset_idx(env_ids)
        self._reset_to_trajectory(env_ids)

    def _reset_to_trajectory(self, env_ids: torch.Tensor) -> None:
        n = len(env_ids)
        if self.reset_to_random:
            new_t = self.dataset.sample_start_indices(n)
        else:
            new_t = self.dataset.traj_starts[0].expand(n).clone()

        eval_mask = env_ids < self.cfg.num_eval_envs
        if eval_mask.any():
            new_t = new_t.clone()
            new_t[eval_mask] = self.dataset.traj_starts[0]

        self.traj_t[env_ids] = new_t
        t = self.dataset.clamp(new_t)

        q0 = self.dataset.q_meas[t]
        qd0 = self.dataset.q_meas_vel[t]

        full_pos = self.robot.data.joint_pos[env_ids].clone()
        full_pos[:, self.actuated_dof_indices] = q0
        full_vel = torch.zeros_like(full_pos)
        full_vel[:, self.actuated_dof_indices] = qd0

        self.robot.write_joint_state_to_sim(full_pos, full_vel, env_ids=env_ids)
        self.robot.set_joint_position_target(full_pos, env_ids=env_ids)
        self.joint_pos_cmd[env_ids] = full_pos

        self.residual[env_ids] = 0.0
        self.last_residual[env_ids] = 0.0
        self.actions[env_ids] = 0.0
        self.last_actions[env_ids] = 0.0

    # ------------------------------------------------------------------
    # Observations
    # ------------------------------------------------------------------

    def _get_proprioception(self) -> torch.Tensor:
        # Defensive refresh: guarantees self.joint_pos/vel/normalised_*/
        # joint_pos_error reflect the state written by _reset_to_trajectory
        # even if this runs immediately after a reset.
        self._compute_intermediate_values()

        t = self.dataset.clamp(self.traj_t)

        # The 13-motor "action" feature: back-solve, via roto's own unscale() (the
        # exact inverse of scale()), what raw policy action WOULD have produced the
        # current joint_pos_cmd at the 13 control-level joints. During UAN's own
        # training joint_pos_cmd comes from dataset.q_cmd (replayed real data); during
        # a future downstream-policy embedding it would come from that policy's own
        # scale(self.actions) call instead -- either way this recovers the same
        # roughly-[-1,1] action-space representation, so the network's input contract
        # doesn't change between the two contexts. NOT self.actions (UAN's own 16-dim
        # residual-torque output) -- a different quantity entirely.
        control_action_t = unscale(
            self.joint_pos_cmd[:, self.control_dof_indices], self.control_pos_lower, self.control_pos_upper
        )

        ctx = FeatureContext(
            # roto's OWN normalized buffers (RotoEnv._compute_intermediate_values,
            # unscale() to [-1,1] via the robot's joint limits) -- "normalize as
            # normalized in roto" means reusing these directly, not recomputing.
            joint_pos=self.normalised_joint_pos[:, self.actuated_dof_indices],
            # Motor-level (13), not per-kinematic-DOF (16) -- see features.py.
            joint_vel=self.normalised_joint_vel[:, self.control_dof_indices],
            # roto's own raw joint_pos_error (joint_pos_cmd - joint_pos, unnormalized).
            joint_pos_error=self.joint_pos_error[:, self.actuated_dof_indices],
            action=control_action_t,
            last_residual=self.last_residual,
            real_vel_t=self.dataset.q_meas_vel[t],
            traj_phase=self.dataset.traj_progress(t),
        )
        return self.feature_builder.build(ctx)


# compute_uan_reward and soft_limit_avoidance now live in reward.py (Isaac-free, so they
# can be unit-tested on CPU without booting Isaac Sim) -- imported at the top of this file.

# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

gym.register(
    id="UAN_Shadowlite",
    entry_point=f"{UANShadowLiteEnv.__module__}:{UANShadowLiteEnv.__name__}",
    disable_env_checker=True,
    kwargs={"env_cfg_entry_point": UANShadowLiteEnvCfg},
)

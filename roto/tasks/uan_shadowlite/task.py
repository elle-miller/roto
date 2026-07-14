"""UAN (Unsupervised Actuator Network) task for the ShadowLite hand.

Registers the gym environment id "UAN_Shadowlite". `UANShadowLiteEnv`
subclasses roto's `ShadowLiteEnv` unchanged (same robot USD, same
`SHADOW_HAND_LITE_CFG` implicit-PD actuator config, i.e. identical KP/KD) and
adds nothing to the scene beyond what `ShadowLiteEnv._setup_scene` already
spawns (hand + ground plane) -- no ball/object, nothing mobile.

What changes relative to a normal roto task:

  * The policy's 16 actions are a small RESIDUAL torque correction added on
    top of PhysX's implicit PD, uniformly for all 16 actuated joints
    (`action_scale`/`residual_clip`). PD itself is completely untouched
    (same KP/KD as every other roto task) and chases a real, physically
    in-range setpoint for every joint -- including the 6 mechanically-
    coupled DOFs (FFJ1/FFJ2, MFJ1/MFJ2, RFJ1/RFJ2), whose shared motor only
    records one combined command on real hardware; `dataset.py` splits that
    single value back into individual J1/J2 targets (see its DESIGN NOTE),
    so PD has a genuine, always-in-bounds target to chase there too, just
    like the 10 directly-driven joints. This is why joint limits need no
    separate handling here: PD's own restoring force keeps every joint
    in-range as a side effect of chasing an in-range target, the same way
    it always has for the 10 direct joints.

    (An earlier revision instead gave the network the (near-)entire torque
    for the 6 coupled joints and neutralized PD's target for them, reasoning
    that no real per-joint setpoint existed to chase. That turned out to
    remove PD's only restoring force near a limit, causing real joint-limit
    violations; a `soft_limit_avoidance` safety envelope patched the
    symptom, but splitting the coupled joints' real command into legitimate
    per-joint setpoints -- so PD keeps working as designed -- fixes the
    cause instead, and is simpler: one control law for all 16 joints.)
  * `_pre_physics_step` replays the recorded target (`dataset.q_cmd[t]`)
    into `joint_pos_cmd` for all 16 actuated joints; `_apply_action` sends
    it straight to PhysX's position-target register (`set_joint_position_
    target`, identical to roto's own `RotoEnv._apply_action`) and adds one
    line, `set_joint_effort_target(residual)`, which PhysX sums with the PD
    torque (D1).
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

from roto.assets.path_resolve import resolve_path
from roto.tasks.robots.shadowlite.shadowlite import ShadowLiteEnv, ShadowLiteEnvCfg
from roto.tasks.roto_env import unscale
from roto.tasks.uan_shadowlite.dataset import COUPLED_JOINT_PAIRS, AlignedTrajectoryDataset, DatasetKeys, TrajectoryDataset
from roto.tasks.uan_shadowlite.features import DEFAULT_FEATURES, FeatureBuilder, FeatureContext
from roto.tasks.uan_shadowlite.reward import compute_uan_reward

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
    "torque_sign": 10.0,
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
    # Small correction added on top of PhysX's own PD, which chases the real commanded
    # setpoint for every one of the 16 actuated joints (direct AND coupled -- see
    # dataset.py's DESIGN NOTE for how the 6 coupled joints get a real, physically
    # in-range setpoint too, by splitting their shared motor's combined command).
    "action_scale": 0.5,
    "residual_clip": 5,
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
        # yaml hardcodes one machine's absolute paths; resolve_path retries every known
        # roto checkout root so the same yaml works unmodified on any of them (see
        # roto/assets/path_resolve.py).
        raw_paths = ds_cfg["paths"]
        ds_cfg["paths"] = [resolve_path(p) for p in raw_paths] if isinstance(raw_paths, list) else resolve_path(raw_paths)
        fmt = ds_cfg.get("format", "aligned")
        if fmt == "aligned":
            # The 6 coupled joints' shared motor only records one combined command;
            # AlignedTrajectoryDataset splits it into individual J1/J2 targets using
            # each joint's own physical upper limit as the split point (see its
            # DESIGN NOTE) -- read straight from the loaded articulation, so the
            # split uses exactly the same bounds the sim itself enforces.
            coupled_joint_upper_limits = {
                n: float(self.robot_joint_pos_upper_limits[self.robot.joint_names.index(n)])
                for pair in COUPLED_JOINT_PAIRS.values()
                for n in pair
            }
            self.dataset = AlignedTrajectoryDataset(
                paths=ds_cfg["paths"],
                joint_names=joint_names,
                device=self.device,
                joint_upper_limits=coupled_joint_upper_limits,
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

        # Same small-residual treatment for all 16 actuated joints, direct or coupled --
        # PD (unchanged KP/KD) chases a real, physically in-range setpoint for every one
        # of them (dataset.q_cmd; see dataset.py's DESIGN NOTE for the 6 coupled joints).
        self.action_scale = self._broadcast_per_joint(uan_cfg.get("action_scale", 0.05), n_act)
        self.residual_clip = self._broadcast_per_joint(uan_cfg.get("residual_clip", 0.3), n_act)
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
        # joint_pos_cmd is the real target for ALL 16 joints -- both what PhysX's PD
        # drive chases (_apply_action) and what reward/joint_pos_error/the "action"
        # feature read. For the 6 coupled joints this is dataset.py's split of their
        # shared motor's combined command, not a measured position -- see dataset.py's
        # DESIGN NOTE.
        self.joint_pos_cmd[:, self.actuated_dof_indices] = self.dataset.q_cmd[t]

        # Small residual correction on top of PD, uniform across all 16 joints -- PD
        # itself (unchanged KP/KD) already chases a real, physically in-range setpoint
        # for every joint, so it keeps its own restoring force everywhere, direct or
        # coupled, with no separate joint-limit safety net required.
        self.residual = torch.clamp(self.actions * self.action_scale, -self.residual_clip, self.residual_clip)

    def _apply_action(self) -> None:
        # Same set_joint_position_target call as roto's own RotoEnv._apply_action; PhysX's
        # implicit PD (KP=1.0, KD=0.1) is completely untouched.
        self.robot.set_joint_position_target(
            self.joint_pos_cmd[:, self.actuated_dof_indices], joint_ids=self.actuated_dof_indices
        )
        # The only new physics call: PhysX additively sums this small residual into the
        # actuation force alongside the implicit PD torque (D1).
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

        self.extras["log"] = {
            "pos_l1": ae_sum.clone(),
            "pos_rmse": torch.sqrt(se_sum / self.tracked_idx.numel()).clone(),
            "mean_abs_residual": self.residual.abs().mean(dim=1).clone(),
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


# compute_uan_reward now lives in reward.py (Isaac-free, so it can be unit-tested on CPU
# without booting Isaac Sim) -- imported at the top of this file.

# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

gym.register(
    id="UAN_Shadowlite",
    entry_point=f"{UANShadowLiteEnv.__module__}:{UANShadowLiteEnv.__name__}",
    disable_env_checker=True,
    kwargs={"env_cfg_entry_point": UANShadowLiteEnvCfg},
)

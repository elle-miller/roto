#!/usr/bin/env python3
"""Fit a deployment-time torque scale for rh_FFJ3/rh_FFJ1/rh_FFJ2, so the
SIMULATED joint trajectory matches the REAL recorded trajectory over a full
episode.

Up to three single-joint checkpoints are loaded: rh_FFJ3
(`train_genan_single.py --residual_torque --torque_range ...`) and
rh_FFJ1/rh_FFJ2, now trained as TWO FULLY INDEPENDENT single-output models
(`train_genan_pair_independent.py`) rather than one shared-trunk two-head
network -- each network's own output has no training-time relationship to
the other's (see that script's module docstring), so each of the three
joints legitimately gets its OWN independently-fit scale here, with no
shared-scale constraint anywhere in this script. `UANShadowLiteEnv`
(`roto/tasks/uan_shadowlite/task.py`) already treats its 16-dim `actions` as
a small residual torque ADDED ON TOP of PhysX's own always-active implicit
PD (`_apply_action` calls both `set_joint_position_target` -- driving PD --
and `set_joint_effort_target(residual)` -- see that file's module docstring)
-- exactly the mechanism this script needs, with NO stiffness-zeroing
required (unlike `play_genan.py`, whose checkpoint predicts ABSOLUTE torque
and so must remove PD's competing term). Per user decision: `scale_j *
<network's own RAW tanh output, in (-1,1) -- NOT de-normalized by the
checkpoint's own torque_range>` is injected per joint, so `scale_j` itself
directly IS the max N*m of extra torque injectable for that joint, without
hand-deriving whether the model's absolute-torque formulation double-counts
against PD -- `scale_j` is fit empirically against real trajectory RMSE, so
it absorbs any such mismatch.

Only the joints with a LOADED checkpoint get nonzero `actions`; every other
of the 16 actuated joints gets exactly 0, i.e. pure PhysX PD tracking of the
real command. All active joints are rolled out TOGETHER in the same single
episode/simulation and reported together in one combined plot (one panel per
active joint).

Scale search: EVERY active joint uses the same `scipy.optimize.minimize_scalar
(method="bounded")` (Brent-style golden-section+parabolic 1-D search over
`--bounds`), but each joint's search runs in FULL ISOLATION: only that
joint's own checkpoint is loaded into the rollout at all during its search --
every other joint's model is entirely absent (not merely scale=0), so a
candidate scale can't be scored on a trajectory contaminated by another
joint's torque. This matters beyond just avoiding a jointly-learned-ratio
problem (Findings from this script's development history): rh_FFJ1/rh_FFJ2
are mechanically the SAME finger's linked joints, so even with independent
training and independent torque injection, one joint's torque can physically
drag the other's POSITION through the rigid-body chain regardless of what's
injected into it -- observed directly when scale_ffj1=50 (well outside
--bounds) caused rh_FFJ2's simulated position to blow out to +9/-15 rad in a
COMBINED rollout, despite rh_FFJ2 itself receiving zero torque. Isolating
each joint's search rollout removes that contamination from the FITTING
process entirely; the three joints' searches are now fully independent of
each other (order doesn't matter). Each candidate scale is scored by
averaging over MULTIPLE real trajectory segments (`--fit_traj_idx`, default a
small deterministic subset of all available segments -- see `main()`),
combining position RMSE with a velocity RMSE term (`--velocity_weight`,
finite difference of the already time-aligned q_sim/q_real arrays) so a scale
that merely has a low time-averaged position error but chatters/oscillates
isn't mistaken for a good fit.

After all scales are independently fitted, a separate COMBINED rollout (all
active joints' models loaded and driven TOGETHER, using their independently-
fit scales) is run for reporting on both `--fit_traj_idx` and `--test_traj_idx`
(held-out) -- this is the actual deployment scenario (all networks run
simultaneously in reality) and the place any REMAINING cross-joint coupling
issue (e.g. the mechanical-coupling effect above) would still show up, even
though it no longer contaminates the fitting itself.

Usage:
    python fit_torque_scale.py --checkpoint_ffj3 ffj3.pt \\
        --checkpoint_ffj1 ffj1.pt --checkpoint_ffj2 ffj2.pt --headless
    python fit_torque_scale.py --checkpoint_ffj1 ffj1.pt --checkpoint_ffj2 ffj2.pt \\
        --scale_ffj1 1.0 --scale_ffj2 1.0 --headless  # single rollout, no sweep
"""

import argparse
import os
import sys

# See play_genan.py's matching comment for why this matters at Isaac Sim shutdown.
sys.stdout.reconfigure(line_buffering=True)

from isaaclab.app import AppLauncher

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_ROTO_ROOT = os.path.dirname(_THIS_DIR)
_GENAN_DIR = os.path.join(_ROTO_ROOT, "genan")

parser = argparse.ArgumentParser(description="Fit deployment-time torque scale for rh_FFJ3/rh_FFJ1/rh_FFJ2.")
parser.add_argument("--checkpoint_ffj3", type=str, default=None, help="Single-joint rh_FFJ3 checkpoint. Omit to skip.")
parser.add_argument("--checkpoint_ffj1", type=str, default=None, help="Single-joint rh_FFJ1 checkpoint (independent model). Omit to skip.")
parser.add_argument("--checkpoint_ffj2", type=str, default=None, help="Single-joint rh_FFJ2 checkpoint (independent model). Omit to skip.")
parser.add_argument(
    "--checkpoint_pair", type=str, default=None,
    help="Two-output rh_FFJ1/rh_FFJ2 checkpoint (train_genan_pair.py's shared-trunk model), as an ALTERNATIVE "
         "to --checkpoint_ffj1/--checkpoint_ffj2. Its two output columns get scored/scaled via --scale_ffj1/"
         "--scale_ffj2 same as the independent case. Cannot be combined with --checkpoint_ffj1/--checkpoint_ffj2.",
)
parser.add_argument(
    "--config", type=str, default=os.path.join(_GENAN_DIR, "agents", "shadowlite", "default.yaml"),
    help="Agent yaml providing dataset.paths (genan/sweeper sections are unused here).",
)
parser.add_argument("--dataset", type=str, action="append", default=None, help="Override dataset.paths (repeatable).")
parser.add_argument("--residual_clip", type=float, default=1000.0, help="uan.residual_clip override (see play_genan.py).")
parser.add_argument(
    "--bounds", type=float, nargs=2, default=[0.0, 30.0],
    help="Search bounds for each joint's scale. Default matches shadow_hand_lite.py's "
         "effort_limit_sim=30.0 N*m -- scale is injected as ADDITIONAL effort on top of "
         "PD, so anything above the joint's own effort limit guarantees actuator "
         "saturation (bang-bang/relay oscillation), not a genuinely better fit.",
)
parser.add_argument(
    "--scale_ffj3", type=float, default=None,
    help="Skip the sweep and run ONE rollout at these fixed scales (required for each active joint together).",
)
parser.add_argument("--scale_ffj1", type=float, default=None)
parser.add_argument("--scale_ffj2", type=float, default=None)
parser.add_argument("--traj_idx", type=int, default=0, help="Trajectory segment for --scale_* single-rollout mode.")
parser.add_argument(
    "--velocity_weight", type=float, default=1.0,
    help="Weight on the velocity-RMSE term added to each joint's position RMSE to form its combined "
         "search score (joint_trajectory_score). No confidently-tuned default exists yet -- adjust and compare.",
)
parser.add_argument(
    "--fit_traj_idx", type=int, action="append", default=None,
    help="Trajectory segment(s) to fit scales on (repeatable). Default: first min(4, n_segments-1) segments "
         "-- capped at a small constant, not a percentage, since each candidate scale costs a full rollout "
         "PER fit segment (see main()).",
)
parser.add_argument(
    "--test_traj_idx", type=int, action="append", default=None,
    help="Held-out trajectory segment(s), never used for fitting, evaluated afterward (repeatable). "
         "Default: up to 2 segments right after the fit set.",
)
parser.add_argument("--out_dir", type=str, default=os.path.join(_ROTO_ROOT, "genan_scale_fit"), help="Output plot/table dir.")
# NOTE: --device is intentionally NOT defined here -- AppLauncher.add_app_launcher_args() below already registers it.

AppLauncher.add_app_launcher_args(parser)
args_cli, _unused = parser.parse_known_args()
args_cli.num_envs = 1
args_cli.video = False
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import numpy as np  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402
from scipy.optimize import minimize_scalar  # noqa: E402

sys.path.insert(0, _THIS_DIR)
sys.path.insert(0, _ROTO_ROOT)
sys.path.insert(0, _GENAN_DIR)
from common_utils import LOG_PATH, make_env, update_env_cfg  # noqa: E402
from multimodal_rl.tools.writer import Writer  # noqa: E402

from history import build_delta_history  # noqa: E402
from model import GenANEnsemble  # noqa: E402

from roto.tasks import uan_shadowlite  # noqa: E402,F401
from roto.tasks.uan_shadowlite.task import UANShadowLiteEnvCfg  # noqa: E402

JOINTS = ("rh_FFJ3", "rh_FFJ1", "rh_FFJ2")


def build_env(agent_cfg: dict, residual_clip: float):
    env_cfg = UANShadowLiteEnvCfg()
    env_cfg.dataset = dict(agent_cfg["dataset"])
    env_cfg.uan = dict(agent_cfg.get("uan", {}))
    # actions ARE the residual torque added on top of PD (task.py's own
    # design, module docstring) -- action_scale=1.0 + a large residual_clip
    # makes `residual == actions`, same convention as play_genan.py.
    env_cfg.uan["action_scale"] = 1.0
    env_cfg.uan["residual_clip"] = residual_clip
    env_cfg.uan["reset_to_random"] = False  # deterministic: always start at the trajectory start
    env_cfg.num_eval_envs = 0

    writer = Writer(agent_cfg, play=True)
    env_cfg = update_env_cfg(args_cli, env_cfg, agent_cfg)

    args_cli.task = "UAN_Shadowlite"
    args_cli.gym_env_id = "UAN_Shadowlite"
    env = make_env(agent_cfg, env_cfg, writer, args_cli)
    return env


def load_single_ensemble(checkpoint_path: str, device) -> tuple[GenANEnsemble, dict]:
    """Loads any single-joint, bounded-output checkpoint -- rh_FFJ3's
    (`train_genan_single.py`) or an independently-trained rh_FFJ1/rh_FFJ2
    (`train_genan_pair_independent.py`); both share the same checkpoint
    schema (`num_joints=1`, `torque_range`, `bounded_output=True`), so one
    generic loader covers all three joints.
    """
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    torque_range = ckpt["torque_range"]
    ensemble = GenANEnsemble(
        ckpt["input_dim"], ckpt["num_joints"], ensemble_size=ckpt["ensemble_size"],
        bounded_output=True, torque_range=torque_range,
    )
    ensemble.load_state_dict(ckpt["ensemble_state_dict"])
    ensemble.eval()
    ensemble.to(device)
    return ensemble, ckpt


def load_pair_ensemble(checkpoint_path: str, device) -> tuple[GenANEnsemble, dict]:
    """Loads a two-output rh_FFJ1/rh_FFJ2 checkpoint (`train_genan_pair.py`'s
    shared-trunk model, `num_joints=2`) -- a DIFFERENT checkpoint schema from
    `load_single_ensemble`'s (`num_joints=1`, `joint_idx`/`joint_name`); this
    one has `joint_pair_idx`/`joint_pair_names` instead. Both schemas store
    `joint_names` (the full 16-joint training order), so the same
    `rollout()` joint-order permutation fix applies unchanged either way.
    """
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    torque_range = ckpt["torque_range"]
    ensemble = GenANEnsemble(
        ckpt["input_dim"], ckpt["num_joints"], ensemble_size=ckpt["ensemble_size"],
        bounded_output=True, torque_range=torque_range,
    )
    ensemble.load_state_dict(ckpt["ensemble_state_dict"])
    ensemble.eval()
    ensemble.to(device)
    return ensemble, ckpt


def _history_from_buffer(buffer: list[torch.Tensor], history_len: int, stride: int) -> torch.Tensor:
    """Identical to play_genan.py's `_history_from_buffer` -- duplicated
    (not imported) so this script has no import-time dependency on
    play_genan.py's module-level AppLauncher side effects, matching this
    repo's established convention of small, independent script-local copies
    (see train_genan_single.py's `split_segments` docstring for the same
    rationale).
    """
    frames = []
    for k in range(history_len + 1):
        idx = max(0, len(buffer) - 1 - k * stride)
        frames.append(buffer[idx])
    current = frames[0]
    deltas = [current] + [f - current for f in frames[1:]]
    return torch.cat(deltas, dim=-1)


def reset_to_segment(env, traj_idx: int) -> None:
    """Reset to trajectory segment `traj_idx`'s start, mirroring
    `uan_shadowlite/task.py`'s `_reset_to_trajectory` (lines 348-373) but
    parameterized by segment index instead of hardcoded to segment 0 (that
    file's own non-random-reset default, `reset_to_random=False`). task.py
    stays unmodified -- this is a small, self-contained override scoped to
    this script (same convention as `_history_from_buffer`'s duplication,
    see its docstring). Overwriting ONLY `traj_t` after `env.reset(hard=True)`
    would leave the robot physically parked at segment 0's start position
    while q_cmd/q_meas lookups jump to a different segment's timeline -- the
    joint state must be rewritten too, via the same `write_joint_state_to_sim`
    call task.py uses.
    """
    env.reset(hard=True)
    unwrapped = env.unwrapped
    t0 = unwrapped.dataset.traj_starts[traj_idx].expand(env.num_envs).clone()
    unwrapped.traj_t[:] = t0
    t = unwrapped.dataset.clamp(t0)
    q0 = unwrapped.dataset.q_meas[t]
    qd0 = unwrapped.dataset.q_meas_vel[t]
    full_pos = unwrapped.robot.data.joint_pos.clone()
    full_pos[:, unwrapped.actuated_dof_indices] = q0
    full_vel = torch.zeros_like(full_pos)
    full_vel[:, unwrapped.actuated_dof_indices] = qd0
    unwrapped.robot.write_joint_state_to_sim(full_pos, full_vel)
    unwrapped.robot.set_joint_position_target(full_pos)


def rollout(
    env, ensembles: dict[str, tuple[GenANEnsemble, dict, int]], joint_idx: dict,
    traj_idx: int, num_steps: int, scales: dict[str, float],
):
    """Roll out `num_steps` of trajectory segment `traj_idx` (via
    `reset_to_segment`) with `actions[joint_idx[name]] = scales[name] * <that
    joint's own RAW tanh output column, in (-1,1)>` for every joint present
    in `ensembles`, and 0 for every other of the 16 actuated joints (pure
    PhysX PD there). `ensembles` maps joint name -> (ensemble, ckpt,
    output_col) -- `output_col` picks which output column of `ensemble`'s
    prediction belongs to this joint: `0` for every single-joint checkpoint
    (`load_single_ensemble`, `num_joints=1`), or `0`/`1` for the two SEPARATE
    dict entries a two-output pair checkpoint (`load_pair_ensemble`,
    `num_joints=2`) contributes -- both of THOSE entries share the exact
    same `ensemble`/`ckpt` object (one forward pass's two columns), each
    with its OWN independent `scales[name]`. All loaded models are driven
    TOGETHER in this one rollout (same episode, same simulation). Returns
    (q_sim, q_real), each (num_steps, 16).

    CRITICAL joint-order fix: `unwrapped.dataset.q_cmd`/`q_meas` and
    `unwrapped.joint_pos[:, unwrapped.actuated_dof_indices]` are ALL ordered
    per this SIM's own `actuated_dof_indices` (task.py:152 -- the sorted
    articulation-index order, NOT hardware order), but every checkpoint here
    was trained on `genan/joint_config.py`'s `hardware_joint_order` (stored
    verbatim as `ckpt["joint_names"]`) -- a DIFFERENT permutation of the same
    16 joints (verified: e.g. rh_FFJ1 is column 11 in sim order, column 0 in
    training order). Feeding the network raw sim-order data, as this
    function previously did, silently scrambles every input feature relative
    to what the network learned. Each ensemble gets its OWN permutation
    (`perms[name]`, sim-order index -> that checkpoint's own training-order
    position) applied ONLY when building ITS raw_input -- `q_sim_log`/
    `q_real_log` (used for RMSE/plotting via `joint_idx`, which is itself in
    sim order) are completely unaffected, this fix is scoped to the
    network's input construction only.
    """
    unwrapped = env.unwrapped
    joint_names_env = [unwrapped.robot.joint_names[i] for i in unwrapped.actuated_dof_indices]
    perms = {name: torch.tensor([joint_names_env.index(n) for n in ckpt["joint_names"]], dtype=torch.long)
              for name, (_, ckpt, _) in ensembles.items()}
    q_cmd_reordered = {name: unwrapped.dataset.q_cmd[:, perms[name]] for name in ensembles}

    with torch.inference_mode():
        reset_to_segment(env, traj_idx)
        q_now = unwrapped.joint_pos[:, unwrapped.actuated_dof_indices].clone()
        q_buffers = {name: [q_now[:, perms[name]]] for name in ensembles}
        q_sim_log, q_real_log = [], []
        for _ in range(num_steps):
            t = unwrapped.dataset.clamp(unwrapped.traj_t)
            q_real_log.append(unwrapped.dataset.q_meas[t].clone())
            # Log sim position BEFORE this step -- matches q_real_log's timing
            # exactly (both represent position AT time t, not after advancing
            # to t+1). Logging AFTER env.step() (as this previously did) put
            # q_sim_log[i] one full timestep ahead of q_real_log[i] for the
            # ENTIRE trajectory -- most visible at i=0, where it meant the
            # correctly-reset starting position (which DOES match q_real_log[0]
            # exactly) was never actually logged/plotted at all.
            q_sim_log.append(unwrapped.joint_pos[:, unwrapped.actuated_dof_indices].clone())

            actions = torch.zeros(env.num_envs, unwrapped.cfg.num_actions, device=env.device)

            for name, (ensemble, ckpt, output_col) in ensembles.items():
                hl, stride = ckpt["history_len"], ckpt["stride"]
                u_hist = build_delta_history(q_cmd_reordered[name], t, hl, stride, unwrapped.dataset)
                q_hist = _history_from_buffer(q_buffers[name], hl, stride)
                raw_input = torch.cat([q_hist, u_hist], dim=-1)
                # forward_standardized -- NOT forward() -- gives the raw tanh output in
                # (-1,1), BEFORE the checkpoint's own torque_range de-normalization.
                # `scale` multiplies THIS directly, so `scale` itself IS the max N*m of
                # extra torque injectable (tanh_output in (-1,1)), not a multiplier on
                # an already-scaled value -- per user decision, this makes `scale`'s
                # search bound directly physically meaningful. `output_col` picks the
                # right column for a two-output pair checkpoint (0 always for a
                # single-joint one) -- see rollout()'s docstring.
                pred = ensemble.forward_standardized(raw_input).mean(dim=0)  # (num_envs, num_joints), raw tanh output
                actions[:, joint_idx[name]] = scales[name] * pred[:, output_col]

            env.step(actions)
            q_new = unwrapped.joint_pos[:, unwrapped.actuated_dof_indices].clone()
            for name in ensembles:
                q_buffers[name].append(q_new[:, perms[name]])

    q_sim = torch.cat(q_sim_log, dim=0).cpu().numpy()
    q_real = torch.cat(q_real_log, dim=0).cpu().numpy()
    return q_sim, q_real


def rmse_for_joints(q_sim: np.ndarray, q_real: np.ndarray, joint_idx: dict, names) -> dict:
    return {name: float(np.sqrt(np.mean((q_sim[:, joint_idx[name]] - q_real[:, joint_idx[name]]) ** 2))) for name in names}


def velocity_rmse_for_joints(q_sim: np.ndarray, q_real: np.ndarray, joint_idx: dict, names, dt: float) -> dict:
    """Per-step finite-difference velocity RMSE. q_sim/q_real are already
    time-aligned (same simulation step at the same row index -- real is
    replayed from the log at the exact index the sim advances), so this is
    the same lockstep-comparison trick `rmse_for_joints` uses, one
    `np.diff` deeper. Distinct from `dataset.q_meas_vel` (flagged unreliable
    for J1/J2 as a single-step derivative of the RECORDED velocity channel,
    see losses.py) -- this differentiates POSITION, which is independently
    established as a faithful per-DOF signal.
    """
    result = {}
    for name in names:
        idx = joint_idx[name]
        qdot_sim = np.diff(q_sim[:, idx]) / dt
        qdot_real = np.diff(q_real[:, idx]) / dt
        result[name] = float(np.sqrt(np.mean((qdot_sim - qdot_real) ** 2)))
    return result


def joint_trajectory_score(
    q_sim: np.ndarray, q_real: np.ndarray, joint_idx: dict, names, dt: float, velocity_weight: float,
) -> tuple[float, float, float, dict, dict]:
    """Combined position+velocity score over `names` (one joint, or several
    at once for reporting). Position-only RMSE can't tell smooth tracking
    apart from oscillation with a similar time-average -- this term catches
    that (see this script's development history: FFJ2 previously showed a
    violent square-wave oscillation that an isolated position-RMSE objective
    didn't penalize). Applied uniformly to every active joint's own 1-D
    search now, not just a former "pair" special case. Returns
    (score, pos_rmse_mean, vel_rmse_mean, pos_rmse_dict, vel_rmse_dict).
    """
    pos_rmse = rmse_for_joints(q_sim, q_real, joint_idx, names)
    vel_rmse = velocity_rmse_for_joints(q_sim, q_real, joint_idx, names, dt)
    pos_mean = float(np.mean(list(pos_rmse.values())))
    vel_mean = float(np.mean(list(vel_rmse.values())))
    score = pos_mean + velocity_weight * vel_mean
    return score, pos_mean, vel_mean, pos_rmse, vel_rmse


def main() -> None:
    with open(args_cli.config) as f:
        agent_cfg = yaml.safe_load(f)
    if args_cli.dataset is not None:
        agent_cfg["dataset"]["paths"] = args_cli.dataset
    agent_cfg["log_path"] = LOG_PATH
    agent_cfg["experiment"]["video_dir"] = None

    checkpoint_args = {"rh_FFJ3": args_cli.checkpoint_ffj3, "rh_FFJ1": args_cli.checkpoint_ffj1, "rh_FFJ2": args_cli.checkpoint_ffj2}
    if all(v is None for v in checkpoint_args.values()) and args_cli.checkpoint_pair is None:
        raise ValueError("At least one of --checkpoint_ffj3/--checkpoint_ffj1/--checkpoint_ffj2/--checkpoint_pair is required.")
    if args_cli.checkpoint_pair is not None and (args_cli.checkpoint_ffj1 is not None or args_cli.checkpoint_ffj2 is not None):
        raise ValueError("--checkpoint_pair cannot be combined with --checkpoint_ffj1/--checkpoint_ffj2 (both provide "
                          "rh_FFJ1/rh_FFJ2 -- pick one source).")

    env = build_env(agent_cfg, args_cli.residual_clip)
    device = env.unwrapped.device

    ensembles: dict[str, tuple[GenANEnsemble, dict, int]] = {}
    for name in JOINTS:  # fixed order for deterministic active_joints/print ordering
        path = checkpoint_args[name]
        if path is None:
            continue
        ensemble, ckpt = load_single_ensemble(path, device)
        ensembles[name] = (ensemble, ckpt, 0)
        print(f"[INFO] Loaded {name} checkpoint: {os.path.abspath(path)} (torque_range={ckpt['torque_range']}, "
              f"independently_trained_pair={ckpt.get('independently_trained_pair', False)})")

    if args_cli.checkpoint_pair is not None:
        pair_ensemble, pair_ckpt = load_pair_ensemble(args_cli.checkpoint_pair, device)
        pair_name_a, pair_name_b = pair_ckpt["joint_pair_names"]
        ensembles[pair_name_a] = (pair_ensemble, pair_ckpt, 0)
        ensembles[pair_name_b] = (pair_ensemble, pair_ckpt, 1)
        print(f"[INFO] Loaded pair checkpoint: {os.path.abspath(args_cli.checkpoint_pair)} "
              f"(torque_range={pair_ckpt['torque_range']}, pair={pair_ckpt['joint_pair_names']}) -- "
              f"{pair_name_a}=col0, {pair_name_b}=col1, each with its OWN --scale_*")

    active_joints = list(ensembles.keys())
    print(f"[INFO] Active joints this run: {active_joints}")

    joint_names_env = [env.unwrapped.robot.joint_names[i] for i in env.unwrapped.actuated_dof_indices]
    joint_idx = {name: joint_names_env.index(name) for name in JOINTS}
    print(f"[INFO] Joint indices in the 16-dim action vector: {joint_idx}")

    os.makedirs(args_cli.out_dir, exist_ok=True)
    dt = env.unwrapped.step_dt

    def run(ensembles_subset: dict, scales: dict, traj_idx: int):
        """Roll out with ONLY the joints in `ensembles_subset` active --
        callers control isolation by choosing what to pass here: a single
        `{target: ...}` dict for an isolated per-joint search, or the full
        `ensembles` dict for a combined (all-active-joints-together) check.
        """
        n_steps = int(env.unwrapped.dataset.traj_lengths[traj_idx].item()) - 1
        q_sim, q_real = rollout(env, ensembles_subset, joint_idx, traj_idx, n_steps, scales)
        names = list(ensembles_subset.keys())
        return q_sim, q_real, rmse_for_joints(q_sim, q_real, joint_idx, names)

    def run_multi(ensembles_subset: dict, scales: dict, traj_indices):
        """Average per-joint position RMSE across multiple segments."""
        names = list(ensembles_subset.keys())
        per_seg = [run(ensembles_subset, scales, ti) for ti in traj_indices]
        mean_rmse = {name: float(np.mean([r[2][name] for r in per_seg])) for name in names}
        return per_seg, mean_rmse

    if args_cli.scale_ffj3 is not None or args_cli.scale_ffj1 is not None or args_cli.scale_ffj2 is not None:
        scale_args = {"rh_FFJ3": args_cli.scale_ffj3, "rh_FFJ1": args_cli.scale_ffj1, "rh_FFJ2": args_cli.scale_ffj2}
        missing = [f"--scale_{n[3:].lower()}" for n in active_joints if scale_args[n] is None]
        if missing:
            raise ValueError(f"Missing {missing} for a single rollout (required for the active joints: {active_joints}).")
        scales = {name: (scale_args[name] if name in active_joints else 0.0) for name in JOINTS}
        # COMBINED by design -- this override mode is for hand-testing exact
        # deployment behavior (all loaded joints together), not an isolated
        # per-joint check. Pass only ONE --checkpoint_* if you want isolation.
        print(f"[INFO] Single COMBINED rollout at scales={ {k: v for k, v in scales.items() if k in active_joints} }, "
              f"traj_idx={args_cli.traj_idx}")
        q_sim, q_real, rmse = run(ensembles, scales, args_cli.traj_idx)
        print(f"[RESULT] RMSE: {rmse}")
        _save_plot(q_sim, q_real, joint_idx, active_joints, args_cli.out_dir, "single_rollout")
        env.close()
        return

    # Which segments to fit on vs. hold out for a genuine generalization check
    # (per user decision: "run and compute based on more trajectories and then
    # test out on held out trajectories").
    n_segments = int(env.unwrapped.dataset.traj_starts.shape[0])
    if args_cli.fit_traj_idx is not None or args_cli.test_traj_idx is not None:
        fit_traj_idx = list(args_cli.fit_traj_idx) if args_cli.fit_traj_idx is not None else \
            [i for i in range(n_segments) if i not in set(args_cli.test_traj_idx)]
        test_traj_idx = list(args_cli.test_traj_idx) if args_cli.test_traj_idx is not None else \
            [i for i in range(n_segments) if i not in set(args_cli.fit_traj_idx)]
    elif n_segments < 2:
        fit_traj_idx, test_traj_idx = [0], []
        print("[WARN] Only 1 trajectory segment available -- no held-out segment; fitting and reporting on segment 0 only.")
    else:
        # Capped at a small constant, NOT a percentage -- with this dataset's
        # actual 89 segments, round(0.8*89)=71 fit segments would multiply
        # every objective evaluation's cost by 71x (thousands of full-episode
        # rollouts). A handful of segments is enough to avoid overfitting a
        # single episode; pass --fit_traj_idx/--test_traj_idx explicitly to
        # use more once a run's actual timing is understood.
        n_fit = min(4, n_segments - 1)
        n_test = min(2, n_segments - n_fit)
        fit_traj_idx = list(range(n_fit))
        test_traj_idx = list(range(n_fit, n_fit + n_test))
    print(f"[INFO] {n_segments} total trajectory segments. Fit: {fit_traj_idx}  Test (held-out): {test_traj_idx}")

    bounds = tuple(args_cli.bounds)
    best_scale = {name: 0.0 for name in JOINTS}
    all_results = {}

    def joint_objective(target: str, s: float, traj_indices) -> tuple[float, float, float]:
        """Mean joint_trajectory_score (position+velocity) for `target`,
        simulated in FULL ISOLATION -- every other joint's model is entirely
        ABSENT from this rollout (not merely scale=0), so `target`'s fitted
        scale can't be contaminated by another joint's own candidate scale
        OR by real rigid-body coupling driven by another joint's injected
        torque (see module docstring: rh_FFJ1's torque was observed to
        physically drag rh_FFJ2's position in a combined rollout, even with
        rh_FFJ2 receiving zero injected torque). Per user decision. The
        three joints' searches are now fully independent -- order doesn't
        matter.
        """
        isolated = {target: ensembles[target]}
        scales = {target: s}
        scores, pos_means, vel_means = [], [], []
        for ti in traj_indices:
            q_sim, q_real, _ = run(isolated, scales, ti)
            score, pos_mean, vel_mean, _, _ = joint_trajectory_score(
                q_sim, q_real, joint_idx, [target], dt, args_cli.velocity_weight,
            )
            scores.append(score)
            pos_means.append(pos_mean)
            vel_means.append(vel_mean)
        return float(np.mean(scores)), float(np.mean(pos_means)), float(np.mean(vel_means))

    # ONE generic 1-D bounded search per active joint, each running in FULL
    # ISOLATION (see joint_objective/module docstring) -- scored by
    # joint_trajectory_score (position+velocity, so oscillatory-but-mean-
    # correct solutions aren't mistaken for good fits), averaged over ALL fit
    # segments. The three searches are independent of each other and of
    # execution order.
    for target in active_joints:
        print(f"\n[OPTIMIZE] {target}: bounded search in {bounds} over fit segments {fit_traj_idx} "
              f"(ISOLATED -- {target} is the ONLY active joint in these rollouts)")
        evals = []

        def objective(s, target=target):
            score, pos_mean, vel_mean = joint_objective(target, s, fit_traj_idx)
            evals.append((s, score))
            print(f"  scale={s:8.4f}  {target}_score={score:.6f}  (pos_mean={pos_mean:.6f}  vel_mean={vel_mean:.6f})")
            return score

        result = minimize_scalar(objective, bounds=bounds, method="bounded")
        best_scale[target] = float(result.x)
        all_results[target] = evals
        print(f"[OPTIMIZE] {target}: best scale={result.x:.4f} (score={result.fun:.6f}, {len(evals)} evals)")

    print(f"\n[RESULT] Best-fit scales (each found in isolation): "
          f"{ {k: v for k, v in best_scale.items() if k in active_joints} }")

    # From here on: COMBINED rollouts (all active joints' models loaded and
    # driven TOGETHER, using their independently-fit scales) -- the actual
    # deployment scenario, and where any REMAINING cross-joint coupling would
    # still show up, even though it can no longer contaminate the fit above.
    def report_segment(traj_idx, tag):
        q_sim, q_real, rmse = run(ensembles, best_scale, traj_idx)
        print(f"[RESULT] {tag} traj_idx={traj_idx} (COMBINED): rmse={rmse}")
        score, pos_mean, vel_mean, pos_rmse, vel_rmse = joint_trajectory_score(
            q_sim, q_real, joint_idx, active_joints, dt, args_cli.velocity_weight,
        )
        print(f"[RESULT] {tag} traj_idx={traj_idx} (COMBINED): combined_score={score:.6f} "
              f"(pos_mean={pos_mean:.6f} vel_mean={vel_mean:.6f}) pos={pos_rmse} vel={vel_rmse}")
        _save_plot(q_sim, q_real, joint_idx, active_joints, args_cli.out_dir, f"{tag}_seg{traj_idx}")

    print("\n[RESULT] === Fit segments (combined check) ===")
    for ti in fit_traj_idx:
        report_segment(ti, "fit")

    print("\n[RESULT] === Held-out (test) segments (combined check) ===")
    if not test_traj_idx:
        print("[WARN] No held-out segments available -- generalization not checked this run.")
    for ti in test_traj_idx:
        report_segment(ti, "test")

    zero_scales = {name: 0.0 for name in JOINTS}
    one_scales = {name: (1.0 if name in active_joints else 0.0) for name in JOINTS}
    _, rmse_zero = run_multi(ensembles, zero_scales, fit_traj_idx)
    _, rmse_one = run_multi(ensembles, one_scales, fit_traj_idx)
    print(f"[RESULT] Reference -- scale=0 (PD only) mean RMSE over fit segments: {rmse_zero}")
    print(f"[RESULT] Reference -- scale=1 (unscaled) mean RMSE over fit segments: {rmse_one}")

    _save_sweep_table(all_results, args_cli.out_dir)

    env.close()


def _save_sweep_table(all_results: dict, out_dir: str) -> None:
    path = os.path.join(out_dir, "scale_sweep.txt")
    with open(path, "w") as f:
        for name, results in all_results.items():
            f.write(f"{name}:\n")
            for s, r in results:
                f.write(f"  scale={s:6.3f}  score={r:.6f}\n")
    print(f"[INFO] Saved sweep table to {path}")


def _save_plot(q_sim, q_real, joint_idx: dict, active_joints: list, out_dir: str, tag: str) -> None:
    """One combined figure, one panel per ACTIVE joint -- all active joints
    are driven TOGETHER in the same single rollout, so they're reported
    together in one figure, each with its own independently-fit scale.
    """
    import matplotlib.pyplot as plt

    names = active_joints
    fig, axes = plt.subplots(1, len(names), figsize=(5 * len(names), 4), squeeze=False)
    axes = axes[0]
    for ax, name in zip(axes, names):
        idx = joint_idx[name]
        ax.plot(q_real[:, idx], "k--", label="real", linewidth=1.2)
        ax.plot(q_sim[:, idx], "b", label="sim", linewidth=1.2, alpha=0.8)
        ax.set_title(name, fontsize=10)
        ax.set_xlabel("step")
        ax.set_ylabel("position (rad)")
        ax.legend(fontsize=8)

    fig.tight_layout()
    out_path = os.path.join(out_dir, f"sim_vs_real_{tag}.png")
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    print(f"[INFO] Saved {out_path}")


if __name__ == "__main__":
    # Explicit except (not just try/finally) is required: Isaac Sim's
    # simulation_app.close() has been observed to terminate the process before a bare
    # `finally:` gets a chance to re-raise/print an exception from main() -- silently
    # swallowing the traceback and exiting with code 0 even on a real failure.
    try:
        main()
    except Exception as err:
        print("ERROR DURING PLAYBACK:", err)
        raise
    finally:
        simulation_app.close()

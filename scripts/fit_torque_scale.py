#!/usr/bin/env python3
"""Fit a deployment-time torque scale for rh_FFJ3 (residual model) and
rh_FFJ1/rh_FFJ2 (mimic-pair model), so the SIMULATED joint trajectory
matches the REAL recorded trajectory over a full episode.

Two checkpoints are loaded: a single-joint rh_FFJ3 residual model
(`train_genan_single.py --residual_torque --torque_range ...`) and a
rh_FFJ1/rh_FFJ2 mimic-pair model (`train_genan_pair.py`). `UANShadowLiteEnv`
(`roto/tasks/uan_shadowlite/task.py`) already treats its 16-dim `actions` as
a small residual torque ADDED ON TOP of PhysX's own always-active implicit
PD (`_apply_action` calls both `set_joint_position_target` -- driving PD --
and `set_joint_effort_target(residual)` -- see that file's module docstring)
-- exactly the mechanism this script needs, with NO stiffness-zeroing
required (unlike `play_genan.py`, whose checkpoint predicts ABSOLUTE torque
and so must remove PD's competing term). Per user decision: `scale_j *
<network's own RAW tanh output, in (-1,1) -- NOT de-normalized by the
checkpoint's own torque_range>` is injected uniformly for all three joints,
so `scale_j` itself directly IS the max N*m of extra torque injectable for
that joint, without hand-deriving whether the pair model's absolute-torque
formulation double-counts against PD -- `scale_j` is fit empirically against
real trajectory RMSE, so it absorbs any such mismatch.

Only rh_FFJ3/rh_FFJ1/rh_FFJ2 get nonzero `actions`; the other 13 (untrained)
joints get exactly 0, i.e. pure PhysX PD tracking of the real command.

rh_FFJ1 and rh_FFJ2 each get their OWN independently-fit scale (per user
decision: the fitted value can legitimately differ between them, even though
they share one physical tendon/motor) -- but all three joints are rolled out
TOGETHER in the same single episode/simulation (they always were -- one
rollout call injects all three joints' torques simultaneously) and reported
TOGETHER in one combined plot (three panels, one figure), not as separate
independent plots.

Scale search: `scipy.optimize.minimize_scalar(method="bounded")` per joint
(Brent-style golden-section+parabolic search over `--bounds`, default
[0,50]), one joint at a time, holding the others at their best-found-so-far
value (coordinate descent). NOT a fixed grid -- far fewer, more precisely
placed evaluations, valid as long as each joint's RMSE-vs-scale curve is
unimodal (every evaluation is still logged to scale_sweep.txt so that
assumption can be sanity-checked after the fact).

Usage:
    python fit_torque_scale.py --checkpoint_ffj3 ffj3.pt --checkpoint_pair pair.pt --headless
    python fit_torque_scale.py --checkpoint_ffj3 ffj3.pt --checkpoint_pair pair.pt \\
        --scale_ffj3 1.0 --scale_ffj1 1.0 --scale_ffj2 1.0 --headless  # single rollout, no sweep
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
parser.add_argument("--checkpoint_ffj3", type=str, default=None, help="Single-joint rh_FFJ3 residual checkpoint. Omit to skip rh_FFJ3 entirely.")
parser.add_argument("--checkpoint_pair", type=str, default=None, help="rh_FFJ1/rh_FFJ2 mimic-pair checkpoint. Omit to skip the pair entirely.")
parser.add_argument(
    "--config", type=str, default=os.path.join(_GENAN_DIR, "agents", "shadowlite", "default.yaml"),
    help="Agent yaml providing dataset.paths (genan/sweeper sections are unused here).",
)
parser.add_argument("--dataset", type=str, action="append", default=None, help="Override dataset.paths (repeatable).")
parser.add_argument("--residual_clip", type=float, default=1000.0, help="uan.residual_clip override (see play_genan.py).")
parser.add_argument(
    "--bounds", type=float, nargs=2, default=[0.0, 50.0],
    help="Search bounds for each joint's scale (per user decision: 0-50, not 0-2).",
)
parser.add_argument(
    "--scale_ffj3", type=float, default=None,
    help="Skip the sweep and run ONE rollout at these fixed scales (all three --scale_* required together).",
)
parser.add_argument("--scale_ffj1", type=float, default=None)
parser.add_argument("--scale_ffj2", type=float, default=None)
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


def load_ffj3_ensemble(checkpoint_path: str, device) -> tuple[GenANEnsemble, dict]:
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


def rollout(
    env, ffj3_ensemble, ffj3_ckpt, pair_ensemble, pair_ckpt,
    joint_idx: dict, num_steps: int, scale_ffj3: float, scale_ffj1: float, scale_ffj2: float,
):
    """Roll out `num_steps` with `actions[joint_idx[name]] = scale * <that
    joint's own RAW tanh output, in (-1,1)>` for whichever of rh_FFJ3/
    rh_FFJ1/rh_FFJ2 have a loaded checkpoint (each with its OWN independent
    scale), and 0 for every other of the 16 actuated joints (pure PhysX PD
    there). `ffj3_ensemble`/`pair_ensemble` may be `None` to skip that model
    entirely (e.g. pair-only runs) -- all LOADED models are driven TOGETHER
    in this one rollout (same episode, same simulation). Returns (q_sim,
    q_real), each (num_steps, 16).

    The pair model's two output shares aren't tied to a specific joint by its
    loss (coupled_pair_activity_loss trains both against an activity-weighted
    per-share target -- see losses.py) -- share index 0 is only "rh_FFJ1's
    share" because `joint_pair_names[0]` was rh_FFJ1 at TRAINING time
    (train_genan_pair.py's --joint_a). Read that ordering from the checkpoint
    itself rather than assuming it, so a differently-ordered pair checkpoint
    can't silently swap which joint gets which share/scale.
    """
    unwrapped = env.unwrapped
    q_sim_log, q_real_log = [], []
    if ffj3_ensemble is not None:
        ffj3_hl, ffj3_stride = ffj3_ckpt["history_len"], ffj3_ckpt["stride"]
    if pair_ensemble is not None:
        pair_hl, pair_stride = pair_ckpt["history_len"], pair_ckpt["stride"]
        pair_name_a, pair_name_b = pair_ckpt["joint_pair_names"]
        pair_scale = {pair_name_a: scale_ffj1 if pair_name_a == "rh_FFJ1" else scale_ffj2,
                      pair_name_b: scale_ffj2 if pair_name_b == "rh_FFJ2" else scale_ffj1}

    with torch.inference_mode():
        env.reset(hard=True)
        q_now = unwrapped.joint_pos[:, unwrapped.actuated_dof_indices].clone()
        q_buffer = [q_now]
        for _ in range(num_steps):
            t = unwrapped.dataset.clamp(unwrapped.traj_t)
            q_real_log.append(unwrapped.dataset.q_meas[t].clone())

            actions = torch.zeros(env.num_envs, unwrapped.cfg.num_actions, device=env.device)

            if ffj3_ensemble is not None:
                u_hist_ffj3 = build_delta_history(unwrapped.dataset.q_cmd, t, ffj3_hl, ffj3_stride, unwrapped.dataset)
                q_hist_ffj3 = _history_from_buffer(q_buffer, ffj3_hl, ffj3_stride)
                raw_input_ffj3 = torch.cat([q_hist_ffj3, u_hist_ffj3], dim=-1)
                # forward_standardized -- NOT forward() -- gives the raw tanh output in
                # (-1,1), BEFORE the checkpoint's own torque_range de-normalization.
                # `scale` multiplies THIS directly, so `scale` itself IS the max N*m of
                # extra torque injectable (tanh_output in (-1,1)), not a multiplier on
                # an already-scaled value -- per user decision, this makes `scale`'s
                # 0-50 search bound directly physically meaningful (was previously
                # amplifying an already +-45 N*m value, causing runaway torque).
                ffj3_pred = ffj3_ensemble.forward_standardized(raw_input_ffj3).mean(dim=0)  # (num_envs, 1), raw tanh output
                actions[:, joint_idx["rh_FFJ3"]] = scale_ffj3 * ffj3_pred[:, 0]

            if pair_ensemble is not None:
                u_hist_pair = build_delta_history(unwrapped.dataset.q_cmd, t, pair_hl, pair_stride, unwrapped.dataset)
                q_hist_pair = _history_from_buffer(q_buffer, pair_hl, pair_stride)
                raw_input_pair = torch.cat([q_hist_pair, u_hist_pair], dim=-1)
                pair_pred = pair_ensemble.forward_standardized(raw_input_pair).mean(dim=0)  # (num_envs, 2), raw tanh output
                actions[:, joint_idx[pair_name_a]] = pair_scale[pair_name_a] * pair_pred[:, 0]
                actions[:, joint_idx[pair_name_b]] = pair_scale[pair_name_b] * pair_pred[:, 1]

            env.step(actions)
            q_new = unwrapped.joint_pos[:, unwrapped.actuated_dof_indices].clone()
            q_buffer.append(q_new)
            q_sim_log.append(q_new)

    q_sim = torch.cat(q_sim_log, dim=0).cpu().numpy()
    q_real = torch.cat(q_real_log, dim=0).cpu().numpy()
    return q_sim, q_real


def rmse_for_joints(q_sim: np.ndarray, q_real: np.ndarray, joint_idx: dict, names) -> dict:
    return {name: float(np.sqrt(np.mean((q_sim[:, joint_idx[name]] - q_real[:, joint_idx[name]]) ** 2))) for name in names}


def main() -> None:
    with open(args_cli.config) as f:
        agent_cfg = yaml.safe_load(f)
    if args_cli.dataset is not None:
        agent_cfg["dataset"]["paths"] = args_cli.dataset
    agent_cfg["log_path"] = LOG_PATH
    agent_cfg["experiment"]["video_dir"] = None

    if args_cli.checkpoint_ffj3 is None and args_cli.checkpoint_pair is None:
        raise ValueError("At least one of --checkpoint_ffj3/--checkpoint_pair is required.")

    env = build_env(agent_cfg, args_cli.residual_clip)
    device = env.unwrapped.device

    ffj3_ensemble = ffj3_ckpt = None
    if args_cli.checkpoint_ffj3 is not None:
        ffj3_ensemble, ffj3_ckpt = load_ffj3_ensemble(args_cli.checkpoint_ffj3, device)
        print(f"[INFO] Loaded FFJ3 checkpoint: {os.path.abspath(args_cli.checkpoint_ffj3)} "
              f"(torque_range={ffj3_ckpt['torque_range']})")

    pair_ensemble = pair_ckpt = pair_name_a = pair_name_b = None
    if args_cli.checkpoint_pair is not None:
        pair_ensemble, pair_ckpt = load_pair_ensemble(args_cli.checkpoint_pair, device)
        pair_name_a, pair_name_b = pair_ckpt["joint_pair_names"]
        print(f"[INFO] Loaded pair checkpoint: {os.path.abspath(args_cli.checkpoint_pair)} "
              f"(torque_range={pair_ckpt['torque_range']}, pair={pair_ckpt['joint_pair_names']})")

    # Which of rh_FFJ3/rh_FFJ1/rh_FFJ2 are actually active this run -- everything
    # below (sweep targets, RMSE, plot panels) is scoped to just these.
    active_joints = ([JOINTS[0]] if ffj3_ensemble is not None else []) + (
        [pair_name_a, pair_name_b] if pair_ensemble is not None else []
    )
    print(f"[INFO] Active joints this run: {active_joints}")

    joint_names_env = [env.unwrapped.robot.joint_names[i] for i in env.unwrapped.actuated_dof_indices]
    joint_idx = {name: joint_names_env.index(name) for name in JOINTS}
    print(f"[INFO] Joint indices in the 16-dim action vector: {joint_idx}")

    num_steps = int(env.unwrapped.dataset.traj_lengths[0].item()) - 1
    os.makedirs(args_cli.out_dir, exist_ok=True)

    def run(scale_ffj3, scale_ffj1, scale_ffj2):
        q_sim, q_real = rollout(
            env, ffj3_ensemble, ffj3_ckpt, pair_ensemble, pair_ckpt,
            joint_idx, num_steps, scale_ffj3, scale_ffj1, scale_ffj2,
        )
        return q_sim, q_real, rmse_for_joints(q_sim, q_real, joint_idx, active_joints)

    if args_cli.scale_ffj3 is not None or args_cli.scale_ffj1 is not None or args_cli.scale_ffj2 is not None:
        needed = (["--scale_ffj3"] if ffj3_ensemble is not None else []) + (
            ["--scale_ffj1", "--scale_ffj2"] if pair_ensemble is not None else []
        )
        missing = [n for n in needed if getattr(args_cli, n[2:]) is None]
        if missing:
            raise ValueError(f"Missing {missing} for a single rollout (required for the active joints: {active_joints}).")
        scale_ffj3 = args_cli.scale_ffj3 if ffj3_ensemble is not None else 0.0
        scale_ffj1 = args_cli.scale_ffj1 if pair_ensemble is not None else 0.0
        scale_ffj2 = args_cli.scale_ffj2 if pair_ensemble is not None else 0.0
        print(f"[INFO] Single rollout at scale_ffj3={scale_ffj3}, scale_ffj1={scale_ffj1}, scale_ffj2={scale_ffj2}")
        q_sim, q_real, rmse = run(scale_ffj3, scale_ffj1, scale_ffj2)
        print(f"[RESULT] RMSE: {rmse}")
        _save_plot(q_sim, q_real, joint_idx, active_joints, args_cli.out_dir, "single_rollout")
        env.close()
        return

    # Independent 1-D bounded search per joint (scipy's Brent-based
    # golden-section+parabolic method, NOT a fixed grid -- far fewer, more
    # precisely-placed evaluations than a dense grid, valid as long as each
    # joint's RMSE-vs-scale curve is unimodal, see all_results/scale_sweep.txt
    # to sanity-check that assumption after the fact), one joint at a time,
    # holding the OTHERS at their best-found-so-far value (coordinate
    # descent -- see module docstring), one persistent Isaac boot throughout.
    # Only ACTIVE joints are swept; inactive ones stay at 0 (no checkpoint to
    # predict anything for them, so no torque is ever injected regardless).
    bounds = tuple(args_cli.bounds)
    best_scale = {"rh_FFJ3": 0.0, "rh_FFJ1": 0.0, "rh_FFJ2": 0.0}
    all_results = {}
    for target in active_joints:
        print(f"\n[OPTIMIZE] {target}: bounded search in {bounds} (others held at {best_scale})")
        evals = []

        def objective(s, target=target):
            scales = dict(best_scale)
            scales[target] = s
            _, _, rmse = run(scales["rh_FFJ3"], scales["rh_FFJ1"], scales["rh_FFJ2"])
            evals.append((s, rmse[target]))
            print(f"  scale={s:8.4f}  {target}_rmse={rmse[target]:.6f}")
            return rmse[target]

        result = minimize_scalar(objective, bounds=bounds, method="bounded")
        best_scale[target] = float(result.x)
        all_results[target] = evals
        print(f"[OPTIMIZE] {target}: best scale={result.x:.4f} (rmse={result.fun:.6f}, {len(evals)} evals)")

    print(f"\n[RESULT] Best-fit scales: { {k: v for k, v in best_scale.items() if k in active_joints} }")
    q_sim, q_real, rmse_final = run(best_scale["rh_FFJ3"], best_scale["rh_FFJ1"], best_scale["rh_FFJ2"])
    print(f"[RESULT] Final RMSE at best-fit scales: {rmse_final}")

    _, _, rmse_zero = run(0.0, 0.0, 0.0)
    _, _, rmse_one = run(1.0 if ffj3_ensemble else 0.0, 1.0 if pair_ensemble else 0.0, 1.0 if pair_ensemble else 0.0)
    print(f"[RESULT] Reference -- scale=0 (PD only) RMSE: {rmse_zero}")
    print(f"[RESULT] Reference -- scale=1 (unscaled) RMSE: {rmse_one}")

    _save_sweep_table(all_results, args_cli.out_dir)
    _save_plot(q_sim, q_real, joint_idx, active_joints, args_cli.out_dir, "best_fit")

    env.close()


def _save_sweep_table(all_results: dict, out_dir: str) -> None:
    path = os.path.join(out_dir, "scale_sweep.txt")
    with open(path, "w") as f:
        for name, results in all_results.items():
            f.write(f"{name}:\n")
            for s, r in results:
                f.write(f"  scale={s:6.3f}  rmse={r:.6f}\n")
    print(f"[INFO] Saved sweep table to {path}")


def _save_plot(q_sim, q_real, joint_idx: dict, active_joints: list, out_dir: str, tag: str) -> None:
    """One combined figure, one panel per ACTIVE joint -- all active joints
    are driven TOGETHER in the same single rollout (per user decision), so
    they're reported together in one figure, even though each joint has its
    own independently-fit scale and gets its own panel.
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

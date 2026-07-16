#!/usr/bin/env python3
"""Plot simulated vs real joint POSITION for rh_FFJ1/rh_FFJ2, separately, over
several trajectory segments, using a fixed per-joint deployment scale on the
rh_FFJ1/rh_FFJ2 mimic-pair model (train_genan_pair.py checkpoint).

Same residual-torque deployment mechanism as fit_torque_scale.py (see that
file's module docstring): `UANShadowLiteEnv` (task.py) always keeps its own
implicit PD active (`_apply_action` calls `set_joint_position_target`), and
`actions` -- here `scale_j * <that joint's own RAW tanh output, in (-1,1),
from ensemble.forward_standardized>` -- is injected ADDITIONALLY via
`set_joint_effort_target`, i.e.:

    tau_applied = Kp*(q_cmd - q_meas) - Kd*qdot_meas   (PhysX's own PD, always on)
                + scale_j * tanh_output_j              (this script's injected residual)

No stiffness-zeroing, unlike play_genan.py -- that script's checkpoints
predict ABSOLUTE torque (Decision 1), this one predicts a small residual ON
TOP of PD by design (fit_torque_scale.py's own scale search already found
the scale empirically against real trajectory RMSE, so it's already tuned
for this additive mechanism, not the sole-torque one).

Unlike fit_torque_scale.py (one rollout, one combined multi-panel figure),
this script rolls out `--segments` (plural, default 6 evenly-spaced ones)
separately -- each reset via the exact same 3 calls task.py's own
`_reset_to_trajectory` uses (`write_joint_state_to_sim`, `set_joint_position_
target`, `joint_pos_cmd` bookkeeping) so no `task.py` edit is needed -- and
saves ONE PNG per (joint, segment), not a combined figure, matching plot_
single.py/plot_pair.py's established per-segment-file convention.

Usage:
    python plot_pair_trajectories.py --checkpoint_pair pair.pt \\
        --scale_ffj1 33.7469 --scale_ffj2 18.9965 --headless
    python plot_pair_trajectories.py --checkpoint_pair pair.pt \\
        --scale_ffj1 33.7469 --scale_ffj2 18.9965 --segments 5 20 40 60 75 88 --headless
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

parser = argparse.ArgumentParser(description="Plot sim-vs-real position for rh_FFJ1/rh_FFJ2 over several segments.")
parser.add_argument("--checkpoint_pair", type=str, required=True, help="rh_FFJ1/rh_FFJ2 mimic-pair checkpoint.")
parser.add_argument("--scale_ffj1", type=float, required=True, help="Deployment scale for rh_FFJ1's share.")
parser.add_argument("--scale_ffj2", type=float, required=True, help="Deployment scale for rh_FFJ2's share.")
parser.add_argument(
    "--config", type=str, default=os.path.join(_GENAN_DIR, "agents", "shadowlite", "default.yaml"),
    help="Agent yaml providing dataset.paths (genan/sweeper sections are unused here).",
)
parser.add_argument("--dataset", type=str, action="append", default=None, help="Override dataset.paths (repeatable).")
parser.add_argument("--residual_clip", type=float, default=1000.0, help="uan.residual_clip override (see fit_torque_scale.py).")
parser.add_argument(
    "--segments", type=int, nargs="+", default=None,
    help="Trajectory segment indices to roll out (default: 6 evenly-spaced segments across the dataset).",
)
parser.add_argument("--num_segments", type=int, default=6, help="How many evenly-spaced segments if --segments is omitted.")
parser.add_argument("--out_dir", type=str, default=os.path.join(_ROTO_ROOT, "genan", "genan_plots", "pair_position_trajectories"))
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

sys.path.insert(0, _THIS_DIR)
sys.path.insert(0, _ROTO_ROOT)
sys.path.insert(0, _GENAN_DIR)
from common_utils import LOG_PATH, make_env, update_env_cfg  # noqa: E402
from multimodal_rl.tools.writer import Writer  # noqa: E402

from history import build_delta_history  # noqa: E402
from model import GenANEnsemble  # noqa: E402

from roto.tasks import uan_shadowlite  # noqa: E402,F401
from roto.tasks.uan_shadowlite.task import UANShadowLiteEnvCfg  # noqa: E402


def build_env(agent_cfg: dict, residual_clip: float):
    env_cfg = UANShadowLiteEnvCfg()
    env_cfg.dataset = dict(agent_cfg["dataset"])
    env_cfg.uan = dict(agent_cfg.get("uan", {}))
    # actions ARE the residual torque added on top of PD (task.py's own design,
    # see module docstring) -- action_scale=1.0 + a large residual_clip makes
    # `residual == actions`, same convention as play_genan.py/fit_torque_scale.py.
    env_cfg.uan["action_scale"] = 1.0
    env_cfg.uan["residual_clip"] = residual_clip
    env_cfg.uan["reset_to_random"] = False  # we drive resets ourselves, per segment
    env_cfg.num_eval_envs = 0

    writer = Writer(agent_cfg, play=True)
    env_cfg = update_env_cfg(args_cli, env_cfg, agent_cfg)

    args_cli.task = "UAN_Shadowlite"
    args_cli.gym_env_id = "UAN_Shadowlite"
    env = make_env(agent_cfg, env_cfg, writer, args_cli)
    return env


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
    """Duplicated (not imported) from fit_torque_scale.py/play_genan.py -- see
    fit_torque_scale.py's matching comment for why (AppLauncher import-time
    side effects).
    """
    frames = []
    for k in range(history_len + 1):
        idx = max(0, len(buffer) - 1 - k * stride)
        frames.append(buffer[idx])
    current = frames[0]
    deltas = [current] + [f - current for f in frames[1:]]
    return torch.cat(deltas, dim=-1)


def reset_to_segment(env, seg_idx: int) -> None:
    """Same 3 calls task.py's own `_reset_to_trajectory` makes
    (write_joint_state_to_sim / set_joint_position_target / joint_pos_cmd
    bookkeeping), just targeting an arbitrary segment index instead of
    `traj_starts[0]` or a random one -- no `task.py` edit needed.
    """
    unwrapped = env.unwrapped
    env.reset(hard=True)  # standard reset (goes to traj_starts[0]); overwritten immediately below
    n = env.num_envs
    t0 = int(unwrapped.dataset.traj_starts[seg_idx])
    new_t = torch.full((n,), t0, dtype=torch.long, device=unwrapped.device)
    unwrapped.traj_t[:] = new_t

    q0 = unwrapped.dataset.q_meas[new_t]
    qd0 = unwrapped.dataset.q_meas_vel[new_t]
    full_pos = unwrapped.robot.data.joint_pos.clone()
    full_pos[:, unwrapped.actuated_dof_indices] = q0
    full_vel = torch.zeros_like(full_pos)
    full_vel[:, unwrapped.actuated_dof_indices] = qd0

    unwrapped.robot.write_joint_state_to_sim(full_pos, full_vel)
    unwrapped.robot.set_joint_position_target(full_pos)
    unwrapped.joint_pos_cmd[:] = full_pos


def rollout_segment(
    env, pair_ensemble: GenANEnsemble, pair_ckpt: dict, joint_idx: dict,
    pair_scale: dict, seg_idx: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Roll out one full segment. Returns (q_sim, q_real), each (num_steps, 16)."""
    unwrapped = env.unwrapped
    hl, stride = pair_ckpt["history_len"], pair_ckpt["stride"]
    name_a, name_b = pair_ckpt["joint_pair_names"]
    num_steps = int(unwrapped.dataset.traj_lengths[seg_idx].item()) - 1

    q_sim_log, q_real_log = [], []
    with torch.inference_mode():
        reset_to_segment(env, seg_idx)
        q_now = unwrapped.joint_pos[:, unwrapped.actuated_dof_indices].clone()
        q_buffer = [q_now]
        for _ in range(num_steps):
            t = unwrapped.dataset.clamp(unwrapped.traj_t)
            q_real_log.append(unwrapped.dataset.q_meas[t].clone())

            u_hist = build_delta_history(unwrapped.dataset.q_cmd, t, hl, stride, unwrapped.dataset)
            q_hist = _history_from_buffer(q_buffer, hl, stride)
            raw_input = torch.cat([q_hist, u_hist], dim=-1)
            # forward_standardized -- NOT forward() -- gives the raw tanh output in (-1,1),
            # BEFORE torque_range de-normalization; `scale` multiplies THIS directly, matching
            # fit_torque_scale.py's own convention (the scales given were fit against exactly
            # this quantity).
            pred = pair_ensemble.forward_standardized(raw_input).mean(dim=0)  # (num_envs, 2)

            actions = torch.zeros(env.num_envs, unwrapped.cfg.num_actions, device=env.device)
            actions[:, joint_idx[name_a]] = pair_scale[name_a] * pred[:, 0]
            actions[:, joint_idx[name_b]] = pair_scale[name_b] * pred[:, 1]

            env.step(actions)
            q_new = unwrapped.joint_pos[:, unwrapped.actuated_dof_indices].clone()
            q_buffer.append(q_new)
            q_sim_log.append(q_new)

    q_sim = torch.cat(q_sim_log, dim=0).cpu().numpy()
    q_real = torch.cat(q_real_log, dim=0).cpu().numpy()
    return q_sim, q_real


def _save_joint_plot(q_sim, q_real, joint_idx: int, joint_name: str, seg_idx: int, out_dir: str) -> str:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rmse = float(np.sqrt(np.mean((q_sim[:, joint_idx] - q_real[:, joint_idx]) ** 2)))

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(q_real[:, joint_idx], "k--", label="real", linewidth=1.4)
    ax.plot(q_sim[:, joint_idx], "b", label="sim", linewidth=1.4, alpha=0.8)
    ax.set_xlabel("step")
    ax.set_ylabel("position (rad)")
    ax.set_title(f"{joint_name} -- segment {seg_idx}, RMSE={rmse:.4f}")
    ax.legend()
    fig.tight_layout()

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{joint_name}_segment_{seg_idx}.png")
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    return out_path, rmse


def main() -> None:
    with open(args_cli.config) as f:
        agent_cfg = yaml.safe_load(f)
    if args_cli.dataset is not None:
        agent_cfg["dataset"]["paths"] = args_cli.dataset
    agent_cfg["log_path"] = LOG_PATH
    agent_cfg["experiment"]["video_dir"] = None

    env = build_env(agent_cfg, args_cli.residual_clip)
    device = env.unwrapped.device

    pair_ensemble, pair_ckpt = load_pair_ensemble(args_cli.checkpoint_pair, device)
    name_a, name_b = pair_ckpt["joint_pair_names"]
    print(f"[INFO] Loaded pair checkpoint: {os.path.abspath(args_cli.checkpoint_pair)} "
          f"(torque_range={pair_ckpt['torque_range']}, pair={pair_ckpt['joint_pair_names']})")

    pair_scale = {name_a: args_cli.scale_ffj1 if name_a == "rh_FFJ1" else args_cli.scale_ffj2,
                  name_b: args_cli.scale_ffj2 if name_b == "rh_FFJ2" else args_cli.scale_ffj1}
    print(f"[INFO] Deployment scales: {pair_scale}")

    joint_names_env = [env.unwrapped.robot.joint_names[i] for i in env.unwrapped.actuated_dof_indices]
    joint_idx = {name: joint_names_env.index(name) for name in (name_a, name_b)}

    num_available = env.unwrapped.dataset.traj_starts.shape[0]
    if args_cli.segments is not None:
        segments = args_cli.segments
    else:
        n = min(args_cli.num_segments, num_available)
        segments = [int(round(i)) for i in np.linspace(0, num_available - 1, n)]
    print(f"[INFO] Rolling out segments: {segments} (of {num_available} available)")

    rmse_table = {name_a: [], name_b: []}
    for seg_idx in segments:
        print(f"[INFO] Segment {seg_idx}: rolling out {int(env.unwrapped.dataset.traj_lengths[seg_idx].item()) - 1} steps...")
        q_sim, q_real = rollout_segment(env, pair_ensemble, pair_ckpt, joint_idx, pair_scale, seg_idx)
        for name in (name_a, name_b):
            out_path, rmse = _save_joint_plot(q_sim, q_real, joint_idx[name], name, seg_idx, args_cli.out_dir)
            rmse_table[name].append(rmse)
            print(f"[INFO] Saved {out_path} (RMSE={rmse:.4f})")

    print("\n{:<10s} {:>10s} {:>10s}".format("segment", name_a, name_b))
    for i, seg_idx in enumerate(segments):
        print(f"{seg_idx:<10d} {rmse_table[name_a][i]:10.4f} {rmse_table[name_b][i]:10.4f}")
    print("{:<10s} {:>10.4f} {:>10.4f}".format("MEAN", np.mean(rmse_table[name_a]), np.mean(rmse_table[name_b])))

    env.close()


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

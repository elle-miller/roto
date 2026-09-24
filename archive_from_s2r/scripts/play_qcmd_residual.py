#!/usr/bin/env python3
"""Evaluate a trained residual-q_cmd checkpoint (train_qcmd_residual.py) in
real Isaac Sim -- source of truth for the whole approach.

Implementation note on HOW the residual is injected: `UANShadowLiteEnv`
(task.py) already has an unmodified residual-TORQUE action path
(`_apply_action` always calls `set_joint_position_target(q_cmd_heuristic)`
AND `set_joint_effort_target(residual)`, Kp/Kd left fully active -- unlike
play_genan.py, which zeroes Kp). Rather than overriding `_pre_physics_step`
to write a modified position target directly, this script computes the
EQUIVALENT torque and feeds it through that existing path:

    tau = Kp*(q_cmd_heuristic + Delta_q_cmd - q_meas) - Kd*qdot
        = tau_PD_baseline + Kp*Delta_q_cmd

`tau_PD_baseline` is already what set_joint_position_target(q_cmd_heuristic)
produces via the ACTIVE (not zeroed) Kp/Kd -- so injecting exactly
`Kp*Delta_q_cmd` as the residual via `set_joint_effort_target` reproduces the
same physics as if `q_cmd_heuristic + Delta_q_cmd` had been the position
target directly, with zero changes to task.py, matching this repo's
established convention of reusing UANShadowLiteEnv unmodified (DESIGN.md
Decision 3). NO Kp-zeroing anywhere in this script -- that's the whole point
of the residual-q_cmd approach over the abandoned torque-scaling one.

Usage:
    python play_qcmd_residual.py --checkpoint qcmd_residual_default.pt --headless
"""

import argparse
import os
import sys

sys.stdout.reconfigure(line_buffering=True)

from isaaclab.app import AppLauncher

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_ROTO_ROOT = os.path.dirname(_THIS_DIR)
_GENAN_DIR = os.path.join(_ROTO_ROOT, "genan")

parser = argparse.ArgumentParser(description="Evaluate a residual-q_cmd checkpoint against real trajectory data.")
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--config", type=str, default=os.path.join(_GENAN_DIR, "agents", "shadowlite", "default.yaml"))
parser.add_argument("--dataset", type=str, action="append", default=None)
parser.add_argument("--residual_clip", type=float, default=1000.0)
parser.add_argument("--traj_idx", type=int, default=0, help="Trajectory segment to roll out.")
parser.add_argument("--out", type=str, default=os.path.join(_ROTO_ROOT, "qcmd_residual_sim_vs_real.png"))
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
from pd_gains import load_pd_gains  # noqa: E402

from roto.tasks import uan_shadowlite  # noqa: E402,F401
from roto.tasks.uan_shadowlite.task import UANShadowLiteEnvCfg  # noqa: E402


def build_env(agent_cfg: dict, residual_clip: float):
    env_cfg = UANShadowLiteEnvCfg()
    env_cfg.dataset = dict(agent_cfg["dataset"])
    env_cfg.uan = dict(agent_cfg.get("uan", {}))
    # actions ARE the residual torque added on top of ALWAYS-ACTIVE PD (task.py's
    # own design, unmodified -- see module docstring). NOT zeroing Kp anywhere.
    env_cfg.uan["action_scale"] = 1.0
    env_cfg.uan["residual_clip"] = residual_clip
    env_cfg.uan["reset_to_random"] = False
    env_cfg.num_eval_envs = 0

    writer = Writer(agent_cfg, play=True)
    env_cfg = update_env_cfg(args_cli, env_cfg, agent_cfg)
    args_cli.task = "UAN_Shadowlite"
    args_cli.gym_env_id = "UAN_Shadowlite"
    return make_env(agent_cfg, env_cfg, writer, args_cli)


def load_ensemble(checkpoint_path: str, device):
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    ensemble = GenANEnsemble(ckpt["input_dim"], ckpt["num_joints"], ensemble_size=ckpt["ensemble_size"],
                              bounded_output=True, torque_range=1.0)
    ensemble.load_state_dict(ckpt["ensemble_state_dict"])
    ensemble.eval()
    ensemble.to(device)
    return ensemble, ckpt


def _history_from_buffer(buffer: list, history_len: int, stride: int) -> torch.Tensor:
    """Identical layout to play_genan.py's helper of the same name -- duplicated,
    not imported, matching this repo's small-independent-script convention."""
    frames = []
    for k in range(history_len + 1):
        idx = max(0, len(buffer) - 1 - k * stride)
        frames.append(buffer[idx])
    current = frames[0]
    deltas = [current] + [f - current for f in frames[1:]]
    return torch.cat(deltas, dim=-1)


def rollout(env, ensemble, ckpt, perm: torch.Tensor, kp_by_sim_idx: torch.Tensor,
            max_delta_rad: float, num_steps: int, use_residual: bool):
    """perm: sim-order-index -> checkpoint-training-order-position (see
    fit_torque_scale.py's rollout() docstring for why this permutation is
    required -- checkpoints are trained on genan/joint_config.py's
    hardware_joint_order, the ENV's actuated_dof_indices is a different sorted
    order). q_cmd_reordered/q_buffer are built in TRAINING order for the
    network's input; the resulting per-joint torque is scattered back to SIM
    order (via perm) before being written as `actions`.
    """
    unwrapped = env.unwrapped
    history_len, stride = ckpt["history_len"], ckpt["stride"]
    q_sim_log, q_real_log = [], []

    with torch.inference_mode():
        env.reset(hard=True)
        q_cmd_reordered = unwrapped.dataset.q_cmd[:, perm]  # (T, 16) training order, whole dataset
        q_now = unwrapped.joint_pos[:, unwrapped.actuated_dof_indices][:, perm].clone()
        q_buffer = [q_now]
        for _ in range(num_steps):
            t = unwrapped.dataset.clamp(unwrapped.traj_t)
            q_real_log.append(unwrapped.dataset.q_meas[t].clone())

            if use_residual:
                u_hist = build_delta_history(q_cmd_reordered, t, history_len, stride, unwrapped.dataset)
                q_hist = _history_from_buffer(q_buffer, history_len, stride)
                raw_input = torch.cat([q_hist, u_hist], dim=-1)
                raw = ensemble.forward_standardized(raw_input).mean(dim=0)  # (num_envs,16) tanh, training order
                delta_q_cmd_train_order = raw * max_delta_rad
                delta_q_cmd_sim_order = torch.zeros_like(delta_q_cmd_train_order)
                delta_q_cmd_sim_order[:, perm] = delta_q_cmd_train_order  # scatter back to sim order
                actions = kp_by_sim_idx.unsqueeze(0) * delta_q_cmd_sim_order  # Kp * Delta_q_cmd = equivalent residual torque
            else:
                actions = torch.zeros(env.num_envs, unwrapped.cfg.num_actions, device=env.device)

            env.step(actions)
            q_new = unwrapped.joint_pos[:, unwrapped.actuated_dof_indices].clone()
            q_buffer.append(q_new[:, perm])
            q_sim_log.append(q_new)

    q_sim = torch.cat(q_sim_log, dim=0).cpu().numpy()
    q_real = torch.cat(q_real_log, dim=0).cpu().numpy()
    return q_sim, q_real


def main() -> None:
    with open(args_cli.config) as f:
        agent_cfg = yaml.safe_load(f)
    if args_cli.dataset is not None:
        agent_cfg["dataset"]["paths"] = args_cli.dataset
    agent_cfg["log_path"] = LOG_PATH
    agent_cfg["experiment"]["video_dir"] = None

    env = build_env(agent_cfg, args_cli.residual_clip)
    unwrapped = env.unwrapped
    ensemble, ckpt = load_ensemble(args_cli.checkpoint, unwrapped.device)
    print(f"[INFO] Loaded checkpoint: {os.path.abspath(args_cli.checkpoint)} "
          f"(max_delta_rad={ckpt['max_delta_rad']:.4f}, best_val_loss={ckpt['best_val_loss']:.8f})")

    joint_names_env = [unwrapped.robot.joint_names[i] for i in unwrapped.actuated_dof_indices]
    perm = torch.tensor([joint_names_env.index(n) for n in ckpt["joint_names"]], dtype=torch.long)
    print(f"[INFO] Joint-order permutation (sim -> training) verified: "
          f"{[joint_names_env[i] for i in perm.tolist()][:3]}... matches ckpt['joint_names'][:3]={ckpt['joint_names'][:3]}")

    kp_by_sim_idx = torch.zeros(len(joint_names_env), device=unwrapped.device)
    for i, name in enumerate(joint_names_env):
        kp, _ = load_pd_gains(name)
        kp_by_sim_idx[i] = kp

    num_steps = int(unwrapped.dataset.traj_lengths[args_cli.traj_idx].item()) - 1
    print(f"[INFO] Rolling out {num_steps} steps, PD-only baseline (segment {args_cli.traj_idx})...")
    q_sim_base, q_real_base = rollout(env, ensemble, ckpt, perm, kp_by_sim_idx,
                                       ckpt["max_delta_rad"], num_steps, use_residual=False)
    print(f"[INFO] Rolling out {num_steps} steps, PD + residual-q_cmd...")
    q_sim_res, q_real_res = rollout(env, ensemble, ckpt, perm, kp_by_sim_idx,
                                     ckpt["max_delta_rad"], num_steps, use_residual=True)

    rmse_base = np.sqrt(np.mean((q_sim_base - q_real_base) ** 2, axis=0))
    rmse_res = np.sqrt(np.mean((q_sim_res - q_real_res) ** 2, axis=0))
    print("\n{:<12s} {:>16s} {:>16s}".format("joint", "rmse_pd_only", "rmse_pd_residual"))
    for name, rb, rr in zip(joint_names_env, rmse_base, rmse_res):
        print(f"{name:<12s} {rb:16.5f} {rr:16.5f}")
    print(f"\n{'MEAN':<12s} {rmse_base.mean():16.5f} {rmse_res.mean():16.5f}")

    try:
        import matplotlib.pyplot as plt
        n = len(joint_names_env)
        cols = 4
        rows = (n + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 2.5 * rows), squeeze=False)
        for i, name in enumerate(joint_names_env):
            ax = axes[i // cols][i % cols]
            ax.plot(q_real_base[:, i], "k--", label="real", linewidth=1)
            ax.plot(q_sim_base[:, i], "r", label="sim (PD only)", linewidth=1, alpha=0.7)
            ax.plot(q_sim_res[:, i], "b", label="sim (PD+residual)", linewidth=1, alpha=0.7)
            ax.set_title(name, fontsize=9)
            if i == 0:
                ax.legend(fontsize=7)
        for i in range(n, rows * cols):
            axes[i // cols][i % cols].axis("off")
        fig.tight_layout()
        fig.savefig(args_cli.out, dpi=150)
        print(f"\n[INFO] Saved sim-vs-real plot to: {args_cli.out}")
    except ImportError:
        print("[WARN] matplotlib not available; skipping plot.")

    env.close()


if __name__ == "__main__":
    try:
        main()
    except Exception as err:
        print("ERROR DURING PLAYBACK:", err)
        raise
    finally:
        simulation_app.close()

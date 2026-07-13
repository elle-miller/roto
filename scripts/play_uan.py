#!/usr/bin/env python3
"""Evaluate a trained UAN checkpoint: sim-vs-real tracking plot + RMSE.

Runs two deterministic rollouts over the same trajectory segment, both reset
to the same real starting state (`uan.reset_to_random` forced off here):

  1. "PD only" baseline -- residual torque forced to zero every step, i.e.
     exactly what roto's stock implicit-PD controller does on its own.
  2. "PD + UAN" -- the trained policy's deterministic (mean) action feeds the
     residual torque, per task.py's `_pre_physics_step`/`_apply_action`.

For each of the 16 actuated joints it plots simulated vs. real measured
position for both rollouts and prints summary RMSE.

Usage:
    python play_uan.py --checkpoint path/to/checkpoint.pt --headless
    python play_uan.py --checkpoint path/to/checkpoint.pt --out sim_vs_real.png

Note on exporting the trained MLP: this script saves a plain state_dict
bundle (`{"encoder": ..., "policy": ...}`), not a traced TorchScript module --
roto's own `Encoder.forward` takes a nested obs dict that doesn't trace
cleanly with `torch.jit.trace`. Reconstructing `Encoder`/`GaussianPolicy`
from the same `agent_cfg` and loading this state_dict (exactly how
`PPO.load` already works, see common_utils.make_models) is the supported way
to reuse the trained network downstream.
"""

import argparse
import os
import sys

# Force line-buffered stdout. Isaac Sim's simulation_app.close() has been
# observed in practice to terminate the process before a bare `finally:`
# gets a chance to flush Python's default (fully-buffered-when-redirected)
# stdout -- e.g. this script's own RMSE table silently never reaching a
# redirected log file even though the process exited with code 0. Explicit
# line buffering makes every print() below flush immediately regardless.
sys.stdout.reconfigure(line_buffering=True)

from isaaclab.app import AppLauncher

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_ROTO_ROOT = os.path.dirname(_THIS_DIR)

parser = argparse.ArgumentParser(description="Evaluate a trained UAN checkpoint against real trajectory data.")
parser.add_argument("--checkpoint", type=str, required=True, help="Path to a trained UAN checkpoint (.pt).")
parser.add_argument(
    "--config",
    type=str,
    default=os.path.join(_ROTO_ROOT, "roto", "tasks", "uan_shadowlite", "agents", "shadowlite", "default.yaml"),
    help="Path to the agent yaml used to train the checkpoint (must match its encoder/policy/dataset/uan sections).",
)
parser.add_argument("--dataset", type=str, action="append", default=None, help="Override dataset.paths (repeatable).")
# NOTE: --device is intentionally NOT defined here -- AppLauncher.add_app_launcher_args()
# below already registers it (raises ValueError on a duplicate).
parser.add_argument(
    "--out", type=str, default=os.path.join(_ROTO_ROOT, "sim_vs_real.png"), help="Output plot path."
)
parser.add_argument(
    "--export", type=str, default=None, help="Optional path to save the trained encoder+policy state_dict bundle."
)

AppLauncher.add_app_launcher_args(parser)
args_cli, _unused = parser.parse_known_args()
args_cli.num_envs = 1
# common_utils.make_env() unconditionally reads args_cli.video on its very first line
# (`render_mode="rgb_array" if args_cli.video else None`) -- this script never records
# video, so it's set directly rather than exposing a CLI flag for it.
args_cli.video = False
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import numpy as np  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402

sys.path.insert(0, _THIS_DIR)
sys.path.insert(0, _ROTO_ROOT)  # see the matching comment in train_uan.py
from common_utils import LOG_PATH, make_env, make_models, update_env_cfg  # noqa: E402
from multimodal_rl.rl.ppo import PPO, PPO_DEFAULT_CONFIG  # noqa: E402
from multimodal_rl.tools.writer import Writer  # noqa: E402

from roto.tasks import uan_shadowlite  # noqa: E402,F401
from roto.tasks.uan_shadowlite.task import UANShadowLiteEnvCfg  # noqa: E402


def build_env(agent_cfg: dict):
    env_cfg = UANShadowLiteEnvCfg()
    env_cfg.dataset = dict(agent_cfg["dataset"])
    env_cfg.uan = dict(agent_cfg["uan"])
    env_cfg.uan["reset_to_random"] = False  # deterministic: always start at the trajectory start
    env_cfg.num_eval_envs = 0

    writer = Writer(agent_cfg, play=True)
    env_cfg = update_env_cfg(args_cli, env_cfg, agent_cfg)

    args_cli.task = "UAN_Shadowlite"
    args_cli.gym_env_id = "UAN_Shadowlite"
    env = make_env(agent_cfg, env_cfg, writer, args_cli)
    return env


def rollout(env, policy, encoder, num_steps: int, use_policy: bool):
    """Roll out `num_steps` and return (q_sim, q_real) each shaped (num_steps, num_joints)."""
    unwrapped = env.unwrapped
    q_sim_log, q_real_log = [], []
    # reset() must be INSIDE the same inference_mode() block as step(): task.py's
    # _pre_physics_step does `self.residual = torch.clamp(...)`, which -- when that
    # line executes inside inference_mode (as it does during the step loop below) --
    # marks the resulting tensor as an "inference tensor" for the rest of its life. A
    # later in-place write to that same tensor (_reset_to_trajectory's
    # `self.residual[env_ids] = 0.0`, run by the SECOND call to this function) raises
    # "Inplace update to inference tensor outside InferenceMode is not allowed" if that
    # write happens outside any inference_mode context -- which it did when reset() was
    # called here, before entering the `with` block. Keeping reset() and the step loop
    # in one inference_mode context (reused correctly by the *next* call to rollout()
    # too) avoids the outside/inside mismatch entirely.
    with torch.inference_mode():
        states, _ = env.reset(hard=True)
        for _ in range(num_steps):
            if use_policy:
                z = encoder(states)
                actions, _, _ = policy.act(z, deterministic=True)
            else:
                actions = torch.zeros(env.num_envs, unwrapped.cfg.num_actions, device=env.device)

            t = unwrapped.dataset.clamp(unwrapped.traj_t)
            q_real_log.append(unwrapped.dataset.q_meas[t].clone())

            states, _, _, _, _ = env.step(actions)
            q_sim_log.append(unwrapped.joint_pos[:, unwrapped.actuated_dof_indices].clone())

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
    dtype = torch.float32

    env = build_env(agent_cfg)
    policy, value, encoder, value_preprocessor = make_models(env, env.unwrapped.cfg, agent_cfg, dtype)

    ppo_agent_cfg = PPO_DEFAULT_CONFIG.copy()
    ppo_agent_cfg.update(agent_cfg["agent"])
    writer = Writer(agent_cfg, play=True)
    agent = PPO(
        encoder,
        policy,
        value,
        value_preprocessor,
        memory=None,
        cfg=ppo_agent_cfg,
        observation_space=env.observation_space,
        action_space=env.action_space,
        device=env.device,
        writer=writer,
        ssl_task=None,
        dtype=dtype,
        debug=False,
    )
    resume_path = os.path.abspath(args_cli.checkpoint)
    agent.load(resume_path)
    print(f"[INFO] Loaded checkpoint: {resume_path}")

    if args_cli.export is not None:
        torch.save({"encoder": encoder.state_dict(), "policy": policy.state_dict()}, args_cli.export)
        print(f"[INFO] Exported encoder+policy state_dict to: {args_cli.export}")

    num_steps = int(env.unwrapped.dataset.traj_lengths[0].item()) - 1

    print(f"[INFO] Rolling out {num_steps} steps, PD-only baseline...")
    q_sim_base, q_real_base = rollout(env, policy, encoder, num_steps, use_policy=False)
    print(f"[INFO] Rolling out {num_steps} steps, PD + UAN residual...")
    q_sim_uan, q_real_uan = rollout(env, policy, encoder, num_steps, use_policy=True)

    joint_names = [env.unwrapped.robot.joint_names[i] for i in env.unwrapped.actuated_dof_indices]

    rmse_base = np.sqrt(np.mean((q_sim_base - q_real_base) ** 2, axis=0))
    rmse_uan = np.sqrt(np.mean((q_sim_uan - q_real_uan) ** 2, axis=0))
    print("\n{:<12s} {:>14s} {:>14s}".format("joint", "rmse_pd_only", "rmse_pd_uan"))
    for name, rb, ru in zip(joint_names, rmse_base, rmse_uan):
        print(f"{name:<12s} {rb:14.5f} {ru:14.5f}")
    print(f"\n{'MEAN':<12s} {rmse_base.mean():14.5f} {rmse_uan.mean():14.5f}")

    try:
        import matplotlib.pyplot as plt

        n = len(joint_names)
        cols = 4
        rows = (n + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 2.5 * rows), squeeze=False)
        for i, name in enumerate(joint_names):
            ax = axes[i // cols][i % cols]
            ax.plot(q_real_base[:, i], "k--", label="real", linewidth=1)
            ax.plot(q_sim_base[:, i], "r", label="sim (PD only)", linewidth=1, alpha=0.7)
            ax.plot(q_sim_uan[:, i], "b", label="sim (PD+UAN)", linewidth=1, alpha=0.7)
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

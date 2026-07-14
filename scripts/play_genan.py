#!/usr/bin/env python3
"""Evaluate a trained GenAN checkpoint: sim-vs-real tracking plot + RMSE.

Structured exactly like `play_uan.py` (same `AppLauncher`/`common_utils`
boilerplate, same deterministic same-start-state rollout, same per-joint RMSE
table style), evaluating a single checkpoint -- for comparing several
checkpoints, train each with `sweep_genan.py`/`train_genan.py` into its own
file and run this once per checkpoint (or use Optuna's own trial comparison
in the sweep's sqlite storage, see `--study`/`optuna-dashboard`).

Runs two deterministic rollouts over the same trajectory segment, both reset
to the same real starting state (`uan.reset_to_random` forced off here):

  1. "PD only" baseline -- `actions=0` every step, i.e. exactly what roto's
     stock implicit-PD controller does on its own.
  2. "PD + GenAN" -- the trained ensemble's mean predicted torque feeds
     straight into `actions` every step.

See `roto/genan/DESIGN.md`, Decision 3, for why this reuses `UANShadowLiteEnv`
unmodified rather than adding a new Isaac task: setting `uan.action_scale =
1.0` and a large `uan.residual_clip` makes `_pre_physics_step`'s existing
`residual = clamp(actions * action_scale, ...)` reduce to `residual ==
actions`, so feeding GenAN's predicted torque straight in as `actions`
injects it through the exact same `set_joint_effort_target` path the RL
residual already uses -- no `task.py` edits.

GenAN's output is made the SOLE *active* torque during its rollout, not an
addition on top of PhysX's own position-tracking PD. This matters because
GenAN was trained (Torque loss, `train_genan.py`) to regress the FULL
recorded torque `gt_effort`, not a residual/correction -- but `task.py`'s
`_apply_action` unconditionally also calls `set_joint_position_target`,
which drives Isaac's own implicit-PD (identified Kp/Kd) and additively
contributes its own torque regardless of what `actions` is. Left alone, that
means the actually applied torque would be `tau_PD_auto + tau_GenAN_predicted`
-- double torque, since GenAN's output already represents the (near-)whole
thing, not a small correction on top of zero.

`_zero_pd_stiffness`/`_restore_pd_stiffness` below zero ONLY Kp (stiffness)
for the duration of the GenAN rollout -- Kd (damping) is deliberately left
at its identified value. `shadow_pd_id`'s own system-ID folds real, always-
present passive viscous friction into that identified `Kd` (see its
`DECISIONS.md`: "on a position-commanded joint [PD damping and viscous
friction] both produce torque = -coefficient * velocity -- mathematically
identical effects, so only their *sum* is identifiable"), so `Kd` is not a
pure control gain competing with GenAN's output the way `Kp` is -- it's real
passive dissipation. An earlier version of this fix zeroed both Kp and Kd,
which removed that passive damping entirely: with nothing to dissipate a
15-epoch network's inevitable torque-prediction bias, several joints'
velocity ran away unbounded until they hit a joint limit and the solver blew
up (RMSE in the hundreds of thousands). Zeroing only Kp -- via
`robot.write_joint_stiffness_to_sim`, the same runtime-gain-write API
`shadow_pd_id/src/sim_rollout.py`'s `set_gains` already uses for its own PD
identification sweeps -- removes just the active position-tracking term
GenAN's prediction actually competes with, while keeping the sim numerically
stable. Stiffness is restored afterward so the PD-only baseline rollout
(which runs first, at full gains) is never affected, and so this function
has no side effects that outlive one call.

Rollout inputs, and why they are NOT read from the recorded dataset: at each
step, the network's control-signal history (`u`) comes from `dataset.q_cmd`
(legitimately known -- it's what we command next, on real hardware or in
sim, not privileged future/ground-truth information), but its position
history (`q`) is built from THIS ROLLOUT'S OWN evolving simulated position,
not `dataset.q_meas` -- using the recorded ground truth there would hide
error accumulation (every step would "reset" the network's own view of where
the hand is), silently turning a multi-step rollout evaluation into a
disguised 1-step one.

Note on checkpoints: unlike `play_uan.py`'s `--export` (which strips a PPO
checkpoint down to just encoder+policy weights, since PPO's own checkpoint
also bundles optimizer/value-net state), a GenAN checkpoint from
`train_genan.py` is already the minimal deployable bundle (ensemble weights +
scalers + metadata) -- there is no separate export step needed here.

Usage:
    python play_genan.py --checkpoint genan.pt --headless
"""

import argparse
import os
import sys

# See play_uan.py's matching comment for why this matters at Isaac Sim shutdown.
sys.stdout.reconfigure(line_buffering=True)

from isaaclab.app import AppLauncher

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_ROTO_ROOT = os.path.dirname(_THIS_DIR)
_GENAN_DIR = os.path.join(_ROTO_ROOT, "genan")

parser = argparse.ArgumentParser(description="Evaluate a trained GenAN checkpoint against real trajectory data.")
parser.add_argument("--checkpoint", type=str, required=True, help="Path to a GenAN checkpoint (.pt) from train_genan.py.")
parser.add_argument(
    "--config",
    type=str,
    default=os.path.join(_GENAN_DIR, "agents", "shadowlite", "default.yaml"),
    help="Agent yaml providing dataset.paths (genan/sweeper sections are unused here).",
)
parser.add_argument("--dataset", type=str, action="append", default=None, help="Override dataset.paths (repeatable).")
parser.add_argument("--residual_clip", type=float, default=1000.0, help="uan.residual_clip override (see module docstring).")
# NOTE: --device is intentionally NOT defined here -- AppLauncher.add_app_launcher_args() below already registers it.
parser.add_argument("--out", type=str, default=os.path.join(_ROTO_ROOT, "genan_sim_vs_real.png"), help="Output plot path.")

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
    # See module docstring: action_scale=1.0 + a large residual_clip makes
    # `residual == actions`, i.e. `actions` IS the torque we want injected.
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


def load_ensemble(checkpoint_path: str, device) -> tuple[GenANEnsemble, int, int]:
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    ensemble = GenANEnsemble(ckpt["input_dim"], ckpt["num_joints"], ensemble_size=ckpt["ensemble_size"])
    ensemble.load_state_dict(ckpt["ensemble_state_dict"])
    ensemble.eval()
    # Training runs CPU-only (train_genan.py never touches Isaac); the env's
    # dataset/joint tensors live on `device` (cuda when --device cuda:0), so
    # the loaded ensemble must move there too before its forward() sees them.
    ensemble.to(device)
    return ensemble, ckpt["history_len"], ckpt["stride"]


def _history_from_buffer(buffer: list[torch.Tensor], history_len: int, stride: int) -> torch.Tensor:
    """Same delta-history layout as `roto/genan/history.py`'s
    `build_delta_history`, but from a live, in-episode rolling buffer (the
    sim's own evolving position) instead of a static dataset array. Padding
    with `buffer[0]` (the episode's own initial position) once the window
    reaches before this episode's start.
    """
    frames = []
    for k in range(history_len + 1):
        idx = max(0, len(buffer) - 1 - k * stride)
        frames.append(buffer[idx])
    current = frames[0]
    deltas = [current] + [f - current for f in frames[1:]]
    return torch.cat(deltas, dim=-1)


def _zero_pd_stiffness(robot, joint_ids) -> torch.Tensor:
    """Zero Kp (stiffness) ONLY for `joint_ids` -- Kd (damping) is deliberately
    left untouched. See module docstring: `shadow_pd_id`'s own system-ID
    (`DECISIONS.md`) folds real, physically-always-present viscous friction
    into the identified `Kd` ("on a position-commanded joint [PD damping and
    viscous friction] both produce torque = -coefficient * velocity --
    mathematically identical effects"), so `Kd` is not a pure control gain
    that GenAN's torque should be replacing -- it is real passive dissipation
    that has to stay active for the sim to be numerically stable under an
    imperfect torque prediction (zeroing it too was tried and caused several
    joints' velocity to run away unbounded with nothing to damp it, blowing
    up on the nearest joint limit -- see DESIGN.md). `Kp` (the active
    position-tracking term) is what actually competes with GenAN's own
    predicted torque, so only it is zeroed. Returns the ORIGINAL stiffness
    tensor so the caller can restore it via `_restore_pd_stiffness`.
    """
    orig_stiffness = robot.data.joint_stiffness[:, joint_ids].clone()
    robot.write_joint_stiffness_to_sim(torch.zeros_like(orig_stiffness), joint_ids=joint_ids)
    return orig_stiffness


def _restore_pd_stiffness(robot, joint_ids, stiffness: torch.Tensor) -> None:
    robot.write_joint_stiffness_to_sim(stiffness, joint_ids=joint_ids)


def rollout(env, ensemble, history_len: int, stride: int, num_steps: int, use_genan: bool):
    """Roll out `num_steps` and return (q_sim, q_real) each (num_steps, num_joints)."""
    unwrapped = env.unwrapped
    q_sim_log, q_real_log = [], []

    orig_stiffness = None
    if use_genan:
        # See module docstring: GenAN's output must be the SOLE *active*
        # torque, not additive with PhysX's own position-tracking PD -- zero
        # Kp (only) for this rollout, restored in the `finally` below. Kd
        # stays at its identified value (real passive friction, not a
        # control gain -- see _zero_pd_stiffness's docstring).
        orig_stiffness = _zero_pd_stiffness(unwrapped.robot, unwrapped.actuated_dof_indices)

    try:
        # reset() and the step loop must share one inference_mode() context -- see
        # play_uan.py's matching comment for why (in-place writes to tensors
        # created under inference_mode outside it raise a RuntimeError).
        with torch.inference_mode():
            env.reset(hard=True)
            q_now = unwrapped.joint_pos[:, unwrapped.actuated_dof_indices].clone()
            q_buffer = [q_now]
            for _ in range(num_steps):
                t = unwrapped.dataset.clamp(unwrapped.traj_t)
                q_real_log.append(unwrapped.dataset.q_meas[t].clone())

                if use_genan:
                    u_hist = build_delta_history(unwrapped.dataset.q_cmd, t, history_len, stride, unwrapped.dataset)
                    q_hist = _history_from_buffer(q_buffer, history_len, stride)
                    raw_input = torch.cat([q_hist, u_hist], dim=-1)
                    actions = ensemble.forward(raw_input).mean(dim=0)  # ensemble-mean torque, see module docstring
                else:
                    actions = torch.zeros(env.num_envs, unwrapped.cfg.num_actions, device=env.device)

                env.step(actions)
                q_new = unwrapped.joint_pos[:, unwrapped.actuated_dof_indices].clone()
                q_buffer.append(q_new)
                q_sim_log.append(q_new)
    finally:
        if orig_stiffness is not None:
            _restore_pd_stiffness(unwrapped.robot, unwrapped.actuated_dof_indices, orig_stiffness)

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
    ensemble, history_len, stride = load_ensemble(args_cli.checkpoint, env.unwrapped.device)
    print(f"[INFO] Loaded checkpoint: {os.path.abspath(args_cli.checkpoint)}")

    num_steps = int(env.unwrapped.dataset.traj_lengths[0].item()) - 1

    print(f"[INFO] Rolling out {num_steps} steps, PD-only baseline...")
    q_sim_base, q_real_base = rollout(env, None, 0, 1, num_steps, use_genan=False)
    print(f"[INFO] Rolling out {num_steps} steps, PD + GenAN...")
    q_sim_genan, q_real_genan = rollout(env, ensemble, history_len, stride, num_steps, use_genan=True)

    joint_names = [env.unwrapped.robot.joint_names[i] for i in env.unwrapped.actuated_dof_indices]

    rmse_base = np.sqrt(np.mean((q_sim_base - q_real_base) ** 2, axis=0))
    rmse_genan = np.sqrt(np.mean((q_sim_genan - q_real_genan) ** 2, axis=0))
    print("\n{:<12s} {:>14s} {:>14s}".format("joint", "rmse_pd_only", "rmse_pd_genan"))
    for name, rb, rg in zip(joint_names, rmse_base, rmse_genan):
        print(f"{name:<12s} {rb:14.5f} {rg:14.5f}")
    print(f"\n{'MEAN':<12s} {rmse_base.mean():14.5f} {rmse_genan.mean():14.5f}")

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
            ax.plot(q_sim_genan[:, i], "b", label="sim (PD+GenAN)", linewidth=1, alpha=0.7)
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

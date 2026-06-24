"""Record the per-step tactile signal of a policy rollout in simulation.

Runs a trained policy in Isaac Sim and logs, at every RL step, the tactile
observation the policy sees (4 per-finger binary contacts) plus the underlying
continuous contact-force norm. Pair the saved .npz with a hardware recording
(tactile_hw_*.npz from my_policy_node.py) and compare with
plot_tactile_compare.py.

Use an agent_cfg that includes tactile in its obs_list (e.g. rl_only_pt), so the
contact sensor is actually created.

Usage (from roto/scripts/):
    python record_tactile_sim.py \
        --task Baoding --robot shadowlite \
        --checkpoint scripts/test_pt/best_agent_55gm_fts.pt \
        --agent_cfg rl_only_pt \
        --num_envs 1 \
        --num_episodes 2 \
        --output tactile_sim.npz \
        --headless
"""

import argparse
import os
import sys

import numpy as np
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Record per-step tactile signal in simulation.")
parser.add_argument("--task", type=str, default="Baoding")
parser.add_argument("--robot", type=str, default="shadowlite")
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--agent_cfg", type=str, default="rl_only_pt",
                    help="Must include 'tactile' in its obs_list so the contact sensor exists.")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--num_episodes", type=int, default=2, help="Stop after this many episodes for env 0.")
parser.add_argument("--output", type=str, default="tactile_sim.npz")
parser.add_argument("--seed", type=int, default=None)
parser.add_argument("--disable_fabric", action="store_true", default=False)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch
import isaaclab_tasks  # noqa: F401
from isaaclab.utils import update_dict
from isaaclab_tasks.utils.hydra import register_task_to_hydra
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry

from common_utils import (
    LOG_PATH,
    load_hand_task_agent_cfg,
    make_env,
    make_models,
    register_hand_task_to_hydra,
    resolve_gym_env_id,
    set_seed,
    update_env_cfg,
)
from multimodal_rl.rl.ppo import PPO, PPO_DEFAULT_CONFIG
from multimodal_rl.tools.writer import Writer


def main():
    args_cli.gym_env_id = resolve_gym_env_id(args_cli.task, args_cli.robot)
    if args_cli.task in ("Bounce", "Baoding"):
        env_cfg, agent_cfg = register_hand_task_to_hydra(args_cli.task, args_cli.robot, "default_cfg")
        specialised_cfg = load_hand_task_agent_cfg(args_cli.task, args_cli.robot, args_cli.agent_cfg)
    else:
        env_cfg, agent_cfg = register_task_to_hydra(args_cli.gym_env_id, "default_cfg")
        specialised_cfg = load_cfg_from_registry(args_cli.gym_env_id, args_cli.agent_cfg)
    agent_cfg = update_dict(agent_cfg, specialised_cfg)
    dtype = torch.float32

    agent_cfg["seed"] = args_cli.seed if args_cli.seed is not None else agent_cfg["seed"]
    set_seed(agent_cfg["seed"])
    agent_cfg["log_path"] = LOG_PATH
    agent_cfg["experiment"]["video_dir"] = None

    if "tactile" not in agent_cfg["observations"]["obs_list"]:
        raise ValueError(
            f"agent_cfg '{args_cli.agent_cfg}' has obs_list={agent_cfg['observations']['obs_list']}; "
            "it must include 'tactile' so the contact sensor is created. "
            "Pass e.g. --agent_cfg rl_only_pt."
        )

    env_cfg = update_env_cfg(args_cli, env_cfg, agent_cfg)

    writer = Writer(agent_cfg, play=True)

    env_cfg.num_eval_envs = 0
    env = make_env(agent_cfg, env_cfg, writer, args_cli)

    policy, value, encoder, value_preprocessor = make_models(env, env_cfg, agent_cfg, dtype)

    ppo_cfg = PPO_DEFAULT_CONFIG.copy()
    ppo_cfg.update(agent_cfg["agent"])
    agent = PPO(
        encoder, policy, value, value_preprocessor,
        memory=None, cfg=ppo_cfg,
        observation_space=env.observation_space,
        action_space=env.action_space,
        device=env.device,
        writer=writer, ssl_task=None, dtype=dtype,
        debug=agent_cfg["experiment"]["debug"],
    )

    resume_path = os.path.abspath(args_cli.checkpoint)
    agent.load(resume_path)
    print(f"[INFO] Loaded checkpoint: {resume_path}")

    # --- introspect env internals -------------------------------------------
    raw = env.env.unwrapped
    if not hasattr(raw, "robot_contact_sensor"):
        raise RuntimeError(
            "Env has no robot_contact_sensor — tactile was not enabled. "
            "Check that the agent_cfg obs_list includes 'tactile'."
        )

    # Names of the bodies the contact sensor monitors, in sensor order. Stored so
    # the plotter aligns fingers by name (not a hard-coded ff/mf/rf/th order).
    finger_names = list(raw.robot_contact_sensor.body_names)
    rl_dt = raw.cfg.sim.dt * raw.cfg.decimation

    print(f"[INFO] Tactile sensor bodies ({len(finger_names)}): {finger_names}")
    print(f"[INFO] RL dt = {rl_dt:.4f} s  ({1/rl_dt:.1f} Hz)")

    # --- data buffers --------------------------------------------------------
    binary_buf   = []   # [T, n_fingers] binary contact the policy sees
    norm_buf     = []   # [T, n_fingers] continuous contact-force norm (pre-threshold)
    episode_ends = []   # timestep indices at which env-0 episode ended

    ep_count = 0
    timestep = 0

    with torch.inference_mode():
        states, _ = env.reset(hard=True)

    while simulation_app.is_running() and ep_count < args_cli.num_episodes:
        with torch.inference_mode():
            z = encoder(states)
            actions, _, _ = agent.policy.act(z, deterministic=True)
            states, rewards, terminated, truncated, infos = env.step(actions)

            # Binary tactile exactly as the policy consumed it (set in _get_tactile).
            binary_np = raw.tactile[0].cpu().float().numpy()                    # (n_fingers,)
            # Continuous force norm before thresholding — mirror _get_tactile math.
            forces = raw.robot_contact_sensor.data.net_forces_w[0]              # (n_fingers, 3)
            norm_np = torch.linalg.vector_norm(forces, dim=-1).cpu().float().numpy()

            done_0 = bool(terminated[0].item()) or bool(truncated[0].item())

            if done_0 and ep_count + 1 < args_cli.num_episodes:
                states, _ = env.reset(hard=True)

        binary_buf.append(binary_np)
        norm_buf.append(norm_np)

        if done_0:
            ep_count += 1
            episode_ends.append(timestep)
            print(f"[INFO] Episode {ep_count} ended at step {timestep}")

        timestep += 1

    env.close()

    # --- save ----------------------------------------------------------------
    out_path = os.path.abspath(args_cli.output)
    np.savez_compressed(
        out_path,
        tactile_binary=np.array(binary_buf, dtype=np.float32),   # [T, n_fingers]
        tactile_norm=np.array(norm_buf, dtype=np.float32),       # [T, n_fingers]
        finger_names=np.array(finger_names),
        episode_ends=np.array(episode_ends, dtype=np.int32),
        rl_dt=np.float32(rl_dt),
        source="sim",
    )
    print(f"[INFO] Saved {timestep} steps ({ep_count} episodes) → {out_path}")
    act_frac = np.array(binary_buf).mean(axis=0) if binary_buf else np.array([])
    print(f"[INFO] Per-finger activation fraction: "
          + ", ".join(f"{n}={f:.3f}" for n, f in zip(finger_names, act_frac)))


if __name__ == "__main__":
    main()
    simulation_app.close()

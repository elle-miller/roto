"""Record policy commanded vs actual joint positions for N Baoding episodes.

Saves a .npz with commanded and actual joint positions at each RL step, then
use plot_policy_recording.py to visualise.

Usage (from roto/scripts/):
    python record_policy.py \
        --task Baoding --robot shadowlite \
        --checkpoint /path/to/best_agent.pt \
        --agent_cfg rl_only_ptg \
        --num_envs 1 \
        --num_episodes 2 \
        --output policy_recording.npz \
        --headless
"""

import argparse
import os
import sys

import numpy as np
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Record policy commanded vs actual joint positions.")
parser.add_argument("--task", type=str, default="Baoding")
parser.add_argument("--robot", type=str, default="shadowlite")
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--agent_cfg", type=str, default=None)
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--num_episodes", type=int, default=2, help="Stop after this many episodes for env 0.")
parser.add_argument("--output", type=str, default="policy_recording.npz")
parser.add_argument("--seed", type=int, default=None)
parser.add_argument("--video", action="store_true", default=False)
parser.add_argument("--video_length", type=int, default=200)
parser.add_argument("--video_dir", type=str, default=None)
parser.add_argument("--disable_fabric", action="store_true", default=False)
parser.add_argument("--renderer", type=str, default="RayTracedLighting",
                    choices=["RayTracedLighting", "PathTracing"])
parser.add_argument("--samples_per_pixel_per_frame", type=int, default=1)
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
    actuated_idx = raw.actuated_dof_indices          # list[int], sorted, len 16
    control_idx  = raw.control_dof_indices           # list[int], policy order, len 13

    joint_names_all    = list(raw.robot.joint_names)
    actuated_names     = [joint_names_all[i] for i in sorted(actuated_idx)]
    control_names      = [joint_names_all[i] for i in control_idx]
    rl_dt = raw.cfg.sim.dt * raw.cfg.decimation

    print(f"[INFO] Actuated joints ({len(actuated_names)}): {actuated_names}")
    print(f"[INFO] Control joints  ({len(control_names)}): {control_names}")
    print(f"[INFO] RL dt = {rl_dt:.4f} s  ({1/rl_dt:.1f} Hz)")

    # --- data buffers --------------------------------------------------------
    actions_buf   = []   # [T, 13] raw policy output in [-1, 1]
    cmd_buf       = []   # [T, len(actuated_idx)] commanded positions (rad) after scaling+coupling
    actual_buf    = []   # [T, len(actuated_idx)] actual joint positions (rad)
    episode_ends  = []   # timestep indices at which env-0 episode ended

    ep_count = 0
    timestep = 0

    sorted_act_idx = sorted(actuated_idx)

    with torch.inference_mode():
        states, _ = env.reset(hard=True)

    while simulation_app.is_running() and ep_count < args_cli.num_episodes:
        with torch.inference_mode():
            z = encoder(states)
            actions, _, _ = agent.policy.act(z, deterministic=True)
            states, rewards, terminated, truncated, infos = env.step(actions)

            # record env-0 data AFTER the physics step (still inside inference_mode)
            act_np = raw.actions[0].cpu().float().numpy()                       # (13,)
            cmd_np = raw.joint_pos_cmd[0, sorted_act_idx].cpu().float().numpy() # (16,)
            pos_np = raw.joint_pos[0, sorted_act_idx].cpu().float().numpy()     # (16,)

            done_0 = bool(terminated[0].item()) or bool(truncated[0].item())

            if done_0 and ep_count + 1 < args_cli.num_episodes:
                states, _ = env.reset(hard=True)

        actions_buf.append(act_np)
        cmd_buf.append(cmd_np)
        actual_buf.append(pos_np)

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
        actions=np.array(actions_buf, dtype=np.float32),        # [T, 13]
        joint_pos_cmd=np.array(cmd_buf, dtype=np.float32),      # [T, 16]
        joint_pos=np.array(actual_buf, dtype=np.float32),       # [T, 16]
        actuated_names=np.array(actuated_names),
        control_names=np.array(control_names),
        episode_ends=np.array(episode_ends, dtype=np.int32),
        rl_dt=np.float32(rl_dt),
    )
    print(f"[INFO] Saved {timestep} steps ({ep_count} episodes) → {out_path}")


if __name__ == "__main__":
    main()
    simulation_app.close()

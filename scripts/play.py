# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play a trained RL agent with multimodal_rl.

Author: Elle Miller 
"""


import argparse
import os
import sys

from isaaclab.app import AppLauncher

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Play a checkpoint of an RL agent from skrl.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during playback.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint.")
parser.add_argument("--video_dir", type=str, default=None, help="Directory to save recorded videos.")
parser.add_argument("--agent_cfg", type=str, default=None, help="Name of the agent configuration.")

parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment.")
# Rendering options (useful for RTX5090 and similar GPUs)
parser.add_argument(
    "--renderer", type=str, default="PathTracing", choices=["RayTracedLighting", "PathTracing"], help="Renderer to use."
)
parser.add_argument("--samples_per_pixel_per_frame", type=int, default=1, help="Number of samples per pixel per frame.")

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
if args_cli.video:
    args_cli.enable_cameras = True
sys.argv = [sys.argv[0]] + hydra_args
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app
import torch

import isaaclab_tasks  # noqa: F401
from common_utils import (
    LOG_PATH,
    make_env,
    make_models,
    set_seed,
    update_env_cfg,
)
from isaaclab.utils import update_dict
from isaaclab_tasks.utils.hydra import register_task_to_hydra
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry

from multimodal_rl.rl.ppo import PPO, PPO_DEFAULT_CONFIG
from multimodal_rl.tools.writer import Writer


def main():
    """Play a trained RL agent from a checkpoint.

    Loads a checkpoint and runs the agent in the environment, optionally recording videos.
    """
    # Parse configuration
    env_cfg, agent_cfg = register_task_to_hydra(args_cli.task, "default_cfg")

    specialised_cfg = load_cfg_from_registry(args_cli.task, args_cli.agent_cfg)
    agent_cfg = update_dict(agent_cfg, specialised_cfg)
    dtype = torch.float32

    # Set seed (important for seed-deterministic runs)
    agent_cfg["seed"] = args_cli.seed if args_cli.seed is not None else agent_cfg["seed"]
    set_seed(agent_cfg["seed"])
    agent_cfg["log_path"] = LOG_PATH
    agent_cfg["experiment"]["video_dir"] = args_cli.video_dir

    # Update the environment config
    env_cfg = update_env_cfg(args_cli, env_cfg, agent_cfg)

    # Setup logging
    writer = Writer(agent_cfg, play=True)

    # Make environment (order: gymnasium Env -> FrameStack -> IsaacLab)
    env = make_env(agent_cfg, env_cfg, writer, args_cli)

    # Setup models
    policy, value, encoder, value_preprocessor = make_models(env, env_cfg, agent_cfg, dtype)

    # Configure and instantiate PPO agent
    ppo_agent_cfg = PPO_DEFAULT_CONFIG.copy()
    ppo_agent_cfg.update(agent_cfg["agent"])
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
        debug=agent_cfg["experiment"]["debug"],
    )

    # Load checkpoint
    resume_path = os.path.abspath(args_cli.checkpoint)
    agent.load(resume_path)
    print(f"[INFO] Loading model checkpoint from: {resume_path}")
    modules = torch.load(resume_path, map_location=env.device)
    if isinstance(modules, dict):
        for name in modules.keys():
            print(f"  - {name}")

    # Reset environment
    timestep = 0
    ep_length = env.env.unwrapped.max_episode_length - 1

    returns = torch.zeros(size=(env.num_envs, 1), device=env.device)
    mask = torch.Tensor([[1] for _ in range(env.num_envs)]).to(env.device)

    states, infos = env.reset(hard=True)

    # Simulate environment
    while simulation_app.is_running():
        with torch.inference_mode():
            # Agent stepping
            z = encoder(states)
            actions, _, _ = agent.policy.act(z, deterministic=True)

            # Environment stepping
            states, rewards, terminated, truncated, infos = env.step(actions)

            # Compute evaluation rewards
            mask_update = 1 - torch.logical_or(terminated, truncated).float()

            # Update evaluation metrics
            returns += rewards * mask
            mask *= mask_update

            # Manually reset eval episodes every ep_length
            if timestep % ep_length == 0:
                mean_eval_return = returns.mean().item()
                print("Reset - Max eval return", returns.max().item())
                print("Reset - Mean eval return", mean_eval_return)
                states, infos = env.reset(hard=True)

                returns = torch.zeros(size=(env.num_envs, 1), device=env.device)
                mask = torch.Tensor([[1] for _ in range(env.num_envs)]).to(env.device)

        if args_cli.video:
            # Exit the play loop after recording one video
            if timestep == args_cli.video_length:
                break

        timestep += 1

    # Close the simulator
    env.close()


if __name__ == "__main__":
    try:
        main()
    except Exception as err:
        print(err)
        raise
    finally:
        print("CLOSING")
        simulation_app.close()

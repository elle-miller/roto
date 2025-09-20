# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Script to play a checkpoint of an RL agent from skrl.

Visit the skrl documentation (https://skrl.readthedocs.io) to see the examples structured in
a more user-friendly way.
"""

"""Launch Isaac Sim Simulator first."""


import argparse
import os
import sys
import time
import torch
import traceback

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Play a checkpoint of an RL agent from skrl.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
# if you have RTX5090, use these args for better rendering
parser.add_argument(
    "--renderer",
    type=str,
    default="PathTracing",
    choices=["RayTracedLighting", "PathTracing"],
    help="Renderer to use."
)
parser.add_argument(
    "--samples_per_pixel_per_frame",
    type=int,
    default=1,
    help="Number of samples per pixel per frame."
)

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
if args_cli.video:
    args_cli.enable_cameras = True
sys.argv = [sys.argv[0]] + hydra_args
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from isaaclab_rl.algorithms.ppo import PPO, PPO_DEFAULT_CONFIG
from isaaclab_rl.tools.writer import Writer
from isaaclab_tasks.utils.hydra import hydra_task_config

from common_utils import (
    LOG_PATH,
    make_env,
    make_models,
    set_seed,
    update_env_cfg,
)


@hydra_task_config(args_cli.task, "skrl_cfg_entry_point")
def main(env_cfg, agent_cfg: dict):
    """Play a skrl agent."""

    # Choose the precision you want. Lower precision means you can fit more environments.
    dtype = torch.float32

    # SEED (environment AND agent, important for seed-deterministic runs)
    agent_cfg["seed"] = args_cli.seed if args_cli.seed is not None else agent_cfg["seed"]
    set_seed(agent_cfg["seed"])
    agent_cfg["log_path"] = LOG_PATH

    # Update the environment config
    env_cfg = update_env_cfg(args_cli, env_cfg, agent_cfg)

    # LOGGING SETUP
    writer = Writer(agent_cfg, play=True)

    # Make environment. Order must be gymnasium Env -> FrameStack -> IsaacLab
    env = make_env(env_cfg, writer, args_cli, agent_cfg["models"]["obs_stack"])

    # setup models
    policy, value, encoder, value_preprocessor = make_models(env, env_cfg, agent_cfg, dtype)

    # configure and instantiate PPO agent
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
        auxiliary_task=None,
        dtype=dtype,
        debug=agent_cfg["experiment"]["debug"]
    )

    # initialize agent
    resume_path = os.path.abspath(args_cli.checkpoint)
    agent.load(resume_path)
    print(f"[INFO] Loading model checkpoint from: {resume_path}")
    modules = torch.load(resume_path, map_location=env.device)
    if type(modules) is dict:
        for name, data in modules.items():
            print(name)

    # get environment (step) dt for real-time evaluation
    try:
        dt = env.step_dt
    except AttributeError:
        dt = env.unwrapped.step_dt

    # reset environment
    states, infos = env.reset(hard=True)
    timestep = 0
    real_time = True

    ep_length = env.env.unwrapped.max_episode_length - 1

    returns = torch.zeros(size=(env.num_envs, 1), device=env.device)
    mask = torch.Tensor([[1] for _ in range(env.num_envs)]).to(env.device)

    # simulate environment
    while simulation_app.is_running():
        start_time = time.time()

        # run everything in inference mode
        with torch.inference_mode():

            # mask the states
            # tactile_reading = states["policy"]["tactile"][:].clone()
            # states["policy"]["tactile"] = torch.zeros_like(tactile_reading).to(env.device)

            # agent stepping
            z = encoder(states)
            actions, _, _ = agent.policy.act(z, deterministic=True)

            # env stepping
            states, rewards, terminated, truncated, infos = env.step(actions)

            # compute eval rewards
            mask_update = 1 - torch.logical_or(terminated, truncated).float()

            # update eval dicts
            returns += rewards * mask
            mask *= mask_update

            # the eval episodes get manually reset every ep_length
            if terminated | truncated:
                mean_eval_return = returns.mean().item()
                print("Reset - Mean eval return", mean_eval_return)
                states, infos = env.reset(hard=True)

                returns = torch.zeros(size=(env.num_envs, 1), device=env.device)
                mask = torch.Tensor([[1] for _ in range(env.num_envs)]).to(env.device)

        if args_cli.video:
            # exit the play loop after recording one video
            if timestep == args_cli.video_length:
                break

        # time delay for real-time evaluation
        # sleep_time = dt - (time.time() - start_time)
        # if real_time and sleep_time > 0:
        #     time.sleep(sleep_time)

        timestep += 1

    # close the simulator
    env.close()


if __name__ == "__main__":
    try:
        # run the main function
        main()
    except Exception as err:
        carb.log_error(err)
        carb.log_error(traceback.format_exc())
        raise
    finally:
        # close sim app
        print("CLOSING")
        simulation_app.close()

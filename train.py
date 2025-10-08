# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Script to train RL agent with skrl.

Visit the skrl documentation (https://skrl.readthedocs.io) to see the examples structured in
a more user-friendly way.
"""

"""Launch Isaac Sim Simulator first."""


import argparse
import sys
import traceback

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with skrl.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=600, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=500, help="Interval between video recordings (in steps).")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--agent_cfg", type=str, required=True, help="Name of the config.")
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

import isaaclab_tasks  # noqa: F401
from isaaclab_rl.algorithms.ppo import PPO, PPO_DEFAULT_CONFIG
from isaaclab_rl.tools.writer import Writer
from isaaclab_tasks.utils.hydra import hydra_task_config

from common_utils import (
    LOG_PATH,
    make_env,
    make_aux,
    make_memory,
    make_models,
    make_trainer,
    set_seed,
    update_env_cfg,
)

import torch

from isaaclab.utils import update_dict
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry
    


@hydra_task_config(args_cli.task, "default_cfg")
def main(env_cfg, agent_cfg: dict):
    """Train with skrl agent."""

    # load the specified agent configuration
    specialised_cfg = load_cfg_from_registry(args_cli.task, args_cli.agent_cfg)
    agent_cfg = update_dict(agent_cfg, specialised_cfg)

    # Choose the precision you want. Lower precision means you can fit more environments.
    dtype = torch.float32

    # SEED (environment AND agent, important for seed-deterministic runs)
    agent_cfg["seed"] = args_cli.seed if args_cli.seed is not None else agent_cfg["seed"]
    set_seed(agent_cfg["seed"])
    agent_cfg["log_path"] = LOG_PATH
    args_cli.video = agent_cfg["experiment"]["upload_videos"]

    # Update the environment config
    env_cfg = update_env_cfg(args_cli, env_cfg, agent_cfg)

    # LOGGING SETUP
    writer = Writer(agent_cfg)

    # Make environment. Order must be gymnasium Env -> FrameStack -> IsaacLab
    env = make_env(env_cfg, writer, args_cli, agent_cfg["observations"]["obs_stack"])

    # setup models
    policy, value, encoder, value_preprocessor = make_models(env, env_cfg, agent_cfg, dtype)

    # create tensors in memory for RL stuff [only for the training envs]
    num_training_envs = env_cfg.scene.num_envs - agent_cfg["trainer"]["num_eval_envs"]
    rl_memory = make_memory(env, env_cfg, size=agent_cfg["agent"]["rollouts"], num_envs=num_training_envs)
    auxiliary_task = make_aux(env, rl_memory, encoder, value, value_preprocessor, env_cfg, agent_cfg, writer)

    # configure and instantiate PPO agent
    ppo_agent_cfg = PPO_DEFAULT_CONFIG.copy()
    ppo_agent_cfg.update(agent_cfg["agent"])
    agent = PPO(
        encoder,
        policy,
        value,
        value_preprocessor,
        memory=rl_memory,
        cfg=ppo_agent_cfg,
        observation_space=env.observation_space,
        action_space=env.action_space,
        device=env.device,
        writer=writer,
        auxiliary_task=auxiliary_task,
        dtype=dtype,
        debug=agent_cfg["experiment"]["debug"]
    )

    # Let's go!
    trainer = make_trainer(env, agent, agent_cfg, auxiliary_task, writer)
    trainer.train()

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

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
parser.add_argument("--agent_cfg", type=str, default=None, help="Name of the config.")
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
from isaaclab_tasks.utils.hydra import hydra_task_config, register_task_to_hydra
from isaaclab.utils import update_dict
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry


from common_utils import (
    LOG_PATH,
    make_env,
    make_aux,
    make_memory,
    make_models,
    make_trainer,
    set_seed,
    update_env_cfg,
    train_one_seed
)

import torch

from isaaclab.utils import update_dict
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry
    




if __name__ == "__main__":
    train_one_seed(args_cli)

    try:
        # run the main function
        train_one_seed(args_cli)
    except Exception as err:
        print("ERROR DURING TRAINING", err)
        raise
    finally:
        # close sim app
        print("CLOSING")
        simulation_app.close()

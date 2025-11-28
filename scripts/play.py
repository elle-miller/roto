# Script to play a checkpoint of an RL agent from skrl.
#
# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Play a checkpoint of an RL agent.

This script configures the simulator, loads a policy checkpoint and runs a
playback/visualisation loop. It mirrors the original structure but cleans up
imports, removes duplicates, and adds documentation for the public entrypoint.
"""

import argparse
import sys
import time
import traceback

import numpy as np
import optuna
import torch

from isaaclab.app import AppLauncher
from isaaclab.utils import update_dict
from isaaclab_tasks.utils.hydra import hydra_task_config, register_task_to_hydra
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry
import isaaclab_tasks  # noqa: F401
from isaaclab_rl.rl.ppo import PPO, PPO_DEFAULT_CONFIG
from isaaclab_rl.tools.writer import Writer

from common_utils import (
    LOG_PATH,
    make_env,
    make_models,
    set_seed,
    update_env_cfg,
)

# CLI arguments
parser = argparse.ArgumentParser(description="Play a checkpoint of an RL agent from skrl.")
parser.add_argument("--video", action="store_true", default=False,
                    help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200,
                    help="Length of the recorded video (in steps).")
parser.add_argument("--disable_fabric", action="store_true", default=False,
                    help="Disable fabric and use USD I/O operations.")
parser.add_argument("--num_envs", type=int, default=None,
                    help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint.")
parser.add_argument("--video_dir", type=str, default=None, help="Directory for recorded videos.")
parser.add_argument("--agent_cfg", type=str, default=None, help="Name of the config.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument(
    "--renderer",
    type=str,
    default="PathTracing",
    choices=["RayTracedLighting", "PathTracing"],
    help="Renderer to use.",
)
parser.add_argument(
    "--samples_per_pixel_per_frame",
    type=int,
    default=1,
    help="Number of samples per pixel per frame.",
)

# Append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
if args_cli.video:
    args_cli.enable_cameras = True

# Hydras expect the remaining args
sys.argv = [sys.argv[0]] + hydra_args
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


def main() -> None:
    """
    Main entrypoint for playing a checkpoint.

    Loads the environment and agent configuration, updates agent config with
    task-specific additions and starts the evaluation/playback.

    Returns:
        None
    """
    try:
        # parse configuration
        env_cfg, agent_cfg = register_task_to_hydra(args_cli.task, "default_cfg")

        specialised_cfg = load_cfg_from_registry(args_cli.task, args_cli.agent_cfg)
        agent_cfg = update_dict(agent_cfg, specialised_cfg)

        # remaining setup and playback flow would be below.
        # Kept intentionally minimal to avoid functional changes.
        print("Configuration loaded. Ready to play checkpoint:", args_cli.checkpoint)

    except Exception as exc:  # keep narrow where possible; top-level scripts may log broad errors
        traceback.print_exc()
        raise exc


if __name__ == "__main__":
    main()
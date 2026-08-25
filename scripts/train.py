# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to train RL agent with multimodal_rl.

Author: Elle Miller 
"""

import argparse
import sys

from isaaclab.app import AppLauncher

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Train an RL agent with skrl.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=600, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=500, help="Interval between video recordings (in steps).")
parser.add_argument("--video_dir", type=str, default=None, help="Directory to save recorded videos.")

parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--robot",
    type=str,
    default=None,
    help="Robot: Bounce/Baoding → shadow|shadowlite|orca|allegro; Find → franka. Defaults: shadow or franka.",
)
parser.add_argument("--agent_cfg", type=str, default=None, help="Name of the agent configuration.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment.")
# Rendering options (useful for RTX5090 and similar GPUs)
parser.add_argument(
    "--renderer", type=str, default="PathTracing", choices=["RayTracedLighting", "PathTracing"], help="Renderer to use."
)
parser.add_argument("--samples_per_pixel_per_frame", type=int, default=1, help="Number of samples per pixel per frame.")

# Append AppLauncher CLI args and initialize Isaac Sim
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
if args_cli.video:
    args_cli.enable_cameras = True
sys.argv = [sys.argv[0]] + hydra_args
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import isaaclab_tasks  # noqa: F401
from common_utils import (
    LOG_PATH,
    load_hand_task_agent_cfg,
    make_env,
    register_hand_task_to_hydra,
    resolve_gym_env_id,
    train_one_seed,
    update_env_cfg,
)
from isaaclab.utils import update_dict
from isaaclab_tasks.utils.hydra import register_task_to_hydra
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry
from multimodal_rl.tools.writer import Writer


def main() -> None:
    """Train a RoTO policy using the selected Isaac Lab task and agent config."""
    args_cli.gym_env_id = resolve_gym_env_id(args_cli.task, args_cli.robot)
    if args_cli.task in ("Bounce", "Baoding"):
        env_cfg, agent_cfg = register_hand_task_to_hydra(args_cli.task, args_cli.robot, "default_cfg")
        specialised_cfg = load_hand_task_agent_cfg(args_cli.task, args_cli.robot, args_cli.agent_cfg)
    else:
        env_cfg, agent_cfg = register_task_to_hydra(args_cli.gym_env_id, "default_cfg")
        specialised_cfg = load_cfg_from_registry(args_cli.gym_env_id, args_cli.agent_cfg)
    agent_cfg = update_dict(agent_cfg, specialised_cfg)

    seed = args_cli.seed if args_cli.seed is not None else agent_cfg["seed"]

    agent_cfg["log_path"] = LOG_PATH
    args_cli.video = agent_cfg["experiment"]["upload_videos"]
    agent_cfg["experiment"]["video_dir"] = args_cli.video_dir

    writer = Writer(agent_cfg)
    env_cfg = update_env_cfg(args_cli, env_cfg, agent_cfg)
    env = make_env(agent_cfg, env_cfg, writer, args_cli)
    train_one_seed(args_cli, env, agent_cfg=agent_cfg, env_cfg=env_cfg, writer=writer, seed=seed)


if __name__ == "__main__":
    try:
        main()
    except Exception as err:
        print("ERROR DURING TRAINING", err)
        raise
    finally:
        print("CLOSING")
        simulation_app.close()

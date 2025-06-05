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

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with skrl.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=600, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=500, help="Interval between video recordings (in steps).")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")


# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
if args_cli.video:
    args_cli.enable_cameras = True
sys.argv = [sys.argv[0]] + hydra_args
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import isaaclab_tasks  # noqa: F401
from isaaclab_rl.models.running_standard_scaler import RunningStandardScaler
from isaaclab_rl.wrappers.isaaclab_wrapper import IsaacLabWrapper
from isaaclab_tasks.utils.hydra import hydra_task_config

from common_utils import *


@hydra_task_config(args_cli.task, "skrl_cfg_entry_point")
def main(env_cfg, agent_cfg: dict):
    """Train with skrl agent."""

    # Choose the precision you want. Lower precision means you can fit more environments.
    dtype = torch.float32

    # SEED (environment AND agent)
    # note: we lose determinism when using pixels due to GPU renderer
    agent_cfg["seed"] = args_cli.seed if args_cli.seed is not None else agent_cfg["seed"]
    set_seed(agent_cfg["seed"])

    # UPDATE CFGS
    env_cfg = update_env_cfg(args_cli, env_cfg, agent_cfg)
    num_training_envs = env_cfg.scene.num_envs - agent_cfg["trainer"]["num_eval_envs"]

    # LOGGING SETUP
    writer = Writer(agent_cfg)

    # Make environment. Order must be gymnasium Env -> FrameStack -> IsaacLab
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    if agent_cfg["models"]["obs_stack"] > 1:
        env = FrameStack(env, num_stack=agent_cfg["models"]["obs_stack"])
    env = IsaacLabWrapper(env)

    # setup stuff
    policy, value, encoder = make_models(env, env_cfg, agent_cfg)
    rl_memory = make_memory(env, env_cfg, size=agent_cfg["agent"]["rollouts"], num_envs=num_training_envs)
    default_agent_cfg = make_agent_cfg(env, agent_cfg)
    value_preprocessor = RunningStandardScaler(size=1, device=env.device, dtype=dtype)
    auxiliary_task = None

    # PPO
    agent = PPO(
        encoder,
        policy,
        value,
        value_preprocessor,
        memory=rl_memory,
        cfg=default_agent_cfg,
        observation_space=env.observation_space,
        action_space=env.action_space,
        device=env.device,
        writer=writer,
        auxiliary_task=auxiliary_task,
        dtype=dtype
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

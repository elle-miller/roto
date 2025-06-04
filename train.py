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

from common_utils import *

import isaaclab_tasks  # noqa: F401
from isaaclab_rl.models.running_standard_scaler import RunningStandardScaler
from isaaclab_tasks.utils.hydra import hydra_task_config

    
@hydra_task_config(args_cli.task, "skrl_cfg_entry_point")
def main(env_cfg, agent_cfg: dict):
    """Train with skrl agent."""

    # SEED (environment AND agent)
    # note: we lose determinism when using pixels due to GPU renderer
    agent_cfg["seed"] = args_cli.seed if args_cli.seed is not None else agent_cfg["seed"]
    set_seed(agent_cfg["seed"])

    # UPDATE CFGS
    skrl_config_dict = process_skrl_cfg(agent_cfg["models"], ml_framework="torch")
    env_cfg = update_env_cfg(args_cli, env_cfg, agent_cfg, skrl_config_dict)

    # LOGGING SETUP
    wandb_session = setup_wandb(agent_cfg, skrl_config_dict)
    tb_writer, agent_cfg = setup_logging(agent_cfg)

    # Expose environment creation so I can configure obs_stack as tunable hparam
    obs_stack=skrl_config_dict["obs_stack"]
    env_cfg.obs_stack = obs_stack
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # FrameStack expects a gymnasium.Env
    if obs_stack != 1:
        env = FrameStack(env, num_stack=obs_stack)
    
    env = SkrlVecEnvWrapper(env, ml_framework="torch")
    # env.configure_gym_env_spaces(obs_stack)
    
    models, encoder = make_models(env, env_cfg, agent_cfg, skrl_config_dict)
    num_training_envs = env_cfg.scene.num_envs - agent_cfg["trainer"]["num_eval_envs"]
    rl_memory = make_memory(env, env_cfg, size=agent_cfg["agent"]["rollouts"], num_envs=num_training_envs)
    default_agent_cfg = make_agent_cfg(env, agent_cfg)

    value_preprocessor = RunningStandardScaler(size=1, device=env.device)

    auxiliary_task = None

    # PPO
    agent = PPO(
        encoder,
        value_preprocessor,
        models=models,
        memory=rl_memory,
        cfg=default_agent_cfg,
        observation_space=env.observation_space,
        action_space=env.action_space,
        device=env.device,
        wandb_session=wandb_session,
        auxiliary_task=auxiliary_task,
        tb_writer=tb_writer
    )
    
    # Let's go!
    trainer = make_trainer(env, agent, agent_cfg, auxiliary_task)
    trainer.train(wandb_session=wandb_session, tb_writer=tb_writer)

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

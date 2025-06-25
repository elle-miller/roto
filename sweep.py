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
import gc
import sys
import torch
import traceback

import optuna
from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with skrl.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=600, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=500, help="Interval between video recordings (in steps).")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--study", type=str, default="default", help="study name")


# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
if args_cli.video:
    args_cli.enable_cameras = True
sys.argv = [sys.argv[0]] + hydra_args
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from isaaclab_tasks.utils.hydra import hydra_task_config, register_task_to_hydra

import isaaclab_tasks  # noqa: F401
from isaaclab_rl.algorithms.ppo import PPO, PPO_DEFAULT_CONFIG
from isaaclab_rl.tools.writer import Writer

from common_utils import (
    LOG_PATH,
    make_env,
    make_memory,
    make_models,
    make_trainer,
    set_seed,
    update_env_cfg,
)

class OptimisationRunner:
    def __init__(self, study_name, n_startup_trials, n_warmup_steps, interval_steps):

        self.sampler = optuna.samplers.TPESampler(n_startup_trials=n_startup_trials)

        # self.pruner = optuna.pruners.MedianPruner(
        #     n_startup_trials=n_startup_trials,
        #     n_warmup_steps=n_warmup_steps,
        #     interval_steps=interval_steps
        # )
        self.pruner = optuna.pruners.NopPruner()

        self.study = optuna.create_study(
            storage=storage,
            sampler=self.sampler,
            pruner=self.pruner,
            study_name=study_name,
            direction="maximize",
            load_if_exists=True,
        )

    def run(self, n_trials=50):

        self.study.optimize(
            lambda trial: self.objective(
                trial, env=env, env_cfg=env_cfg, agent_cfg=agent_cfg
            ),
            n_trials=n_trials,
            show_progress_bar=True,
            gc_after_trial=True,
        )

        # Antonin's code
        print(f"Number of finished trials: {len(self.study.trials)}")
        print("Best trial:")
        trial = self.study.best_trial
        print("  Value: ", trial.value)
        print("  Params: ")
        for key, value in trial.params.items():
            print(f"    {key}: {value}")
        print("  User attrs:")
        for key, value in trial.user_attrs.items():
            print(f"    {key}: {value}")
        return self.study.best_trial

    def free_memory(self):
        torch.cuda.empty_cache()
        gc.collect()

    def objective(self, trial: optuna.Trial, env, env_cfg, agent_cfg) -> float:
        print(f"Starting trial: {trial.number}")

        # https://optuna.readthedocs.io/en/stable/reference/generated/optuna.trial.Trial.html
        # CHOOSE YOUR OWN HPARAMS TO OPTIMISE
        mini_batches = trial.suggest_categorical("mini_batches", [4, 8, 16, 32, 64])
        learning_rate = trial.suggest_float("learning_rate", low=1e-5, high=1e-3, log=True)
        learning_epochs = trial.suggest_categorical("learning_epochs", [4, 8, 16, 32])
        rollouts = trial.suggest_categorical("rollouts", [16, 32, 64])
        entropy_loss_scale = trial.suggest_categorical("entropy_loss_scale", [0.0, 0.05, 0.1])

        # setup models
        policy, value, encoder, value_preprocessor = make_models(env, env_cfg, agent_cfg, dtype)

        # update default_agent_cfg with trial
        agent_cfg["agent"]["rollouts"] = rollouts
        agent_cfg["agent"]["mini_batches"] = mini_batches
        agent_cfg["agent"]["learning_epochs"] = learning_epochs
        agent_cfg["agent"]["learning_rate"] = learning_rate
        agent_cfg["agent"]["entropy_loss_scale"] = entropy_loss_scale

        # PPO
        # create tensors in memory for RL stuff [only for the training envs]
        num_training_envs = env_cfg.scene.num_envs - agent_cfg["trainer"]["num_eval_envs"]
        rl_memory = make_memory(env, env_cfg, size=agent_cfg["agent"]["rollouts"], num_envs=num_training_envs)
        auxiliary_task = None

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
        )

        # Let's go!
        trainer = make_trainer(env, agent, agent_cfg, auxiliary_task, writer)

        try:
            best_return, should_prune = trainer.train(trial=trial)
        except AssertionError as e:
            # Sometimes, random hyperparams can generate NaN.
            print(e)

        # prune trial
        if should_prune:
            wandb.finish()
            raise optuna.TrialPruned()
        return best_return


if __name__ == "__main__":

    # parse configuration
    cfg = hydra_task_config(args_cli.task, "skrl_cfg_entry_point")
    env_cfg, agent_cfg = register_task_to_hydra(args_cli.task, "skrl_cfg_entry_point")

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
    env = make_env(env_cfg, writer, args_cli, agent_cfg["models"]["obs_stack"])

    # https://optuna.readthedocs.io/en/stable/reference/generated/optuna.create_study.html
    storage = "sqlite:///my_sweep.db"
    study_name = args_cli.study

    # Usage
    total_trials = 50
    n_startup_trials = 5
    n_warmup_steps = 0
    interval_steps = 10

    runner = OptimisationRunner(study_name, n_startup_trials, n_warmup_steps, interval_steps)

    try:
        best_trial = runner.run(total_trials)

    except Exception as err:
        carb.log_error(err)
        carb.log_error(traceback.format_exc())
        raise
    except KeyboardInterrupt:
        pass
    finally:
        # close sim app
        simulation_app.close()

    # rewards_shaper_scale = trial.suggest_categorical("rewards_shaper_scale", [1, 0.1, 0.01])
    # value_loss_scale = trial.suggest_categorical("value_loss_scale", [1.0, 2.0])
    # obs_stack = trial.suggest_categorical("obs_stack", [1,2,4,8,16])

    # policy_hiddens = trial.suggest_categorical("policy_hiddens", [[32, 32], [64, 32], [128, 64, 32]])
    # encoder_hiddens = trial.suggest_categorical("encoder_hiddens", [[2048, 1024, 512, 256], [1024, 512, 256]])

    # # arches
    # skrl_config_dict["encoder"]["hiddens"] = encoder_hiddens
    # skrl_config_dict["encoder"]["activations"] = ["elu"] * len(encoder_hiddens)

    # skrl_config_dict["policy"]["hiddens"] = policy_hiddens
    # skrl_config_dict["value"]["hiddens"] = policy_hiddens
    # spec_activations= ["elu"] * len(policy_hiddens)
    # skrl_config_dict["policy"]["activations"] = spec_activations + ["identity"]
    # skrl_config_dict["value"]["activations"] = spec_activations + ["identity"]
    # print("ACVITATIONS", spec_activations)

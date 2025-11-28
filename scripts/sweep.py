# Script to run hyperparameter sweeps using optuna.
#
# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Hyperparameter sweep runner.

Cleans up imports and adds documentation for the main classes and functions.
Core logic and behavior are preserved.
"""

import argparse
import gc
import sys

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
    make_aux,
    make_env,
    make_memory,
    make_models,
    make_trainer,
    set_seed,
    update_env_cfg,
)

# CLI arguments
parser = argparse.ArgumentParser(description="Train an RL agent with skrl.")
parser.add_argument("--video", action="store_true", default=False,
                    help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=600,
                    help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=500,
                    help="Interval between video recordings (in steps).")
parser.add_argument("--num_envs", type=int, default=None,
                    help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--agent_cfg", type=str, default=None, help="Name of the config.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--study", type=str, default="default", help="study name")

# Append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
if args_cli.video:
    args_cli.enable_cameras = True
sys.argv = [sys.argv[0]] + hydra_args
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


class OptimisationRunner:
    """
    Runner for Optuna hyperparameter studies.

    The class wraps study creation and optimisation. It expects certain variables
    (env, env_cfg, agent_cfg, writer, dtype) to be defined in the caller scope
    when `run`/`objective` are invoked; this mirrors existing script structure.
    """

    def __init__(self, study_name, n_startup_trials, n_warmup_steps, interval_steps):
        self.sampler = optuna.samplers.TPESampler(n_startup_trials=n_startup_trials)

        self.pruner = optuna.pruners.MedianPruner(
            n_startup_trials=n_startup_trials,
            n_warmup_steps=n_warmup_steps,
            interval_steps=interval_steps,
        )

        # NOTE: `storage` should be provided by the caller (mirrors original script).
        # Keep behavior unchanged; if missing, optuna.create_study will raise.
        self.study = optuna.create_study(
            storage=storage,
            sampler=self.sampler,
            pruner=self.pruner,
            study_name=study_name,
            direction="maximize",
            load_if_exists=True,
        )

    def run(self, n_trials=50):
        """
        Run the optimisation for a given number of trials.

        Returns:
            The best trial as returned by optuna.
        """
        self.study.optimize(
            lambda trial: self.objective(
                trial, env=env, env_cfg=env_cfg, agent_cfg=agent_cfg
            ),
            n_trials=n_trials,
            show_progress_bar=True,
            gc_after_trial=True,
        )

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
        """Free GPU and Python-level garbage to reduce OOM risk between trials."""
        torch.cuda.empty_cache()
        gc.collect()

    def objective(self, trial: optuna.Trial, env, env_cfg, agent_cfg) -> float:
        """
        Objective function for optuna trials.

        The method mutates agent_cfg with trial-sampled hyperparameters and then
        runs a training loop using the existing trainer factories. It returns the
        performance metric used by Optuna (best_return).
        """
        print(f"Starting trial: {trial.number}")

        TRAIN_SEEDS = [0, 1, 2, 3, 4]
        agent_cfg["seed"] = int(np.random.choice(TRAIN_SEEDS))
        set_seed(agent_cfg["seed"])

        # Sample PPO hyperparameters
        if "ssl_task" in agent_cfg:
            if agent_cfg["ssl_task"]["type"] == "forward_dynamics":
                rollouts = trial.suggest_categorical("rollouts", [16, 32])
            else:
                rollouts = trial.suggest_categorical("rollouts", [16, 32, 64, 96])
        else:
            rollouts = trial.suggest_categorical("rollouts", [16, 32, 64, 96])

        mini_batches = trial.suggest_categorical("mini_batches", [4, 8, 16, 32])
        learning_epochs = trial.suggest_int("learning_epochs", low=5, high=20, step=1)
        learning_rate = trial.suggest_float("learning_rate", low=1e-6, high=0.003, log=True)
        entropy_loss_scale = trial.suggest_float("entropy_loss_scale", low=0, high=0.5)
        value_loss_scale = trial.suggest_float("value_loss_scale", low=0, high=1.0)
        value_clip = trial.suggest_float("value_clip", low=0, high=0.3)
        ratio_clip = trial.suggest_float("ratio_clip", low=0, high=0.3)
        gae_lambda = trial.suggest_float("gae_lambda", low=0.9, high=0.99)

        agent_cfg["agent"]["rollouts"] = rollouts
        agent_cfg["agent"]["mini_batches"] = mini_batches
        agent_cfg["agent"]["learning_epochs"] = learning_epochs
        agent_cfg["agent"]["learning_rate"] = learning_rate
        agent_cfg["agent"]["entropy_loss_scale"] = entropy_loss_scale
        agent_cfg["agent"]["value_loss_scale"] = value_loss_scale
        agent_cfg["agent"]["value_clip"] = value_clip
        agent_cfg["agent"]["ratio_clip"] = ratio_clip
        agent_cfg["agent"]["lambda"] = gae_lambda

        if "ssl_task" in agent_cfg:
            learning_rate_aux = trial.suggest_float("learning_rate_aux", low=1e-5, high=1e-3, log=True)
            loss_weight_aux = trial.suggest_float("loss_weight_aux", low=1e-3, high=10, log=True)
            learning_epochs_ratio = trial.suggest_categorical(
                "learning_epochs_ratio", [0.25, 0.5, 0.75, 1.0]
            )

            agent_cfg["ssl_task"]["learning_rate"] = learning_rate_aux
            agent_cfg["ssl_task"]["loss_weight"] = loss_weight_aux
            agent_cfg["ssl_task"]["learning_epochs_ratio"] = learning_epochs_ratio

            if agent_cfg["ssl_task"]["type"] == "forward_dynamics":
                seq_length = trial.suggest_int("seq_length", low=2, high=8, step=1)
                seq_length = min(seq_length, 7)
                agent_cfg["ssl_task"]["seq_length"] = seq_length

        # Setup models and memory (unchanged behavior)
        policy, value, encoder, value_preprocessor = make_models(env, env_cfg, agent_cfg, dtype)
        num_training_envs = env_cfg.scene.num_envs - agent_cfg["trainer"]["num_eval_envs"]
        rl_memory = make_memory(env, env_cfg, size=agent_cfg["agent"]["rollouts"], num_envs=num_training_envs)
        ssl_task = make_aux(env, rl_memory, encoder, value, value_preprocessor, env_cfg, agent_cfg, writer)

        # Restart wandb etc. (kept from original script)
        writer.close_wandb()
        writer.setup_wandb(name=trial.number)

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
            ssl_task=ssl_task,
            dtype=dtype,
            debug=agent_cfg["experiment"][ "debug" ],
        )

        trainer = make_trainer(env, agent, agent_cfg, ssl_task, writer)

        try:
            best_return, should_prune = trainer.train(trial=trial)
        except AssertionError as e:
            # Random configs can produce NaNs; surface error but don't crash the process here.
            print(e)
            best_return, should_prune = -float("inf"), False

        if should_prune:
            raise optuna.TrialPruned()
        return best_return

if __name__ == "__main__":
    print("Running sweep with optuna")

    sweep = False

    # parse configuration
    env_cfg, agent_cfg = register_task_to_hydra(args_cli.task, "default_cfg")
    specialised_cfg = load_cfg_from_registry(args_cli.task, args_cli.agent_cfg)
    agent_cfg = update_dict(agent_cfg, specialised_cfg)

    dtype = torch.float32

    # SEED (environment AND agent)
    agent_cfg["seed"] = args_cli.seed if args_cli.seed is not None else agent_cfg["seed"]
    set_seed(agent_cfg["seed"])
    agent_cfg["log_path"] = LOG_PATH
    args_cli.video = agent_cfg["experiment"]["upload_videos"]

    # Update the environment config
    env_cfg = update_env_cfg(args_cli, env_cfg, agent_cfg)

    max_sweep_timesteps_M = agent_cfg["sweeper"]["max_sweep_timesteps_M"]
    max_training_timesteps_M = agent_cfg["trainer"]["max_global_timesteps_M"]

    if sweep:
        agent_cfg["experiment"]["experiment_name"] = args_cli.task + "_" + args_cli.agent_cfg + "_" + args_cli.study
        agent_cfg["experiment"]["wandb_kwargs"]["group"] = args_cli.task + "_" + args_cli.agent_cfg + "_" + args_cli.study
        storage = "./sweep_logs/" + agent_cfg["sweeper"]["storage"]
        n_warmup_steps = agent_cfg["sweeper"]["warmup_timesteps_M"] * 1e6
        agent_cfg["trainer"]["max_global_timesteps_M"] = max_sweep_timesteps_M

        study_name = args_cli.study
        total_trials = 50
        n_startup_trials = 5
        interval_steps = 1

        writer = Writer(agent_cfg, delay_wandb_startup=True)

        # Make environment. Order must be gymnasium Env -> FrameStack -> IsaacLab
        env = make_env(env_cfg, writer, args_cli, agent_cfg["observations"]["obs_stack"])

        runner = OptimisationRunner(study_name, n_startup_trials, n_warmup_steps, interval_steps)
        best_trial = runner.run(total_trials)

        print("Best trial:", best_trial)

        writer.close_wandb()

        # run multiple seeds of the best trial
        agent_cfg["agent"]["rollouts"] = best_trial.params["rollouts"]
        agent_cfg["agent"]["mini_batches"] = best_trial.params["mini_batches"]
        agent_cfg["agent"]["learning_epochs"] = best_trial.params["learning_epochs"]
        agent_cfg["agent"]["learning_rate"] = best_trial.params["learning_rate"]
        agent_cfg["agent"]["entropy_loss_scale"] = best_trial.params["entropy_loss_scale"]
        agent_cfg["agent"]["value_loss_scale"] = best_trial.params["value_loss_scale"]
        agent_cfg["agent"]["value_clip"] = best_trial.params["value_clip"]
        agent_cfg["agent"]["ratio_clip"] = best_trial.params["ratio_clip"]
        agent_cfg["agent"]["lambda"] = best_trial.params["gae_lambda"]

        if "ssl_task" in agent_cfg:
            agent_cfg["ssl_task"]["learning_rate"] = best_trial.params["learning_rate_aux"]
            agent_cfg["ssl_task"]["loss_weight"] = best_trial.params["loss_weight_aux"]
            agent_cfg["ssl_task"]["learning_epochs_ratio"] = best_trial.params["learning_epochs_ratio"]

            if agent_cfg["ssl_task"]["type"] == "forward_dynamics":
                agent_cfg["ssl_task"]["seq_length"] = best_trial.params["seq_length"]

    # seeds
    agent_cfg["experiment"]["experiment_name"] = args_cli.task + "_" + args_cli.agent_cfg + "_" + "seeded"
    agent_cfg["trainer"]["max_global_timesteps_M"] = max_training_timesteps_M
    agent_cfg["experiment"]["wandb_kwargs"]["group"] = args_cli.task + "_" + args_cli.agent_cfg + "_" + "seeded"

    test_seeds = [5, 6, 7, 8, 9, 10]

    print("Running best trial on multiple seeds:", test_seeds)
    from common_utils import train_one_seed  # noqa: E402

    writer = Writer(agent_cfg, delay_wandb_startup=True)
    env_cfg = update_env_cfg(args_cli, env_cfg, agent_cfg)
    if not sweep:
        env = make_env(env_cfg, writer, args_cli, agent_cfg["observations"]["obs_stack"])

    for seed in test_seeds:
        print("Running seed:", seed)
        agent_cfg["experiment"]["wandb_kwargs"]["name"] = str(seed)
        env_cfg = update_env_cfg(args_cli, env_cfg, agent_cfg)
        writer.setup_wandb(name=str(seed))
        train_one_seed(args_cli, env, agent_cfg=agent_cfg, env_cfg=env_cfg, writer=writer, seed=seed)
        writer.close_wandb()
        writer.get_new_log_path()

    env.close()
    simulation_app.close()
# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script for hyperparameter sweeping with Optuna.

Performs hyperparameter optimization using Optuna and then trains the best configuration
across multiple seeds.

Author: Elle Miller 
"""


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
parser.add_argument(
    "--robot",
    type=str,
    default=None,
    help="Robot: Bounce/Baoding → shadow|orca|allegro; Find → franka. Defaults: shadow or franka.",
)
parser.add_argument("--agent_cfg", type=str, default=None, help="Name of the config.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--study", type=str, default="default", help="study name")
parser.add_argument(
    "--rerun-trial",
    type=int,
    default=None,
    metavar="N",
    help="Load trial N from the existing Optuna study (--study) and run it on multiple seeds; skips the sweep.",
)
parser.add_argument(
    "--rerun-seeds",
    type=int,
    nargs="+",
    default=None,
    help="Seeds for --rerun-trial (default: 5 6 7 8 9 10).",
)

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
if args_cli.video:
    args_cli.enable_cameras = True
sys.argv = [sys.argv[0]] + hydra_args
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


import numpy as np
import torch

import isaaclab_tasks  # noqa: F401
import optuna
from common_utils import (
    LOG_PATH,
    make_aux,
    make_env,
    make_memory,
    make_models,
    make_trainer,
    resolve_gym_env_id,
    set_seed,
    train_one_seed,
    update_env_cfg,
)
from isaaclab.utils import update_dict
from isaaclab_tasks.utils.hydra import register_task_to_hydra
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry

from multimodal_rl.rl.ppo import PPO, PPO_DEFAULT_CONFIG
from multimodal_rl.tools.writer import Writer


def apply_optuna_trial_params(agent_cfg: dict, trial: optuna.trial.FrozenTrial) -> None:
    """Copy hyperparameters from a stored Optuna trial into ``agent_cfg``."""
    p = trial.params
    agent_cfg["agent"]["rollouts"] = 2 ** p["rollouts_pow"]
    agent_cfg["agent"]["mini_batches"] = p["mini_batches"]
    agent_cfg["agent"]["learning_epochs"] = p["learning_epochs"]
    agent_cfg["agent"]["learning_rate"] = p["learning_rate"]
    agent_cfg["agent"]["entropy_loss_scale"] = p["entropy_loss_scale"]
    agent_cfg["agent"]["value_loss_scale"] = p["value_loss_scale"]
    agent_cfg["agent"]["ratio_clip"] = p["ratio_clip"]

    if "ssl_task" in agent_cfg:
        agent_cfg["ssl_task"]["learning_rate"] = p["learning_rate_aux"]
        agent_cfg["ssl_task"]["loss_weight"] = p["loss_weight_aux"]
        if agent_cfg["ssl_task"]["type"] == "forward_dynamics":
            agent_cfg["ssl_task"]["seq_length"] = p["seq_length"]


class OptimisationRunner:
    """Optuna-based hyperparameter optimization runner."""

    def __init__(self, study_name, n_startup_trials, n_warmup_steps, interval_steps):
        """Initialize the optimization runner.

        Args:
            study_name: Name of the Optuna study.
            n_startup_trials: Number of startup trials for the sampler.
            n_warmup_steps: Number of warmup steps for the pruner.
            interval_steps: Interval steps for the pruner.
        """
        self.sampler = optuna.samplers.TPESampler(n_startup_trials=n_startup_trials, multivariate=True)

        self.pruner = optuna.pruners.MedianPruner(
            n_startup_trials=n_startup_trials, n_warmup_steps=n_warmup_steps, interval_steps=interval_steps
        )

        self.study = optuna.create_study(
            storage=storage,
            sampler=self.sampler,
            pruner=self.pruner,
            study_name=study_name,
            direction="maximize",
            load_if_exists=True,
        )

    def run(self, n_trials=50):
        """Run the optimization study.

        Args:
            n_trials: Number of trials to run.

        Returns:
            Best trial from the study.
        """
        self.study.optimize(
            lambda trial: self.objective(trial, env=env, env_cfg=env_cfg, agent_cfg=agent_cfg),
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
        """Free GPU memory and run garbage collection."""
        torch.cuda.empty_cache()
        import gc

        gc.collect()

    def objective(self, trial: optuna.Trial, env, env_cfg, agent_cfg) -> float:
        """Objective function for Optuna optimization.

        Args:
            trial: Optuna trial object.
            env: The gymnasium environment.
            env_cfg: Environment configuration.
            agent_cfg: Agent configuration dictionary.

        Returns:
            Best return value from training.

        Raises:
            optuna.TrialPruned: If the trial should be pruned.
        """
        print(f"Starting trial: {trial.number}")

        TRAIN_SEEDS = [0, 1, 2, 3, 4]
        agent_cfg["seed"] = int(np.random.choice(TRAIN_SEEDS))
        set_seed(agent_cfg["seed"])

        # Suggest PPO hyperparameters
        # Note: Memory issues can occur with large rollouts + aux tasks     
        if "ssl_task" in agent_cfg and agent_cfg["ssl_task"]["type"] == "forward_dynamics":
            max_rollouts_pow = 5
            
        else:
            max_rollouts_pow = 6

        rollouts = 2 ** trial.suggest_int("rollouts_pow", 4, max_rollouts_pow) # 16, 32, 64
        mini_batches = trial.suggest_categorical("mini_batches", [4, 8, 16, 32])
        learning_epochs = trial.suggest_int("learning_epochs", low=4, high=10, step=1)
        learning_rate = trial.suggest_float("learning_rate", low=1e-5, high=5e-4, log=True)
        entropy_loss_scale = trial.suggest_float("entropy_loss_scale", 1e-4, 0.01, log=True)
        value_loss_scale = trial.suggest_float("value_loss_scale", low=0.1, high=1.0, log=True)
        ratio_clip = trial.suggest_float("ratio_clip", low=0.1, high=0.2)

        # Cap mini_batches for forward dynamics
        if "ssl_task" in agent_cfg and agent_cfg["ssl_task"]["type"] == "forward_dynamics":
            mini_batches = min(mini_batches, 8)

        agent_cfg["agent"]["rollouts"] = rollouts
        agent_cfg["agent"]["mini_batches"] = mini_batches
        agent_cfg["agent"]["learning_epochs"] = learning_epochs
        agent_cfg["agent"]["learning_rate"] = learning_rate
        agent_cfg["agent"]["entropy_loss_scale"] = entropy_loss_scale
        agent_cfg["agent"]["value_loss_scale"] = value_loss_scale
        agent_cfg["agent"]["ratio_clip"] = ratio_clip

        # Suggest SSL task hyperparameters if applicable
        if "ssl_task" in agent_cfg:
            learning_rate_aux = trial.suggest_float("learning_rate_aux", low=1e-5, high=5e-4, log=True)
            loss_weight_aux = trial.suggest_float("loss_weight_aux", low=1e-3, high=1, log=True)

            agent_cfg["ssl_task"]["learning_rate"] = learning_rate_aux
            agent_cfg["ssl_task"]["loss_weight"] = loss_weight_aux

            if agent_cfg["ssl_task"]["type"] == "forward_dynamics":
                seq_length = trial.suggest_int("seq_length", low=2, high=10, step=1)
                agent_cfg["ssl_task"]["seq_length"] = seq_length

        # Setup models
        policy, value, encoder, value_preprocessor = make_models(env, env_cfg, agent_cfg, dtype)

        # Create tensors in memory for RL (only for the training envs, not eval envs)
        num_training_envs = env_cfg.scene.num_envs - agent_cfg["trainer"]["num_eval_envs"]
        rl_memory = make_memory(env, env_cfg, size=agent_cfg["agent"]["rollouts"], num_envs=num_training_envs)
        ssl_task = make_aux(env, rl_memory, encoder, value, value_preprocessor, env_cfg, agent_cfg, writer)

        # Restart wandb for this trial
        writer.close_wandb()
        writer.setup_wandb(name=trial.number)

        # Configure and instantiate PPO agent
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
            debug=agent_cfg["experiment"]["debug"],
        )

        # Train the agent
        trainer = make_trainer(env, agent, agent_cfg, ssl_task, writer)

        try:
            best_return, should_prune = trainer.train(trial=trial)
        except AssertionError as e:
            # Sometimes random hyperparameters can generate NaN
            print(e)

        # Prune trial if needed
        if should_prune:
            raise optuna.TrialPruned()
        return best_return


if __name__ == "__main__":
    if args_cli.rerun_trial is not None:
        print(f"Rerun trial {args_cli.rerun_trial} on multiple seeds (skipping Optuna sweep)")
    else:
        print("Running sweep with Optuna")

    # Parse configuration
    args_cli.gym_env_id = resolve_gym_env_id(args_cli.task, args_cli.robot)
    env_cfg, agent_cfg = register_task_to_hydra(args_cli.gym_env_id, "default_cfg")

    specialised_cfg = load_cfg_from_registry(args_cli.gym_env_id, args_cli.agent_cfg)
    agent_cfg = update_dict(agent_cfg, specialised_cfg)

    dtype = torch.float32

    # Set seed (important for seed-deterministic runs)
    agent_cfg["seed"] = args_cli.seed if args_cli.seed is not None else agent_cfg["seed"]
    set_seed(agent_cfg["seed"])
    agent_cfg["log_path"] = LOG_PATH
    args_cli.video = agent_cfg["experiment"]["upload_videos"]

    # Update the environment config
    env_cfg = update_env_cfg(args_cli, env_cfg, agent_cfg)

    max_sweep_timesteps_M = agent_cfg["sweeper"]["max_sweep_timesteps_M"]
    max_training_timesteps_M = agent_cfg["trainer"]["max_global_timesteps_M"]

    if args_cli.rerun_trial is not None:
        storage = agent_cfg["sweeper"]["storage"]
        study_name = args_cli.study
        study = optuna.load_study(study_name=study_name, storage=storage)
        trial = next((t for t in study.trials if t.number == args_cli.rerun_trial), None)
        if trial is None:
            raise ValueError(
                f"No trial with number {args_cli.rerun_trial} in study {study_name!r} "
                f"({len(study.trials)} trials in storage)."
            )
        try:
            apply_optuna_trial_params(agent_cfg, trial)
        except KeyError as e:
            raise KeyError(
                f"Trial {args_cli.rerun_trial} is missing hyperparameters (incomplete trial?): {e}"
            ) from e

        agent_cfg["trainer"]["max_global_timesteps_M"] = max_training_timesteps_M
        suffix = f"_trial_{args_cli.rerun_trial}"
        agent_cfg["experiment"]["experiment_name"] = (
            args_cli.task + "_" + args_cli.agent_cfg + "_" + args_cli.study + suffix
        )
        agent_cfg["experiment"]["wandb_kwargs"]["group"] = (
            args_cli.task + "_" + args_cli.agent_cfg + "_" + args_cli.study + suffix
        )
        seeds = args_cli.rerun_seeds if args_cli.rerun_seeds is not None else [5, 6, 7, 8, 9, 10]

        writer = Writer(agent_cfg, delay_wandb_startup=True)
        env_cfg = update_env_cfg(args_cli, env_cfg, agent_cfg)
        env = make_env(agent_cfg, env_cfg, writer, args_cli)

        print(f"Rerun Optuna trial {args_cli.rerun_trial} on seeds: {seeds}")
        for seed in seeds:
            agent_cfg["experiment"]["wandb_kwargs"]["name"] = f"trial_{args_cli.rerun_trial}_seed_{seed}"
            env_cfg = update_env_cfg(args_cli, env_cfg, agent_cfg)
            writer.setup_wandb(name=f"trial_{args_cli.rerun_trial}_seed_{seed}")
            train_one_seed(args_cli, env, agent_cfg=agent_cfg, env_cfg=env_cfg, writer=writer, seed=seed)
            writer.close_wandb()
            writer.get_new_log_path()

        env.close()
        simulation_app.close()
        sys.exit(0)

    # Default path: Optuna sweep, then multi-seed evaluation of the best trial.
    # Setup logging for sweep
    agent_cfg["experiment"]["experiment_name"] = args_cli.task + "_" + args_cli.agent_cfg + "_" + args_cli.study
    agent_cfg["experiment"]["wandb_kwargs"]["group"] = (
        args_cli.task + "_" + args_cli.agent_cfg + "_" + args_cli.study
    )
    storage = agent_cfg["sweeper"]["storage"]
    n_warmup_steps = agent_cfg["sweeper"]["warmup_timesteps_M"] * 1e6
    agent_cfg["trainer"]["max_global_timesteps_M"] = max_sweep_timesteps_M

    study_name = args_cli.study
    total_trials = 40
    n_startup_trials = 8
    interval_steps = 1

    writer = Writer(agent_cfg, delay_wandb_startup=True)

    # Make environment (order: gymnasium Env -> FrameStack -> IsaacLab)
    env = make_env(agent_cfg, env_cfg, writer, args_cli)

    runner = OptimisationRunner(study_name, n_startup_trials, n_warmup_steps, interval_steps)

    # Calculate how many more we need
    # We count all trials (Complete, Pruned, and even Failed)
    # to ensure we don't exceed the total budget.
    trials_already_done = len(runner.study.trials)
    remaining_trials = max(0, total_trials - trials_already_done)

    if remaining_trials > 0:
        print("Running remaining trials:", remaining_trials)
        best_trial = runner.run(remaining_trials)
        print("Best trial:", best_trial)

    else:
        print(f"Study already reached or exceeded {total_trials} trials.")
        exit()

    writer.close_wandb()

    # Apply best trial hyperparameters
    apply_optuna_trial_params(agent_cfg, best_trial)

    # Train best configuration on multiple seeds
    agent_cfg["experiment"]["experiment_name"] = args_cli.task + "_" + args_cli.agent_cfg + "_" + "seeded"
    agent_cfg["trainer"]["max_global_timesteps_M"] = max_training_timesteps_M
    agent_cfg["experiment"]["wandb_kwargs"]["group"] = args_cli.task + "_" + args_cli.agent_cfg + "_" + "seeded"

    test_seeds = [5, 6, 7, 8, 9, 10]

    print("Running best trial on multiple seeds:", test_seeds)

    writer = Writer(agent_cfg, delay_wandb_startup=True)
    env_cfg = update_env_cfg(args_cli, env_cfg, agent_cfg)

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

#!/usr/bin/env python3
"""Hyperparameter sweep for the UAN residual-torque policy, via Optuna.

Same Optuna machinery as roto's own `sweep.py` (TPE sampler + median pruner,
PPO hyperparameters, persistent SQLite storage so `optuna-dashboard` can open
it live) -- but built on top of `train_uan.py`'s own direct-yaml config
loading instead of roto's Hydra/ConfigStore task registration, since
UAN_Shadowlite deliberately isn't registered through that path (see
UAN_PROGRESS.md D6). Everything downstream of env/agent_cfg (make_models,
make_memory, make_trainer, PPO, Trainer.train(trial=...)) is the exact same
generic roto/multimodal_rl code `sweep.py` uses -- copied here, not
reimplemented, so a sweep behaves identically to how roto sweeps every other
task.

Usage:
    python sweep_uan.py --headless
    python sweep_uan.py --headless --num_envs 512 --study my_sweep --n_trials 40

View results live while (or after) it runs:
    optuna-dashboard sqlite:///roto_uan.db      # from the repo root
"""

import argparse
import os
import sys

from isaaclab.app import AppLauncher

sys.stdout.reconfigure(line_buffering=True)

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_ROTO_ROOT = os.path.dirname(_THIS_DIR)

parser = argparse.ArgumentParser(description="Hyperparameter sweep for the UAN residual-torque policy.")
parser.add_argument(
    "--config",
    type=str,
    default=os.path.join(_ROTO_ROOT, "roto", "tasks", "uan_shadowlite", "agents", "shadowlite", "default.yaml"),
    help="Path to the base agent yaml (dataset/uan/encoder/policy/value/agent/sweeper/... sections).",
)
parser.add_argument("--agent_cfg", type=str, default=None, help="Optional yaml merged OVER --config.")
parser.add_argument(
    "--dataset", type=str, action="append", default=None, help="Override dataset.paths (repeatable)."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of parallel environments.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment.")
parser.add_argument("--study", type=str, default="uan_default", help="Optuna study name.")
parser.add_argument("--n_trials", type=int, default=40, help="Total trials in the study (across all runs).")
parser.add_argument("--n_startup_trials", type=int, default=8, help="Random trials before TPE kicks in.")
parser.add_argument(
    "--rerun-trial",
    type=int,
    default=None,
    metavar="N",
    help="Load trial N from the existing study (--study) and train it on multiple seeds; skips the sweep.",
)
parser.add_argument(
    "--rerun-seeds", type=int, nargs="+", default=None, help="Seeds for --rerun-trial (default: 5 6 7 8 9 10)."
)
# NOTE: --device is intentionally NOT defined here -- AppLauncher.add_app_launcher_args()
# below already registers it.
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")

AppLauncher.add_app_launcher_args(parser)
args_cli, _unused = parser.parse_known_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# Everything below touches isaaclab/omni and must be imported AFTER AppLauncher boots.
import numpy as np  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402
from isaaclab.utils import update_dict  # noqa: E402

import optuna  # noqa: E402

sys.path.insert(0, _THIS_DIR)
sys.path.insert(0, _ROTO_ROOT)  # see train_uan.py's matching comment: works around a stale editable install
from common_utils import (  # noqa: E402
    LOG_PATH,
    make_env,
    make_memory,
    make_models,
    make_trainer,
    set_seed,
    train_one_seed,
    update_env_cfg,
)
from multimodal_rl.rl.ppo import PPO, PPO_DEFAULT_CONFIG  # noqa: E402
from multimodal_rl.tools.writer import Writer  # noqa: E402

from roto.tasks import uan_shadowlite  # noqa: E402,F401  (side effect: gym.register)
from roto.tasks.uan_shadowlite.task import UANShadowLiteEnvCfg  # noqa: E402

DTYPE = torch.float32


def _resolve_storage(storage: str) -> str:
    """Resolve a relative sqlite:/// path to an absolute path anchored at the repo root."""
    prefix = "sqlite:///"
    abs_prefix = "sqlite:////"
    if storage.startswith(prefix) and not storage.startswith(abs_prefix):
        db_name = storage[len(prefix) :]
        if not os.path.isabs(db_name):
            return f"{prefix}{os.path.join(_ROTO_ROOT, db_name)}"
    return storage


def load_agent_cfg() -> dict:
    with open(args_cli.config) as f:
        agent_cfg = yaml.safe_load(f)
    if args_cli.agent_cfg is not None:
        with open(args_cli.agent_cfg) as f:
            overlay = yaml.safe_load(f)
        agent_cfg = update_dict(agent_cfg, overlay)
    if args_cli.dataset is not None:
        agent_cfg["dataset"]["paths"] = args_cli.dataset
    return agent_cfg


def build_env_cfg(agent_cfg: dict) -> UANShadowLiteEnvCfg:
    env_cfg = UANShadowLiteEnvCfg()
    env_cfg.dataset = agent_cfg["dataset"]
    env_cfg.uan = agent_cfg["uan"]
    env_cfg = update_env_cfg(args_cli, env_cfg, agent_cfg)
    return env_cfg


def apply_optuna_trial_params(agent_cfg: dict, trial: optuna.trial.FrozenTrial) -> None:
    """Copy hyperparameters from a stored Optuna trial into ``agent_cfg`` (same PPO
    knobs roto's own sweep.py tunes -- identical semantics, so a UAN sweep is directly
    comparable to any other roto sweep)."""
    p = trial.params
    agent_cfg["agent"]["rollouts"] = 2 ** p["rollouts_pow"]
    agent_cfg["agent"]["mini_batches"] = 2 ** p["mini_batches_pow"]
    agent_cfg["agent"]["learning_epochs"] = p["learning_epochs"]
    agent_cfg["agent"]["learning_rate"] = p["learning_rate"]
    agent_cfg["agent"]["entropy_loss_scale"] = p["entropy_loss_scale"]
    agent_cfg["agent"]["value_loss_scale"] = p["value_loss_scale"]
    agent_cfg["agent"]["ratio_clip"] = p["ratio_clip"]


class OptimisationRunner:
    """Optuna-based hyperparameter optimization runner (mirrors sweep.py's)."""

    def __init__(self, study_name, storage, n_startup_trials, n_warmup_steps, interval_steps):
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

    def run(self, env, env_cfg, agent_cfg, writer, n_trials):
        self.study.optimize(
            lambda trial: self.objective(trial, env, env_cfg, agent_cfg, writer),
            n_trials=n_trials,
            show_progress_bar=True,
            gc_after_trial=True,
        )
        trial = self.study.best_trial
        print(f"Number of finished trials: {len(self.study.trials)}")
        print("Best trial:", trial.number, "Value:", trial.value)
        for key, value in trial.params.items():
            print(f"    {key}: {value}")
        return trial

    def objective(self, trial: optuna.Trial, env, env_cfg, agent_cfg, writer) -> float:
        print(f"Starting trial: {trial.number}")

        TRAIN_SEEDS = [0, 1, 2, 3, 4]
        agent_cfg["seed"] = int(np.random.choice(TRAIN_SEEDS))
        set_seed(agent_cfg["seed"])

        rollouts = 2 ** trial.suggest_int("rollouts_pow", 4, 6)
        mini_batches = 2 ** trial.suggest_int("mini_batches_pow", 2, 5)
        learning_epochs = trial.suggest_int("learning_epochs", low=4, high=10, step=1)
        learning_rate = trial.suggest_float("learning_rate", low=1e-5, high=5e-4, log=True)
        entropy_loss_scale = trial.suggest_float("entropy_loss_scale", 1e-4, 0.01, log=True)
        value_loss_scale = trial.suggest_float("value_loss_scale", low=0.1, high=1.0, log=True)
        ratio_clip = trial.suggest_float("ratio_clip", low=0.1, high=0.2)

        agent_cfg["agent"]["rollouts"] = rollouts
        agent_cfg["agent"]["mini_batches"] = mini_batches
        agent_cfg["agent"]["learning_epochs"] = learning_epochs
        agent_cfg["agent"]["learning_rate"] = learning_rate
        agent_cfg["agent"]["entropy_loss_scale"] = entropy_loss_scale
        agent_cfg["agent"]["value_loss_scale"] = value_loss_scale
        agent_cfg["agent"]["ratio_clip"] = ratio_clip

        policy, value, encoder, value_preprocessor = make_models(env, env_cfg, agent_cfg, DTYPE)
        num_training_envs = env_cfg.scene.num_envs - agent_cfg["trainer"]["num_eval_envs"]
        rl_memory = make_memory(env, env_cfg, size=agent_cfg["agent"]["rollouts"], num_envs=num_training_envs)

        writer.close_wandb()
        writer.setup_wandb(name=str(trial.number))

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
            ssl_task=None,
            dtype=DTYPE,
            debug=agent_cfg["experiment"]["debug"],
        )

        trainer = make_trainer(env, agent, agent_cfg, ssl_task=None, writer=writer)

        should_prune = False
        best_return = -float("inf")
        try:
            best_return, should_prune = trainer.train(trial=trial)
        except AssertionError as e:
            # Random hyperparameters can occasionally generate NaN -- report it as a bad
            # trial rather than crashing the whole sweep.
            print(e)

        torch.cuda.empty_cache()
        import gc

        gc.collect()

        if should_prune:
            raise optuna.TrialPruned()
        return best_return


def main() -> None:
    agent_cfg = load_agent_cfg()
    agent_cfg["log_path"] = LOG_PATH
    agent_cfg["experiment"]["upload_videos"] = int(args_cli.video)

    storage = _resolve_storage(agent_cfg["sweeper"]["storage"])
    max_sweep_timesteps_M = agent_cfg["sweeper"]["max_sweep_timesteps_M"]
    max_training_timesteps_M = agent_cfg["trainer"]["max_global_timesteps_M"]
    n_warmup_steps = agent_cfg["sweeper"]["warmup_timesteps_M"] * 1e6

    args_cli.task = "UAN_Shadowlite"
    args_cli.gym_env_id = "UAN_Shadowlite"

    if args_cli.rerun_trial is not None:
        study = optuna.load_study(study_name=args_cli.study, storage=storage)
        trial = next((t for t in study.trials if t.number == args_cli.rerun_trial), None)
        if trial is None:
            raise ValueError(
                f"No trial {args_cli.rerun_trial} in study {args_cli.study!r} ({len(study.trials)} trials in storage)."
            )
        apply_optuna_trial_params(agent_cfg, trial)
        agent_cfg["trainer"]["max_global_timesteps_M"] = max_training_timesteps_M
        suffix = f"_trial_{args_cli.rerun_trial}"
        agent_cfg["experiment"]["experiment_name"] = args_cli.study + suffix
        agent_cfg["experiment"]["wandb_kwargs"]["group"] = args_cli.study + suffix

        seeds = args_cli.rerun_seeds if args_cli.rerun_seeds is not None else [5, 6, 7, 8, 9, 10]
        writer = Writer(agent_cfg, delay_wandb_startup=True)
        env_cfg = build_env_cfg(agent_cfg)
        env = make_env(agent_cfg, env_cfg, writer, args_cli)

        for seed in seeds:
            agent_cfg["experiment"]["wandb_kwargs"]["name"] = f"trial_{args_cli.rerun_trial}_seed_{seed}"
            env_cfg = update_env_cfg(args_cli, env_cfg, agent_cfg)
            writer.setup_wandb(name=f"trial_{args_cli.rerun_trial}_seed_{seed}")
            train_one_seed(args_cli, env, agent_cfg=agent_cfg, env_cfg=env_cfg, writer=writer, seed=seed)
            writer.close_wandb()
            writer.get_new_log_path()

        env.close()
        return

    # Default path: Optuna sweep, then multi-seed evaluation of the best trial.
    agent_cfg["experiment"]["experiment_name"] = args_cli.study
    agent_cfg["experiment"]["wandb_kwargs"]["group"] = args_cli.study
    agent_cfg["trainer"]["max_global_timesteps_M"] = max_sweep_timesteps_M
    agent_cfg["seed"] = args_cli.seed if args_cli.seed is not None else agent_cfg["seed"]
    set_seed(agent_cfg["seed"])

    writer = Writer(agent_cfg, delay_wandb_startup=True)
    env_cfg = build_env_cfg(agent_cfg)
    env = make_env(agent_cfg, env_cfg, writer, args_cli)

    runner = OptimisationRunner(args_cli.study, storage, args_cli.n_startup_trials, n_warmup_steps, interval_steps=1)

    trials_already_done = len(runner.study.trials)
    remaining_trials = max(0, args_cli.n_trials - trials_already_done)
    if remaining_trials == 0:
        print(f"Study {args_cli.study!r} already has {trials_already_done} >= {args_cli.n_trials} trials.")
        env.close()
        return

    print(f"Running {remaining_trials} remaining trial(s). Storage: {storage}")
    print(f"Watch live: optuna-dashboard {storage}")
    best_trial = runner.run(env, env_cfg, agent_cfg, writer, remaining_trials)
    writer.close_wandb()

    # Train the best configuration on multiple seeds, at full length.
    apply_optuna_trial_params(agent_cfg, best_trial)
    agent_cfg["experiment"]["experiment_name"] = args_cli.study + "_seeded"
    agent_cfg["experiment"]["wandb_kwargs"]["group"] = args_cli.study + "_seeded"
    agent_cfg["trainer"]["max_global_timesteps_M"] = max_training_timesteps_M

    test_seeds = [5, 6, 7, 8, 9, 10]
    writer = Writer(agent_cfg, delay_wandb_startup=True)
    env_cfg = update_env_cfg(args_cli, env_cfg, agent_cfg)
    for seed in test_seeds:
        agent_cfg["experiment"]["wandb_kwargs"]["name"] = str(seed)
        env_cfg = update_env_cfg(args_cli, env_cfg, agent_cfg)
        writer.setup_wandb(name=str(seed))
        train_one_seed(args_cli, env, agent_cfg=agent_cfg, env_cfg=env_cfg, writer=writer, seed=seed)
        writer.close_wandb()
        writer.get_new_log_path()

    env.close()


if __name__ == "__main__":
    try:
        main()
    except Exception as err:
        print("ERROR DURING SWEEP:", err)
        raise
    finally:
        print("CLOSING")
        simulation_app.close()

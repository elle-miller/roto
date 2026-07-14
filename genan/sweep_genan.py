#!/usr/bin/env python3
"""Hyperparameter sweep for GenAN training, via Optuna.

Isaac-free (`train_genan.py`'s `train()` never touches Isaac Lab, unlike PPO
training) -- so unlike `scripts/sweep.py`, which must boot Isaac Sim once and
reuse that one env across trials, this script loads the dataset ONCE
(Isaac-free) and reuses it across every trial the same way. Mirrors
`sweep.py`'s structure otherwise: `TPESampler` + `MedianPruner`, sqlite
storage (inspect with `optuna-dashboard sqlite:///roto_genan.db` -- see
`agents/shadowlite/default.yaml`'s `sweeper.storage`), a `--rerun-trial` path
to retrain one specific trial on multiple seeds, and a default path that runs
the full search then retrains the best trial on multiple seeds.

What's actually searched: `lr`, `batch_size`, `history_len`, `stride` --
`ensemble_size` and the MLP architecture are Table-1-fixed values from the
paper, not tunable knobs here (see default.yaml's `genan` section comment).
Objective = best validation torque loss (direction="minimize" -- `sweep.py`'s
own study is direction="maximize" for RL return, this is the opposite sense
since lower loss is better here).

Usage:
    python sweep_genan.py
    python sweep_genan.py --study my_study --total_trials 20
    python sweep_genan.py --rerun-trial 12 --rerun-seeds 1 2 3
"""

from __future__ import annotations

import argparse
import os

import optuna
import torch

from config_utils import load_config
from train_genan import load_dataset_from_cfg, train

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_ROTO_ROOT = os.path.normpath(os.path.join(_THIS_DIR, ".."))  # repo root, same anchor scripts/sweep.py uses
_DEFAULT_CONFIG = os.path.join(_THIS_DIR, "agents", "shadowlite", "default.yaml")


def _resolve_storage(storage: str) -> str:
    """Resolve a relative `sqlite:///` path to an absolute path anchored at the
    repo root -- the same helper `scripts/sweep.py` defines, kept standalone
    here (not imported from there) so this script never needs Isaac Lab.
    """
    prefix = "sqlite:///"
    abs_prefix = "sqlite:////"
    if storage.startswith(prefix) and not storage.startswith(abs_prefix):
        db_name = storage[len(prefix):]
        if not os.path.isabs(db_name):
            return f"{prefix}{os.path.join(_ROTO_ROOT, db_name)}"
    return storage


def suggest_params(trial: optuna.Trial) -> dict:
    """Sample the hyperparameters actually worth searching."""
    return {
        "lr": trial.suggest_float("lr", 1e-5, 1e-3, log=True),
        "batch_size": 2 ** trial.suggest_int("batch_size_pow", 9, 13),
        "history_len": trial.suggest_int("history_len", 1, 6),
        "stride": trial.suggest_int("stride", 1, 3),
    }


def apply_trial_params(genan_cfg: dict, trial: optuna.trial.FrozenTrial) -> dict:
    """Copy a stored trial's hyperparameters into a fresh genan-config dict."""
    cfg = dict(genan_cfg)
    p = trial.params
    cfg["lr"] = p["lr"]
    cfg["batch_size"] = 2 ** p["batch_size_pow"]
    cfg["history_len"] = p["history_len"]
    cfg["stride"] = p["stride"]
    return cfg


class GenANSweepRunner:
    """Optuna-based hyperparameter optimization runner for GenAN."""

    def __init__(self, study_name: str, storage: str, n_startup_trials: int, n_warmup_steps: int, interval_steps: int):
        self.sampler = optuna.samplers.TPESampler(n_startup_trials=n_startup_trials, multivariate=True)
        self.pruner = optuna.pruners.MedianPruner(
            n_startup_trials=n_startup_trials, n_warmup_steps=n_warmup_steps, interval_steps=interval_steps
        )
        self.study = optuna.create_study(
            storage=storage, sampler=self.sampler, pruner=self.pruner,
            study_name=study_name, direction="minimize", load_if_exists=True,
        )

    def objective(self, trial: optuna.Trial, dataset, base_genan_cfg: dict, sweep_epochs: int, seed: int) -> float:
        params = suggest_params(trial)
        _, history_log = train(
            dataset,
            history_len=params["history_len"],
            stride=params["stride"],
            ensemble_size=base_genan_cfg["ensemble_size"],
            epochs=sweep_epochs,
            batch_size=params["batch_size"],
            lr=params["lr"],
            val_frac=base_genan_cfg["val_frac"],
            patience=base_genan_cfg["patience"],
            seed=seed,
            trial=trial,
        )
        return history_log["best_val_loss"]

    def run(self, dataset, base_genan_cfg: dict, sweep_epochs: int, seed: int, n_trials: int) -> optuna.trial.FrozenTrial:
        self.study.optimize(
            lambda trial: self.objective(trial, dataset, base_genan_cfg, sweep_epochs, seed),
            n_trials=n_trials,
            show_progress_bar=True,
            gc_after_trial=True,
        )
        print(f"Number of finished trials: {len(self.study.trials)}")
        best = self.study.best_trial
        print("Best trial:")
        print("  Value (val_loss):", best.value)
        print("  Params:")
        for key, value in best.params.items():
            print(f"    {key}: {value}")
        return best


def retrain_on_seeds(dataset, genan_cfg: dict, epochs: int, seeds: list[int], checkpoint_prefix: str, joint_names: list[str]) -> None:
    for seed in seeds:
        print(f"[INFO] Retraining seed {seed} with {genan_cfg}...")
        ensemble, history_log = train(
            dataset,
            history_len=genan_cfg["history_len"],
            stride=genan_cfg["stride"],
            ensemble_size=genan_cfg["ensemble_size"],
            epochs=epochs,
            batch_size=genan_cfg["batch_size"],
            lr=genan_cfg["lr"],
            val_frac=genan_cfg["val_frac"],
            patience=genan_cfg["patience"],
            seed=seed,
        )
        ckpt_path = f"{checkpoint_prefix}_seed_{seed}.pt"
        torch.save(
            {
                "ensemble_state_dict": ensemble.state_dict(),
                "input_dim": ensemble.members[0].trunk[0].in_features,
                "num_joints": ensemble.num_joints,
                "ensemble_size": ensemble.ensemble_size,
                "history_len": genan_cfg["history_len"],
                "stride": genan_cfg["stride"],
                "joint_names": joint_names,
                "best_val_loss": history_log["best_val_loss"],
            },
            ckpt_path,
        )
        print(f"[INFO] Saved {ckpt_path} (best_val_loss={history_log['best_val_loss']:.6f}).")


def main() -> None:
    parser = argparse.ArgumentParser(description="Optuna hyperparameter sweep for GenAN.")
    parser.add_argument("--config", type=str, default=_DEFAULT_CONFIG)
    parser.add_argument("--agent_cfg", type=str, default=None)
    parser.add_argument("--dataset", type=str, action="append", default=None)
    parser.add_argument("--joints_yaml", type=str, default=None)
    parser.add_argument("--study", type=str, default="genan_default")
    parser.add_argument("--total_trials", type=int, default=None, help="Override sweeper.total_trials.")
    parser.add_argument(
        "--rerun-trial", type=int, default=None, dest="rerun_trial",
        help="Load trial N from the existing study and retrain it on multiple seeds; skips the sweep.",
    )
    parser.add_argument(
        "--rerun-seeds", type=int, nargs="+", default=None, dest="rerun_seeds",
        help="Seeds for --rerun-trial (default: sweeper.test_seeds).",
    )
    args = parser.parse_args()

    cfg = load_config(args.config, args.agent_cfg)
    sweeper_cfg = cfg["sweeper"]
    storage = _resolve_storage(sweeper_cfg["storage"])
    seed = cfg["seed"]

    dataset, joint_names = load_dataset_from_cfg(cfg, args.dataset, args.joints_yaml)
    print(f"[INFO] Loaded {len(dataset.paths)} file(s), {dataset.num_steps} steps, "
          f"{dataset.traj_starts.shape[0]} trajectory segment(s).")

    os.makedirs(cfg["experiment"]["checkpoint_dir"], exist_ok=True)
    checkpoint_prefix = os.path.join(cfg["experiment"]["checkpoint_dir"], cfg["experiment"]["experiment_name"])

    if args.rerun_trial is not None:
        study = optuna.load_study(study_name=args.study, storage=storage)
        trial = next((t for t in study.trials if t.number == args.rerun_trial), None)
        if trial is None:
            raise ValueError(f"No trial with number {args.rerun_trial} in study {args.study!r} ({len(study.trials)} trials in storage).")
        genan_cfg = apply_trial_params(cfg["genan"], trial)
        seeds = args.rerun_seeds if args.rerun_seeds is not None else sweeper_cfg["test_seeds"]
        print(f"[INFO] Rerunning trial {args.rerun_trial} on seeds {seeds}.")
        retrain_on_seeds(
            dataset, genan_cfg, cfg["genan"]["epochs"], seeds, f"{checkpoint_prefix}_trial_{args.rerun_trial}", joint_names
        )
        return

    runner = GenANSweepRunner(
        study_name=args.study,
        storage=storage,
        n_startup_trials=sweeper_cfg["n_startup_trials"],
        n_warmup_steps=sweeper_cfg["warmup_epochs"],
        interval_steps=1,
    )
    total_trials = args.total_trials if args.total_trials is not None else sweeper_cfg["total_trials"]
    remaining_trials = max(0, total_trials - len(runner.study.trials))
    if remaining_trials == 0:
        print(f"Study already reached or exceeded {total_trials} trials.")
        return

    print(f"Running remaining trials: {remaining_trials}  (storage: {storage})")
    best_trial = runner.run(dataset, cfg["genan"], sweeper_cfg["max_sweep_epochs"], seed, remaining_trials)

    best_genan_cfg = apply_trial_params(cfg["genan"], best_trial)
    print(f"[INFO] Retraining best trial on seeds {sweeper_cfg['test_seeds']}: {best_genan_cfg}")
    retrain_on_seeds(dataset, best_genan_cfg, cfg["genan"]["epochs"], sweeper_cfg["test_seeds"], checkpoint_prefix, joint_names)


if __name__ == "__main__":
    main()

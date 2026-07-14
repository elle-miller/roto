"""CPU-only unit tests for train_genan.py's optional `trial` (Optuna pruning)
hook, using a duck-typed fake trial -- no dependency on a real optuna.Trial,
though `optuna` itself (for `optuna.TrialPruned`) must be importable, which it
is in this repo's test environments.
"""

import os
import sys

import numpy as np

_GENAN_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "genan")
sys.path.insert(0, _GENAN_DIR)

import optuna  # noqa: E402

from joint_config import load_joint_config  # noqa: E402
from train_genan import train  # noqa: E402

from test_train_smoke import _make_dataset  # noqa: E402

JOINT_NAMES, _ = load_joint_config()


class _FakeTrial:
    """Duck-typed stand-in for optuna.Trial: only `.report()`/`.should_prune()`
    are used by `train()`.
    """

    def __init__(self, prune_after_reports: int | None = None):
        self.reported: list[tuple[float, int]] = []
        self.prune_after_reports = prune_after_reports

    def report(self, value: float, step: int) -> None:
        self.reported.append((value, step))

    def should_prune(self) -> bool:
        if self.prune_after_reports is None:
            return False
        return len(self.reported) >= self.prune_after_reports


def test_trial_receives_a_report_every_epoch(tmp_path):
    dataset = _make_dataset(tmp_path)
    trial = _FakeTrial(prune_after_reports=None)
    _, history_log = train(
        dataset, history_len=2, stride=1, ensemble_size=2, epochs=5, batch_size=256,
        lr=1e-3, val_frac=0.3, patience=1000, seed=0, trial=trial,
    )
    assert len(trial.reported) == 5
    steps = [step for _, step in trial.reported]
    assert steps == list(range(5))
    reported_losses = [val for val, _ in trial.reported]
    assert np.allclose(reported_losses, history_log["val_loss"])


def test_trial_pruning_raises_trial_pruned(tmp_path):
    dataset = _make_dataset(tmp_path)
    trial = _FakeTrial(prune_after_reports=2)
    try:
        train(
            dataset, history_len=2, stride=1, ensemble_size=2, epochs=20, batch_size=256,
            lr=1e-3, val_frac=0.3, patience=1000, seed=0, trial=trial,
        )
        assert False, "expected optuna.TrialPruned"
    except optuna.TrialPruned:
        pass
    # Pruned right after the 2nd report (epoch index 1) -- training must not
    # have run all the way to epoch 20.
    assert len(trial.reported) == 2


def test_no_trial_means_no_optuna_dependency_on_the_hot_path(tmp_path):
    dataset = _make_dataset(tmp_path)
    # trial=None (the default) must never touch optuna at all.
    _, history_log = train(
        dataset, history_len=2, stride=1, ensemble_size=2, epochs=3, batch_size=256,
        lr=1e-3, val_frac=0.3, patience=1000, seed=0,
    )
    assert len(history_log["val_loss"]) == 3

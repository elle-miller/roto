"""CPU-only unit tests for sweep_genan.py's pure-Python helpers.

Importing sweep_genan.py itself is safe (no dataset/Isaac access happens at
import time, only inside main()) even though it imports `optuna` at module
level -- unlike train_genan.py, a sweep script legitimately requires optuna.
"""

import os
import sys

_GENAN_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "genan")
sys.path.insert(0, _GENAN_DIR)

import optuna  # noqa: E402

from sweep_genan import _resolve_storage, apply_trial_params  # noqa: E402


def test_resolve_storage_anchors_relative_sqlite_path(tmp_path, monkeypatch):
    import sweep_genan

    monkeypatch.setattr(sweep_genan, "_ROTO_ROOT", str(tmp_path))
    resolved = _resolve_storage("sqlite:///roto_genan.db")
    assert resolved == f"sqlite:///{tmp_path}/roto_genan.db"


def test_resolve_storage_leaves_absolute_path_untouched():
    resolved = _resolve_storage("sqlite:////already/absolute.db")
    assert resolved == "sqlite:////already/absolute.db"


def test_resolve_storage_leaves_non_sqlite_untouched():
    resolved = _resolve_storage("postgresql://user:pass@host/db")
    assert resolved == "postgresql://user:pass@host/db"


def test_apply_trial_params_overrides_only_searched_keys():
    genan_cfg = {
        "history_len": 3, "stride": 1, "ensemble_size": 5,
        "lr": 1e-4, "batch_size": 4096, "epochs": 150, "patience": 10, "val_frac": 0.2,
    }
    study = optuna.create_study(direction="minimize")
    trial = study.ask()
    trial.suggest_float("lr", 1e-5, 1e-3, log=True)
    trial.suggest_int("batch_size_pow", 9, 13)
    trial.suggest_int("history_len", 1, 6)
    trial.suggest_int("stride", 1, 3)
    study.tell(trial, 1.0)
    frozen = study.trials[0]

    updated = apply_trial_params(genan_cfg, frozen)

    # Unsearched keys pass through unchanged.
    assert updated["ensemble_size"] == 5
    assert updated["epochs"] == 150
    assert updated["patience"] == 10
    assert updated["val_frac"] == 0.2
    # Searched keys come from the trial, batch_size decoded from its power-of-2 exponent.
    assert updated["lr"] == frozen.params["lr"]
    assert updated["batch_size"] == 2 ** frozen.params["batch_size_pow"]
    assert updated["history_len"] == frozen.params["history_len"]
    assert updated["stride"] == frozen.params["stride"]
    # Original dict is untouched (apply_trial_params returns a copy).
    assert genan_cfg["lr"] == 1e-4

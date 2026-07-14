"""Isaac-free yaml config loading + deep-merge for GenAN scripts.

`train_uan.py`/`sweep.py` use `isaaclab.utils.update_dict` for this; importing
that here would drag Isaac Lab into `train_genan.py`/`sweep_genan.py`, which
are deliberately Isaac-free (see DESIGN.md). This is a small, standalone
reimplementation of the same recursive-merge behavior instead.
"""

from __future__ import annotations

import yaml


def deep_update(base: dict, overlay: dict) -> dict:
    """Recursively merge `overlay` into `base` (in place) and return it.

    A nested dict in `overlay` merges key-by-key into the matching nested
    dict in `base`; any other value (including a list) simply overwrites.
    """
    for key, value in overlay.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            deep_update(base[key], value)
        else:
            base[key] = value
    return base


def load_config(config_path: str, overlay_path: str | None = None) -> dict:
    """Load `config_path`, optionally deep-merging `overlay_path` over it --
    mirrors `train_uan.py`'s `--config` + `--agent_cfg` pattern.
    """
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    if overlay_path is not None:
        with open(overlay_path) as f:
            overlay = yaml.safe_load(f)
        deep_update(cfg, overlay)
    return cfg

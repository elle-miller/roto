# Agent YAML roots live in per-robot subdirectories (shadow/, orca/, allegro/).
# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Peace-sign task registration for the ShadowLite hand."""

import os

import gymnasium as gym

from . import agents
from .peace import PeaceSignCfg, PeaceSignEnv

_AGENTS_DIR = os.path.dirname(agents.__file__)

_PEACE_SHADOWLITE_VARIANT_FILES = {
    "default_cfg": "default.yaml",
    "rl_only_pt": "rl_only_pt.yaml",
    "rl_only_ptd": "rl_only_ptd.yaml",
    "rl_only_ptg": "rl_only_ptg.yaml",
    "tac_recon": "tac_recon.yaml",
    "full_recon": "full_recon.yaml",

    "forward_dynamics": "forward_dynamics.yaml",
    "forward_dynamics_memory": "forward_dynamics_memory.yaml",
    "tac_dynamics": "tac_dynamics.yaml",
    
}


def _variant_paths(robot_subdir: str, variant_files: dict[str, str]) -> dict[str, str]:
    base = os.path.join(_AGENTS_DIR, robot_subdir)
    return {key: os.path.join(base, filename) for key, filename in variant_files.items()}


def peace_make_env(cfg, render_mode: str | None = None, **kwargs):
    """Instantiate PeaceSignEnv, stripping registry kwargs from kwargs."""
    reg_keys = set(_PEACE_SHADOWLITE_VARIANT_FILES) | {"env_cfg_entry_point"}
    for k in reg_keys:
        kwargs.pop(k, None)
    return PeaceSignEnv(cfg=cfg, render_mode=render_mode, **kwargs)


gym.register(
    id="PeaceSign_Shadowlite",
    entry_point=peace_make_env,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": PeaceSignCfg,
        **_variant_paths("shadowlite", _PEACE_SHADOWLITE_VARIANT_FILES),
    },
)
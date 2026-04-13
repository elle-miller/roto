# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Central bounce task registration (all robots share the ``Bounce`` gym id)."""

import os

import gymnasium as gym

from . import agents
from .bounce import (
    BounceAllegroCfg,
    BounceAllegroEnv,
    BounceCfg,
    BounceOrcaCfg,
    BounceOrcaEnv,
    BounceShadowEnv,
    BounceShadowLiteCfg,
    BounceShadowLiteEnv,
)

_AGENTS_DIR = os.path.dirname(agents.__file__)

_SHADOW_VARIANT_FILES = {
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


def bounce_make_env(cfg, render_mode: str | None = None, **kwargs):
    """Instantiate the correct env class from the config type (set from ``--robot`` in training scripts)."""
    reg_keys = set(_SHADOW_VARIANT_FILES) | {"env_cfg_entry_point"}
    for k in reg_keys:
        kwargs.pop(k, None)
    if isinstance(cfg, BounceOrcaCfg):
        return BounceOrcaEnv(cfg=cfg, render_mode=render_mode, **kwargs)
    if isinstance(cfg, BounceAllegroCfg):
        return BounceAllegroEnv(cfg=cfg, render_mode=render_mode, **kwargs)
    if isinstance(cfg, BounceShadowLiteCfg):
        return BounceShadowLiteEnv(cfg=cfg, render_mode=render_mode, **kwargs)
    return BounceShadowEnv(cfg=cfg, render_mode=render_mode, **kwargs)


# Default registry kwargs use Shadow paths; training scripts load Orca/Allegro via ``register_hand_task_to_hydra``.
gym.register(
    id="Bounce",
    entry_point=bounce_make_env,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": BounceCfg,
        **_variant_paths("shadow", _SHADOW_VARIANT_FILES),
    },
)

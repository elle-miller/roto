# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Central bounce task registration (all robots)."""

import os

import gymnasium as gym

from . import agents
from .bounce import BounceAllegroCfg, BounceAllegroEnv, BounceCfg, BounceOrcaCfg, BounceOrcaEnv, BounceShadowEnv

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

_ORCA_ALLEGRO_VARIANT_FILES = {
    "default_cfg": "default.yaml",
    "rl_only_pt": "rl_only_pt.yaml",
    "rl_only_ptg": "rl_only_ptg.yaml",
    "forward_dynamics": "forward_dynamics.yaml",
}


def _variant_paths(robot_subdir: str, variant_files: dict[str, str]) -> dict[str, str]:
    base = os.path.join(_AGENTS_DIR, robot_subdir)
    return {key: os.path.join(base, filename) for key, filename in variant_files.items()}


def _register(gym_id: str, env_cls, cfg_cls, variant_files: dict[str, str], robot_subdir: str) -> None:
    kwargs = {"env_cfg_entry_point": cfg_cls}
    kwargs.update(_variant_paths(robot_subdir, variant_files))
    gym.register(
        id=gym_id,
        entry_point=f"{env_cls.__module__}:{env_cls.__name__}",
        disable_env_checker=True,
        kwargs=kwargs,
    )


_register("Bounce", BounceShadowEnv, BounceCfg, _SHADOW_VARIANT_FILES, "shadow")
_register("Bounce_Orca", BounceOrcaEnv, BounceOrcaCfg, _ORCA_ALLEGRO_VARIANT_FILES, "orca")
_register("Bounce_Allegro", BounceAllegroEnv, BounceAllegroCfg, _ORCA_ALLEGRO_VARIANT_FILES, "allegro")

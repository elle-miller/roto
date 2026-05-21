# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Find task registration (Franka Panda)."""

import os

import gymnasium as gym

from . import agents
from .find import FindCfg, FindEnv

_AGENTS_DIR = os.path.dirname(agents.__file__)

_FRANKA_VARIANT_FILES = {
    "default_cfg": "default.yaml",
    "rl_only_pt": "rl_only_pt.yaml",
    "rl_only_ptg": "rl_only_ptg.yaml",
    "rl_only_ptd": "rl_only_ptd.yaml",
    "tac_recon": "tac_recon.yaml",
    "full_recon": "full_recon.yaml",
    "forward_dynamics": "forward_dynamics.yaml",
    "tac_dynamics": "tac_dynamics.yaml",
    "forward_dynamics_memory": "forward_dynamics_memory.yaml",
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


_register("Find", FindEnv, FindCfg, _FRANKA_VARIANT_FILES, "franka")

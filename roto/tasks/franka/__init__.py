# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""Register Franka gym environments and expose agent configs."""

import gymnasium as gym
import os

from . import agents, find

_VARIANT_FILES = {
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

_AGENTS_DIR = os.path.dirname(agents.__file__)


def _variant_paths(task_name: str) -> dict[str, str]:
    base = os.path.join(_AGENTS_DIR, task_name)
    return {key: os.path.join(base, filename) for key, filename in _VARIANT_FILES.items()}


def _register_find_env() -> None:
    kwargs = {"env_cfg_entry_point": find.FindEnvCfg}
    kwargs.update(_variant_paths("find"))

    gym.register(
        id="Find",
        entry_point="roto.tasks.franka.find:FindEnv",
        kwargs=kwargs,
        disable_env_checker=True,
    )


_register_find_env()

# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""Register Franka gym environments and expose agent configs."""

import gymnasium as gym
import os

from . import agents, find, pick_and_place

_VARIANT_FILES = {
    "default_cfg": "default.yaml",
    # "rl_only_pt": "rl_only_pt.yaml",
    "rl_only_ptg": "rl_only_ptg.yaml",
    # "rl_only_ptd": "rl_only_ptd.yaml",
    # "tac_recon": "tac_recon.yaml",
    # "full_recon": "full_recon.yaml",
    # "forward_dynamics": "forward_dynamics.yaml",
    # "tac_dynamics": "tac_dynamics.yaml",
    # "forward_dynamics_memory": "forward_dynamics_memory.yaml",

}

_AGENTS_DIR = os.path.dirname(agents.__file__)

# Map task IDs to their directory names in agents/
_TASK_DIR_MAP = {
    "pickandplace": "pick_and_place",
}


def _variant_paths(task_name: str) -> dict[str, str]:
    # Use mapped directory name if available, otherwise use task_name as-is
    dir_name = _TASK_DIR_MAP.get(task_name.lower(), task_name.lower())
    base = os.path.join(_AGENTS_DIR, dir_name)
    return {key: os.path.join(base, filename) for key, filename in _VARIANT_FILES.items()}


def _register_task(task_id: str, env_cls, cfg_cls) -> None:
    kwargs = {"env_cfg_entry_point": cfg_cls}
    kwargs.update(_variant_paths(task_id.lower()))

    # Get the correct module name for entry_point (handle pickandplace -> pick_and_place)
    module_name = _TASK_DIR_MAP.get(task_id.lower(), task_id.lower())
    
    gym.register(
        id=task_id,
        entry_point=f"roto.tasks.franka.{module_name}:{env_cls.__name__}",
        disable_env_checker=True,
        kwargs=kwargs,
    )


_register_task("Find", find.FindEnv, find.FindEnvCfg)
_register_task("PickAndPlace", pick_and_place.PickAndPlaceEnv, pick_and_place.PickAndPlaceEnvCfg)

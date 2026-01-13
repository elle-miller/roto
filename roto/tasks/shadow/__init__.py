# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Register Shadow-hand gym environments and expose agent configs."""

import gymnasium as gym
import os

from . import agents, baoding, bounce

_VARIANT_FILES = {
    "default_cfg": "default.yaml",
    # proprioception + tactile
    "rl_only_pt": "rl_only_pt.yaml",
    # proprioception + tactile + depth
    "rl_only_ptd": "rl_only_ptd.yaml",
    # proprioception + tactile + ground truth info
    "rl_only_ptg": "rl_only_ptg.yaml",
    # ssl tasks
    "tac_recon": "tac_recon.yaml",
    "full_recon": "full_recon.yaml",
    "forward_dynamics": "forward_dynamics.yaml",
    "forward_dynamics_memory": "forward_dynamics_memory.yaml",
    "tac_dynamics": "tac_dynamics.yaml",
}

_AGENTS_DIR = os.path.dirname(agents.__file__)


def _variant_paths(task_name: str) -> dict[str, str]:
    base = os.path.join(_AGENTS_DIR, task_name)
    return {key: os.path.join(base, filename) for key, filename in _VARIANT_FILES.items()}


def _register_task(task_id: str, env_cls, cfg_cls) -> None:
    kwargs = {"env_cfg_entry_point": cfg_cls}
    kwargs.update(_variant_paths(task_id.lower()))

    gym.register(
        id=task_id,
        entry_point=f"roto.tasks.shadow.{task_id.lower()}:{env_cls.__name__}",
        disable_env_checker=True,
        kwargs=kwargs,
    )


_register_task("Bounce", bounce.BounceEnv, bounce.BounceCfg)
_register_task("Baoding", baoding.BaodingEnv, baoding.BaodingCfg)

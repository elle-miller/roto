# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Register Shadow-hand gym environments and expose agent configs."""

import gymnasium as gym
import os

from . import agents, allegro_baoding, allegro_bounce

_VARIANT_FILES = {
    "default_cfg": "default.yaml",
    "rl_only_pt": "rl_only_pt.yaml",
    "rl_only_ptg": "rl_only_ptg.yaml",
    # ssl tasks
    "forward_dynamics": "forward_dynamics.yaml",
}

_AGENTS_DIR = os.path.dirname(agents.__file__)


def _variant_paths(task_name: str) -> dict[str, str]:
    # Strip the "allegro" prefix so "allegrobounce" → "bounce" matches the directory name
    short_name = task_name.removeprefix("allegro")
    base = os.path.join(_AGENTS_DIR, short_name)
    return {key: os.path.join(base, filename) for key, filename in _VARIANT_FILES.items()}


def _register_task(task_id: str, env_cls, cfg_cls) -> None:
    kwargs = {"env_cfg_entry_point": cfg_cls}
    kwargs.update(_variant_paths(task_id.lower()))

    gym.register(
        id=task_id,
        entry_point=f"{env_cls.__module__}:{env_cls.__name__}",
        disable_env_checker=True,
        kwargs=kwargs,
    )

_register_task("AllegroBounce", allegro_bounce.BounceEnv, allegro_bounce.BounceCfg)
_register_task("AllegroBaoding", allegro_baoding.BaodingEnv, allegro_baoding.BaodingCfg)

# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
import gymnasium as gym
import os

from . import agents
from . import lift

print("Registering franka environments")

agents_dir = os.path.dirname(agents.__file__)

agent_config = os.path.join(agents_dir, "lift.yaml")

gym.register(
    id="Franka_Lift",
    entry_point="tasks.franka.lift:LiftEnv",
    kwargs={
        "env_cfg_entry_point": lift.LiftEnvCfg,
        "skrl_cfg_entry_point": agent_config,
    },
    disable_env_checker=True,
)

# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
import gymnasium as gym
import os

from . import agents, my_task

print("Registering franka environments")

agents_dir = os.path.dirname(agents.__file__)

agent_config = os.path.join(agents_dir, "ppo.yaml")

gym.register(
    id="Example_Task",
    entry_point="tasks.template.my_task:MyTaskEnv",
    kwargs={
        "env_cfg_entry_point": my_task.MyTaskEnvCfg,
        "skrl_cfg_entry_point": agent_config,
    },
    disable_env_checker=True,
)

# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Shadow Hand environment.
"""

import os
import gymnasium as gym

from . import agents, bounce, baoding

##
# Register Gym environments.
##

print("Registering shadow environments")

agents_dir = os.path.dirname(agents.__file__)

p_file = "rl_only_p.yaml"
pt_file = "rl_only_pt.yaml"
ptg_file = "rl_only_ptg.yaml"
full_recon_file = "full_recon.yaml"
tactile_recon_file = "tactile_recon.yaml"
full_dynamics_file = "full_dynamics.yaml"
tactile_dynamics_file = "tactile_dynamics.yaml"

bounce_rl_only_p = os.path.join(agents_dir, "bounce", p_file)
bounce_rl_only_pt = os.path.join(agents_dir, "bounce", pt_file)
bounce_rl_only_ptg = os.path.join(agents_dir, "bounce", ptg_file)
bounce_tactile_recon = os.path.join(agents_dir, "bounce", tactile_recon_file)
bounce_full_recon = os.path.join(agents_dir, "bounce", full_recon_file)
bounce_full_dynamics = os.path.join(agents_dir, "bounce", full_dynamics_file)
bounce_tactile_dynamics = os.path.join(agents_dir, "bounce", tactile_dynamics_file)

baoding_rl_only_p = os.path.join(agents_dir, "baoding", p_file)
baoding_rl_only_pt = os.path.join(agents_dir, "baoding", pt_file)
baoding_rl_only_ptg = os.path.join(agents_dir, "baoding", ptg_file)
baoding_tactile_recon = os.path.join(agents_dir, "baoding", tactile_recon_file)
baoding_full_recon = os.path.join(agents_dir, "baoding", full_recon_file)
baoding_full_dynamics = os.path.join(agents_dir, "baoding", full_dynamics_file)
baoding_tactile_dynamics = os.path.join(agents_dir, "baoding", tactile_dynamics_file)


gym.register(
    id="Shadow_Bounce_PTG",
    entry_point="tasks.shadow.bounce:BounceEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": bounce.BounceCfg,
        "skrl_cfg_entry_point": bounce_rl_only_ptg,
    },
)

gym.register(
    id="Shadow_Bounce_p",
    entry_point="tasks.shadow.bounce:BounceEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": bounce.BounceCfg,
        "skrl_cfg_entry_point": bounce_rl_only_p,
    },
)

gym.register(
    id="Shadow_Bounce",
    entry_point="tasks.shadow.bounce:BounceEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": bounce.BounceCfg,
        "skrl_cfg_entry_point": bounce_rl_only_pt,
    },
)


gym.register(
    id="Shadow_Bounce_Dynamics",
    entry_point="tasks.shadow.bounce:BounceEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": bounce.BounceCfg,
        "skrl_cfg_entry_point": bounce_full_dynamics,
    },
)

gym.register(
    id="Shadow_Bounce_TactileDynamics",
    entry_point="tasks.shadow.bounce:BounceEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": bounce.BounceCfg,
        "skrl_cfg_entry_point": bounce_tactile_dynamics,
    },
)



# gym.register(
#     id="Shadow_Bounce_Dynamics_Memory_Return",
#     entry_point="tasks.shadow.bounce:BounceEnv",
#     disable_env_checker=True,
#     kwargs={
#         "env_cfg_entry_point": bounce.BounceCfg,
#         "skrl_cfg_entry_point": bounce_full_dynamics_memory_return,
#     },
# )
# gym.register(
#     id="Shadow_Bounce_Dynamics_Memory_TD",
#     entry_point="tasks.shadow.bounce:BounceEnv",
#     disable_env_checker=True,
#     kwargs={
#         "env_cfg_entry_point": bounce.BounceCfg,
#         "skrl_cfg_entry_point": bounce_dynamics_td,
#     },
# )

gym.register(
    id="Shadow_Bounce_Recon",
    entry_point="tasks.shadow.bounce:BounceEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": bounce.BounceCfg,
        "skrl_cfg_entry_point": bounce_full_recon,
    },
)

gym.register(
    id="Shadow_Bounce_TactileRecon",
    entry_point="tasks.shadow.bounce:BounceEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": bounce.BounceCfg,
        "skrl_cfg_entry_point": bounce_tactile_recon,
    },
)


# BAODING

gym.register(
    id="Shadow_Baoding_p",
    entry_point="tasks.shadow.baoding:BaodingEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": baoding.BaodingCfg,
        "skrl_cfg_entry_point": baoding_rl_only_p,
    },
)

gym.register(
    id="Shadow_Baoding",
    entry_point="tasks.shadow.baoding:BaodingEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": baoding.BaodingCfg,
        "skrl_cfg_entry_point": baoding_rl_only_pt,
    },
)


gym.register(
    id="Shadow_Baoding_TactileRecon",
    entry_point="tasks.shadow.baoding:BaodingEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": baoding.BaodingCfg,
        "skrl_cfg_entry_point": baoding_tactile_recon,
    },
)

gym.register(
    id="Shadow_Baoding_Recon",
    entry_point="tasks.shadow.baoding:BaodingEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": baoding.BaodingCfg,
        "skrl_cfg_entry_point": baoding_full_recon,
    },
)

gym.register(
    id="Shadow_Baoding_Dynamics",
    entry_point="tasks.shadow.baoding:BaodingEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": baoding.BaodingCfg,
        "skrl_cfg_entry_point": baoding_full_dynamics,
    },
)

# gym.register(
#     id="Shadow_Baoding_Dynamics_Memory",
#     entry_point="tasks.shadow.baoding:BaodingEnv",
#     disable_env_checker=True,
#     kwargs={
#         "env_cfg_entry_point": baoding.BaodingCfg,
#         "skrl_cfg_entry_point": baoding_dynamics_memory,
#     },
# )

# gym.register(
#     id="Shadow_Baoding_Dynamics_Memory_Return",
#     entry_point="tasks.shadow.baoding:BaodingEnv",
#     disable_env_checker=True,
#     kwargs={
#         "env_cfg_entry_point": baoding.BaodingCfg,
#         "skrl_cfg_entry_point": baoding_dynamics_return,
#     },
# )

# gym.register(
#     id="Shadow_Baoding_Dynamics_Memory_TD",
#     entry_point="tasks.shadow.baoding:BaodingEnv",
#     disable_env_checker=True,
#     kwargs={
#         "env_cfg_entry_point": baoding.BaodingCfg,
#         "skrl_cfg_entry_point": baoding_dynamics_td,
#     },
# )


gym.register(
    id="Shadow_Baoding_TactileDynamics",
    entry_point="tasks.shadow.baoding:BaodingEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": baoding.BaodingCfg,
        "skrl_cfg_entry_point": baoding_tactile_dynamics,
    },
)
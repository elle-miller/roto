# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Shadow Hand environment.
"""

import os
import gymnasium as gym

from . import agents, shadow_hand_env_cfg, bounce, baoding

##
# Register Gym environments.
##

print("Registering shadow environments")

agents_dir = os.path.dirname(agents.__file__)

p_file = "prop.yaml"
pt_file = "prop_tactile.yaml"
ptg_file = "prop_gt_tactile.yaml"
ptb_file = "prop_tactile_binary.yaml"
ptc_file = "prop_tactile_cont.yaml"


# concat
concat_pt = os.path.join(agents_dir, "concat", pt_file)
concat_ptg = os.path.join(agents_dir, "concat", ptg_file)

concat_skrl = os.path.join(agents_dir, "concat", "skrl_ppo.yaml")

repose_ptg = os.path.join(agents_dir, "repose", ptg_file)


# bounce
bounce_p = os.path.join(agents_dir, "bounce", p_file)
bounce_ptg = os.path.join(agents_dir, "bounce", ptg_file)
bounce_ptb = os.path.join(agents_dir, "bounce", ptb_file)
bounce_ptc = os.path.join(agents_dir, "bounce", ptc_file)

bounce_dynamics = os.path.join(agents_dir, "bounce", "prop_tactile_binary_dynamics.yaml")

bounce_dynamics_return = os.path.join(agents_dir, "bounce", "prop_tactile_binary_dynamics_memory_return.yaml")
bounce_dynamics_td = os.path.join(agents_dir, "bounce", "prop_tactile_binary_dynamics_memory_td.yaml")


bounce_tactile_dynamics = os.path.join(agents_dir, "bounce", "prop_tactile_binary_dynamics_tactile.yaml")

bounce_recon = os.path.join(agents_dir, "bounce", "prop_tactile_binary_reconstruction_full.yaml")
bounce_tactile_recon = os.path.join(agents_dir, "bounce", "prop_tactile_binary_reconstruction.yaml")

bounce_ptc_recon = os.path.join(agents_dir, "bounce", "prop_tactile_cont_reconstruction.yaml")

rotate_ptg = os.path.join(agents_dir, "rotate", ptg_file)
rotate_pt = os.path.join(agents_dir, "rotate", pt_file)
rotate_pt_dynamics = os.path.join(agents_dir, "rotate", "prop_tactile_dynamics.yaml")

baoding_p = os.path.join(agents_dir, "baoding", p_file)
baoding_ptb = os.path.join(agents_dir, "baoding", ptb_file)
baoding_recon = os.path.join(agents_dir, "baoding", "prop_tactile_binary_reconstruction_full.yaml")
baoding_dynamics = os.path.join(agents_dir, "baoding", "prop_tactile_binary_dynamics.yaml")

baoding_dynamics_memory = os.path.join(agents_dir, "baoding", "prop_tactile_binary_dynamics_memory.yaml")
baoding_dynamics_return = os.path.join(agents_dir, "baoding", "prop_tactile_binary_dynamics_memory_return.yaml")
baoding_dynamics_td = os.path.join(agents_dir, "baoding", "prop_tactile_binary_dynamics_memory_td.yaml")

baoding_tactile_recon = os.path.join(agents_dir, "baoding", "prop_tactile_binary_reconstruction.yaml")

baoding_tactile_dynamics = os.path.join(agents_dir, "baoding", "prop_tactile_binary_dynamics_tactile.yaml")

bounce_dynamics2 = os.path.join(agents_dir, "bounce", "prop_tactile_binary_dynamics2.yaml")

gym.register(
    id="Shadow",
    entry_point="roto.tasks.shadow.inhand:InHandManipulationEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": shadow_hand_env_cfg.ShadowHandEnvCfg,
        "skrl_cfg_entry_point": concat_ptg,
    },
)

gym.register(
    id="Shadow_Bounce_PTG",
    entry_point="roto.tasks.shadow.bounce:BounceEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": bounce.BounceCfg,
        "skrl_cfg_entry_point": bounce_ptg,
    },
)

gym.register(
    id="Shadow_Bounce_p",
    entry_point="roto.tasks.shadow.bounce:BounceEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": bounce.BounceCfg,
        "skrl_cfg_entry_point": bounce_p,
    },
)

gym.register(
    id="Shadow_Bounce",
    entry_point="roto.tasks.shadow.bounce:BounceEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": bounce.BounceCfg,
        "skrl_cfg_entry_point": bounce_ptb,
    },
)

gym.register(
    id="Shadow_Bounce_PT_c",
    entry_point="roto.tasks.shadow.bounce:BounceEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": bounce.BounceCfg,
        "skrl_cfg_entry_point": bounce_ptc,
    },
)


gym.register(
    id="Shadow_Bounce_Dynamics",
    entry_point="roto.tasks.shadow.bounce:BounceEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": bounce.BounceCfg,
        "skrl_cfg_entry_point": bounce_dynamics,
    },
)

gym.register(
    id="Shadow_Bounce_TactileDynamics",
    entry_point="roto.tasks.shadow.bounce:BounceEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": bounce.BounceCfg,
        "skrl_cfg_entry_point": bounce_tactile_dynamics,
    },
)

gym.register(
    id="Shadow_Bounce_Dynamics",
    entry_point="tasks.shadow.bounce:BounceEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": bounce.BounceCfg,
        "skrl_cfg_entry_point": bounce_dynamics,
    },
)

gym.register(
    id="Shadow_Bounce_Dynamics_Memory_Return",
    entry_point="roto.tasks.shadow.bounce:BounceEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": bounce.BounceCfg,
        "skrl_cfg_entry_point": bounce_dynamics_return,
    },
)
gym.register(
    id="Shadow_Bounce_Dynamics_Memory_TD",
    entry_point="roto.tasks.shadow.bounce:BounceEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": bounce.BounceCfg,
        "skrl_cfg_entry_point": bounce_dynamics_td,
    },
)

gym.register(
    id="Shadow_Bounce_Recon",
    entry_point="roto.tasks.shadow.bounce:BounceEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": bounce.BounceCfg,
        "skrl_cfg_entry_point": bounce_recon,
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



gym.register(
    id="Shadow_skrl",
    entry_point="roto.tasks.shadow.inhand:InHandManipulationEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": shadow_hand_env_cfg.ShadowHandEnvCfg,
        "skrl_cfg_entry_point": concat_skrl,
    },
)


# BAODING

gym.register(
    id="Shadow_Baoding_p",
    entry_point="roto.tasks.shadow.baoding:BaodingEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": baoding.BaodingCfg,
        "skrl_cfg_entry_point": baoding_p,
    },
)

gym.register(
    id="Shadow_Baoding",
    entry_point="tasks.shadow.baoding:BaodingEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": baoding.BaodingCfg,
        "skrl_cfg_entry_point": baoding_ptb,
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
    entry_point="roto.tasks.shadow.baoding:BaodingEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": baoding.BaodingCfg,
        "skrl_cfg_entry_point": baoding_recon,
    },
)

gym.register(
    id="Shadow_Baoding_Dynamics",
    entry_point="tasks.shadow.baoding:BaodingEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": baoding.BaodingCfg,
        "skrl_cfg_entry_point": baoding_dynamics,
    },
)

gym.register(
    id="Shadow_Baoding_Dynamics_Memory",
    entry_point="roto.tasks.shadow.baoding:BaodingEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": baoding.BaodingCfg,
        "skrl_cfg_entry_point": baoding_dynamics_memory,
    },
)

gym.register(
    id="Shadow_Baoding_Dynamics_Memory_Return",
    entry_point="roto.tasks.shadow.baoding:BaodingEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": baoding.BaodingCfg,
        "skrl_cfg_entry_point": baoding_dynamics_return,
    },
)

gym.register(
    id="Shadow_Baoding_Dynamics_Memory_TD",
    entry_point="roto.tasks.shadow.baoding:BaodingEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": baoding.BaodingCfg,
        "skrl_cfg_entry_point": baoding_dynamics_td,
    },
)


gym.register(
    id="Shadow_Baoding_TactileDynamics",
    entry_point="roto.tasks.shadow.baoding:BaodingEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": baoding.BaodingCfg,
        "skrl_cfg_entry_point": baoding_tactile_dynamics,
    },
)
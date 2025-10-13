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
tactile_recon_file = "tac_recon.yaml"
full_dynamics_file = "full_dynamics.yaml"
tactile_dynamics_file = "tac_dynamics.yaml"

bounce_default_cfg = os.path.join(agents_dir, "bounce", "default.yaml")
bounce_rl_only_p = os.path.join(agents_dir, "bounce", p_file)
bounce_rl_only_pt = os.path.join(agents_dir, "bounce", pt_file)
bounce_rl_only_ptg = os.path.join(agents_dir, "bounce", ptg_file)
bounce_tactile_recon = os.path.join(agents_dir, "bounce", tactile_recon_file)
bounce_full_recon = os.path.join(agents_dir, "bounce", full_recon_file)
bounce_full_dynamics = os.path.join(agents_dir, "bounce", full_dynamics_file)
bounce_tactile_dynamics = os.path.join(agents_dir, "bounce", tactile_dynamics_file)

baoding_default_cfg = os.path.join(agents_dir, "baoding", "default.yaml")
baoding_rl_only_p = os.path.join(agents_dir, "baoding", p_file)
baoding_rl_only_pt = os.path.join(agents_dir, "baoding", pt_file)
baoding_rl_only_ptg = os.path.join(agents_dir, "baoding", ptg_file)
baoding_tactile_recon = os.path.join(agents_dir, "baoding", tactile_recon_file)
baoding_full_recon = os.path.join(agents_dir, "baoding", full_recon_file)
baoding_full_dynamics = os.path.join(agents_dir, "baoding", full_dynamics_file)
baoding_tactile_dynamics = os.path.join(agents_dir, "baoding", tactile_dynamics_file)


gym.register(
    id="Bounce",
    entry_point="tasks.shadow.bounce:BounceEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": bounce.BounceCfg,
        "default_cfg": bounce_default_cfg,
        "rl_only_p": bounce_rl_only_p,
        "rl_only_pt": bounce_rl_only_pt,
        "tac_recon": bounce_tactile_recon,
        "full_recon": bounce_full_recon,
        "full_dynamics": bounce_full_dynamics,
        "tac_dynamics": bounce_tactile_dynamics,
    },
)

gym.register(
    id="Baoding",
    entry_point="tasks.shadow.baoding:BaodingEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": baoding.BaodingCfg,
        "default_cfg": baoding_default_cfg,
        "rl_only_p": baoding_rl_only_p,
        "rl_only_pt": baoding_rl_only_pt,
        "tac_recon": baoding_tactile_recon,
        "full_recon": baoding_full_recon,
        "full_dynamics": baoding_full_dynamics,
        "tac_dynamics": baoding_tactile_dynamics,
    }
)


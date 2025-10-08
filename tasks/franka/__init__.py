# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
import gymnasium as gym
import os

# from . import agents, deformable_lift, franka, find, occluded_lift, bowl_lift, shadow_lift, trick_lift  # ik_abs_env_cfg, ik_rel_env_cfg, joint_pos_env_cfg
from . import agents
from . import find #, grasp, control, pick_and_place


##
# Register Gym environments.
##

##
# Joint Position Control
##
print("Registering franka environments")

agents_dir = os.path.dirname(agents.__file__)



p_file = "rl_only_p.yaml"
pt_file = "rl_only_pt.yaml"
ptg_file = "rl_only_ptg.yaml"
full_recon_file = "full_recon.yaml"
tactile_recon_file = "tac_recon.yaml"
full_dynamics_file = "full_dynamics.yaml"
tactile_dynamics_file = "tac_dynamics.yaml"

default_cfg = os.path.join(agents_dir, "find", "default.yaml")
find_rl_only_p = os.path.join(agents_dir, "find", p_file)
find_rl_only_pt = os.path.join(agents_dir, "find", pt_file)
find_rl_only_ptg = os.path.join(agents_dir, "find", ptg_file)
find_tactile_recon = os.path.join(agents_dir, "find", tactile_recon_file)
find_full_recon = os.path.join(agents_dir, "find", full_recon_file)
find_full_dynamics = os.path.join(agents_dir, "find", full_dynamics_file)
find_tactile_dynamics = os.path.join(agents_dir, "find", tactile_dynamics_file)

gym.register(
    id="Find",
    entry_point="tasks.franka.find:FindEnv",
    kwargs={
        "env_cfg_entry_point": find.FindEnvCfg,
        "default_cfg": default_cfg,
        "rl_only_p": find_rl_only_p,
        "rl_only_pt": find_rl_only_pt,
        "tac_recon": find_tactile_recon,
        "full_recon": find_full_recon,
        "full_dynamics": find_full_dynamics,
        "tac_dynamics": find_tactile_dynamics,
    },
    disable_env_checker=True,
)

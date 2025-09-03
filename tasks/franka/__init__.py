# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
import gymnasium as gym
import os

# from . import agents, deformable_lift, franka, touch, occluded_lift, bowl_lift, shadow_lift, trick_lift  # ik_abs_env_cfg, ik_rel_env_cfg, joint_pos_env_cfg
from . import agents
from . import touch #, grasp, control, pick_and_place


##
# Register Gym environments.
##

##
# Joint Position Control
##
print("Registering franka environments")

agents_dir = os.path.dirname(agents.__file__)

pt_file = "prop_tactile.yaml"
ptg_file = "prop_gt_tactile.yaml"

ptb_file = "prop_tactile_binary.yaml"
ptc_file = "prop_tactile_cont.yaml"

p_file = "prop.yaml"

find_concat_p = os.path.join(agents_dir, "find", p_file)

find_concat_ptb = os.path.join(agents_dir, "find", ptb_file)
find_concat_ptc = os.path.join(agents_dir, "find", ptc_file)
find_recon_tactile = os.path.join(agents_dir, "find", "prop_tactile_binary_reconstruction.yaml")
find_recon_full = os.path.join(agents_dir, "find", "prop_tactile_binary_reconstruction_full.yaml")

find_dynamics_ptb = os.path.join(agents_dir, "find", "prop_tactile_binary_dynamics.yaml")
find_tactile_dynamics = os.path.join(agents_dir, "find", "prop_tactile_binary_dynamics_tactile.yaml")

# concat
touch_concat_pt = os.path.join(agents_dir, "touch", "concat", pt_file)
touch_concat_ptg = os.path.join(agents_dir, "touch", "concat", ptg_file)

grasp_concat_pt = os.path.join(agents_dir, "grasp", "concat", pt_file)
grasp_concat_ptg = os.path.join(agents_dir, "grasp", "concat", ptg_file)

# recon
touch_recon_pt = os.path.join(agents_dir, "touch", "reconstruction", pt_file)
grasp_recon_ptg = os.path.join(agents_dir, "grasp", "reconstruction", ptg_file)

# recon
touch_dynamics_pt = os.path.join(agents_dir, "touch", "dynamics", pt_file)
touch_dynamics_ptg = os.path.join(agents_dir, "touch", "dynamics", ptg_file)

### TOUCH ###

gym.register(
    id="FrankaFind_p",
    entry_point="roto.tasks.franka.touch:TouchEnv",
    kwargs={
        "env_cfg_entry_point": touch.TouchEnvCfg,
        "skrl_cfg_entry_point": find_concat_p,
    },
    disable_env_checker=True,
)

gym.register(
    id="FrankaFind_b",
    entry_point="tasks.franka.touch:TouchEnv",
    kwargs={
        "env_cfg_entry_point": touch.TouchEnvCfg,
        "skrl_cfg_entry_point": find_concat_ptb,
    },
    disable_env_checker=True,
)

gym.register(
    id="FrankaFind_c",
    entry_point="roto.tasks.franka.touch:TouchEnv",
    kwargs={
        "env_cfg_entry_point": touch.TouchEnvCfg,
        "skrl_cfg_entry_point": find_concat_ptc,
    },
    disable_env_checker=True,
)

gym.register(
    id="FrankaFind_b_Recon",
    entry_point="roto.tasks.franka.touch:TouchEnv",
    kwargs={
        "env_cfg_entry_point": touch.TouchEnvCfg,
        "skrl_cfg_entry_point": find_recon_full,
    },
    disable_env_checker=True,
)

gym.register(
    id="FrankaFind_b_TactileRecon",
    entry_point="tasks.franka.touch:TouchEnv",
    kwargs={
        "env_cfg_entry_point": touch.TouchEnvCfg,
        "skrl_cfg_entry_point": find_recon_tactile,
    },
    disable_env_checker=True,
)

gym.register(
    id="FrankaFind_b_Dynamics",
    entry_point="tasks.franka.touch:TouchEnv",
    kwargs={
        "env_cfg_entry_point": touch.TouchEnvCfg,
        "skrl_cfg_entry_point": find_dynamics_ptb,
    },
    disable_env_checker=True,
)


gym.register(
    id="FrankaFind_b_TactileDynamics",
    entry_point="roto.tasks.franka.touch:TouchEnv",
    kwargs={
        "env_cfg_entry_point": touch.TouchEnvCfg,
        "skrl_cfg_entry_point": find_tactile_dynamics,
    },
    disable_env_checker=True,
)


########################################################################################################


gym.register(
    id="PropGT_FrankaTouch",
    entry_point="roto.tasks.franka.touch:TouchEnv",
    kwargs={
        "env_cfg_entry_point": touch.TouchEnvCfg,
        "skrl_cfg_entry_point": f"{agents.__name__}:prop_gt.yaml",
    },
    disable_env_checker=True,
)

gym.register(
    id="PropTactileGT_FrankaTouch",
    entry_point="roto.tasks.franka.touch:TouchEnv",
    kwargs={
        "env_cfg_entry_point": touch.TouchEnvCfg,
        "skrl_cfg_entry_point": touch_concat_ptg,
    },
    disable_env_checker=True,
)

gym.register(
    id="PropTactile_FrankaTouch",
    entry_point="roto.tasks.franka.touch:TouchEnv",
    kwargs={
        "env_cfg_entry_point": touch.TouchEnvCfg,
        "skrl_cfg_entry_point": touch_concat_pt,
    },
    disable_env_checker=True,
)

gym.register(
    id="Dynamics_PropTactile_FrankaTouch",
    entry_point="roto.tasks.franka.touch:TouchEnv",
    kwargs={
        "env_cfg_entry_point": touch.TouchEnvCfg,
        "skrl_cfg_entry_point": touch_dynamics_pt,
        },
    disable_env_checker=True,
)
gym.register(
    id="Dynamics_PropTactileGT_FrankaTouch",
    entry_point="roto.tasks.franka.touch:TouchEnv",
    kwargs={
        "env_cfg_entry_point": touch.TouchEnvCfg,
        "skrl_cfg_entry_point": touch_dynamics_ptg,
        },
    disable_env_checker=True,
)

gym.register(
    id="Recon_PropTactile_FrankaTouch",
    entry_point="roto.tasks.franka.touch:TouchEnv",
    kwargs={
        "env_cfg_entry_point": touch.TouchEnvCfg,
        "skrl_cfg_entry_point": touch_recon_pt,
    },
    disable_env_checker=True,
)

### GRASP ###

# gym.register(
#     id="PropGT_FrankaGrasp",
#     entry_point="roto.tasks.franka.grasp:GraspEnv",
#     kwargs={
#         "env_cfg_entry_point": grasp.GraspEnvCfg,
#         "skrl_cfg_entry_point": f"{agents.__name__}:prop_gt.yaml",
#     },
#     disable_env_checker=True,
# )



# gym.register(
#     id="PropTactileGT_FrankaGrasp",
#     entry_point="roto.tasks.franka.grasp:GraspEnv",
#     kwargs={
#         "env_cfg_entry_point": grasp.GraspEnvCfg,
#         "skrl_cfg_entry_point":  grasp_concat_ptg,
#     },
#     disable_env_checker=True,
# )

# gym.register(
#     id="PropTactile_FrankaGrasp",
#     entry_point="roto.tasks.franka.grasp:GraspEnv",
#     kwargs={
#         "env_cfg_entry_point": grasp.GraspEnvCfg,
#         "skrl_cfg_entry_point":grasp_concat_pt,
#     },
#     disable_env_checker=True,
# )

# ### GRASP ###

# gym.register(
#     id="PropGT_FrankaControl",
#     entry_point="roto.tasks.franka.grasp:ControlEnv",
#     kwargs={
#         "env_cfg_entry_point": control.ControlEnvCfg,
#         "skrl_cfg_entry_point": f"{agents.__name__}:prop_gt.yaml",
#     },
#     disable_env_checker=True,
# )

# gym.register(
#     id="PropTactileGT_FrankaControl",
#     entry_point="roto.tasks.franka.grasp:ControlEnv",
#     kwargs={
#         "env_cfg_entry_point": control.ControlEnvCfg,
#         "skrl_cfg_entry_point": f"{agents.__name__}:prop_gt_tactile.yaml",
#     },
#     disable_env_checker=True,
# )

# gym.register(
#     id="PropTactile_FrankaControl",
#     entry_point="roto.tasks.franka.grasp:ControlEnv",
#     kwargs={
#         "env_cfg_entry_point": control.ControlEnvCfg,
#         "skrl_cfg_entry_point": f"{agents.__name__}:prop_tactile.yaml",
#     },
#     disable_env_checker=True,
# )


# ### Pick and Place

# gym.register(
#     id="PropGT_FrankaPAP",
#     entry_point="roto.tasks.franka.pick_and_place:PickAndPlaceEnv",
#     kwargs={
#         "env_cfg_entry_point": pick_and_place.PickAndPlaceEnvCfg,
#         "skrl_cfg_entry_point": f"{agents.__name__}:prop_gt.yaml",
#     },
#     disable_env_checker=True,
# )

# gym.register(
#     id="PropTactileGT_FrankaPAP",
#     entry_point="roto.tasks.franka.pick_and_place:PickAndPlaceEnv",
#     kwargs={
#         "env_cfg_entry_point": pick_and_place.PickAndPlaceEnvCfg,
#         "skrl_cfg_entry_point": grasp_concat_ptg,
#     },
#     disable_env_checker=True,
# )

# gym.register(
#     id="PropTactile_FrankaPAP",
#     entry_point="roto.tasks.franka.pick_and_place:PickAndPlaceEnv",
#     kwargs={
#         "env_cfg_entry_point": pick_and_place.PickAndPlaceEnvCfg,
#         "skrl_cfg_entry_point": f"{agents.__name__}:prop_tactile.yaml",
#     },
#     disable_env_checker=True,
# )


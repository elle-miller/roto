"""Configuration for the ORCA.

"""
import isaaclab.sim as sim_utils

from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

from math import radians


ORCA_HAND_CFG = ArticulationCfg(
    
    # spawn=sim_utils.UsdFileCfg(
    #     # jayaram's path (don't delete)
    #     # usd_dir=f"/home/jayaram/research_threads/NUS_RA/roto/roto/assets/orca/orca.usd",
    #     # elle's path (don't delete)
    #     usd_path=f"/home/elle/code/debug/roto/roto/assets/orca/orca.usd",
    #     articulation_props=sim_utils.ArticulationRootPropertiesCfg(
    #         enabled_self_collisions=True,
    #         solver_position_iteration_count=32,
    #         solver_velocity_iteration_count=16,
    #         sleep_threshold=0.0,
    #     ),
    #     activate_contact_sensors=True,
    # ),

    spawn=sim_utils.UrdfFileCfg(
            # jayaram's path (don't delete)
            # asset_path=f"/home/jayaram/research_threads/NUS_RA/roto/roto/assets/orca/orcahand_right.urdf",
            # usd_dir=f"/home/jayaram/research_threads/NUS_RA/roto/roto/assets/orca/orcahand_right.urdf",
            # elle's path (don't delete)
            asset_path=f"/home/elle/code/debug/roto/roto/assets/orca/orcahand_right.urdf",
            usd_dir=f"/home/elle/code/debug/roto/roto/assets/orca",
            usd_file_name="orcahand_right.usd",
            fix_base=True,
            joint_drive=None,
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False,
                solver_position_iteration_count=32,
                solver_velocity_iteration_count=16,
                sleep_threshold=0.0,
            ),
            activate_contact_sensors=True,
    ),

    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.0),
        # put default joint positions here
        joint_pos={".*": 0.0},
        joint_vel={".*": 0.0},
    ),

    # put different actuator groups here
   actuators={
        "wrist": ImplicitActuatorCfg(
            joint_names_expr=["right_wrist"],
            stiffness=400.0,
            damping=40.0,
        ),
        "fingers": ImplicitActuatorCfg(
            joint_names_expr=["right_.*_(mcp|pip|dip|abd|ip)"], 
            stiffness=20.0,
            damping=2.0,
        ),
    }
)

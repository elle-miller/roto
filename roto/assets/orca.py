"""Configuration for the ORCA.

"""
import isaaclab.sim as sim_utils

from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

from math import radians


AIREC_CFG = ArticulationCfg(
    spawn=sim_utils.UrdfFileCfg(
            asset_path=f"/home/jayaram/research_threads/NUS_RA/roto/roto/assets/orca/orcahand_right.urdf",
            usd_dir=f"/home/jayaram/research_threads/NUS_RA/roto/roto/assets/orca/orcahand_right.urdf",
            usd_file_name="orcahand_right.usd",
            fix_base=False,
            joint_drive=None,
            # joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
            #     gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(
            #         stiffness=1.0,
            #         damping=0.5,
            #     ),
            # ),
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
        "all_joints": ImplicitActuatorCfg(
            joint_names_expr=[".*"],
            stiffness=1.0,
            damping=0.5,
        ),}

)

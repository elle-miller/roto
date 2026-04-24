# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the dexterous hand from Shadow Robot.

The following configurations are available:

* :obj:`SHADOW_HAND_CFG`: Shadow Hand with implicit actuator model.

Reference:

* https://www.shadowrobot.com/dexterous-hand-series/

"""


import isaaclab.sim as sim_utils
from isaaclab.actuators.actuator_cfg import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

##
# Configuration
##


# SHADOW_HAND_LITE_CFG = ArticulationCfg(
#     spawn=sim_utils.UrdfFileCfg(
#         asset_path=f"/home/ayush/icra/roto/roto/assets/shadow_lite/sr_hand.urdf", #change path
#         usd_dir=f"/home/ayush/icra/roto/roto/assets/shadow_lite",
#         usd_file_name="sr_hand_new.usd",
#         scale=(1.0, 1.0, 1.0),
#         fix_base=True,
#         joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
#             gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(
#                 stiffness=30.0,
#                 damping=1.0,
#             ),
#         ),
#         visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.5, 0.5, 0.5)),
#         collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.005, rest_offset=0.0),
#         activate_contact_sensors=True, #testing by making false; default: True
#         rigid_props=sim_utils.RigidBodyPropertiesCfg(
#             disable_gravity=True,
#             retain_accelerations=True,
#             max_depenetration_velocity=1000.0,
#         ),
#         articulation_props=sim_utils.ArticulationRootPropertiesCfg(
#             enabled_self_collisions=True,
#             solver_position_iteration_count=8,
#             solver_velocity_iteration_count=0,
#             sleep_threshold=0.005,
#             stabilization_threshold=0.0005,
#         ),
#         # collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.005, rest_offset=0.0),
#         joint_drive_props=sim_utils.JointDrivePropertiesCfg(drive_type="force"),
#         fixed_tendons_props=sim_utils.FixedTendonPropertiesCfg(limit_stiffness=30.0, damping=0.1),
#     ),
#     init_state=ArticulationCfg.InitialStateCfg(
#         pos=(0.0, 0.0, 0.5),
#         #rot=(0.0, 0.0, -0.7071, 0.7071),
        
#         joint_pos={".*": 0.0},
#     ),
#     actuators={
#         "fingers": ImplicitActuatorCfg(
#             # Matches FF, MF, RF joints 1-4 and TH joints 1, 2, 4, 5
#             joint_names_expr=["rh_[MRF]FJ[1-4]", "rh_THJ[1245]"],
#             effort_limit_sim={
#                 # Distal joints (Fingertips)
#                 "rh_[MRF]FJ1": 0.7245,
#                 # Proximal and Middle joints
#                 "rh_[MRF]FJ[23]": 0.9,
#                 # Knuckle abduction/adduction
#                 "rh_[MRF]FJ4": 0.9,
#                 # Thumb Base
#                 "rh_THJ5": 2.3722, 
#                 "rh_THJ4": 1.45,
#                 # Thumb Fingers
#                 "rh_THJ[12]": 0.99,
#             },
#             stiffness={
#                 "rh_[MRF]FJ[1-4]": 1.0,
#                 "rh_THJ[1245]": 1.0,
#             },
#             damping={
#                 "rh_[MRF]FJ[1-4]": 0.1,
#                 "rh_THJ[1245]": 0.1,
#             },
#         ),
#     },
#     soft_joint_pos_limit_factor=1.0,
# )
"""Configuration of Shadow Hand robot."""
# SHADOW_HAND_LITE_CFG = ArticulationCfg(
#     spawn=sim_utils.UsdFileCfg(
#         # Point to your local USD file
#         usd_path="/home/ayush/Desktop/icra/Roto-on-ShadowLite/roto/assets/shadow_lite/sr_hand.usd",
#         activate_contact_sensors=True,
#         rigid_props=sim_utils.RigidBodyPropertiesCfg(
#             disable_gravity=True,
#             retain_accelerations=True,
#             linear_damping=0.0,
#             angular_damping=0.01,
#             max_linear_velocity=1000.0,
#             max_angular_velocity=1000.0,
#             max_depenetration_velocity=1000.0,
#             max_contact_impulse=1e32,
#         ),
#         articulation_props=sim_utils.ArticulationRootPropertiesCfg(
#             enabled_self_collisions=True,
#             solver_position_iteration_count=8,
#             solver_velocity_iteration_count=0,
#             sleep_threshold=0.005,
#             stabilization_threshold=0.0005,
#         ),
#         collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.005, rest_offset=0.0),
#     ),
#     init_state=ArticulationCfg.InitialStateCfg(
#         pos=(0.0, 0.0, 0.5),
#         rot=(0.0, 0.0, -0.7071, 0.7071),
#         joint_pos={".*": 0.0},
#     ),
#     actuators={
#         "fingers": ImplicitActuatorCfg(
#             joint_names_expr=["rh_[MRF]FJ[1-4]", "rh_THJ[1245]"],
#             effort_limit_sim={
#                 "rh_[MRF]FJ1": 0.7245,
#                 "rh_[MRF]FJ[23]": 0.9,
#                 "rh_[MRF]FJ4": 0.9,
#                 "rh_THJ5": 2.3722, 
#                 "rh_THJ4": 1.45,
#                 "rh_THJ[12]": 0.99,
#             },
#             stiffness={
#                 "rh_[MRF]FJ[1-4]": 1.0,
#                 "rh_THJ[1245]": 1.0,
#             },
#             damping={
#                 "rh_[MRF]FJ[1-4]": 0.1,
#                 "rh_THJ[1245]": 0.1,
#             },
#         ),
#     },
#     soft_joint_pos_limit_factor=1.0,
# )
# """Configuration of Shadow Hand robot."""
SHADOW_HAND_LITE_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path="/home/ayush/icra/roto/roto/assets/shadow_lite/sr_hand_new.usd",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=True,
            retain_accelerations=True,
            max_depenetration_velocity=1000.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=0,
            sleep_threshold=0.005,
            stabilization_threshold=0.0005,
        ),
        collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.005, rest_offset=0.0),
        joint_drive_props=sim_utils.JointDrivePropertiesCfg(drive_type="force"),
        fixed_tendons_props=sim_utils.FixedTendonPropertiesCfg(limit_stiffness=30.0, damping=0.1),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.5),
        # rot=(0.0, 0.0, -0.7071, 0.7071),  # uncomment if rotation needed
        joint_pos={".*": 0.0},
    ),
    actuators={
        "fingers": ImplicitActuatorCfg(
            # FF is NOT actuated (joints exist but are not controlled)
            # Actuated: MF, RF (J1-J4) + Thumb (J1,J2,J4,J5)
            joint_names_expr=["rh_[MRF]FJ[1-4]", "rh_THJ[1245]"],
            effort_limit_sim={
                # Distal joints (fingertips)
                "rh_[MRF]FJ1": 0.7245,
                # Middle and proximal joints
                "rh_[MRF]FJ[23]": 0.9,
                # Knuckle abduction/adduction
                "rh_[MRF]FJ4": 0.9,
                # Thumb base rotation
                "rh_THJ5": 2.3722,
                # Thumb proximal
                "rh_THJ4": 1.45,
                # Thumb middle + distal
                "rh_THJ[12]": 0.99,
            },
            stiffness={
                "rh_[MRF]FJ[1-4]": 1.0,
                "rh_THJ[1245]": 1.0,
            },
            damping={
                "rh_[MRF]FJ[1-4]": 0.1,
                "rh_THJ[1245]": 0.1,
            },
        ),
    },
    soft_joint_pos_limit_factor=1.0,
)
"""Configuration of Shadow Hand Lite robot."""
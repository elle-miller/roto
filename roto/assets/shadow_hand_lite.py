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



_SHADOW_LITE_SPAWN = sim_utils.UsdFileCfg(
    usd_path=f"/home/ayush/Desktop/real_to_sim/roto/roto/assets/shadow_lite/shadow_touchlab_col.usd",
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
    # collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.005, rest_offset=0.0),
    joint_drive_props=sim_utils.JointDrivePropertiesCfg(drive_type="force"),
    #fixed_tendons_props=sim_utils.FixedTendonPropertiesCfg(limit_stiffness=30.0, damping=0.1),
)
"""Configuration of Shadow Hand robot."""
# A single spawn block is shared by both configs below (it never changes).
# _SHADOW_LITE_SPAWN = sim_utils.UrdfFileCfg(
#     asset_path=f"/home/ayush/Desktop/real_to_sim/roto/roto/assets/shadow_lite/sr_hand_touchlab.urdf",  # change path
#     usd_dir=f"/home/ayush/Desktop/real_to_sim/roto/roto/assets/shadow_lite/",
#     usd_file_name="sr_hand_touch_nomimic.usd",
#     scale=(1.0, 1.0, 1.0),

#     fix_base=True,
#     # TouchLab fingertip <collision> is the fingertip_v5_simple.stl mesh.
#     # convex_hull is the Isaac Lab default and exactly what the original PST
#     # fingertip used; the only concavity is the internal mount socket, which
#     # never contacts anything, so the hull of the sensing surface is faithful.
#     collider_type="convex_hull",
#     joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
#         gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(
#             stiffness=30.0,
#             damping=1.0,
#         ),
#     ),
#     visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.5, 0.5, 0.5)),
#     collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.005, rest_offset=0.0),
#     activate_contact_sensors=True,  # testing by making false; default: True
#     rigid_props=sim_utils.RigidBodyPropertiesCfg(
#         disable_gravity=True,
#         retain_accelerations=True,
#         max_depenetration_velocity=1000.0,
#     ),
#     articulation_props=sim_utils.ArticulationRootPropertiesCfg(
#         enabled_self_collisions=True,
#         solver_position_iteration_count=8,
#         solver_velocity_iteration_count=0,
#         sleep_threshold=0.005,
#         stabilization_threshold=0.0005,
#     ),
#     joint_drive_props=sim_utils.JointDrivePropertiesCfg(drive_type="force"),
#     #fixed_tendons_props=sim_utils.FixedTendonPropertiesCfg(limit_stiffness=30.0, damping=0.1),
# )

_SHADOW_LITE_INIT = ArticulationCfg.InitialStateCfg(
    pos=(0.0, 0.0, 0.5),
    rot=(0.0, 0.0, -0.7071, 0.7071),
    joint_pos={".*": 0.0},
)


# ════════════════════════════════════════════════════════════════════════════
# BASELINE config (original gains: effort 0.9 / 0.7245 N·m, stiffness 1.0).
# Commented out — uncomment this and comment the HIGH-EFFORT block below to revert.
# ════════════════════════════════════════════════════════════════════════════
# SHADOW_HAND_LITE_CFG = ArticulationCfg(
#     spawn=_SHADOW_LITE_SPAWN,
#     init_state=_SHADOW_LITE_INIT,
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
#             stiffness={"rh_[MRF]FJ[1-4]": 1.0, "rh_THJ[1245]": 1.0},
#             damping={"rh_[MRF]FJ[1-4]": 0.1, "rh_THJ[1245]": 0.1},
#         ),
#     },
#     soft_joint_pos_limit_factor=1.0,
# )

# _SHADOW_LITE_SPAWN = ArticulationCfg(
#     spawn=sim_utils.UsdFileCfg(
#         usd_path=f"/home/ayush/Desktop/real_to_sim/roto/roto/assets/shadow_lite/shadow_touchlab.usd",
#         activate_contact_sensors=True,
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
#         #fixed_tendons_props=sim_utils.FixedTendonPropertiesCfg(limit_stiffness=30.0, damping=0.1),
#     ),
#     init_state=ArticulationCfg.InitialStateCfg(
#         pos=(0.0, 0.0, 0.5),
#         rot=(0.0, 0.0, -0.7071, 0.7071),
#         joint_pos={".*": 0.0},
#     ),
#     actuators={
#         "fingers": ImplicitActuatorCfg(
#             # All 16 joints actuated: FF/MF/RF J1-J4 + thumb J1,J2,J4,J5
#             joint_names_expr=["rh_[MRF]FJ[1-4]", "rh_THJ[1245]"],
#             effort_limit_sim={
#                 # Finger flexion (incl. coupled J1/J2) — raised 0.7245/0.9 -> 2.5
#                 # so the joint can follow the steep coupled ramps without saturating.
#                 "rh_[MRF]FJ1": .9,
#                 "rh_[MRF]FJ[23]": .9,
#                 # Knuckle abduction/adduction — barely moves, left at baseline
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
#                 "rh_[MRF]FJ[12]": 0.1,
#                 "rh_[MRF]FJ[34]": 0.1,
#                 "rh_THJ[1245]": 0.1,
#             },
#             # velocity_limit_sim={
#             #     "rh_[MRF]FJ[1-4]": 4.0,
#             #     "rh_THJ[1245]": 4.0,
#             # },
#             # armature={
#             #     "rh_[MRF]FJ[1-4]": 0.005,
#             #     "rh_THJ[1245]": 0.005,
#             # },
#         ),
#     },
#     soft_joint_pos_limit_factor=1.0,
# )
# """Configuration of Shadow Hand robot."""
# ════════════════════════════════════════════════════════════════════════════
# ACTIVE: HIGH-EFFORT variant — effort raised to 2.5 N·m on the finger flexion
# joints (FFJ1/2/3, MF, RF) so the coupled joints aren't torque-starved on the
# 2× command ramps. Stiffness/damping unchanged; only the torque ceiling differs.
# ════════════════════════════════════════════════════════════════════════════
# SHADOW_HAND_LITE_CFG = ArticulationCfg(
#     spawn=_SHADOW_LITE_SPAWN,
#     init_state=_SHADOW_LITE_INIT,
#     actuators={
#         "fingers": ImplicitActuatorCfg(
#             # All 16 joints actuated: FF/MF/RF J1-J4 + thumb J1,J2,J4,J5
#             joint_names_expr=["rh_[MRF]FJ[1-4]", "rh_THJ[1245]"],
#             effort_limit_sim={
#                 # Finger flexion (incl. coupled J1/J2) — raised 0.7245/0.9 -> 2.5
#                 # so the joint can follow the steep coupled ramps without saturating.
#                 "rh_[MRF]FJ1": .9,
#                 "rh_[MRF]FJ[23]": .9,
#                 # Knuckle abduction/adduction — barely moves, left at baseline
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
#                 "rh_[MRF]FJ[12]": 0.1,
#                 "rh_[MRF]FJ[34]": 0.1,
#                 "rh_THJ[1245]": 0.1,
#             },
#             # velocity_limit_sim={
#             #     # Override the URDF's 2.0 rad/s cap (which was saturating the
#             #     # coupled join ts). Runtime — applied without USD reconversion.
#             #     "rh_[MRF]FJ[1-4]": 4.0,
#             #     "rh_THJ[1245]": 4.0,
#             # },
#             # armature={
#             #     # rotor inertia — smooths the response and reduces overshoot
#             #     # (matches the real motor's inertia); tune up if still ringing.
#             #     "rh_[MRF]FJ[1-4]": 0.005,
#             #     "rh_THJ[1245]": 0.005,
#             # },
#         ),
#     },
#     soft_joint_pos_limit_factor=1.0,
# )
# """Configuration of Shadow Hand robot (HIGH-EFFORT variant active)."""



# SHADOW_HAND_LITE_TOUCHLAB_CFG = ArticulationCfg(
#     spawn=sim_utils.UrdfFileCfg(
#         asset_path=f"/home/ayush/Desktop/gap/roto/roto/assets/shadow_lite/sr_hand_mimic_touchlab.urdf",
#         usd_dir=f"/home/ayush/Desktop/gap/roto/roto/assets/shadow_lite",
#         usd_file_name="sr_hand_mimic_touchlab.usd",
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
#         activate_contact_sensors=True,
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
#         joint_drive_props=sim_utils.JointDrivePropertiesCfg(drive_type="force"),
#         fixed_tendons_props=sim_utils.FixedTendonPropertiesCfg(limit_stiffness=30.0, damping=0.1),
#     ),
#     init_state=ArticulationCfg.InitialStateCfg(
#         pos=(0.0, 0.0, 0.5),
#         rot=(0.0, 0.0, -0.7071, 0.7071),
#         joint_pos={".*": 0.0},
#     ),
#     actuators={
#         "fingers": ImplicitActuatorCfg(
#             joint_names_expr=["rh_[MRF]FJ[2-4]", "rh_THJ[1245]"],
#             effort_limit_sim={
#                 "rh_[MRF]FJ[23]": 0.9,
#                 "rh_[MRF]FJ4": 0.9,
#                 "rh_THJ5": 2.3722,
#                 "rh_THJ4": 1.45,
#                 "rh_THJ[12]": 0.99,
#             },
#             stiffness={
#                 "rh_[MRF]FJ[2-4]": 1.0,
#                 "rh_THJ[1245]": 1.0,
#             },
#             damping={
#                 "rh_[MRF]FJ[2-4]": 0.1,
#                 "rh_THJ[1245]": 0.1,
#             },
#         ),
#     },
#     soft_joint_pos_limit_factor=1.0,
# )
# """Shadow Hand Lite with TouchLab v5 fingertips (box collision + 16 taxel contact links per finger).

# To monitor per-taxel contact forces in Isaac Lab, update ContactSensorCfg.prim_path to:
#     "/World/envs/env_.*/Robot/rh_(ff|mf|rf|th)_(A|B|C|D|E)[0-9]_taxel"
# This yields net_forces_w shape [N, 64, 3] (16 taxels x 4 fingers).
# Keep the default distal-link path for coarse per-finger sensing ([N, 4, 3]).
# """

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

SHADOW_HAND_LITE_CFG = ArticulationCfg(
    spawn=_SHADOW_LITE_SPAWN,
    init_state=_SHADOW_LITE_INIT,
    actuators={
        "fingers": ImplicitActuatorCfg(
            # All 16 joints actuated: FF/MF/RF J1-J4 + thumb J1,J2,J4,J5
            joint_names_expr=["rh_[MRF]FJ[1-4]", "rh_THJ[1245]"],
            # Uniform 30 N·m effort limit across all 16 actuated joints (per-joint-group
            # limits of ~0.9-2.4 N·m removed at user's request) -- applies to every joint
            # matched by joint_names_expr above since ImplicitActuatorCfg accepts a plain
            # float here, not just a dict.
            effort_limit_sim=30.0,
            stiffness={
                # Identified from real hardware via shadow_pd_id (sim-in-the-loop
                # Kp/Kd/Fc fit against real command/response logs, see
                # shadow_pd_id/results/params/*_gains.yaml + DECISIONS.md). Fit
                # only against training excitation (chirp/ramp/step) -- the
                # held-out-trajectory check (validate.py) was never run live, so
                # these are the best offline fit, not independently validated.
                "rh_FFJ4": 19.3834,
                "rh_MFJ4": 19.3834,
                "rh_RFJ4": 19.3834,
                "rh_THJ5": 2.1784,
                "rh_FFJ3": 17.7143,
                "rh_MFJ3": 17.7143,
                "rh_RFJ3": 17.7143,
                "rh_THJ4": 9.4651,
                "rh_FFJ2": 16.4757,
                "rh_MFJ2": 16.4757,
                "rh_RFJ2": 18.5329,
                "rh_THJ2": 2.1784,
                "rh_THJ1": 14.3495,
                # J1 mimic joints have no independent command, so shadow_pd_id never
                # excited/identified them directly -- per user decision, given the same
                # Kp as their driver J2 (same finger, same physical actuator/tendon
                # driving both) rather than the old generic placeholder (Kp=1.0).
                "rh_FFJ1": 16.4757,  # = rh_FFJ2
                "rh_MFJ1": 16.4757,  # = rh_MFJ2
                "rh_RFJ1": 18.5329,  # = rh_RFJ2
            },
            damping={
                "rh_FFJ4": 1.7063,
                "rh_MFJ4": 1.7063,
                "rh_RFJ4": 1.7063,
                "rh_THJ5": 0.0770,
                "rh_FFJ3": 1.1835,
                "rh_MFJ3": 1.1835,
                "rh_RFJ3": 1.1835,
                "rh_THJ4": 1.8569,
                "rh_FFJ2": 0.0094,
                "rh_MFJ2": 0.0094,
                "rh_RFJ2": 0.6777,
                "rh_THJ2": 0.0770,
                "rh_THJ1": 0.7221,
                "rh_FFJ1": 0.0094,  # = rh_FFJ2
                "rh_MFJ1": 0.0094,  # = rh_MFJ2
                "rh_RFJ1": 0.6777,  # = rh_RFJ2
            },
            # velocity_limit_sim={
            #     # Override the URDF's 2.0 rad/s cap (which was saturating the
            #     # coupled joints). Runtime — applied without USD reconversion.
            #     "rh_[MRF]FJ[1-4]": 4.0,
            #     "rh_THJ[1245]": 4.0,
            # },
            # armature={
            #     # rotor inertia — smooths the response and reduces overshoot
            #     # (matches the real motor's inertia); tune up if still ringing.
            #     "rh_[MRF]FJ[1-4]": 0.005,
            #     "rh_THJ[1245]": 0.005,
            # },
        ),
    },
    soft_joint_pos_limit_factor=1.0,
)
"""Configuration of Shadow Hand robot (HIGH-EFFORT variant active)."""
# """Configuration of Shadow Hand robot."""
# SHADOW_HAND_LITE_CFG = ArticulationCfg(
#     spawn=sim_utils.UsdFileCfg(
#         usd_path="/home/ayush/icra/roto/roto/assets/shadow_lite/sr_hand_new.usd",
#         activate_contact_sensors=True,
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
#         collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.005, rest_offset=0.0),
#         joint_drive_props=sim_utils.JointDrivePropertiesCfg(drive_type="force"),
#         fixed_tendons_props=sim_utils.FixedTendonPropertiesCfg(limit_stiffness=30.0, damping=0.1),
#     ),
#     init_state=ArticulationCfg.InitialStateCfg(
#         pos=(0.0, 0.0, 0.5),
#         # rot=(0.0, 0.0, -0.7071, 0.7071),  # uncomment if rotation needed
#         joint_pos={".*": 0.0},
#     ),
#     actuators={
#         "fingers": ImplicitActuatorCfg(
#             # FF is NOT actuated (joints exist but are not controlled)
#             # Actuated: MF, RF (J1-J4) + Thumb (J1,J2,J4,J5)
#             joint_names_expr=["rh_[MRF]FJ[1-4]", "rh_THJ[1245]"],
#             effort_limit_sim={
#                 # Distal joints (fingertips)
#                 "rh_[MRF]FJ1": 0.7245,
#                 # Middle and proximal joints
#                 "rh_[MRF]FJ[23]": 0.9,
#                 # Knuckle abduction/adduction
#                 "rh_[MRF]FJ4": 0.9,
#                 # Thumb base rotation
#                 "rh_THJ5": 2.3722,
#                 # Thumb proximal
#                 "rh_THJ4": 1.45,
#                 # Thumb middle + distal
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
# """Configuration of Shadow Hand Lite robot."""
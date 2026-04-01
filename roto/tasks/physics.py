# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.sim import PhysxCfg, SimulationCfg
from isaaclab.sim.schemas.schemas_cfg import CollisionPropertiesCfg

PHYSICS_DT = 1 / 240
DECIMATION = 4
SLEEP_THRESHOLD = 0.0
STABILIZATION_THRESHOLD = 0.002
MAX_DEPENETRATION_VELOCITY = 5.0


# High-precision contact for dexterous manipulation
contact_props = CollisionPropertiesCfg(
    collision_enabled=True,
    
    # 1. Eliminate the Air Gap
    # rest_offset = 0.0 ensures the ball actually touches the bed/fingers.
    # contact_offset = 0.002 (2mm) gives the solver enough "warning" to 
    # prepare for the collision without being too computationally expensive.
    contact_offset=0.002,
    rest_offset=0.0,
    
    # 2. Add Rotational Resistance (The "Grip")
    # Torsional friction prevents the ball from spinning like a frictionless bearing.
    # A 5mm patch radius is a realistic approximation for a finger-tip or ball compression.
    torsional_patch_radius=0.005, 
    min_torsional_patch_radius=0.001,
)

physx_optimized = PhysxCfg(
    # 1. Solver Type: TGS (1) is significantly more stable for articulations than PGS (0)
    solver_type=1,
    
    # 2. Iterations: Higher counts prevent fingers from 'sinking' into objects 
    # and improve the stability of grasps.
    min_position_iteration_count=8,
    max_position_iteration_count=255,
    min_velocity_iteration_count=1,
    max_velocity_iteration_count=255,
    
    # 3. Determinism & Stability
    enable_enhanced_determinism=True, # Critical if you are training RL policies
    
    # 4. Friction & Contact: Fine-tuning how friction is applied across surfaces.
    # friction_offset_threshold: Lowering this makes friction kick in sooner.
    # friction_correlation_distance: Merges contacts; 0.005 (5mm) is a good balance for hands.
    friction_offset_threshold=0.01,
    friction_correlation_distance=0.005,
    bounce_threshold_velocity=0.2, # Prevents tiny micro-bounces during sliding
    
    # 5. CCD (Continuous Collision Detection): 
    # Prevents fingers from "tunneling" through thin objects during fast movements.
    enable_ccd=True,

    # Refined GPU Buffer Management if needed
    # gpu_max_rigid_patch_count=2**19,       # Covers your ~233k requirement with 100% margin
    # gpu_max_rigid_contact_count=2**22,     # Increased to 4M to support high-contact density
    # gpu_heap_capacity=2**27,               # Bumped slightly to support the contact increase
    # gpu_total_aggregate_pairs_capacity=2**21, # Still massive, but more reasonable
    # gpu_found_lost_pairs_capacity=2**21,
)

robot_material = sim_utils.RigidBodyMaterialCfg(
    static_friction=1.0,
    dynamic_friction=1.0, 
    restitution=0.0      
)

roto_sim_cfg = SimulationCfg(
    dt=PHYSICS_DT,
    render_interval=DECIMATION,
    physics_material=robot_material,
    physx=physx_optimized
)


### BOUNCE PHYSICS ###
stress_ball_radius_m = 0.035
stress_ball_mass_g = 30
tennis_ball_radius_m = 0.033
tennis_ball_mass_g = 57
bounce_ball_mass_g = stress_ball_mass_g
bounce_ball_radius_m = stress_ball_radius_m
bounce_ball_mass_props = sim_utils.MassPropertiesCfg(mass=bounce_ball_mass_g/1000)

bounce_ball_rigid_props = sim_utils.RigidBodyPropertiesCfg(
    kinematic_enabled=False,
    disable_gravity=False,
    enable_gyroscopic_forces=True,
    solver_position_iteration_count=12, # Increased for stability
    solver_velocity_iteration_count=4,  # MUST be > 0 for bounce physics
    sleep_threshold=SLEEP_THRESHOLD,              # Lowered to prevent premature "falling asleep"
    stabilization_threshold=STABILIZATION_THRESHOLD,
    max_depenetration_velocity=MAX_DEPENETRATION_VELOCITY,
)

bounce_ball_material = sim_utils.RigidBodyMaterialCfg(
    static_friction=0.8, 
    dynamic_friction=0.8, 
    restitution=0.1
)


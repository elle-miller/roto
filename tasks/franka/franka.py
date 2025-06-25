# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Author: Elle Miller 2025

Shared Franka parent environment
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np
import torch
from collections.abc import Sequence

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg, RigidObject, RigidObjectCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg, ViewerCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import (
    ContactSensor,
    ContactSensorCfg,
    FrameTransformer,
    FrameTransformerCfg,
    OffsetCfg,
    TiledCamera,
    TiledCameraCfg,
)
from isaaclab.sim import PhysxCfg, SimulationCfg
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab.sim.spawners.materials.physics_materials_cfg import RigidBodyMaterialCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.math import (
    quat_conjugate,
    quat_from_angle_axis,
    quat_mul,
    sample_uniform,
    saturate,
)

from isaaclab.markers.config import FRAME_MARKER_CFG  # isort: skip

# from isaaclab_assets.robots.franka import FRANKA_PANDA_CFG  # isort: skip
from assets.franka import FRANKA_PANDA_CFG  # isort: skip


@configclass
class FrankaEnvCfg(DirectRLEnvCfg):
    # physics sim
    physics_dt = 1 / 120  # 0.002 #1 / 500 # 120 # 500 Hz

    # number of physics step per control step
    decimation = 2  # 10 # # 50 Hz

    # the number of physics simulation steps per rendering steps (default=1)
    render_interval = 2
    episode_length_s = 5.0  # 5 * 120 / 2 = 300 timesteps

    num_observations = 0
    num_actions = 9
    num_states = 0

    # isaac 4.5 stuff
    action_space = num_actions
    observation_space = num_observations
    state_space = num_states

    # configure this to get the right dimensions for fusion network
    obs_stack = 1

    # reset config
    reset_object_position_noise = 0.05
    default_object_pos = [0.5, 0, 0.055]

    # lift stuff
    minimal_height = 0.04
    reaching_object_scale = 1
    contact_reward_scale = 10
    lift_object_scale = 15.0
    object_goal_tracking_scale = 16.0
    joint_vel_penalty_scale = 0  # -0.01
    object_out_of_bounds = 1.5

    # reach stuff
    min_reach_dist = 0.05

    # simulation
    sim: SimulationCfg = SimulationCfg(
        dt=physics_dt,
        render_interval=decimation,
        physics_material=RigidBodyMaterialCfg(
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        physx=PhysxCfg(
            bounce_threshold_velocity=0.2,
            # gpu_max_rigid_contact_count=2**25, # default 2**23
            # gpu_max_rigid_patch_count=2**25, #23, default 5 * 2 ** 15.
            # gpu_max_soft_body_contacts=2**24, # default 2**20
            # gpu_collision_stack_size=2**26, # default 2**26
        ),
    )

    # temp
    replicate_physics = True
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=4096, env_spacing=1.5, replicate_physics=replicate_physics
    )

    default_object_pos = [0.5, 0, 0.03]
    eye = (3, 3, 3)
    lookat = (0, 0, 0)

    viewer: ViewerCfg = ViewerCfg(eye=eye, lookat=lookat, resolution=(1920, 1080))

    # robot
    robot_cfg: ArticulationCfg = FRANKA_PANDA_CFG.replace(prim_path="/World/envs/env_.*/Robot")

    # panda finger bodies are called : 'panda_leftfinger', 'panda_rightfinger``
    # We set the update period to 0 to update the sensor at the same frequency as the simulation
    # contact sensors are called 'left_contact_sensor' and 'right_contact_sensor'
    marker_cfg = FRAME_MARKER_CFG.copy()
    marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
    marker_cfg.prim_path = "/Visuals/ContactCfg"
    left_contact_cfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/left_contact_sensor",
        update_period=0.0,
        history_length=1,
        debug_vis=False,
        visualizer_cfg=marker_cfg,
        filter_prim_paths_expr=["/World/envs/env_.*/Object"],
    )
    right_contact_cfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/right_contact_sensor",
        update_period=0.0,
        history_length=1,
        debug_vis=False,
        visualizer_cfg=marker_cfg,
        filter_prim_paths_expr=["/World/envs/env_.*/Object"],
    )

    wholebody_contact_cfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/panda_.*",
        update_period=0.0,
        history_length=1,
        debug_vis=False,
        visualizer_cfg=marker_cfg,
        # filter_prim_paths_expr=["/World/envs/env_.*/Object"],
    )

    # Normalisation numbers
    tactile_min_val = 0
    tactile_max_val = 20.0
    vel_max_magnitude = 3

    actuated_joint_names = [
        "panda_joint1",
        "panda_joint2",
        "panda_joint3",
        "panda_joint4",
        "panda_joint5",
        "panda_joint6",
        "panda_joint7",
        "panda_finger_joint1",
        "panda_finger_joint2",
    ]

    object_cfg: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Object",
        init_state=RigidObjectCfg.InitialStateCfg(pos=default_object_pos, rot=[1, 0, 0, 0]),
        spawn=UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/DexCube/dex_cube_instanceable.usd",
            scale=(0.8, 0.8, 0.8),
            rigid_props=RigidBodyPropertiesCfg(
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=1,
                max_angular_velocity=1000.0,
                max_linear_velocity=1000.0,
                max_depenetration_velocity=5.0,
                disable_gravity=False,
            ),
            # mass_props=sim_utils.MassPropertiesCfg(mass=1000.0)
        ),
    )

    # Listens to the required transforms
    marker_cfg = FRAME_MARKER_CFG.copy()
    marker_cfg.markers["frame"].scale = (0.01, 0.01, 0.01)
    marker_cfg.prim_path = "/Visuals/EndEffectorFrameTransformer"
    ee_config: FrameTransformerCfg = FrameTransformerCfg(
        # source frame
        prim_path="/World/envs/env_.*/Robot/panda_link0",
        debug_vis=False,
        # visualizer_cfg=marker_cfg,
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path="/World/envs/env_.*/Robot/panda_hand",
                name="end_effector",
                offset=OffsetCfg(
                    pos=[0.0, 0.0, 0.1034],
                ),
            ),
        ],
    )

    # contact sensors - ONLY RECORD FORCES FROM THE OBJECT FOR NOW
    left_sensor_cfg: FrameTransformerCfg = FrameTransformerCfg(
        prim_path="/World/envs/env_.*/Robot/panda_link0",
        debug_vis=False,
        visualizer_cfg=marker_cfg,
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path="/World/envs/env_.*/Robot/left_contact_sensor",
                name="left_sensor",
                offset=OffsetCfg(
                    pos=[0.0, 0.0, 0.0],
                ),
            ),
        ],
    )
    right_sensor_cfg: FrameTransformerCfg = FrameTransformerCfg(
        prim_path="/World/envs/env_.*/Robot/panda_link0",
        debug_vis=False,
        visualizer_cfg=marker_cfg,
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path="/World/envs/env_.*/Robot/right_contact_sensor",
                name="right_sensor",
                offset=OffsetCfg(
                    pos=[0.0, 0.0, 0.0],
                ),
            ),
        ],
    )
    img_dim = 84
    eye = [1.2, -0.3, 0.5]
    target = [0, 0, 0]
    tiled_camera: TiledCameraCfg = TiledCameraCfg(
        prim_path="/World/envs/env_.*/Camera",
        offset=TiledCameraCfg.OffsetCfg(pos=(0.0, -0.0, 0.0), rot=(1, 0, 0, 0), convention="world"),
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.01, 3.2)
        ),
        width=img_dim,
        height=img_dim,
        debug_vis=True,
    )

    # defaults to be overwritten
    write_image_to_file = False
    obs_list = ["prop", "gt"]
    aux_list = []
    normalise_prop = True
    normalise_pixels = True
    num_cameras = 1
    object_type = "rigid"


class FrankaEnv(DirectRLEnv):
    # pre-physics step calls
    #   |-- _pre_physics_step(action)
    #   |-- _apply_action()
    # post-physics step calls
    #   |-- _get_dones()
    #   |-- _get_rewards()
    #   |-- _reset_idx(env_ids)
    #   |-- _get_observations()

    cfg: FrankaEnvCfg

    def __init__(self, cfg: FrankaEnvCfg, render_mode: str | None = None, **kwargs):

        self.obs_stack = cfg.obs_stack
        super().__init__(cfg, render_mode, **kwargs)

        self.dtype = torch.float32
        self.binary_tactile = cfg.binary_tactile

        # create auxiliary variables for computing applied action, observations and rewards
        self.robot_dof_lower_limits = self.robot.data.soft_joint_pos_limits[0, :, 0].to(device=self.device)
        self.robot_dof_upper_limits = self.robot.data.soft_joint_pos_limits[0, :, 1].to(device=self.device)
        self.robot_dof_targets = torch.zeros((self.num_envs, self.robot.num_joints), device=self.device)

        # list of actuated joints
        self.actuated_dof_indices = list()
        for joint_name in cfg.actuated_joint_names:
            self.actuated_dof_indices.append(self.robot.joint_names.index(joint_name))
        self.actuated_dof_indices.sort()

        # create empty tensors
        self.actions = torch.zeros((self.num_envs, 9), device=self.device)

        self.joint_pos = torch.zeros((self.num_envs, 9), device=self.device)
        self.joint_vel = torch.zeros((self.num_envs, 9), device=self.device)
        self.normalised_joint_pos = torch.zeros((self.num_envs, 9), device=self.device)
        self.normalised_joint_vel = torch.zeros((self.num_envs, 9), device=self.device)
        self.aperture = torch.zeros((self.num_envs,), device=self.device)
        self.object_pos = torch.zeros((self.num_envs, 3), device=self.device)
        self.normalised_forces = torch.zeros((self.num_envs, 2), device=self.device)
        self.unnormalised_forces = torch.zeros((self.num_envs, 2), device=self.device)
        self.in_contact = torch.zeros((self.num_envs, 1), device=self.device)
        self.tactile = torch.zeros((self.num_envs, 2), device=self.device)

        self.object_rot = torch.zeros((self.num_envs, 4), device=self.device)
        self.ee_pos = torch.zeros((self.num_envs, 3), device=self.device)
        self.ee_rot = torch.zeros((self.num_envs, 4), device=self.device)

        self.object_ee_distance = torch.zeros((self.num_envs, 3), device=self.device)
        self.object_ee_rotation = torch.zeros((self.num_envs, 4), device=self.device)
        self.object_ee_angular_distance = torch.zeros((self.num_envs,), device=self.device)
        self.object_ee_euclidean_distance = torch.zeros((self.num_envs,), device=self.device)

        # save reward weights so they can be adjusted online
        self.reaching_object_scale = cfg.reaching_object_scale
        self.contact_reward_scale = cfg.contact_reward_scale
        self.lift_object_scale = cfg.lift_object_scale
        self.joint_vel_penalty_scale = cfg.joint_vel_penalty_scale
        self.object_goal_tracking_scale = cfg.object_goal_tracking_scale

        # unit tensors
        self.x_unit_tensor = torch.tensor([1, 0, 0], dtype=torch.float, device=self.device).repeat((self.num_envs, 1))
        self.y_unit_tensor = torch.tensor([0, 1, 0], dtype=torch.float, device=self.device).repeat((self.num_envs, 1))
        self.z_unit_tensor = torch.tensor([0, 0, 1], dtype=torch.float, device=self.device).repeat((self.num_envs, 1))

        # camera stuff
        self.count = 0

        self.extras["log"] = {
            "reach_reward": None,
            "lift_reward": None,
            "dist_reward": None,
            "contact_reward": None,
            "joint_vel_penalty": None,
            "object_ee_distance": None,
            "object_goal_tracking": None,
            "object_goal_tracking_finegrained": None,
            "tactile": None,
            "unnormalised_forces_left_x": None,
            "unnormalised_forces_right_x": None,
            "normalised_forces_left_x": None,
            "normalised_forces_right_x": None,
        }

        self.extras["counters"] = {
            "timesteps_to_find_object_easy": None,
            "timesteps_to_find_object_med": None,
            "timesteps_to_find_object_hard": None,
            "object_found_easy": None,
            "object_found_med": None,
            "object_found_hard": None,
        }

    def set_spaces(self, single_obs, obs, single_action, action):
        self.single_observation_space = single_obs
        self.observation_space = obs
        self.single_action_space = single_action
        self.action_space = action

    def _add_object_to_scene(self):
        # if using a different object, setup in respective env
        if self.cfg.object_type == "rigid":
            print("SETTING UP RIGID OBJECT", self.cfg.object_cfg)
            self.object = RigidObject(self.cfg.object_cfg)
            self.scene.rigid_objects["object"] = self.object

    def _setup_scene(self):
        self.robot = Articulation(self.cfg.robot_cfg)

        self._add_object_to_scene()

        # FrameTransformer provides interface for reporting the transform of
        # one or more frames (target frames) wrt to another frame (source frame)
        self.ee_frame = FrameTransformer(self.cfg.ee_config)
        self.ee_frame.set_debug_vis(False)
        self.left_sensor_frame = FrameTransformer(self.cfg.left_sensor_cfg)
        self.right_sensor_frame = FrameTransformer(self.cfg.right_sensor_cfg)

        # add ground plane
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg(size=(10000, 10000)))

        # clone and replicate (no need to filter for this environment)
        self.scene.clone_environments(copy_from_source=False)

        # register to scene
        self.scene.articulations["robot"] = self.robot
        self.scene.sensors["ee_frame"] = self.ee_frame
        self.scene.sensors["left_sensor_frame"] = self.left_sensor_frame
        self.scene.sensors["right_sensor_frame"] = self.right_sensor_frame

        yellow = (1.0, 0.96, 0.0)
        orange = (1.0, 0.5, 0.0)
        light_cfg = sim_utils.DomeLightCfg(intensity=1000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)
        light_cfg_1 = sim_utils.SphereLightCfg(intensity=10000.0, color=yellow)
        light_cfg_1.func("/World/ds", light_cfg_1, translation=(1, 0, 1))
        light_cfg_2 = sim_utils.SphereLightCfg(intensity=10000.0, color=orange)
        light_cfg_2.func("/World/disk", light_cfg_2, translation=(-1, 0, 1))

        if "pixels" in self.cfg.obs_list or "pixels" in self.cfg.aux_list:
            print("This won't work at the moment - talk to Elle")
            from omni.isaac.core.utils.extensions import enable_extension

            enable_extension("omni.replicator.core")
            import omni.replicator.core as rep

            rep.settings.set_render_rtx_realtime(antialiasing="DLAA")
            self._tiled_camera = TiledCamera(self.cfg.tiled_camera)
            self.scene.sensors["tiled_camera"] = self._tiled_camera

        if "tactile" in self.cfg.obs_list:
            self.left_contact_sensor = ContactSensor(self.cfg.left_contact_cfg)
            self.scene.sensors["left_contact_sensor"] = self.left_contact_sensor

            self.right_contact_sensor = ContactSensor(self.cfg.right_contact_cfg)
            self.scene.sensors["right_contact_sensor"] = self.right_contact_sensor

            self.wholebody_contact_sensor = ContactSensor(self.cfg.wholebody_contact_cfg)
            self.scene.sensors["wholebody_contact_sensor"] = self.wholebody_contact_sensor

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        """
        Store actions from policy in a class variable
        """
        self.last_action = self.robot_dof_targets[:, self.actuated_dof_indices]
        self.actions = actions.clone()

    def _apply_action(self) -> None:
        """
        The _apply_action(self) API is called decimation number of times for each RL step, prior to taking each physics step.
        This provides more flexibility for environments where actions should be applied for each physics step.
        """
        scaled_actions = self.scale_action(self.actions)

        self.robot.set_joint_position_target(scaled_actions, joint_ids=self.actuated_dof_indices)

    def scale_action(self, action):
        self.robot_dof_targets[:, self.actuated_dof_indices] = scale(
            action,
            self.robot_dof_lower_limits[self.actuated_dof_indices],
            self.robot_dof_upper_limits[self.actuated_dof_indices],
        )

        self.robot_dof_targets[:, self.actuated_dof_indices] = saturate(
            self.robot_dof_targets[:, self.actuated_dof_indices],
            self.robot_dof_lower_limits[self.actuated_dof_indices],
            self.robot_dof_upper_limits[self.actuated_dof_indices],
        )
        return self.robot_dof_targets[:, self.actuated_dof_indices]

    def get_observations(self):
        # public method
        return self._get_observations()

    def _get_observations(self) -> dict:

        obs_dict = {}
        for k in self.cfg.obs_list:
            if k == "prop":
                obs_dict[k] = self._get_proprioception()
            elif k == "pixels":
                obs_dict[k] = self._get_images()
            elif k == "gt":
                obs_dict[k] = self._get_gt()
            elif k == "tactile":
                obs_dict[k] = self._get_tactile()
            else:
                print("Unknown observations type!")

        obs_dict = {"policy": obs_dict}

        return obs_dict

    def _get_proprioception(self):
        prop = torch.cat(
            (
                self.normalised_joint_pos,
                self.normalised_joint_vel,
                self.aperture.unsqueeze(1),
                self.ee_pos,
                self.ee_rot,
                self.actions,
            ),
            dim=-1,
        )

        return prop

    def _get_gt(self):

        gt = torch.cat(
            (
                # xyz diffs (3,)
                self.object_ee_distance,
                # rotation quaternion (4,)
                self.object_ee_rotation,
                # rotation difference (1,)
                self.object_ee_angular_distance.unsqueeze(1),
                # euclidean distances (1,) [transform from (num_envs,) to (num_envs,1)]
                self.object_ee_euclidean_distance.unsqueeze(1),
            ),
            dim=-1,
        )
        return gt

    def _read_force_matrix(self, filter=False):
        # separate into left and right for frame transform force_matrix_w net_forces_w
        if filter:
            forcesL_world = self.left_contact_sensor.data.force_matrix_w[:].clone().reshape(self.num_envs, 3)
            forcesR_world = self.right_contact_sensor.data.force_matrix_w[:].clone().reshape(self.num_envs, 3)
        else:
            forcesL_world = self.left_contact_sensor.data.net_forces_w[:].clone().reshape(self.num_envs, 3)
            forcesR_world = self.right_contact_sensor.data.net_forces_w[:].clone().reshape(self.num_envs, 3)

        return forcesL_world, forcesR_world

    def _normalise_forces(self, forcesL, forcesR):
        # only return the normal component
        return_forces = torch.abs(torch.cat((forcesL, forcesR), dim=1))

        # Clip the tensor values and normalise 0 to 1
        self.unnormalised_forces = torch.clamp(
            return_forces, min=self.cfg.tactile_min_val, max=self.cfg.tactile_max_val
        )
        self.normalised_forces = (self.unnormalised_forces - self.cfg.tactile_min_val) / (
            self.cfg.tactile_max_val - self.cfg.tactile_min_val
        )

        return self.normalised_forces

    def _get_tactile(self):
        # contact sensor data is [num_envs, 2, 3]
        forcesL_world, forcesR_world = self._read_force_matrix()

        # absolute value the whole thing, and sum it
        forcesL_net = torch.linalg.vector_norm(forcesL_world, dim=1, keepdim=True)
        forcesR_net = torch.linalg.vector_norm(forcesR_world, dim=1, keepdim=True)

        if self.binary_tactile:
            if self.dtype == torch.float16:
                forcesL_norm = (forcesL_net > 0).half()
                forcesR_norm = (forcesR_net > 0).half()
            else:
                forcesL_norm = (forcesL_net > 0).float()
                forcesR_norm = (forcesR_net > 0).float()

            tactile = torch.cat(
                (
                    forcesL_norm,
                    forcesR_norm,
                ),
                dim=-1,
            )
            self.tactile = tactile
            return tactile
        else:

            self.normalised_forces = self._normalise_forces(forcesL_net, forcesR_net)
            self.tactile = self.normalised_forces
            return self.normalised_forces

    def _get_images(self):

        camera_data = self._tiled_camera.data.output["rgb"].clone()

        # normalize the camera data for better training results
        # convert to float 32 to subtract mean, then back to uint8 for memory storage
        if self.cfg.normalise_pixels:
            camera_data = camera_data / 255.0
            mean_tensor = torch.mean(camera_data, dim=(1, 2), keepdim=True)
            camera_data -= mean_tensor
            camera_data *= 255
            camera_data = camera_data.to(torch.uint8)

        return camera_data

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        super()._reset_idx(env_ids)

        # reset goals
        self._reset_target_pose(env_ids)

        # reset object
        if self.cfg.object_type == "rigid":
            self._reset_object_pose(env_ids)

        # reset robot
        self._reset_robot(env_ids)

        # refresh intermediate values for _get_observations()
        self._compute_intermediate_values(reset=True, env_ids=env_ids)

    def _reset_object_pose(self, env_ids):
        object_default_state = self.object.data.default_root_state.clone()[env_ids]
        pos_noise = sample_uniform(-1.0, 1.0, (len(env_ids), 3), device=self.device)

        # global object positions (for writing to sim)
        object_default_state[:, 0:3] = (
            object_default_state[:, 0:3]
            + self.cfg.reset_object_position_noise * pos_noise
            + self.scene.env_origins[env_ids]
        )
        rot_noise = sample_uniform(-1.0, 1.0, (len(env_ids), 2), device=self.device)  # noise for X and Y rotation
        object_default_state[:, 3:7] = randomize_rotation(
            rot_noise[:, 0], rot_noise[:, 1], self.x_unit_tensor[env_ids], self.y_unit_tensor[env_ids]
        )
        object_default_state[:, 7:] = torch.zeros_like(self.object.data.default_root_state[env_ids, 7:])
        self.object.write_root_state_to_sim(object_default_state, env_ids)

    def _reset_robot(self, env_ids):
        joint_pos = self.robot.data.default_joint_pos[env_ids] + sample_uniform(
            -0.125,
            0.125,
            (len(env_ids), self.robot.num_joints),
            self.device,
        )
        joint_pos = torch.clamp(joint_pos, self.robot_dof_lower_limits, self.robot_dof_upper_limits)
        joint_vel = torch.zeros_like(joint_pos)
        self.robot.set_joint_position_target(joint_pos, env_ids=env_ids)
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)

    def _reset_target_pose(self, env_ids):
        pass

    def _compute_intermediate_values(self, env_ids: torch.Tensor | None = None):
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES

        # get robot data
        self.joint_pos[env_ids] = self.robot.data.joint_pos[env_ids]
        self.joint_vel[env_ids] = self.robot.data.joint_vel[env_ids]
        self.ee_pos[env_ids] = self.ee_frame.data.target_pos_source[..., 0, :][env_ids]
        self.ee_rot[env_ids] = self.ee_frame.data.target_quat_source[..., 0, :][env_ids]

        # aperture between 0-0.08, lets scale to 0-1
        max_aperture = 0.08
        self.aperture = (self.joint_pos[:, 7] + self.joint_pos[:, 8]) / max_aperture

        # normalise joint pos
        self.normalised_joint_pos[env_ids] = unscale(
            self.joint_pos[env_ids], self.robot_dof_lower_limits, self.robot_dof_upper_limits
        )

        # joint vel roughly between -2.5, 2.5, so dividing by 3.
        self.normalised_joint_vel[env_ids] = self.joint_vel[env_ids] / self.cfg.vel_max_magnitude

        # object
        self.object_pos[env_ids] = self.object.data.root_pos_w[env_ids] - self.scene.env_origins[env_ids]

        # deformable doesn't have quat
        if self.cfg.object_type == "rigid":
            self.object_rot[env_ids] = self.object.data.root_quat_w[env_ids]

        # relative distances
        self.object_ee_distance[env_ids] = self.object_pos[env_ids] - self.ee_pos[env_ids]
        self.object_ee_euclidean_distance[env_ids] = torch.norm(self.object_ee_distance[env_ids], dim=1)
        self.object_ee_rotation[env_ids] = quat_mul(self.object_rot[env_ids], quat_conjugate(self.ee_rot[env_ids]))
        self.object_ee_angular_distance[env_ids] = rotation_distance(self.object_rot[env_ids], self.ee_rot[env_ids])

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        self._compute_intermediate_values()

        # no termination at the moment
        out_of_reach = torch.norm(self.object_pos, dim=1) >= self.cfg.object_out_of_bounds
        termination = out_of_reach

        time_out = self.episode_length_buf >= self.max_episode_length - 1

        return termination, time_out


# scales an input between lower and upper
@torch.jit.script
def scale(x, lower, upper):
    return 0.5 * (x + 1.0) * (upper - lower) + lower


# scales an input between 1 and -1
@torch.jit.script
def unscale(x, lower, upper):
    return (2.0 * x - upper - lower) / (upper - lower)


@torch.jit.script
def randomize_rotation(rand0, rand1, x_unit_tensor, y_unit_tensor):
    return quat_mul(
        quat_from_angle_axis(rand0 * np.pi, x_unit_tensor), quat_from_angle_axis(rand1 * np.pi, y_unit_tensor)
    )


@torch.jit.script
def rotation_distance(object_rot, target_rot):
    # Orientation alignment for the cube in hand and goal cube
    quat_diff = quat_mul(object_rot, quat_conjugate(target_rot))
    return 2.0 * torch.asin(torch.clamp(torch.norm(quat_diff[:, 1:4], p=2, dim=-1), max=1.0))  # changed quat convention


### reward functions


@torch.jit.script
def distance_reward(object_ee_distance, std: float = 0.1):
    r_reach = 1 - torch.tanh(object_ee_distance / std)
    return r_reach


@torch.jit.script
def lift_reward(object_pos, minimal_height: float, episode_timestep_counter):
    # reward for lifting object
    object_height = object_pos[:, 2]
    is_lifted = torch.where(object_height > minimal_height, 1.0, 0.0)
    is_lifted *= (episode_timestep_counter > 15).float()
    return is_lifted


@torch.jit.script
def object_goal_reward(object_goal_distance, r_lift, std: float = 0.1):
    # tracking
    std = 0.3
    object_goal_tracking = 1 - torch.tanh(object_goal_distance / std)
    # only recieve reward if object is lifted
    object_goal_tracking *= (r_lift > 0).float()
    return object_goal_tracking


@torch.jit.script
def joint_vel_penalty(robot_joint_vel):
    r_joint_vel = torch.sum(torch.square(robot_joint_vel), dim=1)
    return r_joint_vel

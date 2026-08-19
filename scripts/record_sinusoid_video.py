"""Play a simultaneous multi-joint sinusoidal excitation of the Shadow Hand
Lite (TouchLab variant) in IsaacLab sim and record it to a short video + a
handful of high-res screenshots.

All 13 policy joints move at once, each on its own sinusoid (own frequency
and phase offset, and its own amplitude since each joint's physical range
differs) so the motion looks organic rather than robotic/synchronized.

Purely for producing a visual asset (e.g. a dissertation figure image with the
background removed) — it does not save any .npz trajectory data. For the
actual sim-vs-hardware quantitative comparison, see collect_traj_sim.py.

Usage (from roto/scripts/):
    # Default (elevated, slightly-front) camera:
    python record_sinusoid_video.py --headless --enable_cameras \
        --video_path ../videos/sim_multi_default.mp4 \
        --screenshot_dir ../videos/sim_multi_default_screenshots

    # Level, front-on camera:
    python record_sinusoid_video.py --headless --enable_cameras \
        --cam_eye 0 -0.9 0.45 --cam_target 0 -0.05 0.45 \
        --video_path ../videos/sim_multi_front.mp4 \
        --screenshot_dir ../videos/sim_multi_front_screenshots
"""

import argparse
import importlib.util
import math
import os

import numpy as np
from isaaclab.app import AppLauncher

# The `roto` package name may resolve to a stale editable install elsewhere
# on this machine (an older checkout with the BioTac/FSR hand instead of the
# TouchLab hand). Load the current project's modules directly by path so the
# correct assets/tasks are always used regardless of site-packages state.
ROTO_ROOT = "/home/ayush/icra/roto/roto"


def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


parser = argparse.ArgumentParser(description="Record a simultaneous multi-joint sinusoid in sim.")
parser.add_argument("--video_path", type=str, default="../videos/sim_multi.mp4")
parser.add_argument("--screenshot_dir", type=str, default="../videos/sim_multi_screenshots")
parser.add_argument("--num_screenshots", type=int, default=6, help="Evenly-spaced full-res stills to save.")
parser.add_argument("--video_fps", type=float, default=30.0)
parser.add_argument("--resolution", type=int, nargs=2, default=(1920, 1080), metavar=("W", "H"))
parser.add_argument("--cam_eye", type=float, nargs=3, default=(0, -0.6, 1.0))
parser.add_argument("--cam_target", type=float, nargs=3, default=(0, -0.3, 0.5))
parser.add_argument("--duration", type=float, default=8.0, help="Total motion duration in seconds (5-10s typical).")
parser.add_argument("--base_freq", type=float, default=0.6, help="Base sinusoid frequency (Hz); each joint is offset from this.")
parser.add_argument("--amplitude_frac", type=float, default=0.8, help="Amplitude as fraction of each joint's half-range.")
parser.add_argument("--settle_secs", type=float, default=0.3, help="Zero-hold seconds before the motion starts.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch
import imageio
import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
import omni.replicator.core as rep

shadow_hand_lite = _load_module("roto_assets_shadow_hand_lite", os.path.join(ROTO_ROOT, "assets", "shadow_hand_lite.py"))
physics = _load_module("roto_tasks_physics", os.path.join(ROTO_ROOT, "tasks", "physics.py"))
SHADOW_HAND_LITE_CFG = shadow_hand_lite.SHADOW_HAND_LITE_CFG
PHYSICS_DT = physics.PHYSICS_DT
roto_sim_cfg = physics.roto_sim_cfg

# Must match collect_traj_sim.py / collect_traj_hw.py exactly.
POLICY_JOINT_ORDER = [
    "rh_FFJ4", "rh_MFJ4", "rh_RFJ4", "rh_THJ5",
    "rh_FFJ3", "rh_MFJ3", "rh_RFJ3", "rh_THJ4",
    "rh_FFJ2", "rh_MFJ2", "rh_RFJ2", "rh_THJ2", "rh_THJ1",
]
LOWER_LIMITS = np.array([-0.3491, -0.3491, -0.3491, -1.0472, -0.2618, -0.2618, -0.2618,
                          0.0, 0.0, 0.0, 0.0, -0.6981, -0.2618], dtype=np.float32)
UPPER_LIMITS = np.array([0.3491, 0.3491, 0.3491, 1.0472, 1.5708, 1.5708, 1.5708,
                          1.2217, 1.5708, 1.5708, 1.5708, 0.6981, 1.5708], dtype=np.float32)
COUPLED_J1_NAMES = {8: "rh_FFJ1", 9: "rh_MFJ1", 10: "rh_RFJ1"}
COUPLING_THETA = 0.785

N_JOINTS = len(POLICY_JOINT_ORDER)
# Each joint gets its own frequency (small deterministic spread around base_freq)
# and phase offset, so the 13 joints ripple rather than pulse in lockstep.
FREQ_SCALE = 1.0 + 0.4 * (np.arange(N_JOINTS) / (N_JOINTS - 1) - 0.5)   # 0.8x - 1.2x base_freq
PHASE_OFFSET = 2.0 * math.pi * np.arange(N_JOINTS) / N_JOINTS           # spread over one full cycle


def _coupling(proxy, upper):
    j2 = float(np.clip(proxy * (upper / COUPLING_THETA), 0.0, upper))
    j1 = float(np.clip((proxy - COUPLING_THETA) / (upper - COUPLING_THETA) * upper, 0.0, upper))
    return j2, j1


def main():
    video_path = os.path.abspath(args_cli.video_path)
    screenshot_dir = os.path.abspath(args_cli.screenshot_dir)
    os.makedirs(os.path.dirname(video_path), exist_ok=True)
    os.makedirs(screenshot_dir, exist_ok=True)

    sim = sim_utils.SimulationContext(roto_sim_cfg)
    sim.set_camera_view(eye=tuple(args_cli.cam_eye), target=tuple(args_cli.cam_target))

    sim_utils.GroundPlaneCfg().func("/World/ground", sim_utils.GroundPlaneCfg())
    # Same HDR dome light the actual Baoding/Bounce training scenes use
    # (roto/tasks/baoding/baoding.py::_setup_scene) — a flat white DomeLightCfg
    # gives a lifeless grey render; this is what makes train.py's videos look right.
    hdr_path = os.path.join(ROTO_ROOT, "assets", "rooms", "qwantani_dusk_2_4k.hdr")
    light_cfg = sim_utils.DomeLightCfg(
        color=(0.81, 0.86, 1.28),
        intensity=1000.0,
        texture_file=hdr_path,
        texture_format="latlong",
    )
    light_cfg.func("/World/bglight", light_cfg)

    robot_cfg = SHADOW_HAND_LITE_CFG.replace(prim_path="/World/Robot")
    robot = Articulation(robot_cfg)

    sim.reset()

    width, height = args_cli.resolution
    render_product = rep.create.render_product("/OmniverseKit_Persp", (width, height))
    rgb_annotator = rep.AnnotatorRegistry.get_annotator("rgb", device="cpu")
    rgb_annotator.attach([render_product])

    def grab_frame():
        data = rgb_annotator.get_data()
        return np.frombuffer(data, dtype=np.uint8).reshape(*data.shape)[:, :, :3]

    joint_names = list(robot.data.joint_names)
    n_robot_joints = len(joint_names)
    name_to_ridx = {n: i for i, n in enumerate(joint_names)}
    policy_to_ridx = [name_to_ridx[pj] for pj in POLICY_JOINT_ORDER]
    coupled_j1_ridx = {pi: name_to_ridx[jn] for pi, jn in COUPLED_J1_NAMES.items()}

    robot.write_root_pose_to_sim(robot.data.default_root_state[:, :7])

    center = (UPPER_LIMITS + LOWER_LIMITS) / 2.0
    amp = args_cli.amplitude_frac * (UPPER_LIMITS - LOWER_LIMITS) / 2.0

    zero_target = torch.zeros(1, n_robot_joints, device="cpu")
    video_writer = imageio.get_writer(video_path, fps=args_cli.video_fps, macro_block_size=1)
    steps_per_video_frame = max(1, round((1.0 / args_cli.video_fps) / PHYSICS_DT))

    # Warm-up so the renderer isn't returning stale/empty frames.
    for _ in range(10):
        robot.set_joint_position_target(zero_target)
        robot.write_data_to_sim()
        sim.step()
        robot.update(PHYSICS_DT)
    grab_frame()

    frames = []
    step_counter = 0

    settle_steps = int(args_cli.settle_secs / PHYSICS_DT)
    for _ in range(settle_steps):
        robot.set_joint_position_target(zero_target)
        robot.write_data_to_sim()
        should_render = (step_counter % steps_per_video_frame == 0)
        sim.step(render=should_render)
        robot.update(PHYSICS_DT)
        if should_render:
            frame = grab_frame()
            video_writer.append_data(frame)
            frames.append(frame)
        step_counter += 1

    motion_steps = int(args_cli.duration / PHYSICS_DT)
    for step in range(motion_steps):
        t = step * PHYSICS_DT
        proxy = center + amp * np.sin(2.0 * math.pi * args_cli.base_freq * FREQ_SCALE * t + PHASE_OFFSET)
        proxy = np.clip(proxy, LOWER_LIMITS, UPPER_LIMITS).astype(np.float32)

        target = torch.zeros(1, n_robot_joints, device="cpu")
        for pi in range(N_JOINTS):
            if pi in COUPLED_J1_NAMES:
                j2_tgt, j1_tgt = _coupling(float(proxy[pi]), float(UPPER_LIMITS[pi]))
                target[0, policy_to_ridx[pi]] = j2_tgt
                target[0, coupled_j1_ridx[pi]] = j1_tgt
            else:
                target[0, policy_to_ridx[pi]] = float(proxy[pi])

        robot.set_joint_position_target(target)
        robot.write_data_to_sim()
        should_render = (step_counter % steps_per_video_frame == 0)
        sim.step(render=should_render)
        robot.update(PHYSICS_DT)

        if should_render:
            frame = grab_frame()
            video_writer.append_data(frame)
            frames.append(frame)
        step_counter += 1

    video_writer.close()
    print(f"\n[INFO] Saved video ({len(frames)} frames @ {args_cli.video_fps} fps) -> {video_path}")

    n_shots = min(args_cli.num_screenshots, len(frames))
    shot_indices = np.linspace(0, len(frames) - 1, n_shots, dtype=int)
    for i, fi in enumerate(shot_indices):
        out_path = os.path.join(screenshot_dir, f"frame_{i:02d}.png")
        imageio.imwrite(out_path, frames[fi])
    print(f"[INFO] Saved {n_shots} screenshots -> {screenshot_dir}")


if __name__ == "__main__":
    main()
    simulation_app.close()

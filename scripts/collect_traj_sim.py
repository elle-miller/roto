"""Collect per-joint sinusoidal trajectories in IsaacLab simulation.

For each of the 13 Shadow Hand Lite policy joints, commands a sinusoid while
holding all other joints at zero, and records commanded vs actual position.

Usage (from roto/scripts/):
    python collect_traj_sim.py --headless
    python collect_traj_sim.py --headless --freq 0.5 --cycles 4 --amplitude_frac 0.8
    python collect_traj_sim.py --headless --joint_idx 0   # test only joint 0
"""

import argparse
import math
import os
import sys

import numpy as np
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Collect per-joint sinusoidal trajectories in sim.")
parser.add_argument("--output_dir", type=str, default="../trajectories/sim")
parser.add_argument("--freq",           type=float, default=0.5,  help="Sinusoid frequency (Hz)")
parser.add_argument("--cycles",         type=float, default=4.0,  help="Number of sinusoid cycles per joint")
parser.add_argument("--amplitude_frac", type=float, default=0.8,  help="Amplitude as fraction of half-range")
parser.add_argument("--settle_secs",    type=float, default=1.0,  help="Zero-hold seconds before each sinusoid")
parser.add_argument("--joint_idx",      type=int,   default=None, help="Test only this policy joint index (0-12)")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch
import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation

from roto.assets.shadow_hand_lite import SHADOW_HAND_LITE_CFG
from roto.tasks.physics import PHYSICS_DT, roto_sim_cfg

# ---------------------------------------------------------------------------
# Joint ordering and limits — must match run_shadow.py exactly
# ---------------------------------------------------------------------------

POLICY_JOINT_ORDER = [
    "rh_FFJ4",   # 0
    "rh_MFJ4",   # 1
    "rh_RFJ4",   # 2
    "rh_THJ5",   # 3
    "rh_FFJ3",   # 4
    "rh_MFJ3",   # 5
    "rh_RFJ3",   # 6
    "rh_THJ4",   # 7
    "rh_FFJ2",   # 8  — coupled proxy (J2 driver)
    "rh_MFJ2",   # 9  — coupled proxy
    "rh_RFJ2",   # 10 — coupled proxy
    "rh_THJ2",   # 11
    "rh_THJ1",   # 12
]

LOWER_LIMITS = np.array([
    -0.3491,  # FFJ4
    -0.3491,  # MFJ4
    -0.3491,  # RFJ4
    -1.0472,  # THJ5
    -0.2618,  # FFJ3
    -0.2618,  # MFJ3
    -0.2618,  # RFJ3
     0.0,     # THJ4
     0.0,     # FFJ2 proxy
     0.0,     # MFJ2 proxy
     0.0,     # RFJ2 proxy
    -0.6981,  # THJ2
    -0.2618,  # THJ1
], dtype=np.float32)

UPPER_LIMITS = np.array([
    0.3491,   # FFJ4
    0.3491,   # MFJ4
    0.3491,   # RFJ4
    1.0472,   # THJ5
    1.5708,   # FFJ3
    1.5708,   # MFJ3
    1.5708,   # RFJ3
    1.2217,   # THJ4
    1.5708,   # FFJ2 proxy
    1.5708,   # MFJ2 proxy
    1.5708,   # RFJ2 proxy
    0.6981,   # THJ2
    1.5708,   # THJ1
], dtype=np.float32)

# Coupled J1 joint names corresponding to policy indices 8, 9, 10
COUPLED_J1_NAMES = {8: "rh_FFJ1", 9: "rh_MFJ1", 10: "rh_RFJ1"}
COUPLING_THETA = 0.785   # rad — matches roto_env._handle_coupled_joints


def _coupling(proxy, upper):
    """Decode a J2 proxy value into (j2_target, j1_target) using the same
    split as roto_env._handle_coupled_joints."""
    j2 = float(np.clip(proxy * (upper / COUPLING_THETA), 0.0, upper))
    j1 = float(np.clip((proxy - COUPLING_THETA) / (upper - COUPLING_THETA) * upper, 0.0, upper))
    return j2, j1


def main():
    output_dir = os.path.abspath(args_cli.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    # -- Simulation setup (same physics config as training) ------------------
    sim = sim_utils.SimulationContext(roto_sim_cfg)
    sim.set_camera_view(eye=(0, -0.6, 1.0), target=(0, -0.3, 0.5))

    sim_utils.GroundPlaneCfg().func("/World/ground", sim_utils.GroundPlaneCfg())
    sim_utils.DomeLightCfg(intensity=1000.0, color=(1.0, 1.0, 1.0)).func(
        "/World/light", sim_utils.DomeLightCfg(intensity=1000.0)
    )

    robot_cfg = SHADOW_HAND_LITE_CFG.replace(prim_path="/World/Robot")
    robot = Articulation(robot_cfg)

    sim.reset()

    # -- Build joint name → robot index maps ---------------------------------
    joint_names = list(robot.data.joint_names)
    n_robot_joints = len(joint_names)
    name_to_ridx = {n: i for i, n in enumerate(joint_names)}

    print("\n=== Robot joint names ===")
    for i, n in enumerate(joint_names):
        print(f"  [{i:2d}] {n}")

    # Policy joint → robot joint index
    policy_to_ridx = []
    for pj in POLICY_JOINT_ORDER:
        if pj not in name_to_ridx:
            raise RuntimeError(f"Policy joint '{pj}' not found in robot joints: {joint_names}")
        policy_to_ridx.append(name_to_ridx[pj])

    # Coupled J1 joints → robot index
    coupled_j1_ridx = {pi: name_to_ridx[jn] for pi, jn in COUPLED_J1_NAMES.items()
                       if jn in name_to_ridx}

    # -- Set root pose (fixed base, so this just anchors it) -----------------
    robot.write_root_pose_to_sim(robot.data.default_root_state[:, :7])

    # -- Determine which joints to test --------------------------------------
    joints_to_test = [args_cli.joint_idx] if args_cli.joint_idx is not None else list(range(13))

    # -- Per-joint collection loop -------------------------------------------
    for pi in joints_to_test:
        joint_name = POLICY_JOINT_ORDER[pi]
        lower = float(LOWER_LIMITS[pi])
        upper = float(UPPER_LIMITS[pi])
        center = (upper + lower) / 2.0
        amp    = args_cli.amplitude_frac * (upper - lower) / 2.0

        print(f"\n[{pi:2d}] {joint_name}  lower={lower:.3f}  upper={upper:.3f}  "
              f"center={center:.3f}  amp={amp:.3f}")

        # -- Settle at zero --------------------------------------------------
        settle_steps = int(args_cli.settle_secs / PHYSICS_DT)
        zero_target = torch.zeros(1, n_robot_joints, device="cpu")
        for _ in range(settle_steps):
            robot.set_joint_position_target(zero_target)
            robot.write_data_to_sim()
            sim.step()
            robot.update(PHYSICS_DT)

        # -- Sinusoidal excitation -------------------------------------------
        traj_steps = int((args_cli.cycles / args_cli.freq) / PHYSICS_DT)
        ts, cmds, actual_pos_buf, actual_vel_buf = [], [], [], []

        for step in range(traj_steps):
            t = step * PHYSICS_DT
            proxy = center + amp * math.sin(2.0 * math.pi * args_cli.freq * t)
            proxy = float(np.clip(proxy, lower, upper))

            target = torch.zeros(1, n_robot_joints, device="cpu")

            if pi in COUPLED_J1_NAMES:
                j2_tgt, j1_tgt = _coupling(proxy, upper)
                target[0, policy_to_ridx[pi]]   = j2_tgt
                target[0, coupled_j1_ridx[pi]]  = j1_tgt
            else:
                target[0, policy_to_ridx[pi]] = proxy

            robot.set_joint_position_target(target)
            robot.write_data_to_sim()
            sim.step()
            robot.update(PHYSICS_DT)

            full_pos = robot.data.joint_pos[0].cpu().numpy()
            full_vel = robot.data.joint_vel[0].cpu().numpy()

            actual_pos = np.array([full_pos[policy_to_ridx[j]] for j in range(13)],
                                  dtype=np.float32)
            actual_vel = np.array([full_vel[policy_to_ridx[j]] for j in range(13)],
                                  dtype=np.float32)

            ts.append(t)
            cmds.append(proxy)
            actual_pos_buf.append(actual_pos)
            actual_vel_buf.append(actual_vel)

        # -- Save ------------------------------------------------------------
        fname = os.path.join(output_dir, f"joint_{pi:02d}_{joint_name}.npz")
        np.savez(
            fname,
            t=np.array(ts, dtype=np.float32),
            cmd=np.array(cmds, dtype=np.float32),
            actual_pos=np.array(actual_pos_buf, dtype=np.float32),
            actual_vel=np.array(actual_vel_buf, dtype=np.float32),
            joint_name=np.array(joint_name),
            joint_idx=np.array(pi),
            lower=np.array(lower, dtype=np.float32),
            upper=np.array(upper, dtype=np.float32),
        )
        print(f"  → Saved {fname}  ({len(ts)} steps @ {1/PHYSICS_DT:.0f} Hz)")

    print("\nAll joints done.")


if __name__ == "__main__":
    main()
    simulation_app.close()

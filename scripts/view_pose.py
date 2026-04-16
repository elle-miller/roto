"""Visualise a static hand pose in Isaac Lab — no training, no RL."""

import argparse
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="View a static hand pose.")
parser.add_argument("--robot", type=str, default="shadowlite")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch
import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from roto.tasks.robots.shadowlite.shadowlite import SHADOW_HAND_LITE_CFG

# ── Target pose — edit these until the hand looks right ──────────────────────
TARGET_JOINT_POS = {
            # ── Index finger (FF) — EXTENDED and spread outward ──────────────────
            "rh_FFJ4": -0.25,   # abduct index away from middle (toward thumb side)
            "rh_FFJ3":  0.0,    # MCP straight
            "rh_FFJ2":  0.0,    # PIP straight
            "rh_FFJ1":  0.0,    # DIP straight (coupled)

            # ── Middle finger (MF) — EXTENDED and spread outward ─────────────────
            "rh_MFJ4":  0.25,   # abduct middle away from index
            "rh_MFJ3":  0.0,
            "rh_MFJ2":  0.0,
            "rh_MFJ1":  0.0,

            # ── Ring finger (RF) — CURLED ─────────────────────────────────────────
            "rh_RFJ4":  0.0,    # no abduction
            "rh_RFJ3":  1.55,    # MCP curl
            "rh_RFJ2":  1.55,    # PIP curl
            "rh_RFJ1":  1.2,    # DIP curl (coupled, will follow J2)

            # ── Thumb (TH) — tucked toward palm center ────────────────────────────
            "rh_THJ5": 0.3,    # rotate thumb inward
            "rh_THJ4":  1.2,    # abduct thumb across palm
            "rh_THJ2":  0.5,    # slight flex
            "rh_THJ1":  1.5,    # distal curl
        }
# ─────────────────────────────────────────────────────────────────────────────


def main():
    # Basic sim setup
    sim_cfg = sim_utils.SimulationCfg(dt=0.01)
    sim = sim_utils.SimulationContext(sim_cfg)
    sim.set_camera_view(eye=(0, 0, 1.2), target=(0, -0.3, 0.5))

    # Ground plane + light
    sim_utils.GroundPlaneCfg().func("/World/ground", sim_utils.GroundPlaneCfg())
    sim_utils.DomeLightCfg(intensity=1000.0, color=(1.0, 1.0, 1.0)).func(
        "/World/light", sim_utils.DomeLightCfg(intensity=1000.0)
    )

    # Spawn the hand
    robot_cfg = SHADOW_HAND_LITE_CFG.replace(prim_path="/World/Robot")
    robot = Articulation(robot_cfg)

    sim.reset()

    # Build target tensor from joint names
    joint_names = robot.data.joint_names
    print("\n=== Joint names on this robot ===")
    for i, n in enumerate(joint_names):
        print(f"  [{i:2d}] {n}")

    target = torch.zeros(1, len(joint_names))
    for i, name in enumerate(joint_names):
        if name in TARGET_JOINT_POS:
            target[0, i] = TARGET_JOINT_POS[name]
            print(f"  SET {name} = {TARGET_JOINT_POS[name]}")
        else:
            if name not in ("rh_WRJ1", "rh_WRJ2"):  # ignore wrist
                print(f"  WARN: {name} not in TARGET_JOINT_POS — staying at 0")

    # Hold the pose forever — no physics, just set positions each step
    robot.set_joint_position_target(target)
    robot.write_root_pose_to_sim(robot.data.default_root_state[:, :7])

    print("\nViewer running — inspect the pose. Ctrl+C to quit.\n")
    step = 0
    while simulation_app.is_running():
        robot.set_joint_position_target(target)
        robot.write_data_to_sim()
        sim.step()
        robot.update(sim.get_physics_dt())
        
        step += 1
        if step % 100 == 0:
            pos = robot.data.joint_pos[0]
            print("\nCurrent joint positions:")
            for i, name in enumerate(joint_names):
                print(f"  {name:20s}: target={target[0,i]:.3f}  actual={pos[i]:.3f}")
        


if __name__ == "__main__":
    main()
    simulation_app.close()
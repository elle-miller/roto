"""Diagnose finger self-collision: which bodies touch, and at what J2 angle.

Free space (balls parked), no policy. Widens the robot contact sensor to cover every
ff/mf/rf segment (knuckle/proximal/middle/distal), slowly curls FF+MF+RF to full, and
reports per-body contact force each step. In free space the only contacts are
self-collisions, so a body showing force is hitting another finger. Prints, per finger:
the J2 angle where contact first appears (collision onset) and the max J2 reached.

Also sweeps the J4 abduction (spread) to find a splay that removes the collision so
J2 can reach its 100° limit.

    python diag_self_contact.py --headless
    python diag_self_contact.py --spread_deg -20 -30 -40 --headless     # try wider splay
"""

import argparse
import math
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Finger self-collision diagnostic.")
parser.add_argument("--task", type=str, default="Baoding")
parser.add_argument("--robot", type=str, default="shadowlite")
parser.add_argument("--agent_cfg", type=str, default="rl_only_pt")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--ramp_steps", type=int, default=150)
parser.add_argument("--hold_steps", type=int, default=40)
parser.add_argument("--force_thresh", type=float, default=0.05, help="N contact force counted as 'touching'.")
parser.add_argument("--spread_deg", type=float, nargs="+", default=[-20.0],
                    help="FF/RF J4 abduction values (deg) to test (MF kept at 0). Negative = current sign.")
parser.add_argument("--stiffness", type=float, default=20.0)
parser.add_argument("--damping", type=float, default=2.0)
parser.add_argument("--video", action="store_true", default=False)
parser.add_argument("--video_length", type=int, default=200)
parser.add_argument("--disable_fabric", action="store_true", default=False)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import types
import torch
import isaaclab_tasks  # noqa: F401
from isaaclab.utils import update_dict
from isaaclab.sensors import ContactSensorCfg
from isaaclab_tasks.utils.hydra import register_task_to_hydra
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry

from common_utils import (
    LOG_PATH, load_hand_task_agent_cfg, make_env, register_hand_task_to_hydra,
    resolve_gym_env_id, set_seed, update_env_cfg,
)
from multimodal_rl.tools.writer import Writer

DEG = 180.0 / math.pi
RAD = math.pi / 180.0


def _no_dones(self):
    z = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
    return z, z.clone()


def main():
    args_cli.gym_env_id = resolve_gym_env_id(args_cli.task, args_cli.robot)
    if args_cli.task in ("Bounce", "Baoding"):
        env_cfg, agent_cfg = register_hand_task_to_hydra(args_cli.task, args_cli.robot, "default_cfg")
        specialised_cfg = load_hand_task_agent_cfg(args_cli.task, args_cli.robot, args_cli.agent_cfg)
    else:
        env_cfg, agent_cfg = register_task_to_hydra(args_cli.gym_env_id, "default_cfg")
        specialised_cfg = load_cfg_from_registry(args_cli.gym_env_id, args_cli.agent_cfg)
    agent_cfg = update_dict(agent_cfg, specialised_cfg)
    agent_cfg["seed"] = args_cli.seed
    set_seed(agent_cfg["seed"])
    agent_cfg["log_path"] = LOG_PATH
    agent_cfg["experiment"]["video_dir"] = None
    env_cfg = update_env_cfg(args_cli, env_cfg, agent_cfg)

    # deterministic + free space
    if hasattr(env_cfg, "events"):
        env_cfg.events = None
    if hasattr(env_cfg, "ball_friction_range"):
        env_cfg.ball_friction_range = None
    if hasattr(env_cfg, "reset_joint_pos_noise"):
        env_cfg.reset_joint_pos_noise = 0.0
    if hasattr(env_cfg, "settle_steps"):
        env_cfg.settle_steps = 0
    for bcfg in (env_cfg.ball_1_cfg, env_cfg.ball_2_cfg):
        bcfg.spawn.rigid_props.kinematic_enabled = True
        bcfg.init_state.pos = (5.0, 5.0, 5.0)

    # widen the robot contact sensor to every ff/mf/rf segment
    env_cfg.robot_contact_sensor_cfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/rh_(ff|mf|rf)(knuckle|proximal|middle|distal)",
        update_period=0.0, history_length=1,
    )

    writer = Writer(agent_cfg, play=True)
    env_cfg.num_eval_envs = 0
    env = make_env(agent_cfg, env_cfg, writer, args_cli)
    raw = env.env.unwrapped
    raw._get_dones = types.MethodType(_no_dones, raw)
    raw.robot.write_joint_stiffness_to_sim(float(args_cli.stiffness))
    raw.robot.write_joint_damping_to_sim(float(args_cli.damping))

    control_names = list(raw.cfg.control_joint_names)
    driver_names = list(raw.cfg.coupled_joint_map.values())
    sweep_idx = [control_names.index(n) for n in driver_names]
    drv = raw.coupled_driver_indices
    fingers = [n.replace("rh_", "").replace("J2", "") for n in driver_names]

    sensor = raw.robot_contact_sensor
    body_names = list(sensor.body_names)
    print(f"[INFO] contact-sensor bodies ({len(body_names)}): {body_names}")

    # J4 joint ids for abduction override
    jn = raw.robot.joint_names
    ff4, mf4, rf4 = jn.index("rh_FFJ4"), jn.index("rh_MFJ4"), jn.index("rh_RFJ4")

    def set_spread(deg):
        # FF and RF abduct by `deg` (sign as URDF), MF stays at 0
        q = torch.zeros((raw.num_envs, raw.robot.num_joints), device=raw.device)
        cur = raw.robot.data.joint_pos.clone()
        cur[:, ff4] = deg * RAD
        cur[:, rf4] = deg * RAD
        cur[:, mf4] = 0.0
        raw.robot.write_joint_state_to_sim(cur, torch.zeros_like(cur))

    def run(spread_deg):
        onset = {f: None for f in fingers}              # J2 deg at first contact
        bodyhit = {f: set() for f in fingers}
        j2max = {f: 0.0 for f in fingers}
        # everything inside one inference_mode so reset + set_spread + step are
        # consistent (writing joint state inside inference_mode otherwise poisons
        # tensors that the next reset touches outside it).
        with torch.inference_mode():
            env.reset(hard=True)
            set_spread(spread_deg)
            steps = args_cli.ramp_steps + args_cli.hold_steps
            for k in range(steps):
                s = min(1.0, -1.0 + 2.0 * k / args_cli.ramp_steps)
                a = torch.zeros((raw.num_envs, len(control_names)), dtype=torch.float32, device=env.device)
                a[:, sweep_idx] = s
                # keep abduction commanded too (so it doesn't relax back)
                env.step(a)
                set_spread(spread_deg)
                f_w = torch.linalg.vector_norm(sensor.data.net_forces_w[0], dim=-1)  # [B]
                j2a = raw.robot.data.joint_pos[0, drv] * DEG
                for fi, f in enumerate(fingers):
                    j2max[f] = max(j2max[f], j2a[fi].item())
                for bi, bn in enumerate(body_names):
                    if f_w[bi].item() > args_cli.force_thresh:
                        fkey = next((f for f in fingers if bn[3:5].lower() == f[:2].lower()), None)
                        if fkey is not None:
                            bodyhit[fkey].add(bn.replace("rh_", ""))
                            if onset[fkey] is None:
                                onset[fkey] = j2a[fingers.index(fkey)].item()
        return onset, bodyhit, j2max

    print("\n" + "=" * 78)
    print("SELF-CONTACT DIAGNOSTIC (free space — any force = finger-finger collision)")
    print("=" * 78)
    for sp in args_cli.spread_deg:
        onset, bodyhit, j2max = run(sp)
        print(f"\n--- FF/RF abduction = {sp:+.0f}° (MF=0) ---")
        print(f"{'finger':>6} | {'J2 max':>7} | {'contact onset J2':>16} | bodies in contact")
        for f in fingers:
            on = f"{onset[f]:.0f}°" if onset[f] is not None else "none"
            print(f"{f:>6} | {j2max[f]:6.1f}° | {on:>16} | {sorted(bodyhit[f])}")
    print("\nReading it: if J2 max << 100° AND contact onset ~ that angle, the finger is")
    print("stopped by self-collision. Compare abduction values: the spread that yields")
    print("'none' contact and J2 max ≈ 100° is the splay to put in shadowlite.py init_state.")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()

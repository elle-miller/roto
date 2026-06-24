"""Verify the coupled-finger sequencing: is J2 commanded to its full range and
actually getting there, and does the J1 mimic ever fire before the split?

No policy. Drives FF/MF/RF from open to full curl (slow ramp + hold) and records
J2 and J1 (commanded AND actual) every step, in degrees. Prints the RUNTIME joint
limits (which come from the loaded USD, not the URDF text), so you can see whether
J2 tops out at 90 deg or 100 deg. Runs once with the gate OFF (raw theta-split)
and once with the gate ON, and flags every step where J1-actual moves before J2
reaches --split_deg.

    python test_coupling.py --task Baoding --robot shadowlite --agent_cfg rl_only_pt --headless
    python test_coupling.py --split_deg 100 --frac 0.7 --headless
"""

import argparse
import math
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Test coupled-finger J2->J1 sequencing.")
parser.add_argument("--task", type=str, default="Baoding")
parser.add_argument("--robot", type=str, default="shadowlite")
parser.add_argument("--agent_cfg", type=str, default="rl_only_pt")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--seed", type=int, default=None)
parser.add_argument("--split_deg", type=float, default=100.0, help="J2 angle (deg) J1 should wait for.")
parser.add_argument("--frac", type=float, default=None,
                    help="Gate fraction for the ON pass. Default: radians(split_deg)/J2_limit.")
parser.add_argument("--ramp_steps", type=int, default=120)
parser.add_argument("--hold_steps", type=int, default=60)
parser.add_argument("--eps_deg", type=float, default=2.0, help="J1 angle (deg) counted as 'fired'.")
parser.add_argument("--stiffness", type=float, default=None, help="Override stiffness on the coupled joints.")
parser.add_argument("--damping", type=float, default=None, help="Override damping on the coupled joints.")
parser.add_argument("--effort", type=float, default=None, help="Override effort_limit on the coupled joints.")
parser.add_argument("--video", action="store_true", default=False)
parser.add_argument("--video_length", type=int, default=200)
parser.add_argument("--disable_fabric", action="store_true", default=False)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch
import isaaclab_tasks  # noqa: F401
from isaaclab.utils import update_dict
from isaaclab_tasks.utils.hydra import register_task_to_hydra
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry

from common_utils import (
    LOG_PATH, load_hand_task_agent_cfg, make_env, register_hand_task_to_hydra,
    resolve_gym_env_id, set_seed, update_env_cfg,
)
from multimodal_rl.tools.writer import Writer

DEG = 180.0 / math.pi


def main():
    args_cli.gym_env_id = resolve_gym_env_id(args_cli.task, args_cli.robot)
    if args_cli.task in ("Bounce", "Baoding"):
        env_cfg, agent_cfg = register_hand_task_to_hydra(args_cli.task, args_cli.robot, "default_cfg")
        specialised_cfg = load_hand_task_agent_cfg(args_cli.task, args_cli.robot, args_cli.agent_cfg)
    else:
        env_cfg, agent_cfg = register_task_to_hydra(args_cli.gym_env_id, "default_cfg")
        specialised_cfg = load_cfg_from_registry(args_cli.gym_env_id, args_cli.agent_cfg)
    agent_cfg = update_dict(agent_cfg, specialised_cfg)
    agent_cfg["seed"] = args_cli.seed if args_cli.seed is not None else agent_cfg["seed"]
    set_seed(agent_cfg["seed"])
    agent_cfg["log_path"] = LOG_PATH
    agent_cfg["experiment"]["video_dir"] = None
    env_cfg = update_env_cfg(args_cli, env_cfg, agent_cfg)

    # deterministic: no friction DR, no reset noise, no early termination
    if hasattr(env_cfg, "events"):
        env_cfg.events = None
    if hasattr(env_cfg, "ball_friction_range"):
        env_cfg.ball_friction_range = None
    if hasattr(env_cfg, "reset_joint_pos_noise"):
        env_cfg.reset_joint_pos_noise = 0.0

    writer = Writer(agent_cfg, play=True)
    env_cfg.num_eval_envs = 0
    env = make_env(agent_cfg, env_cfg, writer, args_cli)
    raw = env.env.unwrapped

    import types
    raw._get_dones = types.MethodType(
        lambda self: (torch.zeros(self.num_envs, dtype=torch.bool, device=self.device),
                      torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)), raw)

    control_names = list(raw.cfg.control_joint_names)
    driver_names = list(raw.cfg.coupled_joint_map.values())   # FFJ2, MFJ2, RFJ2
    dep_names = list(raw.cfg.coupled_joint_map.keys())         # FFJ1, MFJ1, RFJ1
    sweep_idx = [control_names.index(n) for n in driver_names]
    drv, dep = raw.coupled_driver_indices, raw.coupled_dependent_indices
    fingers = [n.replace("rh_", "") for n in driver_names]

    j2u = raw.robot_joint_pos_upper_limits[drv]               # RUNTIME limits (from USD)
    j1u = raw.robot_joint_pos_upper_limits[dep]
    j2u_np, j1u_np = j2u.cpu().numpy(), j1u.cpu().numpy()

    print("\n" + "=" * 70)
    print("RUNTIME joint limits (from the loaded USD — what actually governs motion)")
    print("=" * 70)
    for i, f in enumerate(fingers):
        print(f"  {f:>6}: J2 upper = {j2u_np[i]:.3f} rad = {j2u_np[i]*DEG:5.1f}°   |   "
              f"{dep_names[i].replace('rh_',''):>6}: J1 upper = {j1u_np[i]:.3f} rad = {j1u_np[i]*DEG:5.1f}°")
    print(f"  coupling_theta = {raw.coupling_theta:.3f} rad = {raw.coupling_theta*DEG:.1f}°  "
          f"(for a J2-does-its-full-range split, theta should be J2_upper/2 = "
          f"{j2u_np.min()/2:.3f} rad = {j2u_np.min()/2*DEG:.1f}°)")

    # optional: stiffen/strengthen the coupled joints so J2 can reach 100°
    if any(v is not None for v in (args_cli.stiffness, args_cli.damping, args_cli.effort)):
        jt = list(drv) + list(dep)
        if args_cli.stiffness is not None:
            raw.robot.write_joint_stiffness_to_sim(float(args_cli.stiffness), joint_ids=jt)
        if args_cli.damping is not None:
            raw.robot.write_joint_damping_to_sim(float(args_cli.damping), joint_ids=jt)
        if args_cli.effort is not None:
            raw.robot.write_joint_effort_limit_to_sim(float(args_cli.effort), joint_ids=jt)
        print(f"[INFO] coupled-joint gains overridden: stiffness={args_cli.stiffness} "
              f"damping={args_cli.damping} effort={args_cli.effort}")

    eps = args_cli.eps_deg / DEG
    split_rad = args_cli.split_deg / DEG

    def run(gate_on):
        raw.couple_gate_j1_on_measured = gate_on
        if gate_on:
            raw.couple_gate_lo_frac = (args_cli.frac if args_cli.frac is not None
                                       else float(split_rad / j2u_np.min()))
        j2c, j2a, j1c, j1a = [], [], [], []
        with torch.inference_mode():
            env.reset(hard=True)
            for k in range(args_cli.ramp_steps + args_cli.hold_steps):
                s = min(1.0, -1.0 + 2.0 * k / args_cli.ramp_steps)   # ramp then hold at +1
                a = torch.full((raw.num_envs, len(control_names)), 0.0,
                               dtype=torch.float32, device=env.device)
                a[:, sweep_idx] = s
                env.step(a)
                j2c.append(raw.joint_pos_cmd[0, drv].clone())
                j2a.append(raw.robot.data.joint_pos[0, drv].clone())
                j1c.append(raw.joint_pos_cmd[0, dep].clone())
                j1a.append(raw.robot.data.joint_pos[0, dep].clone())
        return (torch.stack(j2c), torch.stack(j2a), torch.stack(j1c), torch.stack(j1a))

    for gate_on in (False, True):
        tag = "GATE ON " if gate_on else "GATE OFF"
        j2c, j2a, j1c, j1a = run(gate_on)
        frac = raw.couple_gate_lo_frac if gate_on else None
        print("\n" + "-" * 70)
        print(f"{tag}" + (f"  (couple_gate_lo_frac={frac:.3f} -> gate opens at J2="
                          f"{frac*j2u_np.min()*DEG:.1f}°)" if gate_on else "  (raw theta-split)"))
        print(f"{'finger':>6} | {'J2cmd max':>9} {'J2act max':>9} | {'J1act max':>9} | "
              f"{'J1 fires @ J2act':>16} | early-fire?")
        for i, f in enumerate(fingers):
            j2cm = j2c[:, i].max().item() * DEG
            j2am = j2a[:, i].max().item() * DEG
            j1am = j1a[:, i].max().item() * DEG
            fired = (j1a[:, i] > eps)
            if fired.any():
                k0 = int(torch.nonzero(fired)[0])
                fire_at = j2a[k0, i].item() * DEG          # J2 actual when J1 first moved
            else:
                fire_at = float("nan")
            # steps where J1 moved while J2 had NOT yet reached the split
            violations = int(((j1a[:, i] > eps) & (j2a[:, i] < split_rad)).sum())
            flag = "—" if (fire_at != fire_at) else \
                   (f"NO ({fire_at:.0f}°≥{args_cli.split_deg:.0f})" if fire_at >= args_cli.split_deg - args_cli.eps_deg
                    else f"YES @ {fire_at:.0f}° ({violations} steps)")
            print(f"{f:>6} | {j2cm:>8.1f}° {j2am:>8.1f}° | {j1am:>8.1f}° | {fire_at:>15.1f}° | {flag}")

    print("\nHow to read it:")
    print("  • 'J2cmd max' near the J2 limit confirms J2 IS commanded to its full range.")
    print("  • 'J2act max' << J2cmd means J2 can't physically get there (too soft — stiffen it).")
    print("  • 'J1 fires @ J2act' is the J2 angle at which J1 actually started moving;")
    print("    it should be ≥ split_deg. GATE OFF usually shows early firing; GATE ON should not.")
    print("  • If J2 never reaches split_deg, a gate set to that split keeps J1 at 0 (dead fingertip).")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()

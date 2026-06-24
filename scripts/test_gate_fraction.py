"""Concretely test the J1/J2 gating mechanism and the couple_gate_lo_frac knob.

No policy involved. It fully curls the FF/MF/RF fingers and holds, then measures
the steady-state J2 (driver) and J1 (mimic) angles — first with gating OFF
(baseline), then with gating ON across a sweep of couple_gate_lo_frac values.

This answers "why can't frac be 1?": the gate opens only when measured J2 >=
frac * J2_limit, but J2 physically tops out below its limit. The script prints
J2's achievable ceiling and the max usable frac, and shows J1 dying out as frac
approaches/exceeds that ceiling.

Run headless (numbers only) or with the viewer:
    python test_gate_fraction.py --task Baoding --robot shadowlite --agent_cfg rl_only_pt --headless
    python test_gate_fraction.py --fracs 1.0 0.9 0.8 0.7 0.6 0.5 --settle 150 --headless
"""

import argparse
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Test the coupled-finger gate fraction.")
parser.add_argument("--task", type=str, default="Baoding")
parser.add_argument("--robot", type=str, default="shadowlite")
parser.add_argument("--agent_cfg", type=str, default=None)
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--seed", type=int, default=None)
parser.add_argument("--fracs", type=float, nargs="+", default=[1.0, 0.9, 0.8, 0.7, 0.6, 0.5],
                    help="couple_gate_lo_frac values to test (gating ON).")
parser.add_argument("--settle", type=int, default=150,
                    help="Steps to hold full curl so J2/J1 reach steady state.")
parser.add_argument("--hold", type=float, default=0.0, help="Action held on non-swept joints.")
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
    LOG_PATH,
    load_hand_task_agent_cfg,
    make_env,
    register_hand_task_to_hydra,
    resolve_gym_env_id,
    set_seed,
    update_env_cfg,
)
from multimodal_rl.tools.writer import Writer


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
    writer = Writer(agent_cfg, play=True)
    env_cfg.num_eval_envs = 0
    env = make_env(agent_cfg, env_cfg, writer, args_cli)

    raw = env.env.unwrapped
    control_names = list(raw.cfg.control_joint_names)
    driver_names = list(raw.cfg.coupled_joint_map.values())     # FFJ2, MFJ2, RFJ2
    dep_names = list(raw.cfg.coupled_joint_map.keys())          # FFJ1, MFJ1, RFJ1
    sweep_idx = [control_names.index(n) for n in driver_names]
    drv = raw.coupled_driver_indices
    dep = raw.coupled_dependent_indices
    j2u = raw.robot_joint_pos_upper_limits[drv]                # (3,) J2 limits

    def run_segment(gate_on, frac):
        """Reset, fully curl FF/MF/RF, hold, return mean steady J2 & J1 (3,) each."""
        raw.couple_gate_j1_on_measured = bool(gate_on)
        raw.couple_gate_lo_frac = float(frac)
        with torch.inference_mode():
            env.reset(hard=True)
            action = torch.full((raw.num_envs, len(control_names)), args_cli.hold,
                                dtype=torch.float32, device=env.device)
            action[:, sweep_idx] = 1.0          # full curl request
            j2_hist, j1_hist = [], []
            for k in range(args_cli.settle):
                env.step(action)
                if k >= args_cli.settle - 20:    # average last 20 steps
                    j2_hist.append(raw.robot.data.joint_pos[0, drv].clone())
                    j1_hist.append(raw.robot.data.joint_pos[0, dep].clone())
        j2 = torch.stack(j2_hist).mean(0).cpu().numpy()
        j1 = torch.stack(j1_hist).mean(0).cpu().numpy()
        return j2, j1

    j2u_np = j2u.cpu().numpy()
    fingers = [n.replace("rh_", "").replace("J2", "") for n in driver_names]  # FF, MF, RF

    print("\n" + "=" * 78)
    print("J2 ceiling test (gating OFF, full curl) — how far can each J2 actually reach?")
    print("=" * 78)
    j2_off, j1_off = run_segment(gate_on=False, frac=1.0)
    print(f"{'finger':>8} | {'J2 limit':>9} {'J2 reached':>11} {'% of limit':>11} | {'J1 (ungated)':>12}")
    for i, f in enumerate(fingers):
        print(f"{f:>8} | {j2u_np[i]:9.3f} {j2_off[i]:11.3f} {100*j2_off[i]/j2u_np[i]:10.1f}% | {j1_off[i]:12.3f}")
    ceiling = float((j2_off / j2u_np).min())
    print(f"\n  J2 ceiling (worst finger) = {ceiling:.2f} of its limit.")
    print(f"  => the gate can ONLY open if couple_gate_lo_frac < {ceiling:.2f}.")
    print(f"  => frac >= {ceiling:.2f} (e.g. 1.0) leaves J1 stuck at ~0 (fingertips never curl).")

    print("\n" + "=" * 78)
    print("Gate-fraction sweep (gating ON, full curl)")
    print("=" * 78)
    print(f"{'frac':>5} | {'gate opens at J2':>16} | "
          + " ".join(f"{f+'J1':>7}" for f in fingers) + " | note")
    for frac in args_cli.fracs:
        j2, j1 = run_segment(gate_on=True, frac=frac)
        gate_lo = frac * j2u_np
        opens = (j2 >= gate_lo)
        j1max = float(j1.max())
        note = "J1 dead (gate never opens)" if j1max < 0.05 else \
               ("J1 partial" if j1max < 0.6 else "J1 curls — sequential")
        print(f"{frac:>5.2f} | {gate_lo[0]:16.3f} | "
              + " ".join(f"{j1[i]:7.3f}" for i in range(len(fingers))) + f" | {note}")

    print("\nReading it: a frac whose 'gate opens at J2' exceeds the J2 reached above")
    print("means the gate never opens -> J1 stays ~0. Pick the largest frac that still")
    print("gives 'J1 curls' for the strictest-but-working sequencing.\n")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()

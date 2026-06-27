"""Validate the stateful backlash coupling + the per-episode DR (unlock R, hand tilt).

No policy. Drives the FF/MF/RF curl proxy to exact COMBINED-motor angles m (ffj0
frame, 0–180°) and reads back the commanded J2/J1, so the asymmetric law can be
checked directly against the three scenarios:

  seg1  fresh curl    m: 0→180   — J2 reaches 100° first, J1=0 below 100°, then J1 0→80°
  seg2  uncurl/stop   m: 180→120 — J2 unlocks at R, J1 unwinds (overlap over [100,R])
  seg3  re-curl       m: 120→180 — J1 FROZEN over [120,R], then resumes to 80° at 180°
  seg4  full uncurl   m: 180→90  — J2 unlocks at R, J1 hits 0 at m=100°, stays 0 below

It forces couple_release=R (default 136°) so the expected corners are deterministic.
Then it resets a few times to confirm R and hand-tilt are randomized per episode.

    python test_backlash_coupling.py --headless
    python test_backlash_coupling.py --R 130 --headless
"""

import argparse
import math
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Test the stateful backlash coupling + DR.")
parser.add_argument("--task", type=str, default="Baoding")
parser.add_argument("--robot", type=str, default="shadowlite")
parser.add_argument("--agent_cfg", type=str, default="rl_only_pt")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--R", type=float, default=136.0, help="Forced unlock angle (deg, combined) for the deterministic drive.")
parser.add_argument("--stop", type=float, default=120.0, help="Combined angle (deg) to stop the partial uncurl at.")
parser.add_argument("--step_deg", type=float, default=2.0, help="Combined-angle step between waypoints.")
parser.add_argument("--resets", type=int, default=6, help="Number of resets to sample R / tilt DR over.")
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

    # deterministic + no settle so the driven command isn't overridden
    if hasattr(env_cfg, "events"):
        env_cfg.events = None
    if hasattr(env_cfg, "ball_friction_range"):
        env_cfg.ball_friction_range = None
    if hasattr(env_cfg, "reset_joint_pos_noise"):
        env_cfg.reset_joint_pos_noise = 0.0
    if hasattr(env_cfg, "settle_steps"):
        env_cfg.settle_steps = 0

    writer = Writer(agent_cfg, play=True)
    env_cfg.num_eval_envs = 0
    env = make_env(agent_cfg, env_cfg, writer, args_cli)
    raw = env.env.unwrapped
    raw._get_dones = types.MethodType(_no_dones, raw)

    assert getattr(raw, "couple_asymmetric_backward", False), \
        "couple_asymmetric_backward must be True for this test (check shadowlite cfg)."

    control_names = list(raw.cfg.control_joint_names)
    driver_names  = list(raw.cfg.coupled_joint_map.values())   # FFJ2 MFJ2 RFJ2
    sweep_idx = [control_names.index(n) for n in driver_names]
    drv = raw.coupled_driver_indices
    dep = raw.coupled_dependent_indices
    fingers = [n.replace("rh_", "").replace("J2", "") for n in driver_names]

    theta = raw.coupling_theta
    j2u = raw.robot_joint_pos_upper_limits[drv][0].item()   # ~1.745 (100°)
    j1u = raw.robot_joint_pos_upper_limits[dep][0].item()   # ~1.396 (80°)
    lower = raw.robot_joint_pos_lower_limits[drv][0].item() # 0

    def proxy_for_m(m_rad):
        """Invert the forward split: combined motor m (rad) -> driver proxy (rad)."""
        if m_rad <= j2u:
            return m_rad * theta / j2u
        return theta + (m_rad - j2u) / j1u * (j2u - theta)

    def action_for_m(m_deg):
        p = proxy_for_m(m_deg * RAD)
        return 2.0 * (p - lower) / (j2u - lower) - 1.0   # scale() inverse, [-1,1]

    def drive_to(m_deg):
        a = torch.zeros((raw.num_envs, len(control_names)), dtype=torch.float32, device=env.device)
        a[:, sweep_idx] = action_for_m(m_deg)
        env.step(a)
        # commanded J2/J1 (the model output), finger 0
        j2 = raw.joint_pos_cmd[0, drv].clone() * DEG
        j1 = raw.joint_pos_cmd[0, dep].clone() * DEG
        m  = raw.prev_m[0].clone() * DEG
        return m, j2, j1

    def ramp(m_from, m_to):
        """Step from m_from to m_to in step_deg increments, return recorded arrays."""
        n = max(1, int(abs(m_to - m_from) / args_cli.step_deg))
        rows = []
        for k in range(1, n + 1):
            mt = m_from + (m_to - m_from) * k / n
            m, j2, j1 = drive_to(mt)
            rows.append((m[0].item(), j2[0].item(), j1[0].item()))  # finger 0 (FF)
        return rows

    # ---- deterministic drive on finger 0 (FF), R forced -----------------------
    with torch.inference_mode():
        env.reset(hard=True)
        raw.couple_release[:] = args_cli.R * RAD      # force unlock angle
        # clear state for a clean fresh start
        raw.prev_m[:] = 0.0; raw.couple_dir[:] = 1.0; raw.j1_state[:] = 0.0
        raw.couple_frozen_flag[:] = False; raw.couple_frozen_val[:] = 0.0

        seg1 = ramp(0.0, 180.0)
        seg2 = ramp(180.0, args_cli.stop)
        seg3 = ramp(args_cli.stop, 180.0)
        seg4 = ramp(180.0, 90.0)

    R = args_cli.R
    print("\n" + "=" * 76)
    print(f"BACKLASH COUPLING (finger {fingers[0]}, forced R = {R:.0f}°, stop = {args_cli.stop:.0f}°)")
    print("=" * 76)

    def at(rows, m_target):
        return min(rows, key=lambda r: abs(r[0] - m_target))

    checks = []

    # seg1 fresh curl: J1 ~0 below 100°, J2 reaches ~100° at m=100, J1>0 above 100°
    j1_below = max(j1 for (m, j2, j1) in seg1 if m < 95.0)
    j2_at100 = at(seg1, 100.0)[1]
    j1_at140 = at(seg1, 140.0)[2]
    checks.append(("seg1 J1 quiet for m<95°",      j1_below < 2.0,  f"max J1={j1_below:.1f}°"))
    checks.append(("seg1 J2≈100° at m=100°",       abs(j2_at100 - 100) < 6, f"J2={j2_at100:.1f}°"))
    checks.append(("seg1 J1 fires by m=140°",      j1_at140 > 20.0, f"J1={j1_at140:.1f}°"))

    # seg2 uncurl: J2 starts dropping near R; J1 tracks down = m-100
    j2_just_below_R = at(seg2, R - 6)[1]
    j1_at_stop = at(seg2, args_cli.stop)[2]
    checks.append((f"seg2 J2 dropped below 100° by m={R-6:.0f}°", j2_just_below_R < 99.0, f"J2={j2_just_below_R:.1f}°"))
    checks.append((f"seg2 J1≈(m-100) at stop",      abs(j1_at_stop - (args_cli.stop - 100)) < 4,
                   f"J1={j1_at_stop:.1f}° vs {args_cli.stop-100:.0f}°"))

    # seg3 re-curl: J1 FROZEN over [stop, R], then resumes; J2 climbs back to 100° at R
    frozen_band = [j1 for (m, j2, j1) in seg3 if args_cli.stop - 1 <= m <= R - 1]
    frozen_var = (max(frozen_band) - min(frozen_band)) if frozen_band else 99
    j1_after_R = at(seg3, min(180, R + 20))[2]
    j2_back_at_R = at(seg3, R)[1]
    checks.append((f"seg3 J1 frozen over [{args_cli.stop:.0f},{R:.0f}]°", frozen_var < 2.5, f"Δ={frozen_var:.1f}°"))
    checks.append((f"seg3 J1 resumes after R",      j1_after_R > frozen_band[-1] + 3 if frozen_band else False,
                   f"J1={j1_after_R:.1f}°"))
    checks.append((f"seg3 J2 back to ~100° at m=R", j2_back_at_R > 96, f"J2={j2_back_at_R:.1f}°"))

    # seg4 full uncurl: J1 hits ~0 at m=100°, stays 0 below
    j1_at100 = at(seg4, 100.0)[2]
    j1_below100 = max(j1 for (m, j2, j1) in seg4 if m < 98.0)
    checks.append(("seg4 J1≈0 at m=100°",           j1_at100 < 4.0, f"J1={j1_at100:.1f}°"))
    checks.append(("seg4 J1 stays 0 for m<98°",     j1_below100 < 3.0, f"max J1={j1_below100:.1f}°"))

    print(f"\n{'check':<42} {'result':<8} detail")
    print("-" * 76)
    npass = 0
    for name, ok, detail in checks:
        mark = "✓ PASS" if ok else "✗ FAIL"
        if ok:
            npass += 1
        print(f"{name:<42} {mark:<8} {detail}")
    print("-" * 76)
    print(f"{npass}/{len(checks)} checks passed.")

    # ---- trace dump (every ~10° for eyeballing the hysteresis loop) -----------
    def trace(label, rows):
        print(f"\n  {label}:  " + "  ".join(
            f"m{r[0]:3.0f}|J2{r[1]:3.0f}|J1{r[2]:3.0f}"
            for r in rows[::max(1, len(rows)//9)]))
    trace("seg1 curl  ", seg1)
    trace("seg2 uncurl", seg2)
    trace("seg3 recurl", seg3)
    trace("seg4 uncurl", seg4)

    # ---- DR check: R and hand tilt vary per episode ---------------------------
    print("\n" + "=" * 76)
    print(f"DOMAIN RANDOMIZATION over {args_cli.resets} resets (release R, hand tilt)")
    print("=" * 76)
    q0 = torch.tensor((0.0, 0.0, -0.7071, 0.7071), device=raw.device)
    lo_t, hi_t = raw.cfg.hand_tilt_range_deg
    print(f"{'reset':>5} | {'R per finger (deg)':<28} | tilt(deg)")
    with torch.inference_mode():
        for i in range(args_cli.resets):
            env.reset(hard=True)
            env.step(torch.zeros((raw.num_envs, len(control_names)), dtype=torch.float32, device=env.device))
            Rs = (raw.couple_release[0] * DEG).tolist()
            q = raw.robot.data.root_quat_w[0]
            dot = float(torch.clamp(torch.abs((q * q0).sum()), max=1.0))
            tilt = 2.0 * math.acos(dot) * DEG     # angle from the 0° pose
            rstr = " ".join(f"{r:5.1f}" for r in Rs)
            print(f"{i:>5} | {rstr:<28} | {tilt:6.2f}")
    print(f"\n  Expected: R in [{raw.couple_release_lo*DEG:.0f},{raw.couple_release_hi*DEG:.0f}]°, "
          f"tilt in [{lo_t:.0f},{hi_t:.0f}]°, both varying per reset.")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()

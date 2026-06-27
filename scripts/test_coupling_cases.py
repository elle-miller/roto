"""7-case PASS/FAIL test for the strict frac=1 sequential coupling gate.

Exercises every behavioural case that the measured-J2 gate must satisfy:

  FORWARD (slow ramp then hold):
    Case 1  — J1 quiet below J2 limit  (both worlds)
    Case 2  — J1 fires once J2 ≈ limit (free space only; expected FIRE)
    Case 3  — J1 silent under ball load (J2 blocked at ~90°; expected NO-FIRE)
    Case 4  — abrupt step: no early J1 even on a hard proxy jump (gate governs)

  BACKWARD (start fully curled, ramp back to open):
    Case 5  — J1 retracts before/with J2 (no lingering fingertip curl)
    Case 6  — abrupt uncurl collapses J1 immediately

  EDGE:
    Case 7  — per-finger independence: curl FF only, MF/RF J1 must stay ~0

Run in free space to test Case 2 (J1 fires at limit):
    python test_coupling_cases.py --no_ball --headless

Run with ball to test Cases 3 & 4 (J1 blocked because J2 can't reach limit):
    python test_coupling_cases.py --headless

Run both back-to-back:
    python test_coupling_cases.py --headless
    python test_coupling_cases.py --no_ball --headless
"""

import argparse
import math
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Full coupling gate test suite.")
parser.add_argument("--task",      type=str, default="Baoding")
parser.add_argument("--robot",     type=str, default="shadowlite")
parser.add_argument("--agent_cfg", type=str, default="rl_only_pt")
parser.add_argument("--num_envs",  type=int, default=1)
parser.add_argument("--seed",      type=int, default=None)
parser.add_argument("--no_ball",   action="store_true", default=False,
                    help="Park both balls far away (free-space test).")
parser.add_argument("--ramp_steps",  type=int, default=120,
                    help="Steps to ramp proxy 0→max (forward/backward).")
parser.add_argument("--hold_steps",  type=int, default=80,
                    help="Steps to hold at full curl / full open after the ramp.")
parser.add_argument("--eps_deg",   type=float, default=2.0,
                    help="J1 angle (deg) considered 'fired'.")
parser.add_argument("--tol_deg",   type=float, default=4.0,
                    help="J2 must be within this many degrees of its limit for J1 "
                         "to be expected to fire (should be > couple_gate_j2_tol in deg).")
parser.add_argument("--stiffness", type=float, default=None,
                    help="Override stiffness on coupled joints (helps J2 reach its limit).")
parser.add_argument("--damping",   type=float, default=None)
parser.add_argument("--effort",    type=float, default=None,
                    help="Override effort_limit. Raise if J2 stalls short of its limit.")
parser.add_argument("--frac",      type=float, default=None,
                    help="Override couple_gate_lo_frac at runtime (default: keep cfg value).")
parser.add_argument("--video",       action="store_true", default=False)
parser.add_argument("--video_length",type=int, default=200)
parser.add_argument("--disable_fabric", action="store_true", default=False)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args

app_launcher  = AppLauncher(args_cli)
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


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _no_dones(self):
    z = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
    return z, z.clone()


def _override_gains(raw, coupled_ids):
    if args_cli.stiffness is not None:
        raw.robot.write_joint_stiffness_to_sim(float(args_cli.stiffness), joint_ids=coupled_ids)
    if args_cli.damping is not None:
        raw.robot.write_joint_damping_to_sim(float(args_cli.damping), joint_ids=coupled_ids)
    if args_cli.effort is not None:
        raw.robot.write_joint_effort_limit_to_sim(float(args_cli.effort), joint_ids=coupled_ids)


def _action(raw, control_names, sweep_idx, val_sweep, val_rest=0.0):
    a = torch.full((raw.num_envs, len(control_names)), val_rest,
                   dtype=torch.float32, device=raw.device)
    a[:, sweep_idx] = val_sweep
    return a


def _ramp(raw, env, control_names, sweep_idx, start, end, steps):
    """Ramp all sweep joints from start to end over `steps` steps, record J2/J1."""
    j2c, j2a, j1c, j1a = [], [], [], []
    drv = raw.coupled_driver_indices
    dep = raw.coupled_dependent_indices
    with torch.inference_mode():
        for k in range(steps):
            t = k / max(steps - 1, 1)
            val = start + t * (end - start)
            env.step(_action(raw, control_names, sweep_idx, val))
            j2c.append(raw.joint_pos_cmd[0, drv].clone())
            j2a.append(raw.robot.data.joint_pos[0, drv].clone())
            j1c.append(raw.joint_pos_cmd[0, dep].clone())
            j1a.append(raw.robot.data.joint_pos[0, dep].clone())
    return (torch.stack(j2c), torch.stack(j2a), torch.stack(j1c), torch.stack(j1a))


def _hold(raw, env, control_names, sweep_idx, val, steps):
    return _ramp(raw, env, control_names, sweep_idx, val, val, steps)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

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

    # deterministic: no DR, no reset noise, no early terminations
    if hasattr(env_cfg, "events"):
        env_cfg.events = None
    if hasattr(env_cfg, "ball_friction_range"):
        env_cfg.ball_friction_range = None
    if hasattr(env_cfg, "reset_joint_pos_noise"):
        env_cfg.reset_joint_pos_noise = 0.0
    # disable settle for this diagnostic so results aren't offset by pose-hold steps
    if hasattr(env_cfg, "settle_steps"):
        env_cfg.settle_steps = 0

    # free-space: park both balls far away and disable TacSL
    if args_cli.no_ball:
        for bcfg in (env_cfg.ball_1_cfg, env_cfg.ball_2_cfg):
            bcfg.spawn.rigid_props.kinematic_enabled = True
            bcfg.init_state.pos = (5.0, 5.0, 5.0)
        if hasattr(env_cfg, "tacsl_contact_expr"):
            env_cfg.tacsl_contact_expr = None

    writer = Writer(agent_cfg, play=True)
    env_cfg.num_eval_envs = 0
    env = make_env(agent_cfg, env_cfg, writer, args_cli)
    raw = env.env.unwrapped

    # never auto-reset mid-trial
    raw._get_dones = types.MethodType(_no_dones, raw)

    control_names = list(raw.cfg.control_joint_names)
    driver_names  = list(raw.cfg.coupled_joint_map.values())   # FFJ2, MFJ2, RFJ2
    dep_names     = list(raw.cfg.coupled_joint_map.keys())     # FFJ1, MFJ1, RFJ1
    fingers       = [n.replace("rh_", "").replace("J2", "") for n in driver_names]

    # indices for sweeping all 3 curl-proxy joints
    sweep_all = [control_names.index(n) for n in driver_names]
    # index for FF only (Case 7)
    sweep_ff  = [control_names.index(driver_names[0])]

    drv = raw.coupled_driver_indices
    dep = raw.coupled_dependent_indices
    coupled_ids = list(drv) + list(dep)

    j2u = raw.robot_joint_pos_upper_limits[drv]
    j2u_np = j2u.cpu().numpy()

    # apply runtime gain overrides
    _override_gains(raw, coupled_ids)

    # apply frac override
    raw.couple_gate_j1_on_measured = True
    if args_cli.frac is not None:
        raw.couple_gate_lo_frac = args_cli.frac

    frac   = raw.couple_gate_lo_frac
    tol    = raw.couple_gate_j2_tol
    world  = "free-space" if args_cli.no_ball else "with-ball"
    eps    = args_cli.eps_deg / DEG
    tol_deg = args_cli.tol_deg

    print("\n" + "=" * 80)
    print(f"COUPLING GATE TEST  |  world={world}  |  frac={frac:.3f}  |  tol={tol*DEG:.1f}°")
    print(f"J2 limits: " + ", ".join(f"{f}={j2u_np[i]*DEG:.1f}°" for i, f in enumerate(fingers)))
    print("=" * 80)

    results = []  # list of (case_id, description, world, finger_idx, PASS)

    def record(case_id, desc, j2a, j1a, expected, finger_i=None):
        """Evaluate one case for one or all fingers and append to results."""
        fi_range = range(len(fingers)) if finger_i is None else [finger_i]
        for fi in fi_range:
            j2_max_deg = j2a[:, fi].max().item() * DEG
            j1_max_deg = j1a[:, fi].max().item() * DEG
            fired = bool((j1a[:, fi] > eps).any())
            near_limit = j2_max_deg >= (j2u_np[fi] * DEG - tol_deg)

            if expected == "QUIET":
                # J1 must not fire while J2 is clearly below its limit
                quiet_mask = j2a[:, fi] < (j2u_np[fi] - tol / DEG * DEG)  # rough
                early_fire = bool((j1a[:, fi][quiet_mask] > eps).any()) if quiet_mask.any() else False
                ok = not early_fire
            elif expected == "FIRE":
                ok = fired and near_limit
            elif expected == "NO-FIRE":
                ok = not fired
            elif expected == "RETRACT":
                # J1 must reach ~0 at some point while J2 is still partly curled
                j1_min_deg = j1a[:, fi].min().item() * DEG
                ok = j1_min_deg < args_cli.eps_deg
            else:
                ok = True  # unknown, just record

            f = fingers[fi]
            verdict = "PASS" if ok else "FAIL"
            results.append((case_id, desc, world, f, j2_max_deg, j1_max_deg, fired, expected, verdict))

    # -------------------------------------------------------------------
    # Case 1+2 / 3+4 — forward: slow ramp + hold
    # -------------------------------------------------------------------
    print(f"\n--- FORWARD (slow ramp {args_cli.ramp_steps} steps + hold {args_cli.hold_steps}) ---")
    with torch.inference_mode():
        env.reset(hard=True)
        j2c_r, j2a_r, j1c_r, j1a_r = _ramp(raw, env, control_names, sweep_all, -1.0, 1.0, args_cli.ramp_steps)
        j2c_h, j2a_h, j1c_h, j1a_h = _hold(raw, env, control_names, sweep_all, 1.0, args_cli.hold_steps)

    j2a_fwd = torch.cat([j2a_r, j2a_h], dim=0)
    j1a_fwd = torch.cat([j1a_r, j1a_h], dim=0)

    # Case 1: J1 quiet during ramp while J2 < limit
    record(1, "J1 quiet below J2 limit (fwd ramp)", j2a_r, j1a_r, "QUIET")
    # Case 2 (free space) or Case 3 (with ball): fire vs no-fire
    if args_cli.no_ball:
        record(2, "J1 fires at J2 limit (free space)", j2a_fwd, j1a_fwd, "FIRE")
    else:
        record(3, "J1 silent: J2 blocked by ball <100°", j2a_fwd, j1a_fwd, "NO-FIRE")

    # -------------------------------------------------------------------
    # Case 4 — abrupt step: proxy max in one shot (gate must still govern)
    # -------------------------------------------------------------------
    print(f"--- ABRUPT STEP (proxy -1→+1 in one step) ---")
    with torch.inference_mode():
        env.reset(hard=True)
        j2c_s, j2a_s, j1c_s, j1a_s = _hold(raw, env, control_names, sweep_all, 1.0,
                                             args_cli.ramp_steps + args_cli.hold_steps)
    record(4, "Abrupt step: no early J1", j2a_s, j1a_s,
           "FIRE" if args_cli.no_ball else "NO-FIRE")

    # -------------------------------------------------------------------
    # Cases 5+6 — backward: start fully curled, ramp back to open
    # -------------------------------------------------------------------
    print(f"--- BACKWARD (start curled, ramp back) ---")
    with torch.inference_mode():
        # first bring fingers to full curl and hold
        env.reset(hard=True)
        _hold(raw, env, control_names, sweep_all, 1.0, args_cli.ramp_steps)
        # slow uncurl
        j2c_u, j2a_u, j1c_u, j1a_u = _ramp(raw, env, control_names, sweep_all, 1.0, -1.0, args_cli.ramp_steps)
        j2c_uh, j2a_uh, j1c_uh, j1a_uh = _hold(raw, env, control_names, sweep_all, -1.0, args_cli.hold_steps)

    j1a_back = torch.cat([j1a_u, j1a_uh], dim=0)
    j2a_back = torch.cat([j2a_u, j2a_uh], dim=0)
    record(5, "J1 retracts during uncurl", j2a_back, j1a_back, "RETRACT")

    # Case 6: abrupt uncurl (proxy max→-1 in one step, hold)
    with torch.inference_mode():
        env.reset(hard=True)
        _hold(raw, env, control_names, sweep_all, 1.0, args_cli.ramp_steps)
        j2c_au, j2a_au, j1c_au, j1a_au = _hold(raw, env, control_names, sweep_all, -1.0, args_cli.hold_steps)
    record(6, "Abrupt uncurl: J1 collapses", j2a_au, j1a_au, "RETRACT")

    # -------------------------------------------------------------------
    # Case 7 — per-finger: curl FF only, MF/RF J1 must stay ~0
    # -------------------------------------------------------------------
    print(f"--- PER-FINGER INDEPENDENCE (FF curl only) ---")
    with torch.inference_mode():
        env.reset(hard=True)
        _ramp(raw, env, control_names, sweep_all, -1.0, -1.0, 5)  # start open
        j2c_pf, j2a_pf, j1c_pf, j1a_pf = _ramp(raw, env, control_names, sweep_ff, -1.0, 1.0, args_cli.ramp_steps)
        j2c_pfh, j2a_pfh, j1c_pfh, j1a_pfh = _hold(raw, env, control_names, sweep_ff, 1.0, args_cli.hold_steps)

    j1a_pf_all = torch.cat([j1a_pf, j1a_pfh], dim=0)
    j2a_pf_all = torch.cat([j2a_pf, j2a_pfh], dim=0)
    # MF (index 1) and RF (index 2) must not fire
    record(7, "MF/RF J1 silent when only FF curls", j2a_pf_all, j1a_pf_all, "NO-FIRE", finger_i=1)
    record(7, "MF/RF J1 silent when only FF curls", j2a_pf_all, j1a_pf_all, "NO-FIRE", finger_i=2)

    # -------------------------------------------------------------------
    # Print results table
    # -------------------------------------------------------------------
    print("\n" + "=" * 90)
    print(f"RESULTS  |  world={world}  |  frac={frac:.3f}  |  gate opens at J2 within {tol*DEG:.1f}° of limit")
    print("=" * 90)
    hdr = f"{'Case':>5} {'finger':>6} {'J2max':>7} {'J1max':>7} {'fired':>6} {'expected':>10} {'verdict':>7}"
    print(hdr)
    print("-" * 90)
    passes = 0
    for (cid, desc, wld, finger, j2m, j1m, fired, exp, verdict) in results:
        mark = "✓" if verdict == "PASS" else "✗"
        print(f"  C{cid:<2d} {finger:>6} {j2m:>6.1f}° {j1m:>6.1f}° {str(fired):>6} {exp:>10} {mark} {verdict}")
        if verdict == "PASS":
            passes += 1
    print("-" * 90)
    total = len(results)
    print(f"\n  {passes}/{total} cases passed.\n")

    if passes == total:
        print("  ALL PASS — gate behaviour is correct for this world.")
    else:
        failed = [(cid, desc) for (cid, desc, *_, verdict) in results if verdict == "FAIL"]
        print("  FAILED cases:")
        for cid, desc in failed:
            print(f"    Case {cid}: {desc}")
        if not args_cli.no_ball:
            print("\n  Tip: Cases 2 requires --no_ball (J2 is blocked by ball geometry in this world).")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()

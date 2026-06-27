"""Adversarial failure-case suite for the stateful backlash coupling.

Drives edge-case proxy trajectories through the REAL env code path and asserts
invariants every step plus targeted edge-case checks. This is the airtight gate to
run before training (and after any future coupling tweak).

Cases:
   1 NaN guard / R->180°         7 reverse during resume
   2 R=100° symmetric            8 per-finger independence
   3 R<100° degenerate           9 multi-env reset subset
   4 abrupt steps               10 monotonic pure sweeps
   5 deadband chatter           11 settle interaction
   6 double reversal in zone    12 hand-tilt achieved

    python test_backlash_robustness.py --headless --num_envs 4
"""

import argparse
import math
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Backlash coupling robustness suite.")
parser.add_argument("--task", type=str, default="Baoding")
parser.add_argument("--robot", type=str, default="shadowlite")
parser.add_argument("--agent_cfg", type=str, default="rl_only_pt")
parser.add_argument("--num_envs", type=int, default=4)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--resets", type=int, default=8, help="Resets for the hand-tilt DR check.")
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


class Results:
    def __init__(self):
        self.rows = []
    def add(self, cid, name, ok, detail=""):
        self.rows.append((cid, name, bool(ok), detail))
    def report(self):
        print("\n" + "=" * 84)
        print("BACKLASH ROBUSTNESS RESULTS")
        print("=" * 84)
        npass = 0
        for cid, name, ok, detail in self.rows:
            mark = "✓ PASS" if ok else "✗ FAIL"
            if ok:
                npass += 1
            print(f"  C{cid:<2} {name:<34} {mark:<7} {detail}")
        print("-" * 84)
        print(f"  {npass}/{len(self.rows)} checks passed.")
        fails = [f"C{c} {n}" for c, n, ok, _ in self.rows if not ok]
        if fails:
            print("  FAILED: " + "; ".join(fails))
        else:
            print("  ALL PASS — coupling is airtight on these cases.")
        return npass == len(self.rows)


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

    if hasattr(env_cfg, "events"):
        env_cfg.events = None
    if hasattr(env_cfg, "ball_friction_range"):
        env_cfg.ball_friction_range = None
    if hasattr(env_cfg, "reset_joint_pos_noise"):
        env_cfg.reset_joint_pos_noise = 0.0
    settle_default = getattr(env_cfg, "settle_steps", 0)
    if hasattr(env_cfg, "settle_steps"):
        env_cfg.settle_steps = 0      # off by default; case 11 re-enables locally

    writer = Writer(agent_cfg, play=True)
    env_cfg.num_eval_envs = 0
    env = make_env(agent_cfg, env_cfg, writer, args_cli)
    raw = env.env.unwrapped
    raw._get_dones = types.MethodType(_no_dones, raw)
    assert getattr(raw, "couple_asymmetric_backward", False), \
        "couple_asymmetric_backward must be True (check shadowlite cfg)."

    control_names = list(raw.cfg.control_joint_names)
    driver_names  = list(raw.cfg.coupled_joint_map.values())
    sweep_idx = [control_names.index(n) for n in driver_names]
    drv = raw.coupled_driver_indices
    dep = raw.coupled_dependent_indices
    fingers = [n.replace("rh_", "").replace("J2", "") for n in driver_names]

    theta = raw.coupling_theta
    j2u = raw.robot_joint_pos_upper_limits[drv][0].item()
    j1u = raw.robot_joint_pos_upper_limits[dep][0].item()
    lower = raw.robot_joint_pos_lower_limits[drv][0].item()
    j2_top_d, j1_span_d = j2u * DEG, j1u * DEG
    N = raw.num_envs
    EPS = 0.5  # deg tolerance on bounds

    def proxy_for_m(m_rad):
        if m_rad <= j2u:
            return m_rad * theta / j2u
        return theta + (m_rad - j2u) / j1u * (j2u - theta)

    def action_for_m(m_deg, idxs=None):
        a = torch.full((N, len(control_names)), 0.0, dtype=torch.float32, device=env.device)
        val = 2.0 * (proxy_for_m(m_deg * RAD) - lower) / (j2u - lower) - 1.0
        for ix in (idxs if idxs is not None else sweep_idx):
            a[:, ix] = val
        return a

    R = Results()
    inv_bad = {"nan": 0, "j2": 0, "j1": 0, "consist": 0}

    def step_m(m_deg, idxs=None, hold_others_open=False):
        a = action_for_m(m_deg, idxs)
        if hold_others_open:
            held = [i for i in sweep_idx if (idxs is None or i not in idxs)]
            for ix in held:
                a[:, ix] = -1.0
        env.step(a)
        # invariants over env 0, all fingers
        j2c = raw.joint_pos_cmd[0, drv]
        j1c = raw.joint_pos_cmd[0, dep]
        j1s = raw.j1_state[0]
        pm  = raw.prev_m[0]
        if not (torch.isfinite(j2c).all() and torch.isfinite(j1c).all()
                and torch.isfinite(j1s).all() and torch.isfinite(pm).all()):
            inv_bad["nan"] += 1
        if (j2c < -EPS * RAD).any() or (j2c > j2u + EPS * RAD).any():
            inv_bad["j2"] += 1
        if (j1c < -EPS * RAD).any() or (j1c > j1u + EPS * RAD).any():
            inv_bad["j1"] += 1
        if (torch.abs(j1s - j1c) > 1e-4).any():
            inv_bad["consist"] += 1
        m_deg_actual = float(raw.prev_m[0, 0].item()) * DEG
        return (m_deg_actual, j2c.cpu().numpy() * DEG, j1c.cpu().numpy() * DEG)

    def reset_clean(R_deg=None):
        env.reset(hard=True)
        raw.prev_m[:] = 0.0; raw.couple_dir[:] = 1.0; raw.j1_state[:] = 0.0
        raw.couple_frozen_flag[:] = False; raw.couple_frozen_val[:] = 0.0
        if R_deg is not None:
            raw.couple_release[:] = R_deg * RAD

    def ramp_m(m0, m1, idxs=None, hold_open=False, step=2.0):
        n = max(1, int(abs(m1 - m0) / step))
        out = []
        for k in range(1, n + 1):
            out.append(step_m(m0 + (m1 - m0) * k / n, idxs, hold_open))
        return out

    with torch.inference_mode():
        # -- C1 NaN guard / R near 180 --------------------------------------
        for Rv in (179.9, 180.0):
            reset_clean(Rv)
            ramp_m(0, 180); ramp_m(180, 120); ramp_m(120, 180); ramp_m(180, 0)
        R.add(1, "NaN guard / R->180°", inv_bad["nan"] == 0, f"nan steps={inv_bad['nan']}")

        # -- C2 R=100 symmetric (no hysteresis, no freeze) -------------------
        reset_clean(100.0)
        fwd = ramp_m(0, 180)
        frozen_ever = bool(raw.couple_frozen_flag.any().item())
        bwd = ramp_m(180, 0)

        def j1_near(seq, m_target):
            m, _, j1 = min(seq, key=lambda r: abs(r[0] - m_target))
            return j1[0]
        # with R=100 the forward and backward J1(m) curves coincide
        dmatch = max(abs(j1_near(fwd, mt) - j1_near(bwd, mt)) for mt in (120, 140, 160))
        sym_ok = (not frozen_ever) and dmatch < 3.0
        R.add(2, "R=100° symmetric (no freeze)", sym_ok,
              f"frozen_ever={frozen_ever} max|fwd-bwd|={dmatch:.1f}°")

        # -- C3 R<100 degenerate, must not break ----------------------------
        before = dict(inv_bad)
        reset_clean(90.0)
        ramp_m(0, 180); ramp_m(180, 0)
        degen_ok = (inv_bad["nan"] == before["nan"] and inv_bad["j2"] == before["j2"]
                    and inv_bad["j1"] == before["j1"])
        R.add(3, "R<100° degenerate robust", degen_ok, "bounded, no NaN")

        # -- C4 abrupt single-step jumps ------------------------------------
        reset_clean(136.0)
        _, _, j1a = step_m(180.0)        # 0 -> 180 in one step
        step_m(0.0); step_m(180.0); step_m(0.0)
        R.add(4, "abrupt steps bounded", inv_bad["nan"] == 0,
              f"fresh-curl J1={j1a[0]:.0f}° (≈80)")

        # -- C5 deadband chatter near 130° ----------------------------------
        reset_clean(136.0)
        ramp_m(0, 130)
        flips = 0
        prev_fr = bool(raw.couple_frozen_flag[0, 0].item())
        j1_vals = []
        for k in range(60):
            wob = 130.0 + (0.02 if k % 2 == 0 else -0.02)   # << deadband (~0.11°)
            _, _, j1 = step_m(wob)
            j1_vals.append(j1[0])
            fr = bool(raw.couple_frozen_flag[0, 0].item())
            if fr != prev_fr:
                flips += 1
            prev_fr = fr
        j1_range = max(j1_vals) - min(j1_vals)
        R.add(5, "deadband: no freeze flicker", flips == 0 and j1_range < 1.0,
              f"flips={flips} J1range={j1_range:.2f}°")

        # -- C6 double reversal inside the zone -----------------------------
        reset_clean(136.0)
        ramp_m(0, 180)
        ramp_m(180, 115)             # uncurl into zone
        up1 = ramp_m(115, 130)       # re-curl (should be frozen below R)
        ramp_m(130, 110)             # uncurl again
        up2 = ramp_m(110, 128)       # re-curl again (freeze must re-arm)
        # J1 (index 2) should stay ~flat while re-curling below R
        froz1 = max(j[2][0] for j in up1) - min(j[2][0] for j in up1)
        froz2 = max(j[2][0] for j in up2) - min(j[2][0] for j in up2)
        R.add(6, "double reversal freezes", froz1 < 2.5 and froz2 < 2.5,
              f"ΔJ1 up1={froz1:.1f}° up2={froz2:.1f}°")

        # -- C7 reverse during resume ---------------------------------------
        reset_clean(136.0)
        ramp_m(0, 180); ramp_m(180, 120)        # freeze armed at ~20°
        ramp_m(120, 160)                         # resume (rising, m>R)
        # now reverse down; J1 must not exceed demand clamp(m-100)
        bad = 0
        for m_, j2a, j1a in ramp_m(160, 130):
            # J1 must not exceed the falling demand clamp(m-100) (+tol) on the way down
            demand = max(0.0, m_ - 100.0)
            if j1a[0] > demand + 2.0:
                bad += 1
        R.add(7, "reverse during resume sane", bad == 0, "J1 ≤ demand")

        # -- C8 per-finger independence (drive FF only) ---------------------
        reset_clean(136.0)
        seq = ramp_m(0, 180, idxs=[sweep_idx[0]], hold_open=True)
        last_j1 = seq[-1][2]      # j1 array [FF, MF, RF]
        ff_moved = last_j1[0] > 30
        others_quiet = (abs(last_j1[1]) < 3 and abs(last_j1[2]) < 3)
        R.add(8, "per-finger independence", ff_moved and others_quiet,
              f"FF J1={last_j1[0]:.0f}° MF/RF={last_j1[1]:.0f}/{last_j1[2]:.0f}°")

        # -- C9 multi-env reset subset --------------------------------------
        if N >= 2:
            reset_clean(136.0)
            ramp_m(0, 180); ramp_m(180, 120)     # both envs into freeze
            frz_before = raw.couple_frozen_flag.clone()
            raw._reset_idx(torch.tensor([0], device=raw.device))
            e0_clear = (not bool(raw.couple_frozen_flag[0].any())) and \
                       float(raw.j1_state[0].abs().max()) < 1e-4
            e1_keep = bool(frz_before[1].any()) == bool(raw.couple_frozen_flag[1].any())
            r0 = raw.couple_release[0] * DEG
            r0_in = bool(((r0 >= raw.couple_release_lo * DEG - 1e-3) &
                          (r0 <= raw.couple_release_hi * DEG + 1e-3)).all())
            R.add(9, "multi-env reset subset", e0_clear and e1_keep and r0_in,
                  f"e0_clear={e0_clear} e1_keep={e1_keep} R0∈range={r0_in}")
        else:
            R.add(9, "multi-env reset subset", True, "skipped (num_envs<2)")

        # -- C10 monotonic pure sweeps --------------------------------------
        reset_clean(136.0)
        seq = ramp_m(0, 180)
        j2s = [s[1][0] for s in seq]; j1s = [s[2][0] for s in seq]
        mono_up = all(j1s[i + 1] >= j1s[i] - 0.6 for i in range(len(j1s) - 1)) and \
                  all(j2s[i + 1] >= j2s[i] - 0.6 for i in range(len(j2s) - 1))
        seq = ramp_m(180, 0)
        j2s = [s[1][0] for s in seq]; j1s = [s[2][0] for s in seq]
        mono_dn = all(j1s[i + 1] <= j1s[i] + 0.6 for i in range(len(j1s) - 1)) and \
                  all(j2s[i + 1] <= j2s[i] + 0.6 for i in range(len(j2s) - 1))
        R.add(10, "monotonic pure sweeps", mono_up and mono_dn,
              f"up={mono_up} down={mono_dn}")

        # -- C11 settle interaction: backlash state frozen during settle ----
        settle_ok = True
        detail11 = "no settle in cfg"
        if hasattr(raw.cfg, "settle_steps"):
            n_settle = max(10, settle_default)
            raw.cfg.settle_steps = n_settle
            env.reset(hard=True)             # arms settle_counter = n_settle
            j1_during = []
            for _ in range(n_settle):
                step_m(180.0)                # command FULL curl through settle
                j1_during.append(float(raw.j1_state[0].abs().max()) * DEG)
            # while settling, the state must stay frozen near open (J1 ~ 0)
            frozen_during_settle = max(j1_during) < 3.0
            # after settle ends, the policy curl must engage normally
            seq = ramp_m(180, 180)           # one more step post-settle (full curl)
            engaged_after = seq[-1][2][0] > 30.0
            settle_ok = (frozen_during_settle and engaged_after and inv_bad["nan"] == 0)
            detail11 = (f"J1_in_settle≤{max(j1_during):.1f}° engaged_after="
                        f"{seq[-1][2][0]:.0f}°")
            raw.cfg.settle_steps = 0
        R.add(11, "settle freezes backlash state", settle_ok, detail11)

        # -- C12 hand-tilt achieved + varies --------------------------------
        q0 = torch.tensor((0.0, 0.0, -0.7071, 0.7071), device=raw.device)
        lo_t, hi_t = raw.cfg.hand_tilt_range_deg
        tilts = []
        for _ in range(args_cli.resets):
            env.reset(hard=True)
            env.step(torch.zeros((N, len(control_names)), dtype=torch.float32, device=env.device))
            q = raw.robot.data.root_quat_w[0]
            dot = float(torch.clamp(torch.abs((q * q0).sum()), max=1.0))
            tilts.append(2.0 * math.acos(dot) * DEG)
        in_range = all(lo_t - 1.5 <= t <= hi_t + 1.5 for t in tilts)
        varies = (max(tilts) - min(tilts)) > 1.0 if hi_t > lo_t else True
        R.add(12, "hand-tilt DR achieved", in_range and varies,
              f"tilts {min(tilts):.1f}–{max(tilts):.1f}° (range [{lo_t:.0f},{hi_t:.0f}])")

    # global invariants summary
    inv_ok = all(v == 0 for v in inv_bad.values())
    R.add(0, "per-step invariants (all cases)", inv_ok,
          f"nan={inv_bad['nan']} j2oob={inv_bad['j2']} j1oob={inv_bad['j1']} "
          f"consist={inv_bad['consist']}")

    ok = R.report()
    env.close()
    return ok


if __name__ == "__main__":
    main()
    simulation_app.close()

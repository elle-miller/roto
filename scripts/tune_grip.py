"""Sweep stiffness/damping/effort of the coupled finger joints to find values that
let the fingers grip and LIFT the ball.

No policy. Builds the Baoding env ONCE, and for each (stiffness, damping, effort)
combo it sets the gains on the 6 coupled joints (FFJ1/MFJ1/RFJ1 + FFJ2/MFJ2/RFJ2)
at runtime, runs a scripted grip-and-lift (curl FF/MF/RF onto the ball and hold),
and measures how far the ball is lifted and whether it stays up. Results are
ranked and written to CSV so you can pick gains, then put them in
shadow_hand_lite.py.

    python tune_grip.py --task Baoding --robot shadowlite --agent_cfg rl_only_pt --headless
    python tune_grip.py --stiffness 5 15 40 --damping 0.5 2 --effort 0.9 2.0 --headless
"""

import argparse
import csv
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Tune coupled-finger gains for ball lifting.")
parser.add_argument("--task", type=str, default="Baoding")
parser.add_argument("--robot", type=str, default="shadowlite")
parser.add_argument("--agent_cfg", type=str, default="rl_only_pt")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--seed", type=int, default=None)
parser.add_argument("--stiffness", type=float, nargs="+", default=[3.0, 10.0, 30.0])
parser.add_argument("--damping", type=float, nargs="+", default=[0.5, 2.0])
parser.add_argument("--effort", type=float, nargs="+", default=[0.9, 2.0],
                    help="effort_limit_sim (N·m). Keep <= ~2.0 (URDF motor torque) for sim-to-real honesty.")
parser.add_argument("--ramp_steps", type=int, default=60, help="Steps to ramp open->full curl.")
parser.add_argument("--hold_steps", type=int, default=120, help="Steps to hold full curl.")
parser.add_argument("--settle", type=int, default=20, help="Steps to let the ball settle before gripping.")
parser.add_argument("--hold", type=float, default=0.0, help="Action held on non-swept joints.")
parser.add_argument("--lift_thresh", type=float, default=0.01, help="Min ball rise (m) to count as a lift.")
parser.add_argument("--held_speed", type=float, default=0.05, help="Max ball speed (m/s) to count as held.")
parser.add_argument("--ball_mass_g", type=float, default=100.0, help="Ball mass (grams), applied to both balls.")
parser.add_argument("--ball_diameter_in", type=float, default=1.5,
                    help="Ball diameter (inches), applied to both balls. Baoding default is 1.5.")
parser.add_argument("--two_balls", action="store_true", default=False,
                    help="Keep both balls. Default: single ball (ball_2 parked far away, frozen).")
parser.add_argument("--realistic", action="store_true", default=False,
                    help="Keep friction DR, reset noise, and early terminations. Default OFF for a clean, "
                         "deterministic gain-tuning test (these otherwise cause random motion / auto-resets).")
parser.add_argument("--out", type=str, default="tune_grip_results.csv")
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

    # --- ball overrides (mass + diameter) for the lift test ------------------
    mass_kg = args_cli.ball_mass_g / 1000.0
    radius_m = (args_cli.ball_diameter_in / 2.0) * 0.0254          # inches -> m
    for bcfg in (env_cfg.ball_1_cfg, env_cfg.ball_2_cfg):
        bcfg.spawn.mass_props.mass = mass_kg
        bcfg.spawn.radius = radius_m
    if not args_cli.two_balls:
        # park ball_2 far away and freeze it so it can't interfere with the grasp
        env_cfg.ball_2_cfg.spawn.rigid_props.kinematic_enabled = True
        env_cfg.ball_2_cfg.init_state.pos = (5.0, 5.0, 5.0)
    print(f"[INFO] balls: mass={args_cli.ball_mass_g} g, diameter={args_cli.ball_diameter_in} in"
          f"{' (single)' if not args_cli.two_balls else ' (two)'}")

    # --- clean, deterministic test: kill the confounds (default) -------------
    # Friction is re-randomized every reset (ball friction as low as 0.2 -> ball
    # slips), the reset adds joint-pos noise, and the episode auto-resets when a
    # ball falls or the balls go "out of reach" (parking ball_2 far triggers this
    # every step). All of that shows up as "random motion". Turn it off so the
    # gains are the only variable.
    if not args_cli.realistic:
        if hasattr(env_cfg, "events"):
            env_cfg.events = None                 # fingertip/segment friction DR
        if hasattr(env_cfg, "ball_friction_range"):
            env_cfg.ball_friction_range = None    # ball friction DR
        if hasattr(env_cfg, "reset_joint_pos_noise"):
            env_cfg.reset_joint_pos_noise = 0.0   # deterministic reset pose
        print("[INFO] clean test: friction DR + reset noise + early terminations DISABLED "
              "(use --realistic to keep them)")

    writer = Writer(agent_cfg, play=True)
    env_cfg.num_eval_envs = 0
    env = make_env(agent_cfg, env_cfg, writer, args_cli)

    raw = env.env.unwrapped
    raw.couple_gate_j1_on_measured = False   # pure actuator test, no sequencing gate

    if not args_cli.realistic:
        # Never auto-reset mid-trial: the env otherwise resets when a ball falls
        # (z<0.2) or the balls go out of reach (parking ball_2 far triggers it
        # every step) — which makes the hand jump around ("random motion").
        import types

        def _no_dones(self):
            z = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
            return z, z.clone()

        raw._get_dones = types.MethodType(_no_dones, raw)

    control_names = list(raw.cfg.control_joint_names)
    driver_names = list(raw.cfg.coupled_joint_map.values())     # FFJ2, MFJ2, RFJ2
    sweep_idx = [control_names.index(n) for n in driver_names]
    coupled_ids = list(raw.coupled_driver_indices) + list(raw.coupled_dependent_indices)  # 6 joint ids
    has_contact = hasattr(raw, "robot_contact_sensor")

    print(f"[INFO] Tuning joint ids {coupled_ids} "
          f"({driver_names} + {list(raw.cfg.coupled_joint_map.keys())})")
    print(f"[INFO] contact sensor available: {has_contact}")

    def ball_height():
        return (raw.ball_1.data.root_pos_w[0, 2] - raw.scene.env_origins[0, 2]).item()

    def ball_speed():
        return torch.linalg.vector_norm(raw.ball_1.data.root_lin_vel_w[0]).item()

    def finger_force():
        if not has_contact:
            return float("nan")
        f = raw.robot_contact_sensor.data.net_forces_w[0]          # [4, 3] ff/mf/rf/th
        return torch.linalg.vector_norm(f[:3], dim=-1).mean().item()  # mean of FF/MF/RF tips

    def set_gains(k, d, e):
        raw.robot.write_joint_stiffness_to_sim(float(k), joint_ids=coupled_ids)
        raw.robot.write_joint_damping_to_sim(float(d), joint_ids=coupled_ids)
        raw.robot.write_joint_effort_limit_to_sim(float(e), joint_ids=coupled_ids)
        # read back to confirm the runtime write actually applied
        applied = raw.robot.data.joint_stiffness[0, coupled_ids]
        if not torch.allclose(applied, torch.full_like(applied, float(k)), atol=1e-3):
            print(f"  [WARN] stiffness read-back {applied.tolist()} != {k} — runtime set may not stick")

    def step(curl_value):
        action = torch.full((raw.num_envs, len(control_names)), args_cli.hold,
                            dtype=torch.float32, device=env.device)
        action[:, sweep_idx] = curl_value
        env.step(action)

    def run_trial(k, d, e):
        set_gains(k, d, e)
        with torch.inference_mode():
            env.reset(hard=True)
            for _ in range(args_cli.settle):
                step(-1.0)                       # fingers open, let the ball settle
            z0 = ball_height()
            for j in range(args_cli.ramp_steps):  # ramp open -> full curl
                step(-1.0 + 2.0 * j / args_cli.ramp_steps)
            zs, vs, fs = [], [], []
            for _ in range(args_cli.hold_steps):  # hold full curl
                step(1.0)
                zs.append(ball_height()); vs.append(ball_speed()); fs.append(finger_force())
        zs = torch.tensor(zs); vs = torch.tensor(vs); fs = torch.tensor(fs)
        lift_max = float(zs.max().item() - z0)
        lift_final = float(zs[-20:].mean().item() - z0)
        speed_final = float(vs[-20:].mean().item())
        force_mean = float(fs.mean().item())
        return lift_max, lift_final, speed_final, force_mean

    # --- sweep ---------------------------------------------------------------
    rows = []
    combos = [(k, d, e) for k in args_cli.stiffness for d in args_cli.damping for e in args_cli.effort]
    print(f"\n[INFO] Running {len(combos)} combos × "
          f"{args_cli.settle + args_cli.ramp_steps + args_cli.hold_steps} steps each ...\n")
    print(f"{'stiff':>6} {'damp':>5} {'eff':>5} | {'liftMax':>8} {'liftFin':>8} {'ballSpd':>8} {'force':>7} | verdict")
    print("-" * 72)
    for (k, d, e) in combos:
        lmax, lfin, spd, frc = run_trial(k, d, e)
        held = spd < args_cli.held_speed
        lifted = lfin > args_cli.lift_thresh
        verdict = "LIFTS & holds" if (lifted and held) else \
                  ("lifts, not held" if lifted else ("grips, no lift" if frc > 0.3 else "no grip"))
        rows.append(dict(stiffness=k, damping=d, effort=e, lift_max=lmax, lift_final=lfin,
                         ball_speed=spd, finger_force=frc, verdict=verdict))
        print(f"{k:>6.1f} {d:>5.2f} {e:>5.2f} | {lmax:>8.4f} {lfin:>8.4f} {spd:>8.4f} {frc:>7.2f} | {verdict}")

    # --- save + recommend ----------------------------------------------------
    with open(args_cli.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"\n[INFO] Wrote {args_cli.out}")

    held_rows = [r for r in rows if r["ball_speed"] < args_cli.held_speed and r["lift_final"] > args_cli.lift_thresh]
    if held_rows:
        best = max(held_rows, key=lambda r: r["lift_final"])
        print(f"\n[RECOMMEND] stiffness={best['stiffness']} damping={best['damping']} "
              f"effort={best['effort']}  → lifts {best['lift_final']*100:.1f} cm, held "
              f"(force {best['finger_force']:.2f} N).")
    else:
        best = max(rows, key=lambda r: r["finger_force"] if r["finger_force"] == r["finger_force"] else -1)
        print("\n[RECOMMEND] No combo lifted & held. Strongest grip was "
              f"stiffness={best['stiffness']} damping={best['damping']} effort={best['effort']} "
              f"(force {best['finger_force']:.2f} N). Try higher stiffness/effort, or adjust the "
              "lift motion / thumb support (--hold) if it's pressing the ball instead of scooping it.")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()

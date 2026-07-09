#!/usr/bin/env python3
"""Validate identified gains for one joint against its HELD-OUT trajectory.

WHY THIS FILE EXISTS: gains that fit the training trajectories well might
just be memorizing those specific motions. The only real test is a
trajectory the optimizer never saw -- the `held_out` random trajectory
make_trajectories.py set aside for exactly this purpose (see DECISIONS.md;
it's tagged via `meta_held_out`, not by directory location, precisely so
this check can't accidentally run against a training file).

This rolls out optimize.py's selected {Kp, Kd, Fc} against that held-out
command trajectory (one more live Isaac Sim rollout -- the same known stall
risk as collect_rollouts.py applies here, see DECISIONS.md), compares against
the REAL measured position from the held-out log, and reports RMSE in
physical units (rad, rad/s) plus an overlay plot. Run once per joint;
appends/updates that joint's section in results/REPORT.md so the report
accumulates as each joint is validated.
"""

from __future__ import annotations

import argparse
import os
import re
import sys

import numpy as np
from isaaclab.app import AppLauncher

_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _HERE)

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--gains_file", type=str, required=True, help="results/params/<joint>_gains.yaml from optimize.py")
parser.add_argument("--held_out_file", type=str, required=True,
                     help="Collected .npz for the held-out trajectory (meta_held_out=True) for this same joint.")
parser.add_argument("--tolerance_rad", type=float, default=0.1,
                     help="Position RMSE (rad) above which a joint is flagged as failing (see module docstring).")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import yaml  # noqa: E402

from load_data import load_joint_config  # noqa: E402
from sim_rollout import SimRolloutEngine  # noqa: E402


def rmse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean((a - b) ** 2)))


def main():
    with open(args_cli.gains_file, encoding="utf-8") as f:
        gains = yaml.safe_load(f)

    held = np.load(args_cli.held_out_file, allow_pickle=True)
    if "meta_held_out" in held and not bool(held["meta_held_out"]):
        raise ValueError(
            f"{args_cli.held_out_file} has meta_held_out=False -- this looks like a TRAINING "
            "file, not the held-out one. Refusing to validate against it (that would defeat "
            "the point of held-out validation)."
        )

    joint_idx = int(held["joint_idx"])
    joint_name = str(held["joint_name"])
    if joint_name != gains["joint_name"]:
        raise ValueError(f"gains are for {gains['joint_name']!r} but held_out_file is for {joint_name!r}.")

    cmd = held["cmd"]
    real_q = held["actual_pos"][:, joint_idx]
    real_v = held["actual_vel"][:, joint_idx]
    # Must match what was used to collect `held` -- see collect_rollouts.py's
    # identical comment (DECISIONS.md, 2026-07-08: non-excited fingers parked
    # out of the way, not held at zero).
    default_pose = np.asarray(held["default_pose"], dtype=np.float32) if "default_pose" in held else None

    joint_cfg = load_joint_config()
    engine = SimRolloutEngine(joint_cfg, simulation_app=simulation_app)
    engine.set_gains(joint_name, kp=gains["kp"], kd=gains["kd"], fc=gains["fc"])
    engine.reset(default_pose)
    sim_q = engine.rollout(joint_idx, cmd, default_pose)
    engine.close()

    dt = 1.0 / joint_cfg["control_rate_hz"]
    sim_v = np.gradient(sim_q, dt)

    pos_rmse = rmse(sim_q, real_q)
    vel_rmse = rmse(sim_v, real_v)
    passed = pos_rmse <= args_cli.tolerance_rad

    print(f"[validate] {joint_name}: pos_rmse={pos_rmse:.5f} rad  vel_rmse={vel_rmse:.5f} rad/s  "
          f"tolerance={args_cli.tolerance_rad} rad  {'PASS' if passed else 'FAIL'}")

    # Overlay plot
    t = np.arange(len(cmd)) * dt
    fig, (ax_pos, ax_vel) = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
    ax_pos.plot(t, cmd, "k--", lw=1, label="commanded")
    ax_pos.plot(t, real_q, lw=1.2, label="real (held-out)")
    ax_pos.plot(t, sim_q, lw=1.2, label=f"sim (Kp={gains['kp']:.2f} Kd={gains['kd']:.2f} Fc={gains['fc']:.3f})")
    ax_pos.set_ylabel("position (rad)")
    ax_pos.set_title(f"{joint_name} — held-out validation (pos RMSE={pos_rmse:.4f} rad)")
    ax_pos.legend(fontsize=8)

    ax_vel.plot(t, real_v, lw=1.2, label="real")
    ax_vel.plot(t, sim_v, lw=1.2, label="sim")
    ax_vel.set_ylabel("velocity (rad/s)")
    ax_vel.set_xlabel("t (s)")
    ax_vel.set_title(f"velocity RMSE={vel_rmse:.4f} rad/s")
    ax_vel.legend(fontsize=8)

    fig.tight_layout()
    plot_dir = os.path.join(_PROJECT_ROOT, "results", "plots")
    os.makedirs(plot_dir, exist_ok=True)
    plot_path = os.path.join(plot_dir, f"{joint_name}_validation.png")
    fig.savefig(plot_path, dpi=110)
    plt.close(fig)
    print(f"[validate] Saved {plot_path}")

    _update_report(joint_name, gains, pos_rmse, vel_rmse, args_cli.tolerance_rad, passed, plot_path)


def _update_report(joint_name, gains, pos_rmse, vel_rmse, tolerance, passed, plot_path):
    """Append or replace this joint's section in results/REPORT.md.

    Written in plain language per this project's standing communication
    convention -- REPORT.md is meant to be readable by someone who wasn't
    part of building this, not just a dump of numbers.
    """
    report_path = os.path.join(_PROJECT_ROOT, "results", "REPORT.md")
    verdict = "PASSED" if passed else "FAILED"
    caveat = (
        ""
        if passed
        else (
            "\n> This joint did not meet the tolerance. Before assuming the gains are wrong, "
            "consider whether this joint has strong backlash/stiction on fast reversals -- "
            "that's a known limitation of a pure PD+Coulomb-friction model (see the project "
            "plan's own \"gotchas\"), and the signal to consider a residual-network correction "
            "for this joint specifically, not to keep re-tuning PD gains indefinitely.\n"
        )
    )
    section = (
        f"## {joint_name}\n\n"
        f"**Result: {verdict}** (tolerance: {tolerance} rad position RMSE)\n\n"
        f"- Identified gains: Kp={gains['kp']:.4f} N·m/rad, Kd={gains['kd']:.4f} N·m·s/rad, "
        f"Fc={gains['fc']:.4f} N·m (Fv fixed at 0 — see DECISIONS.md)\n"
        f"- Position RMSE on held-out trajectory: {pos_rmse:.5f} rad\n"
        f"- Velocity RMSE on held-out trajectory: {vel_rmse:.5f} rad/s\n"
        f"- Plot: `{os.path.relpath(plot_path, _PROJECT_ROOT)}`\n"
        f"{caveat}\n"
    )

    if os.path.exists(report_path):
        with open(report_path, encoding="utf-8") as f:
            content = f.read()
    else:
        content = (
            "# Validation Report\n\n"
            "Per-joint results of rolling out `optimize.py`'s selected gains against each "
            "joint's held-out trajectory (never used for fitting). A joint failing its "
            "tolerance is not automatically a bug in the gains -- see the note under any "
            "failed joint below.\n\n"
        )

    heading = f"## {joint_name}\n"
    pattern = re.compile(rf"{re.escape(heading)}.*?(?=\n## |\Z)", re.DOTALL)
    if pattern.search(content):
        content = pattern.sub(section, content)
    else:
        content = content.rstrip() + "\n\n" + section

    with open(report_path, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"[validate] Updated {report_path}")


if __name__ == "__main__":
    main()

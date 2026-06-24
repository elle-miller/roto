"""Diagnose the coupled-finger mimic (J1 follows J2) from a sim recording.

Loads a record_policy.py npz and, for each coupled finger (FF/MF/RF), checks how
the J1 mimic relates to its J2 driver.

IDEAL behaviour = SEQUENTIAL coupling (the sim "theta-split" in
_handle_coupled_joints, roto_env.py): one curl proxy per finger drives J2 to its
90 deg max FIRST, and only once J2 is maxed does J1 start moving. The combined
hardware actuator (ffj0) spans 2x90 = 180 deg, so the J2->J1 handover at ffj0=90
deg lands at proxy = 45 deg = theta = 0.785 rad. This tool checks whether the sim
actually follows that ideal sequential law, and overlays a proportional (J1=J2)
curve only as a NOT-ideal contrast.

Usage:
    python plot_mimic_check.py --rec mimic_recording.npz
    python plot_mimic_check.py --rec mimic_recording.npz --out mimic.png
    python plot_mimic_check.py --rec mimic_recording.npz --theta 0.785 --j2_upper 1.5708 --j1_upper 1.5708
"""

import argparse

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# Coupled finger: J2 driver -> its J1 mimic (matches coupled_joint_map in shadowlite.py).
COUPLED = {"rh_FFJ2": "rh_FFJ1", "rh_MFJ2": "rh_MFJ1", "rh_RFJ2": "rh_RFJ1"}


def theta_split_law(proxy, theta, j2u, j1u):
    """Mirror roto_env._handle_coupled_joints: proxy -> (j2_cmd, j1_cmd)."""
    j2 = np.clip(proxy * (j2u / theta), 0.0, j2u)
    j1 = np.clip((proxy - theta) / (j2u - theta) * j1u, 0.0, j1u)
    return j2, j1


def main():
    parser = argparse.ArgumentParser(description="Check the J1/J2 coupled-finger mimic in sim.")
    parser.add_argument("--rec", required=True, help="record_policy.py npz")
    parser.add_argument("--out", default="mimic.png", help="Output figure base (PNG or PDF).")
    parser.add_argument("--theta", type=float, default=0.785, help="coupling_theta (rad).")
    parser.add_argument("--j2_upper", type=float, default=1.5708, help="J2 upper limit (rad).")
    parser.add_argument("--j1_upper", type=float, default=1.5708, help="J1 upper limit (rad).")
    parser.add_argument("--j1_eps", type=float, default=0.01,
                        help="J1 threshold (rad) for the 'J1 moves early' check.")
    args = parser.parse_args()

    d = np.load(args.rec, allow_pickle=True)
    cmd = d["joint_pos_cmd"]                       # [T, 16]
    act = d["joint_pos"]                           # [T, 16]
    names = [str(n) for n in d["actuated_names"]]  # 16 names for the columns
    rl_dt = float(d["rl_dt"]) if "rl_dt" in d.files else 1.0 / 60.0
    ep_ends = [int(e) for e in d["episode_ends"]] if "episode_ends" in d.files else []
    col = {n: i for i, n in enumerate(names)}

    missing = [j for pair in COUPLED.items() for j in pair if j not in col]
    if missing:
        raise ValueError(f"Recording is missing coupled joints {missing}. "
                         f"actuated_names = {names}")

    theta, j2u, j1u = args.theta, args.j2_upper, args.j1_upper
    fingers = list(COUPLED.keys())
    T = cmd.shape[0]
    t = np.arange(T) * rl_dt

    def shade(ax):
        ep0 = 0
        for ei, ee in enumerate(ep_ends):
            ax.axvspan(ep0 * rl_dt, ee * rl_dt, alpha=0.05, color=plt.cm.tab10.colors[ei % 10])
            ep0 = ee + 1

    # -------------------------------------------------------------------------
    # Figure 1: time series of J2 & J1 (commanded + actual) per finger
    # -------------------------------------------------------------------------
    fig1, axes = plt.subplots(len(fingers), 1, figsize=(11, 2.6 * len(fingers)), sharex=True)
    if len(fingers) == 1:
        axes = [axes]
    fig1.suptitle("Coupled-finger joints over time: J2 driver & J1 mimic", fontsize=13, fontweight="bold")
    for ax, j2name in zip(axes, fingers):
        j1name = COUPLED[j2name]
        ax.plot(t, cmd[:, col[j2name]], color="steelblue", lw=1.3, ls="--", label="J2 cmd")
        ax.plot(t, act[:, col[j2name]], color="steelblue", lw=1.0, label="J2 actual")
        ax.plot(t, cmd[:, col[j1name]], color="crimson", lw=1.3, ls="--", label="J1 cmd")
        ax.plot(t, act[:, col[j1name]], color="crimson", lw=1.0, label="J1 actual")
        ax.axhline(j2u, color="grey", lw=0.6, ls=":")
        shade(ax)
        ax.set_title(f"{j2name} (J2)  /  {j1name} (J1)", fontsize=9, pad=2)
        ax.set_ylabel("rad", fontsize=8)
        ax.grid(True, lw=0.3, alpha=0.5)
    axes[-1].set_xlabel("time (s)", fontsize=9)
    fig1.legend(handles=[
        Line2D([0], [0], color="steelblue", lw=1.5, ls="--", label="J2 cmd"),
        Line2D([0], [0], color="steelblue", lw=1.5, label="J2 actual"),
        Line2D([0], [0], color="crimson", lw=1.5, ls="--", label="J1 cmd"),
        Line2D([0], [0], color="crimson", lw=1.5, label="J1 actual"),
    ], loc="upper right", fontsize=8, ncol=2)
    fig1.tight_layout(rect=[0, 0, 1, 0.96])

    # -------------------------------------------------------------------------
    # Figure 2: J1 vs J2, with sim theta-split law and proportional reference
    # -------------------------------------------------------------------------
    sweep = np.linspace(0.0, j2u, 200)               # proxy sweep
    j2_curve, j1_curve = theta_split_law(sweep, theta, j2u, j1u)

    fig2, axes2 = plt.subplots(1, len(fingers), figsize=(4.2 * len(fingers), 4.2))
    if len(fingers) == 1:
        axes2 = [axes2]
    fig2.suptitle("J1 mimic vs J2 driver  (actual positions)", fontsize=13, fontweight="bold")
    for ax, j2name in zip(axes2, fingers):
        j1name = COUPLED[j2name]
        ax.scatter(act[:, col[j2name]], act[:, col[j1name]], s=6, color="black",
                   alpha=0.4, label="actual (sim)")
        ax.plot(j2_curve, j1_curve, color="darkorange", lw=2.0, label="ideal: sequential (θ-split)")
        lim = max(j2u, j1u)
        ax.plot([0, lim], [0, lim], color="seagreen", lw=1.2, ls=":", label="proportional (NOT ideal)")
        ax.set_title(f"{j2name} → {j1name}", fontsize=9)
        ax.set_xlabel("J2 (rad)", fontsize=8)
        ax.set_ylabel("J1 (rad)", fontsize=8)
        ax.grid(True, lw=0.3, alpha=0.5)
        ax.set_aspect("equal", adjustable="box")
        ax.legend(fontsize=7, loc="upper left")
    fig2.tight_layout(rect=[0, 0, 1, 0.94])

    # -------------------------------------------------------------------------
    # Save
    # -------------------------------------------------------------------------
    stem = args.out.rsplit(".", 1)[0]
    ext = args.out.rsplit(".", 1)[1] if "." in args.out else "png"
    out1, out2 = f"{stem}_timeseries.{ext}", f"{stem}_j1_vs_j2.{ext}"
    fig1.savefig(out1, dpi=150, bbox_inches="tight")
    fig2.savefig(out2, dpi=150, bbox_inches="tight")
    print(f"Saved:\n  {out1}\n  {out2}")

    # -------------------------------------------------------------------------
    # Printed per-finger checks
    # -------------------------------------------------------------------------
    print("\n--- Mimic checks per coupled finger (ideal = sequential θ-split) ---")
    print(f"{'finger':>16} | {'cmd|J1-ideal|':>14} | {'act|J1-ideal|':>14} | {'mean|J1act-J1cmd|':>17} | {'J1 moves early':>14}")
    print("-" * 90)
    for j2name in fingers:
        j1name = COUPLED[j2name]
        cmd_j2, cmd_j1 = cmd[:, col[j2name]], cmd[:, col[j1name]]
        act_j2, act_j1 = act[:, col[j2name]], act[:, col[j1name]]
        # Ideal sequential J1 expected from each step's proxy (2*proxy = J2+J1).
        cmd_proxy = 0.5 * (cmd_j2 + cmd_j1)
        _, cmd_j1_ideal = theta_split_law(cmd_proxy, theta, j2u, j1u)
        act_proxy = 0.5 * (act_j2 + act_j1)
        _, act_j1_ideal = theta_split_law(act_proxy, theta, j2u, j1u)
        cmd_err = np.max(np.abs(cmd_j1 - cmd_j1_ideal))      # does the env COMMAND the ideal split
        act_err = np.mean(np.abs(act_j1 - act_j1_ideal))     # does the hand ACHIEVE the ideal split
        track = np.mean(np.abs(act_j1 - cmd_j1))             # soft-PD tracking of J1
        early = np.mean((act_j1 > args.j1_eps) & (act_j2 < 0.95 * j2u))  # J1 before J2 maxed
        print(f"{j2name:>16} | {cmd_err:>14.4f} | {act_err:>14.4f} | {track:>17.4f} | {early * 100:>13.1f}%")

    print("\nInterpretation (ideal = J1 stays 0 until J2 reaches 90°, then J1 ramps):")
    print("  cmd|J1-ideal| ≈ 0    → env COMMANDS the ideal sequential split (correct by design).")
    print("  act|J1-ideal| small  → the hand ACHIEVES sequential coupling (what you want).")
    print("  act|J1-ideal| large  → actual J1/J2 deviate from sequential (mimic not behaving).")
    print("  J1 moves early       → % of steps J1 has moved while J2 < 90° (ideal ≈ 0%;")
    print("                          high = J1 leading/with J2, i.e. not sequential).")
    print("  mean|J1act-J1cmd|    → how far actual J1 lags its command (soft-PD tracking).")


if __name__ == "__main__":
    main()

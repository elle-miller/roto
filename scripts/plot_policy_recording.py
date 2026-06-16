"""Plot commanded vs actual joint positions from a policy recording.

Usage:
    python plot_policy_recording.py policy_recording.npz
    python plot_policy_recording.py policy_recording.npz --out baoding_joints.pdf
"""

import argparse
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("npz", help="Path to .npz saved by record_policy.py")
    parser.add_argument("--out", default=None, help="Output file (PDF or PNG). Default: <npz_stem>.pdf")
    args = parser.parse_args()

    data = np.load(args.npz, allow_pickle=True)
    actions      = data["actions"]           # [T, 13]
    cmd          = data["joint_pos_cmd"]     # [T, 16]
    actual       = data["joint_pos"]         # [T, 16]
    act_names    = list(data["actuated_names"])
    ctrl_names   = list(data["control_names"])
    ep_ends      = list(data["episode_ends"])
    rl_dt        = float(data["rl_dt"])

    T = actions.shape[0]
    t = np.arange(T) * rl_dt

    n_joints = cmd.shape[1]  # 16 actuated joints

    # -------------------------------------------------------------------------
    # Figure 1: commanded vs actual for all 16 actuated joints
    # -------------------------------------------------------------------------
    ncols = 4
    nrows = (n_joints + ncols - 1) // ncols
    fig1, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4, nrows * 2.8), sharex=True)
    fig1.suptitle("Policy commanded vs actual joint positions", fontsize=13, fontweight="bold")
    axes_flat = axes.flatten()

    ep_colors = plt.cm.tab10.colors

    for j, ax in enumerate(axes_flat):
        if j >= n_joints:
            ax.set_visible(False)
            continue

        ax.plot(t, cmd[:, j],    color="steelblue",  lw=1.2, ls="--", label="commanded")
        ax.plot(t, actual[:, j], color="darkorange",  lw=1.0, ls="-",  label="actual")

        # episode boundary lines
        ep0 = 0
        for ei, ee in enumerate(ep_ends):
            color = ep_colors[ei % len(ep_colors)]
            ax.axvspan(ep0 * rl_dt, ee * rl_dt, alpha=0.06, color=color)
            if ee < T:
                ax.axvline(ee * rl_dt, color=color, lw=0.8, ls=":")
            ep0 = ee + 1

        ax.set_title(act_names[j], fontsize=8, pad=2)
        ax.set_ylabel("rad", fontsize=7)
        ax.tick_params(labelsize=7)
        ax.grid(True, lw=0.3, alpha=0.5)

    # shared x label
    for ax in axes_flat[(nrows - 1) * ncols:]:
        ax.set_xlabel("time (s)", fontsize=8)

    legend_elements = [
        Line2D([0], [0], color="steelblue", lw=1.5, ls="--", label="commanded"),
        Line2D([0], [0], color="darkorange", lw=1.5, ls="-",  label="actual"),
    ]
    fig1.legend(handles=legend_elements, loc="upper right", fontsize=9)
    fig1.tight_layout(rect=[0, 0, 1, 0.97])

    # -------------------------------------------------------------------------
    # Figure 2: tracking error (cmd - actual) per joint
    # -------------------------------------------------------------------------
    fig2, axes2 = plt.subplots(nrows, ncols, figsize=(ncols * 4, nrows * 2.8), sharex=True)
    fig2.suptitle("Joint tracking error  (commanded − actual)  [rad]", fontsize=13, fontweight="bold")
    axes2_flat = axes2.flatten()

    error = cmd - actual

    for j, ax in enumerate(axes2_flat):
        if j >= n_joints:
            ax.set_visible(False)
            continue

        ax.plot(t, error[:, j], color="crimson", lw=0.9)
        ax.axhline(0, color="k", lw=0.5, ls="--")

        ep0 = 0
        for ei, ee in enumerate(ep_ends):
            color = ep_colors[ei % len(ep_colors)]
            ax.axvspan(ep0 * rl_dt, ee * rl_dt, alpha=0.06, color=color)
            if ee < T:
                ax.axvline(ee * rl_dt, color=color, lw=0.8, ls=":")
            ep0 = ee + 1

        ax.set_title(act_names[j], fontsize=8, pad=2)
        ax.set_ylabel("rad", fontsize=7)
        ax.tick_params(labelsize=7)
        ax.grid(True, lw=0.3, alpha=0.5)

    for ax in axes2_flat[(nrows - 1) * ncols:]:
        ax.set_xlabel("time (s)", fontsize=8)

    fig2.tight_layout(rect=[0, 0, 1, 0.97])

    # -------------------------------------------------------------------------
    # Figure 3: raw policy actions (normalised [-1, 1]) for 13 control joints
    # -------------------------------------------------------------------------
    n_ctrl = actions.shape[1]
    ncols3 = 4
    nrows3 = (n_ctrl + ncols3 - 1) // ncols3
    fig3, axes3 = plt.subplots(nrows3, ncols3, figsize=(ncols3 * 4, nrows3 * 2.8), sharex=True)
    fig3.suptitle("Raw policy actions (normalised, −1 … +1)", fontsize=13, fontweight="bold")
    axes3_flat = axes3.flatten()

    for j, ax in enumerate(axes3_flat):
        if j >= n_ctrl:
            ax.set_visible(False)
            continue

        ax.plot(t, actions[:, j], color="mediumseagreen", lw=0.9)
        ax.axhline(0, color="k", lw=0.5, ls="--")
        ax.set_ylim(-1.1, 1.1)

        ep0 = 0
        for ei, ee in enumerate(ep_ends):
            color = ep_colors[ei % len(ep_colors)]
            ax.axvspan(ep0 * rl_dt, ee * rl_dt, alpha=0.06, color=color)
            if ee < T:
                ax.axvline(ee * rl_dt, color=color, lw=0.8, ls=":")
            ep0 = ee + 1

        ax.set_title(ctrl_names[j], fontsize=8, pad=2)
        ax.tick_params(labelsize=7)
        ax.grid(True, lw=0.3, alpha=0.5)

    for ax in axes3_flat[(nrows3 - 1) * ncols3:]:
        ax.set_xlabel("time (s)", fontsize=8)

    fig3.tight_layout(rect=[0, 0, 1, 0.97])

    # -------------------------------------------------------------------------
    # Save
    # -------------------------------------------------------------------------
    stem = args.npz.replace(".npz", "")
    out1 = args.out or f"{stem}_cmd_vs_actual.png"
    out2 = f"{stem}_error.png"
    out3 = f"{stem}_raw_actions.png"

    fig1.savefig(out1, dpi=150, bbox_inches="tight")
    fig2.savefig(out2, dpi=150, bbox_inches="tight")
    fig3.savefig(out3, dpi=150, bbox_inches="tight")

    print(f"Saved:\n  {out1}\n  {out2}\n  {out3}")

    # also print quick stats
    print("\n--- Tracking error summary (mean |error| per joint, in rad) ---")
    mean_abs_err = np.abs(error).mean(axis=0)
    for name, err in zip(act_names, mean_abs_err):
        print(f"  {name:12s}  {err:.4f} rad  ({np.degrees(err):.2f}°)")


if __name__ == "__main__":
    main()

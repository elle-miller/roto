"""Compare sim vs hardware joint trajectories.

Loads a sim recording (record_policy.py) and a hardware replay (replay_traj_hw.py)
and overlays, per joint. The plot adapts to the replay mode (stored in the hw npz):

    command  replay → 3 lines: command, sim-actual, hw-actual
                      (same command into both → gap = hw-actual − sim-actual)
    sim_pos  replay → 2 lines: sim joint pos (hw target), hw-actual
                      (hw told to follow sim's positions → gap = hw-actual − sim pos)

Everything is in 13 policy-order "proxy" space; coupled fingers (FF/MF/RF) are
the mean of their J2 driver and J1 mimic, matching how the command is issued.

Usage:
    python plot_traj_compare.py --sim policy_recording.npz --hw traj_hw_command_20260623.npz
    python plot_traj_compare.py --sim policy_recording.npz --hw traj_hw_sim_pos_20260623.npz
"""

import argparse

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

POLICY_JOINT_ORDER = [
    "rh_FFJ4", "rh_MFJ4", "rh_RFJ4", "rh_THJ5",
    "rh_FFJ3", "rh_MFJ3", "rh_RFJ3", "rh_THJ4",
    "rh_FFJ2", "rh_MFJ2", "rh_RFJ2",
    "rh_THJ2", "rh_THJ1",
]
COUPLED_DEP = {"rh_FFJ2": "rh_FFJ1", "rh_MFJ2": "rh_MFJ1", "rh_RFJ2": "rh_RFJ1"}


def to_proxy13(data16, actuated_names):
    """[T,16] actuated-order -> [T,13] policy-order proxy (coupled = mean(J2,J1))."""
    actuated_names = [str(n) for n in actuated_names]
    col = {n: i for i, n in enumerate(actuated_names)}
    out = []
    for jn in POLICY_JOINT_ORDER:
        if jn in COUPLED_DEP:
            out.append(0.5 * (data16[:, col[jn]] + data16[:, col[COUPLED_DEP[jn]]]))
        else:
            out.append(data16[:, col[jn]])
    return np.stack(out, axis=1).astype(np.float32)


def shade_episodes(ax, ep_ends, T, rl_dt):
    colors = plt.cm.tab10.colors
    ep0 = 0
    for ei, ee in enumerate(ep_ends):
        c = colors[ei % len(colors)]
        ax.axvspan(ep0 * rl_dt, ee * rl_dt, alpha=0.06, color=c)
        if ee < T:
            ax.axvline(ee * rl_dt, color=c, lw=0.8, ls=":")
        ep0 = ee + 1


def main():
    parser = argparse.ArgumentParser(description="Plot sim (and optionally hardware) joint trajectories.")
    parser.add_argument("--sim", required=True, help="record_policy.py npz (sim)")
    parser.add_argument("--hw", default=None,
                        help="replay_traj_hw.py npz (hardware). Omit to plot sim command vs sim output only.")
    parser.add_argument("--out", default="traj_compare.png", help="Output figure (PNG or PDF).")
    parser.add_argument("--gap", action="store_true", default=False,
                        help="Also save the hw-actual minus sim-actual gap figure (hardware mode only).")
    args = parser.parse_args()

    sim = np.load(args.sim, allow_pickle=True)

    # Sim side: reconstruct proxy command + actual from the 16-DOF recording.
    sim_cmd = to_proxy13(sim["joint_pos_cmd"], sim["actuated_names"])
    sim_act = to_proxy13(sim["joint_pos"], sim["actuated_names"])
    rl_dt = float(sim["rl_dt"]) if "rl_dt" in sim.files else 1.0 / 60.0
    ep_ends = [int(e) for e in sim["episode_ends"]] if "episode_ends" in sim.files else []

    if args.hw is None:
        # ---- sim-only: policy command vs sim output ------------------------
        replay_source = "sim_only"
        hw_act = None
        T = sim_cmd.shape[0]
        sim_cmd, sim_act = sim_cmd[:T], sim_act[:T]
        t = np.arange(T) * rl_dt
        title1 = "Policy command vs sim output"
        lines = [(sim_cmd, "command (policy)", "black", "--"),
                 (sim_act, "sim output", "steelblue", "-")]
    else:
        # ---- hardware comparison ------------------------------------------
        hw = np.load(args.hw, allow_pickle=True)
        hw_names = [str(n) for n in hw["control_names"]]
        hw_ref = hw["replayed13"] if "replayed13" in hw.files else hw["cmd13"]  # back-compat
        hw_act = hw["hw_actual13"]
        if hw_names != POLICY_JOINT_ORDER:
            idx = [hw_names.index(jn) for jn in POLICY_JOINT_ORDER]
            hw_ref = hw_ref[:, idx]
            hw_act = hw_act[:, idx]
        replay_source = str(hw["replay_source"]) if "replay_source" in hw.files else "command"

        # Align lengths (hardware replays exactly T steps, but be defensive).
        T = min(sim_cmd.shape[0], hw_act.shape[0])
        if sim_cmd.shape[0] != hw_act.shape[0]:
            print(f"[warn] length mismatch sim={sim_cmd.shape[0]} hw={hw_act.shape[0]}; truncating to {T}")
        sim_cmd, sim_act = sim_cmd[:T], sim_act[:T]
        hw_ref, hw_act = hw_ref[:T], hw_act[:T]
        t = np.arange(T) * rl_dt

        # The reference the hardware was told to follow depends on the mode:
        #   command : ref = policy command   → compare hw-actual vs sim-actual
        #   sim_pos : ref = sim joint pos     → compare hw-actual vs that same sim pos
        if replay_source == "sim_pos":
            sim_reference = sim_act
            ref_label = "sim joint pos (hw target)"
            title1 = "Hardware tracking of sim joint positions"
            gap_label = "hw-actual − sim joint pos"
            lines = [(sim_reference, "sim joint pos (target)", "black", "--"),
                     (hw_act, "hw actual", "darkorange", "-")]
        else:
            sim_reference = sim_cmd
            ref_label = "command"
            title1 = "Same command: sim-actual vs hw-actual"
            gap_label = "hw-actual − sim-actual"
            lines = [(sim_act, "sim actual", "steelblue", "-"),
                     (hw_act, "hw actual", "darkorange", "-")]

        ref_mismatch = float(np.abs(sim_reference - hw_ref).max())
        if ref_mismatch > 1e-3:
            print(f"[warn] hw reference ({ref_label}) differs from sim by up to {ref_mismatch:.4f} rad "
                  "(clipping or a mismatched recording pair?).")

    # Gap (hardware-actual vs sim-actual); only meaningful with hardware.
    gap = (hw_act - sim_act) if hw_act is not None else None
    n = len(POLICY_JOINT_ORDER)
    ncols, nrows = 4, (n + 3) // 4

    # -------------------------------------------------------------------------
    # Figure 1: per-joint trajectories
    # -------------------------------------------------------------------------
    fig1, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4, nrows * 2.8), sharex=True)
    fig1.suptitle(f"Joint trajectories ({replay_source} replay): {title1}", fontsize=13, fontweight="bold")
    axf = axes.flatten()
    for j, ax in enumerate(axf):
        if j >= n:
            ax.set_visible(False)
            continue
        for arr, _lbl, color, ls in lines:
            ax.plot(t, arr[:, j], color=color, lw=1.1, ls=ls)
        shade_episodes(ax, ep_ends, T, rl_dt)
        ax.set_title(POLICY_JOINT_ORDER[j], fontsize=8, pad=2)
        ax.set_ylabel("rad", fontsize=7)
        ax.tick_params(labelsize=7)
        ax.grid(True, lw=0.3, alpha=0.5)
    for ax in axf[(nrows - 1) * ncols:]:
        ax.set_xlabel("time (s)", fontsize=8)
    fig1.legend(handles=[Line2D([0], [0], color=c, lw=1.5, ls=ls, label=lbl)
                         for _a, lbl, c, ls in lines], loc="upper right", fontsize=9)
    fig1.tight_layout(rect=[0, 0, 1, 0.97])

    # -------------------------------------------------------------------------
    # Save
    # -------------------------------------------------------------------------
    stem = args.out.rsplit(".", 1)[0]
    ext = args.out.rsplit(".", 1)[1] if "." in args.out else "png"
    out1 = f"{stem}_{replay_source}_traj.{ext}"
    fig1.savefig(out1, dpi=150, bbox_inches="tight")
    saved = [out1]

    # Optional Figure 2: hw-actual − sim-actual gap, per joint (only with --gap)
    if args.gap and gap is not None:
        fig2, axes2 = plt.subplots(nrows, ncols, figsize=(ncols * 4, nrows * 2.8), sharex=True)
        fig2.suptitle(f"Sim-to-real gap  ({gap_label})  [rad]", fontsize=13, fontweight="bold")
        axf2 = axes2.flatten()
        for j, ax in enumerate(axf2):
            if j >= n:
                ax.set_visible(False)
                continue
            ax.plot(t, gap[:, j], color="crimson", lw=0.9)
            ax.axhline(0, color="k", lw=0.5, ls="--")
            shade_episodes(ax, ep_ends, T, rl_dt)
            ax.set_title(POLICY_JOINT_ORDER[j], fontsize=8, pad=2)
            ax.set_ylabel("rad", fontsize=7)
            ax.tick_params(labelsize=7)
            ax.grid(True, lw=0.3, alpha=0.5)
        for ax in axf2[(nrows - 1) * ncols:]:
            ax.set_xlabel("time (s)", fontsize=8)
        fig2.tight_layout(rect=[0, 0, 1, 0.97])
        out2 = f"{stem}_{replay_source}_gap.{ext}"
        fig2.savefig(out2, dpi=150, bbox_inches="tight")
        saved.append(out2)

    print("Saved:\n  " + "\n  ".join(saved))

    # -------------------------------------------------------------------------
    # Summary table
    # -------------------------------------------------------------------------
    if replay_source == "sim_only":
        sim_track = np.abs(sim_act - sim_cmd).mean(axis=0)
        print("\n--- Sim tracking error: command vs sim output (mean abs, rad) ---")
        print(f"{'joint':>10} | {'sim track':>10}")
        print("-" * 25)
        for jn, a in zip(POLICY_JOINT_ORDER, sim_track):
            print(f"{jn:>10} | {a:>10.4f}")
        print(f"\n{'mean':>10} | {sim_track.mean():>10.4f}")
        print(f"\nsteps={T}  duration={T * rl_dt:.1f}s  rl_dt={rl_dt:.4f}s")
        return

    s2r = np.abs(gap).mean(axis=0)
    if replay_source == "sim_pos":
        print("\n--- Hardware tracking of sim joint positions (mean abs, rad) ---")
        print(f"{'joint':>10} | {'hw vs sim_pos':>14}")
        print("-" * 30)
        for jn, c in zip(POLICY_JOINT_ORDER, s2r):
            print(f"{jn:>10} | {c:>14.4f}")
        print(f"\n{'mean':>10} | {s2r.mean():>14.4f}")
    else:
        sim_track = np.abs(sim_act - sim_cmd).mean(axis=0)
        hw_track = np.abs(hw_act - sim_cmd).mean(axis=0)
        print("\n--- Per-joint tracking error & sim-to-real gap (mean abs, rad) ---")
        print(f"{'joint':>10} | {'sim track':>9} {'hw track':>9} | {'sim2real gap':>12}")
        print("-" * 50)
        for jn, a, b, c in zip(POLICY_JOINT_ORDER, sim_track, hw_track, s2r):
            print(f"{jn:>10} | {a:>9.4f} {b:>9.4f} | {c:>12.4f}")
        print(f"\n{'mean':>10} | {sim_track.mean():>9.4f} {hw_track.mean():>9.4f} | {s2r.mean():>12.4f}")
    print(f"\nsteps={T}  duration={T * rl_dt:.1f}s  rl_dt={rl_dt:.4f}s")


if __name__ == "__main__":
    main()

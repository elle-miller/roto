#!/usr/bin/env python3
"""
Overlay plots for the 4-run replay-motion-test comparison
(balls/noballs x position/trajectory control).

Plots only -- no derived/coupled signals, no printed conclusions.
Every 13-dim series is actuator space (actuator_order); every 16-dim
series is joint space (joint_order). Cross-correlation alignment only
shifts the time axis for cross-run overlays; it never touches values.
"""
import os

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter1d

BASE = os.path.dirname(os.path.abspath(__file__))
PLOTS = os.path.join(BASE, "plots")

# Display margins around the detected motion, in seconds.
PAD_BEFORE = 1.0
PAD_AFTER = 1.0

# "My hand" -- the sim-policy rollout on ayush's hand.
FILES = {
    ("balls", "position"): "baoding_simgap_ayush_balls/position_baoding_sim_policy_log_padtac_new_rollout_20260719_163012.aligned.npz",
    ("balls", "trajectory"): "baoding_simgap_ayush_balls/trajectory_baoding_sim_policy_log_padtac_new_rollout_20260719_163300.aligned.npz",
    ("noballs", "position"): "baoding_simgap_ayush_noballs/position_baoding_sim_policy_log_padtac_new_rollout_20260719_162853.aligned.npz",
    ("noballs", "trajectory"): "baoding_simgap_ayush_noballs/trajectory_baoding_sim_policy_log_padtac_new_rollout_20260719_163159.aligned.npz",
}

# "The other hand" -- the same replayed motion on nalin's hand (identical schema).
NALIN_FILES = {
    ("balls", "position"): "baoding_simgap_ayush_balls/position_baoding_balls_nalin_speed1p0.aligned.npz",
    ("balls", "trajectory"): "baoding_simgap_ayush_balls/trajectory_baoding_balls_nalin_speed1p0.aligned.npz",
    ("noballs", "position"): "baoding_simgap_ayush_noballs/position_baoding_noballs_nalin_speed1p0.aligned.npz",
    ("noballs", "trajectory"): "baoding_simgap_ayush_noballs/trajectory_baoding_noballs_nalin_speed1p0.aligned.npz",
}


def load_run(rel_path):
    d = np.load(os.path.join(BASE, rel_path), allow_pickle=True)
    out = {k: d[k] for k in d.files}
    out["t_rel"] = out["t"] - out["t"][0]
    # Position error computed the way roto_env.py does it:
    #   joint_pos_error = joint_pos_cmd - joint_pos   (command minus measured, same step)
    # Here the logged position command is `action` (rad) and the measured position is
    # `act_pos`, both in 13-dim actuator space -- so no coupling is needed. This is a
    # different signal from the stored `act_err` (which is ~ act_pos - action, i.e. the
    # opposite sign convention plus a small controller-side offset); we plot both.
    out["pos_err_calc"] = out["action"] - out["act_pos"]
    return out


def load_all(files=FILES):
    return {key: load_run(path) for key, path in files.items()}


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def grid_plot(series, names, title, outpath, ylabel, ncols=4, figsize_per=(3.2, 2.2)):
    """series: list of {"label","t","data"(N,D),"color","ls"}. names: D column labels."""
    D = len(names)
    ncols = min(ncols, D)
    nrows = int(np.ceil(D / ncols))
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(figsize_per[0] * ncols, figsize_per[1] * nrows), squeeze=False
    )
    for i, name in enumerate(names):
        ax = axes[i // ncols][i % ncols]
        for s in series:
            ax.plot(
                s["t"],
                s["data"][:, i],
                label=s["label"],
                color=s.get("color"),
                linestyle=s.get("ls", "-"),
                linewidth=1.0,
                alpha=s.get("alpha", 0.9),
            )
        ax.set_title(name, fontsize=9)
        ax.tick_params(labelsize=7)
        ax.grid(alpha=0.3)
    for j in range(D, nrows * ncols):
        axes[j // ncols][j % ncols].axis("off")

    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.suptitle(title, fontsize=11)
    fig.legend(handles, labels, loc="upper center", ncol=len(series), fontsize=9, bbox_to_anchor=(0.5, 0.965))
    fig.text(0.5, 0.005, f"x: time since motion onset (s)   y: {ylabel}", ha="center", fontsize=9)
    fig.subplots_adjust(hspace=0.55, wspace=0.35, top=0.90, bottom=0.06)

    ensure_dir(os.path.dirname(outpath))
    fig.savefig(outpath, dpi=140, bbox_inches="tight")
    plt.close(fig)


def motion_bounds(run, window_sec=1.0, thresh_frac=0.15):
    """(t_onset, t_offset): the first and last times the `action` setpoint is under
    sustained motion, detected via a rolling standard deviation (window_sec wide)
    thresholded relative to its own peak. A rolling measure (vs. raw step diffs)
    means an isolated settle/step that is flat immediately before and after does not
    count as motion -- only genuine, sustained movement does.

    This is the single alignment reference used everywhere: every run is later shifted
    so its onset lands at t=0, so the sync rule is identical across all comparisons.
    Values are never touched -- this only reads the setpoint to find the onset."""
    t = run["t_rel"]
    data = run["action"]
    dt = np.median(np.diff(t))
    w = max(1, int(round(window_sec / dt)))
    mean = uniform_filter1d(data, size=w, axis=0, mode="nearest")
    meansq = uniform_filter1d(data**2, size=w, axis=0, mode="nearest")
    rolling_std = np.sqrt(np.clip(meansq - mean**2, 0, None))
    activity = rolling_std.max(axis=1)
    if activity.max() <= 0:
        return t[0], t[-1]
    idx = np.where(activity > thresh_frac * activity.max())[0]
    if len(idx) == 0:
        return t[0], t[-1]
    return t[idx[0]], t[idx[-1]]


def prepare_run(run, onset, t_lo, t_hi):
    """Copy of `run` on an onset-referenced time axis (t_onset -> 0), cropped to the
    shared display window [t_lo, t_hi]. Every per-sample array is masked to the same
    samples; a "t_plot" entry holds the shifted, cropped time. Only the time offset
    and which samples are drawn change -- no data value is modified."""
    t_shifted = run["t_rel"] - onset
    mask = (t_shifted >= t_lo) & (t_shifted <= t_hi)
    n = t_shifted.shape[0]
    out = {k: (v[mask] if isinstance(v, np.ndarray) and v.ndim >= 1 and v.shape[0] == n else v)
           for k, v in run.items()}
    out["t_plot"] = t_shifted[mask]
    return out


def plot_within_test(runs):
    outdir = os.path.join(PLOTS, "within_test")
    for (cond, mode), full_run in runs.items():
        onset, offset = motion_bounds(full_run)
        t_lo, t_hi = -PAD_BEFORE, (offset - onset) + PAD_AFTER
        run = prepare_run(full_run, onset, t_lo, t_hi)
        prefix = f"{cond}_{mode}"
        t = run["t_plot"]
        act_order = list(run["actuator_order"])
        joint_order = list(run["joint_order"])

        grid_plot(
            [
                {"label": "action (setpoint)", "t": t, "data": run["action"], "color": "tab:blue", "ls": "--"},
                {"label": "act_pos (achieved)", "t": t, "data": run["act_pos"], "color": "tab:orange"},
            ],
            act_order,
            f"{cond} / {mode} ctrl — actuator position: setpoint vs achieved",
            os.path.join(outdir, f"{prefix}_act_positions.png"),
            "rad",
        )
        grid_plot(
            [{"label": "command (effort)", "t": t, "data": run["command"], "color": "tab:red"}],
            act_order,
            f"{cond} / {mode} ctrl — /command effort (±600 PWM)",
            os.path.join(outdir, f"{prefix}_effort_cmd.png"),
            "effort units",
        )
        grid_plot(
            [{"label": "act_vel", "t": t, "data": run["act_vel"], "color": "tab:green"}],
            act_order,
            f"{cond} / {mode} ctrl — actuator velocity",
            os.path.join(outdir, f"{prefix}_act_vel.png"),
            "rad/s",
        )
        grid_plot(
            [{"label": "act_err (from data)", "t": t, "data": run["act_err"], "color": "tab:purple"}],
            act_order,
            f"{cond} / {mode} ctrl — actuator error (act_err, from data)",
            os.path.join(outdir, f"{prefix}_act_err.png"),
            "rad",
        )
        grid_plot(
            [{"label": "pos_err = action − act_pos", "t": t, "data": run["pos_err_calc"], "color": "tab:brown"}],
            act_order,
            f"{cond} / {mode} ctrl — computed pos error (roto_env: cmd − measured)",
            os.path.join(outdir, f"{prefix}_pos_err_calc.png"),
            "rad",
        )
        grid_plot(
            [{"label": "gt_pos", "t": t, "data": run["gt_pos"], "color": "tab:orange"}],
            joint_order,
            f"{cond} / {mode} ctrl — joint position (/joint_states)",
            os.path.join(outdir, f"{prefix}_joint_positions.png"),
            "rad",
        )
        grid_plot(
            [{"label": "gt_vel", "t": t, "data": run["gt_vel"], "color": "tab:green"}],
            joint_order,
            f"{cond} / {mode} ctrl — joint velocity",
            os.path.join(outdir, f"{prefix}_gt_vel.png"),
            "rad/s",
        )
        grid_plot(
            [{"label": "gt_effort", "t": t, "data": run["gt_effort"], "color": "tab:red"}],
            joint_order,
            f"{cond} / {mode} ctrl — joint effort (measured torque)",
            os.path.join(outdir, f"{prefix}_gt_effort.png"),
            "torque units",
        )


def plot_traj_vs_pos(runs):
    outdir = os.path.join(PLOTS, "traj_vs_pos")
    for cond in ["balls", "noballs"]:
        run_pos_full = runs[(cond, "position")]
        run_traj_full = runs[(cond, "trajectory")]
        on_p, off_p = motion_bounds(run_pos_full)
        on_t, off_t = motion_bounds(run_traj_full)
        dur_p, dur_t = off_p - on_p, off_t - on_t
        t_lo, t_hi = -PAD_BEFORE, max(dur_p, dur_t) + PAD_AFTER

        run_pos = prepare_run(run_pos_full, on_p, t_lo, t_hi)
        run_traj = prepare_run(run_traj_full, on_t, t_lo, t_hi)
        t_pos = run_pos["t_plot"]
        t_traj = run_traj["t_plot"]
        act_order = list(run_pos["actuator_order"])
        joint_order = list(run_pos["joint_order"])
        suffix = f"(aligned at motion onset; pos {dur_p:.2f}s vs traj {dur_t:.2f}s)"

        def pair(key):
            return [
                {"label": "position ctrl", "t": t_pos, "data": run_pos[key], "color": "tab:blue"},
                {"label": "trajectory ctrl", "t": t_traj, "data": run_traj[key], "color": "tab:orange"},
            ]

        grid_plot(pair("action"), act_order, f"{cond} — setpoint (action) {suffix}",
                   os.path.join(outdir, f"{cond}_setpoint.png"), "rad")
        grid_plot(pair("act_pos"), act_order, f"{cond} — actuator position {suffix}",
                   os.path.join(outdir, f"{cond}_act_pos.png"), "rad")
        grid_plot(pair("command"), act_order, f"{cond} — /command effort {suffix}",
                   os.path.join(outdir, f"{cond}_effort_cmd.png"), "effort units")
        grid_plot(pair("act_vel"), act_order, f"{cond} — actuator velocity {suffix}",
                   os.path.join(outdir, f"{cond}_act_vel.png"), "rad/s")
        grid_plot(pair("pos_err_calc"), act_order, f"{cond} — computed pos error (roto_env: cmd − measured) {suffix}",
                   os.path.join(outdir, f"{cond}_pos_err_calc.png"), "rad")
        grid_plot(pair("act_err"), act_order, f"{cond} — actuator error (act_err, from data) {suffix}",
                   os.path.join(outdir, f"{cond}_act_err.png"), "rad")
        grid_plot(pair("gt_pos"), joint_order, f"{cond} — joint position {suffix}",
                   os.path.join(outdir, f"{cond}_gt_pos.png"), "rad")
        grid_plot(pair("gt_vel"), joint_order, f"{cond} — joint velocity {suffix}",
                   os.path.join(outdir, f"{cond}_gt_vel.png"), "rad/s")
        grid_plot(pair("gt_effort"), joint_order, f"{cond} — joint effort {suffix}",
                   os.path.join(outdir, f"{cond}_gt_effort.png"), "torque units")


def plot_ball_vs_noball(runs):
    outdir = os.path.join(PLOTS, "ball_vs_noball")
    for mode in ["position", "trajectory"]:
        run_balls_full = runs[("balls", mode)]
        run_noballs_full = runs[("noballs", mode)]
        on_b, off_b = motion_bounds(run_balls_full)
        on_n, off_n = motion_bounds(run_noballs_full)
        dur_b, dur_n = off_b - on_b, off_n - on_n
        t_lo, t_hi = -PAD_BEFORE, max(dur_b, dur_n) + PAD_AFTER

        run_balls = prepare_run(run_balls_full, on_b, t_lo, t_hi)
        run_noballs = prepare_run(run_noballs_full, on_n, t_lo, t_hi)
        t_b = run_balls["t_plot"]
        t_nb = run_noballs["t_plot"]
        act_order = list(run_balls["actuator_order"])
        joint_order = list(run_balls["joint_order"])
        suffix = f"(aligned at motion onset; balls {dur_b:.2f}s vs no-balls {dur_n:.2f}s)"

        def pair(key):
            return [
                {"label": "balls", "t": t_b, "data": run_balls[key], "color": "tab:blue"},
                {"label": "no balls", "t": t_nb, "data": run_noballs[key], "color": "tab:red"},
            ]

        grid_plot(pair("act_pos"), act_order, f"{mode} ctrl — actuator position {suffix}",
                   os.path.join(outdir, f"{mode}_act_pos.png"), "rad")
        grid_plot(pair("command"), act_order, f"{mode} ctrl — /command effort {suffix}",
                   os.path.join(outdir, f"{mode}_effort_cmd.png"), "effort units")
        grid_plot(pair("act_vel"), act_order, f"{mode} ctrl — actuator velocity {suffix}",
                   os.path.join(outdir, f"{mode}_act_vel.png"), "rad/s")
        grid_plot(pair("pos_err_calc"), act_order, f"{mode} ctrl — computed pos error (roto_env: cmd − measured) {suffix}",
                   os.path.join(outdir, f"{mode}_pos_err_calc.png"), "rad")
        grid_plot(pair("act_err"), act_order, f"{mode} ctrl — actuator error (act_err, from data) {suffix}",
                   os.path.join(outdir, f"{mode}_act_err.png"), "rad")
        grid_plot(pair("gt_pos"), joint_order, f"{mode} ctrl — joint position {suffix}",
                   os.path.join(outdir, f"{mode}_gt_pos.png"), "rad")
        grid_plot(pair("gt_vel"), joint_order, f"{mode} ctrl — joint velocity {suffix}",
                   os.path.join(outdir, f"{mode}_gt_vel.png"), "rad/s")
        grid_plot(pair("gt_effort"), joint_order, f"{mode} ctrl — joint effort {suffix}",
                   os.path.join(outdir, f"{mode}_gt_effort.png"), "torque units")

        tactile_names = [f"tac_{i}" for i in range(run_balls["gt_tactile"].shape[1])]
        grid_plot(pair("gt_tactile"), tactile_names, f"{mode} ctrl — tactile {suffix}",
                   os.path.join(outdir, f"{mode}_tactile.png"), "raw tactile units",
                   ncols=8, figsize_per=(2.0, 1.5))


def plot_hand_vs_hand(runs_mine, runs_other):
    """Overlay my hand (ayush) vs the other hand (nalin) for the same replayed motion.
    Per (ball condition, control mode): position, velocity, actuator error, and the
    roto_env computed pos error -- in both actuator (13) and joint (16) spaces where
    both exist. Each run is shifted to its own motion onset (t=0), the single alignment
    rule used everywhere; values are untouched."""
    outdir = os.path.join(PLOTS, "hand_vs_hand")
    for cond in ["balls", "noballs"]:
        for mode in ["position", "trajectory"]:
            run_m_full = runs_mine[(cond, mode)]
            run_o_full = runs_other[(cond, mode)]
            on_m, off_m = motion_bounds(run_m_full)
            on_o, off_o = motion_bounds(run_o_full)
            dur_m, dur_o = off_m - on_m, off_o - on_o
            t_lo, t_hi = -PAD_BEFORE, max(dur_m, dur_o) + PAD_AFTER

            run_m = prepare_run(run_m_full, on_m, t_lo, t_hi)
            run_o = prepare_run(run_o_full, on_o, t_lo, t_hi)
            t_m, t_o = run_m["t_plot"], run_o["t_plot"]
            act_order = list(run_m["actuator_order"])
            joint_order = list(run_m["joint_order"])
            prefix = f"{cond}_{mode}"
            suffix = f"({cond}, {mode} ctrl; onset-aligned; ayush {dur_m:.2f}s vs nalin {dur_o:.2f}s)"

            def pair(key):
                return [
                    {"label": "ayush (my hand)", "t": t_m, "data": run_m[key], "color": "tab:blue"},
                    {"label": "nalin (other hand)", "t": t_o, "data": run_o[key], "color": "tab:green"},
                ]

            # actuator space (13)
            grid_plot(pair("act_pos"), act_order, f"actuator position — {suffix}",
                       os.path.join(outdir, f"{prefix}_act_pos.png"), "rad")
            grid_plot(pair("act_vel"), act_order, f"actuator velocity — {suffix}",
                       os.path.join(outdir, f"{prefix}_act_vel.png"), "rad/s")
            grid_plot(pair("act_err"), act_order, f"actuator error (act_err, from data) — {suffix}",
                       os.path.join(outdir, f"{prefix}_act_err.png"), "rad")
            grid_plot(pair("pos_err_calc"), act_order, f"computed pos error (roto_env: cmd − measured) — {suffix}",
                       os.path.join(outdir, f"{prefix}_pos_err_calc.png"), "rad")
            # joint space (16)
            grid_plot(pair("gt_pos"), joint_order, f"joint position — {suffix}",
                       os.path.join(outdir, f"{prefix}_gt_pos.png"), "rad")
            grid_plot(pair("gt_vel"), joint_order, f"joint velocity — {suffix}",
                       os.path.join(outdir, f"{prefix}_gt_vel.png"), "rad/s")


def main():
    runs = load_all(FILES)
    plot_within_test(runs)
    plot_traj_vs_pos(runs)
    plot_ball_vs_noball(runs)

    runs_nalin = load_all(NALIN_FILES)
    plot_hand_vs_hand(runs, runs_nalin)
    print("Done. Plots written to", PLOTS)


if __name__ == "__main__":
    main()

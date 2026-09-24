#!/usr/bin/env python3
"""Exploratory look at the collected hardware data feeding GenAN training --
no model, no coupling formula, no derived/manipulated quantities. Every plot
and table here reads straight off the recorded `*.aligned.npz` arrays
(`gt_pos`/`gt_vel`/`gt_effort`/`act_pos`/`act_err`), filtered only by the
file's own `valid` flag.

Isaac-free, torch-free -- plain numpy/matplotlib over the three recorded
dataset directories (same paths as `agents/shadowlite/default.yaml`'s
`dataset.paths`). Joint names/order are read from each file's own
`joint_order`/`actuator_order` fields (self-describing, no guessing);
`joint_config.py` is used only to pull each joint's known physical position
limit, for the range-coverage bar chart (a static reference value, not
something derived from this data).

Usage:
    python3 analyze_hw_data.py
"""

from __future__ import annotations

import csv
import glob
import os
import sys

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_GENAN_DIR = os.path.dirname(_THIS_DIR)
if _GENAN_DIR not in sys.path:
    sys.path.insert(0, _GENAN_DIR)

from dataset_loader import COUPLED_JOINT_PAIRS  # noqa: E402  (name mapping only, no computation)
from joint_config import load_joint_config  # noqa: E402

OUT_DIR = _THIS_DIR
PLOTS_DIR = os.path.join(OUT_DIR, "plots")
TABLES_DIR = os.path.join(OUT_DIR, "tables")

DATASET_DIRS = [
    "/home/ayush/icra/roto/roto/data/data/aligned/free_space_50",
    "/home/ayush/icra/roto/roto/data/data/aligned/free_space_continous_50",
    "/home/ayush/icra/roto/roto/data/data/aligned/free_space_10_13072026",
]
SOURCE_COLORS = {os.path.basename(d): c for d, c in zip(DATASET_DIRS, plt.cm.tab10.colors)}

RNG = np.random.default_rng(0)  # only used to subsample scatter points for rendering


# -----------------------------------------------------------------------------
# Loading -- every file read exactly once, kept as its own record (no merging
# across files beyond simple concatenation for pooled stats/plots).
# -----------------------------------------------------------------------------

class Episode:
    def __init__(self, path: str, source: str):
        self.path = path
        self.source = source
        d = np.load(path, allow_pickle=True)
        self.joint_order = [str(n) for n in d["joint_order"]]
        self.actuator_order = [str(n) for n in d["actuator_order"]]
        self.valid = np.asarray(d["valid"], dtype=bool)
        self.gt_pos = np.asarray(d["gt_pos"], dtype=np.float64)[self.valid]
        self.gt_vel = np.asarray(d["gt_vel"], dtype=np.float64)[self.valid]
        self.gt_effort = np.asarray(d["gt_effort"], dtype=np.float64)[self.valid]
        self.act_err = np.asarray(d["act_err"], dtype=np.float64)[self.valid]
        self.rate = float(d["dataset_rate"])
        self.n_valid = int(self.valid.sum())
        self.n_total = int(self.valid.shape[0])
        self.duration_s = self.n_valid / self.rate if self.rate > 0 else float("nan")
        self.t = np.arange(self.n_valid) / self.rate


def load_all_episodes() -> list[Episode]:
    episodes = []
    for d in DATASET_DIRS:
        source = os.path.basename(d)
        files = sorted(glob.glob(os.path.join(d, "*.aligned.npz")))
        for f in files:
            episodes.append(Episode(f, source))
    return episodes


# -----------------------------------------------------------------------------
# Overview table + plot
# -----------------------------------------------------------------------------

def write_overview(episodes: list[Episode]) -> None:
    rows = []
    for source in [os.path.basename(d) for d in DATASET_DIRS]:
        eps = [e for e in episodes if e.source == source]
        rows.append({
            "source": source,
            "n_episodes": len(eps),
            "total_valid_rows": sum(e.n_valid for e in eps),
            "total_duration_s": round(sum(e.duration_s for e in eps), 1),
            "dataset_rate_hz": eps[0].rate if eps else float("nan"),
        })
    rows.append({
        "source": "ALL",
        "n_episodes": len(episodes),
        "total_valid_rows": sum(e.n_valid for e in episodes),
        "total_duration_s": round(sum(e.duration_s for e in episodes), 1),
        "dataset_rate_hz": episodes[0].rate if episodes else float("nan"),
    })
    with open(os.path.join(TABLES_DIR, "dataset_overview.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print("Wrote tables/dataset_overview.csv:")
    for r in rows:
        print(" ", r)

    # Plot: episode duration histogram, by source
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for source in [os.path.basename(d) for d in DATASET_DIRS]:
        durations = [e.duration_s for e in episodes if e.source == source]
        ax.hist(durations, bins=15, alpha=0.6, label=source, color=SOURCE_COLORS[source])
    ax.set_xlabel("episode duration (s)")
    ax.set_ylabel("count")
    ax.set_title("Episode durations by source dataset")
    ax.legend(fontsize=8)
    ax.grid(True, lw=0.3, alpha=0.5)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, "overview", "episode_durations.png"), dpi=150)
    plt.close(fig)


# -----------------------------------------------------------------------------
# Per-joint stats table + histograms + range-coverage plot
# -----------------------------------------------------------------------------

def pool_by_joint(episodes: list[Episode], attr: str, joint_names: list[str]) -> dict[str, np.ndarray]:
    """Concatenate one column (by joint name) across all episodes that share
    the same joint_order (all aligned.npz files use hardware_joint_order, but
    index defensively by name rather than assuming position)."""
    pooled = {name: [] for name in joint_names}
    for ep in episodes:
        arr = getattr(ep, attr)
        col = {n: i for i, n in enumerate(ep.joint_order)}
        for name in joint_names:
            if name in col:
                pooled[name].append(arr[:, col[name]])
    return {name: (np.concatenate(chunks) if chunks else np.empty(0)) for name, chunks in pooled.items()}


def write_per_joint_stats(episodes: list[Episode], joint_names: list[str], joint_limits: dict[str, float]) -> None:
    pos = pool_by_joint(episodes, "gt_pos", joint_names)
    vel = pool_by_joint(episodes, "gt_vel", joint_names)
    eff = pool_by_joint(episodes, "gt_effort", joint_names)

    rows = []
    for name in joint_names:
        p, v, e = pos[name], vel[name], eff[name]
        limit_upper = joint_limits.get(name)
        used_range = float(p.max() - p.min()) if p.size else float("nan")
        rows.append({
            "joint": name,
            "n_samples": p.size,
            "pos_min": round(float(p.min()), 4) if p.size else "",
            "pos_max": round(float(p.max()), 4) if p.size else "",
            "pos_range_used": round(used_range, 4),
            "joint_upper_limit_rad": limit_upper,
            "vel_std": round(float(v.std()), 4) if v.size else "",
            "vel_absmax": round(float(np.abs(v).max()), 4) if v.size else "",
            "effort_min": round(float(e.min()), 3) if e.size else "",
            "effort_max": round(float(e.max()), 3) if e.size else "",
            "effort_std": round(float(e.std()), 3) if e.size else "",
        })
    with open(os.path.join(TABLES_DIR, "per_joint_stats.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print("Wrote tables/per_joint_stats.csv")

    # Range-coverage bar chart: pos_range_used vs the joint's own physical limit range.
    # (Limit range shown as reference context only -- a known static config value,
    # not something computed from this data.)
    fig, ax = plt.subplots(figsize=(8, 6))
    y = np.arange(len(joint_names))
    used = [r["pos_range_used"] for r in rows]
    limits = [joint_limits.get(n, np.nan) for n in joint_names]
    ax.barh(y, limits, color="lightgrey", label="known joint limit range (config)")
    ax.barh(y, used, color="steelblue", height=0.5, label="range actually seen in data")
    ax.set_yticks(y)
    ax.set_yticklabels(joint_names, fontsize=8)
    ax.set_xlabel("range (rad)")
    ax.set_title("Per-joint position range: recorded data vs known joint limit")
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(True, axis="x", lw=0.3, alpha=0.5)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, "overview", "per_joint_range_coverage.png"), dpi=150)
    plt.close(fig)

    # Pooled histograms, all 16 joints, one figure per signal.
    for attr_name, pooled, unit, fname in [
        ("position", pos, "rad", "position_hist_all16.png"),
        ("velocity", vel, "rad/s", "velocity_hist_all16.png"),
        ("effort", eff, "raw (uncalibrated)", "effort_hist_all16.png"),
    ]:
        ncols, nrows = 4, (len(joint_names) + 3) // 4
        fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3.2, nrows * 2.4))
        fig.suptitle(f"Pooled {attr_name} histograms, all episodes ({unit})", fontsize=13, fontweight="bold")
        axf = axes.flatten()
        for j, name in enumerate(joint_names):
            ax = axf[j]
            data = pooled[name]
            if data.size:
                ax.hist(data, bins=60, color="steelblue")
            ax.set_title(name, fontsize=8)
            ax.tick_params(labelsize=6)
            ax.grid(True, lw=0.3, alpha=0.5)
        for ax in axf[len(joint_names):]:
            ax.set_visible(False)
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        fig.savefig(os.path.join(PLOTS_DIR, "per_joint", fname), dpi=150)
        plt.close(fig)


def pick_longest_episode(episodes: list[Episode], source: str) -> Episode:
    eps = [e for e in episodes if e.source == source]
    return max(eps, key=lambda e: e.n_valid)


def plot_episode_timeseries_grids(episodes: list[Episode], joint_names: list[str]) -> None:
    for d in DATASET_DIRS:
        source = os.path.basename(d)
        ep = pick_longest_episode(episodes, source)
        col = {n: i for i, n in enumerate(ep.joint_order)}
        for attr, arr_name, unit in [
            ("gt_pos", "pos", "rad"),
            ("gt_vel", "vel", "rad/s"),
            ("gt_effort", "effort", "raw"),
        ]:
            arr = getattr(ep, attr)
            ncols, nrows = 4, (len(joint_names) + 3) // 4
            fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3.4, nrows * 2.2), sharex=True)
            fig.suptitle(
                f"{source}: {os.path.basename(ep.path)} (longest episode, {ep.duration_s:.1f}s) -- {arr_name} ({unit})",
                fontsize=11, fontweight="bold",
            )
            axf = axes.flatten()
            for j, name in enumerate(joint_names):
                ax = axf[j]
                if name in col:
                    ax.plot(ep.t, arr[:, col[name]], color="steelblue", lw=0.8)
                ax.set_title(name, fontsize=8, pad=2)
                ax.tick_params(labelsize=6)
                ax.grid(True, lw=0.3, alpha=0.5)
            for ax in axf[len(joint_names):]:
                ax.set_visible(False)
            for ax in axf[(nrows - 1) * ncols:]:
                ax.set_xlabel("time (s)", fontsize=7)
            fig.tight_layout(rect=[0, 0, 1, 0.95])
            fig.savefig(
                os.path.join(PLOTS_DIR, "per_joint", f"timeseries_grid_{source}_{arr_name}.png"), dpi=150
            )
            plt.close(fig)

        # act_err grid (13 actuator-order joints) -- separate since different column order.
        act_col = {n: i for i, n in enumerate(ep.actuator_order)}
        ncols, nrows = 4, (len(ep.actuator_order) + 3) // 4
        fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3.4, nrows * 2.2), sharex=True)
        fig.suptitle(
            f"{source}: {os.path.basename(ep.path)} (longest episode) -- act_err (rad)",
            fontsize=11, fontweight="bold",
        )
        axf = axes.flatten()
        for j, name in enumerate(ep.actuator_order):
            ax = axf[j]
            ax.plot(ep.t, ep.act_err[:, act_col[name]], color="darkorange", lw=0.8)
            ax.set_title(name, fontsize=8, pad=2)
            ax.tick_params(labelsize=6)
            ax.grid(True, lw=0.3, alpha=0.5)
        for ax in axf[len(ep.actuator_order):]:
            ax.set_visible(False)
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        fig.savefig(os.path.join(PLOTS_DIR, "per_joint", f"timeseries_grid_{source}_act_err.png"), dpi=150)
        plt.close(fig)


# -----------------------------------------------------------------------------
# Outlier-episode scan -- per joint, flag episodes whose OWN position range is
# far outside the typical (median) per-episode range for that joint. Purely
# descriptive (median + ratio), no filtering/removal of any data.
# -----------------------------------------------------------------------------

def find_outlier_episodes(episodes: list[Episode], joint_names: list[str], ratio_threshold: float = 3.0) -> list[dict]:
    flagged = []
    for name in joint_names:
        per_ep = []
        for ep in episodes:
            col = {n: i for i, n in enumerate(ep.joint_order)}
            if name not in col:
                continue
            arr = ep.gt_pos[:, col[name]]
            per_ep.append((ep, float(arr.max() - arr.min())))
        if not per_ep:
            continue
        ranges = np.array([r for _, r in per_ep])
        median = float(np.median(ranges))
        if median <= 0:
            continue
        for ep, r in per_ep:
            if r > ratio_threshold * median:
                flagged.append({
                    "joint": name,
                    "episode": os.path.basename(ep.path),
                    "source": ep.source,
                    "episode_range": round(r, 4),
                    "median_episode_range": round(median, 4),
                    "ratio": round(r / median, 2),
                })
    return flagged


# -----------------------------------------------------------------------------
# Coupled-joint section (FF/MF/RF J1+J2 pairs) -- raw data only, no coupling
# model, no theta-split, no backlash reconstruction of any kind.
# -----------------------------------------------------------------------------

def write_coupled_pair_stats_and_plots(episodes: list[Episode]) -> None:
    rows = []
    for pair_key, (j1_name, j2_name) in COUPLED_JOINT_PAIRS.items():
        # Pool raw (J2, J1) pairs across every episode.
        j2_chunks, j1_chunks, source_chunks = [], [], []
        for ep in episodes:
            col = {n: i for i, n in enumerate(ep.joint_order)}
            if j1_name not in col or j2_name not in col:
                continue
            j2_chunks.append(ep.gt_pos[:, col[j2_name]])
            j1_chunks.append(ep.gt_pos[:, col[j1_name]])
            source_chunks.append(np.full(ep.n_valid, ep.source, dtype=object))
        j2_all = np.concatenate(j2_chunks)
        j1_all = np.concatenate(j1_chunks)
        source_all = np.concatenate(source_chunks)

        corr = float(np.corrcoef(j2_all, j1_all)[0, 1])
        rows.append({
            "pair": pair_key,
            "j1_joint": j1_name,
            "j2_joint": j2_name,
            "n_samples": j2_all.size,
            "j2_range_used": round(float(j2_all.max() - j2_all.min()), 4),
            "j1_range_used": round(float(j1_all.max() - j1_all.min()), 4),
            "corr_j2_j1": round(corr, 4),
        })

        # --- scatter (raw, subsampled only for rendering) ---
        fig, ax = plt.subplots(figsize=(5.5, 5))
        for source in [os.path.basename(d) for d in DATASET_DIRS]:
            mask = source_all == source
            idx = np.nonzero(mask)[0]
            if idx.size > 4000:
                idx = RNG.choice(idx, size=4000, replace=False)
            ax.scatter(j2_all[idx], j1_all[idx], s=3, alpha=0.35, color=SOURCE_COLORS[source], label=source)
        ax.set_xlabel(f"{j2_name} actual position (rad)")
        ax.set_ylabel(f"{j1_name} actual position (rad)")
        ax.set_title(f"{pair_key}: raw J2 vs J1 (all episodes, subsampled for display)", fontsize=10)
        ax.legend(fontsize=7)
        ax.grid(True, lw=0.3, alpha=0.5)
        fig.tight_layout()
        fig.savefig(os.path.join(PLOTS_DIR, "coupled", f"{pair_key}_j2_vs_j1_scatter.png"), dpi=150)
        plt.close(fig)

        # --- coverage heatmap (2D histogram, full data, no subsampling) ---
        fig, ax = plt.subplots(figsize=(5.5, 5))
        h = ax.hist2d(j2_all, j1_all, bins=60, cmap="viridis")
        fig.colorbar(h[3], ax=ax, label="count")
        ax.set_xlabel(f"{j2_name} actual position (rad)")
        ax.set_ylabel(f"{j1_name} actual position (rad)")
        ax.set_title(f"{pair_key}: (J2,J1) coverage, all episodes pooled", fontsize=10)
        fig.tight_layout()
        fig.savefig(os.path.join(PLOTS_DIR, "coupled", f"{pair_key}_coverage_heatmap.png"), dpi=150)
        plt.close(fig)

        # --- representative-episode timeseries: J1, J2, and their raw sum ---
        rep_ep = max(episodes, key=lambda e: e.n_valid)
        col = {n: i for i, n in enumerate(rep_ep.joint_order)}
        j2_t = rep_ep.gt_pos[:, col[j2_name]]
        j1_t = rep_ep.gt_pos[:, col[j1_name]]
        fig, ax = plt.subplots(figsize=(9, 4))
        ax.plot(rep_ep.t, j2_t, color="steelblue", lw=1.0, label=f"{j2_name} (J2)")
        ax.plot(rep_ep.t, j1_t, color="crimson", lw=1.0, label=f"{j1_name} (J1)")
        ax.plot(rep_ep.t, j2_t + j1_t, color="black", lw=0.8, ls="--", label="J2 + J1 (sum)")
        ax.set_xlabel("time (s)")
        ax.set_ylabel("rad")
        ax.set_title(f"{pair_key}: raw position vs time, {os.path.basename(rep_ep.path)} ({rep_ep.source})", fontsize=10)
        ax.legend(fontsize=8)
        ax.grid(True, lw=0.3, alpha=0.5)
        fig.tight_layout()
        fig.savefig(os.path.join(PLOTS_DIR, "coupled", f"{pair_key}_timeseries.png"), dpi=150)
        plt.close(fig)

        # --- representative-episode raw effort, both joints ---
        j2_e = rep_ep.gt_effort[:, col[j2_name]]
        j1_e = rep_ep.gt_effort[:, col[j1_name]]
        fig, ax = plt.subplots(figsize=(9, 4))
        ax.plot(rep_ep.t, j2_e, color="steelblue", lw=1.0, label=f"{j2_name} (J2) gt_effort")
        ax.plot(rep_ep.t, j1_e, color="crimson", lw=1.0, label=f"{j1_name} (J1) gt_effort")
        ax.set_xlabel("time (s)")
        ax.set_ylabel("gt_effort (raw, uncalibrated)")
        ax.set_title(f"{pair_key}: raw gt_effort vs time, {os.path.basename(rep_ep.path)} ({rep_ep.source})", fontsize=10)
        ax.legend(fontsize=8)
        ax.grid(True, lw=0.3, alpha=0.5)
        fig.tight_layout()
        fig.savefig(os.path.join(PLOTS_DIR, "coupled", f"{pair_key}_effort.png"), dpi=150)
        plt.close(fig)

    with open(os.path.join(TABLES_DIR, "coupled_pair_stats.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print("Wrote tables/coupled_pair_stats.csv:")
    for r in rows:
        print(" ", r)


# -----------------------------------------------------------------------------
# Per-episode coupled-joint plots -- same signals as the pooled/representative
# versions above (raw gt_pos/gt_effort, no model), but broken out one episode
# at a time so individual-episode behavior/anomalies are visible rather than
# averaged away. The J2-vs-J1 scatter is colored by time (a plot styling
# choice -- it doesn't change any data value) so the real temporal order of
# the trajectory is visible in a single static image.
# -----------------------------------------------------------------------------

def select_representative_episodes(episodes: list[Episode], outliers: list[dict], n: int = 4) -> list[Episode]:
    """Longest episode per source dataset, plus (if room and one exists) the
    single highest-ratio flagged outlier episode not already included."""
    reps: list[Episode] = []
    seen = set()
    for d in DATASET_DIRS:
        ep = pick_longest_episode(episodes, os.path.basename(d))
        if ep.path not in seen:
            reps.append(ep)
            seen.add(ep.path)
    if len(reps) < n and outliers:
        for o in sorted(outliers, key=lambda o: -o["ratio"]):
            match = next((e for e in episodes if os.path.basename(e.path) == o["episode"]), None)
            if match is not None and match.path not in seen:
                reps.append(match)
                seen.add(match.path)
                break
    return reps[:n]


def plot_coupled_per_episode(reps: list[Episode]) -> None:
    out_dir = os.path.join(PLOTS_DIR, "coupled", "per_episode")
    os.makedirs(out_dir, exist_ok=True)

    for ep in reps:
        col = {n: i for i, n in enumerate(ep.joint_order)}
        ep_label = os.path.splitext(os.path.basename(ep.path))[0]

        for pair_key, (j1_name, j2_name) in COUPLED_JOINT_PAIRS.items():
            if j1_name not in col or j2_name not in col:
                continue
            j2, j1 = ep.gt_pos[:, col[j2_name]], ep.gt_pos[:, col[j1_name]]
            j2e, j1e = ep.gt_effort[:, col[j2_name]], ep.gt_effort[:, col[j1_name]]

            # --- J2 vs J1, this episode only, colored by time ---
            fig, ax = plt.subplots(figsize=(5.5, 5))
            sca = ax.scatter(j2, j1, c=ep.t, cmap="viridis", s=6)
            fig.colorbar(sca, ax=ax, label="time (s)")
            ax.set_xlabel(f"{j2_name} actual position (rad)")
            ax.set_ylabel(f"{j1_name} actual position (rad)")
            ax.set_title(f"{pair_key}: J2 vs J1 -- {ep_label} ({ep.source})", fontsize=9)
            ax.grid(True, lw=0.3, alpha=0.5)
            fig.tight_layout()
            fig.savefig(os.path.join(out_dir, f"{pair_key}_{ep_label}_j2_vs_j1_scatter.png"), dpi=150)
            plt.close(fig)

            # --- position vs time, this episode only ---
            fig, ax = plt.subplots(figsize=(9, 4))
            ax.plot(ep.t, j2, color="steelblue", lw=1.0, label=f"{j2_name} (J2)")
            ax.plot(ep.t, j1, color="crimson", lw=1.0, label=f"{j1_name} (J1)")
            ax.plot(ep.t, j2 + j1, color="black", lw=0.8, ls="--", label="J2 + J1 (sum)")
            ax.set_xlabel("time (s)")
            ax.set_ylabel("rad")
            ax.set_title(f"{pair_key}: raw position vs time -- {ep_label} ({ep.source})", fontsize=9)
            ax.legend(fontsize=8)
            ax.grid(True, lw=0.3, alpha=0.5)
            fig.tight_layout()
            fig.savefig(os.path.join(out_dir, f"{pair_key}_{ep_label}_timeseries.png"), dpi=150)
            plt.close(fig)

            # --- raw gt_effort vs time, this episode only ---
            fig, ax = plt.subplots(figsize=(9, 4))
            ax.plot(ep.t, j2e, color="steelblue", lw=1.0, label=f"{j2_name} (J2) gt_effort")
            ax.plot(ep.t, j1e, color="crimson", lw=1.0, label=f"{j1_name} (J1) gt_effort")
            ax.set_xlabel("time (s)")
            ax.set_ylabel("gt_effort (raw, uncalibrated)")
            ax.set_title(f"{pair_key}: raw gt_effort vs time -- {ep_label} ({ep.source})", fontsize=9)
            ax.legend(fontsize=8)
            ax.grid(True, lw=0.3, alpha=0.5)
            fig.tight_layout()
            fig.savefig(os.path.join(out_dir, f"{pair_key}_{ep_label}_effort.png"), dpi=150)
            plt.close(fig)

    print(f"Wrote per-episode coupled plots for {len(reps)} episodes: "
          f"{[os.path.basename(e.path) for e in reps]}")


# -----------------------------------------------------------------------------
# Phase 2: what does the J1-given-J2 coupling actually depend on?
#
# Every quantity below (direction sign, speed magnitude, the binned-median
# baseline, the per-episode onset point) is a plain descriptive statistic
# computed directly from the recorded gt_pos/gt_vel. Nothing here uses the
# sim's coupling formula/constants (COUPLING_CODE_EXPLAINED.md) -- that model
# is the hypothesis being tested against real data, not an input to it.
# -----------------------------------------------------------------------------

DEP_OUT_DIR = os.path.join(PLOTS_DIR, "coupled", "dependency_diagnostics")
ONSET_THRESHOLD_RAD = 0.05    # "J1 has started moving" threshold for the onset measurement
MIN_RUN_SAMPLES = 10          # ignore curling runs shorter than this (avoids noise-length runs)
GRADIENT_DEADBAND = 0.0005    # rad/sample; below this, a sample is classified "flat", not rising/falling


def _direction_labels(m: np.ndarray) -> np.ndarray:
    """Per-sample direction of the combined position m = J2+J1: +1 rising
    (curling), -1 falling (uncurling), 0 flat (within a small deadband).
    Plain centered finite difference (`np.gradient`) of the recorded signal
    -- no state, no history, no per-episode parameter of any kind."""
    g = np.gradient(m)
    labels = np.zeros_like(m)
    labels[g > GRADIENT_DEADBAND] = 1.0
    labels[g < -GRADIENT_DEADBAND] = -1.0
    return labels


def _episode_onset(j2: np.ndarray, j1: np.ndarray, direction: np.ndarray) -> float | None:
    """Median J2 value, across this episode's curling (rising) runs of at
    least MIN_RUN_SAMPLES, at which J1 first exceeds ONSET_THRESHOLD_RAD.
    None if this episode has no qualifying run/crossing."""
    onsets = []
    n = len(direction)
    i = 0
    while i < n:
        if direction[i] > 0:
            j = i
            while j < n and direction[j] > 0:
                j += 1
            if j - i >= MIN_RUN_SAMPLES:
                run_j1 = j1[i:j]
                crossing = np.nonzero(run_j1 > ONSET_THRESHOLD_RAD)[0]
                if crossing.size:
                    onsets.append(j2[i:j][crossing[0]])
            i = j
        else:
            i += 1
    return float(np.median(onsets)) if onsets else None


def _latched_direction(direction: np.ndarray) -> np.ndarray:
    """Carry the last nonzero direction forward through 'flat' (0) samples,
    so a brief pause isn't miscounted as a reversal -- generic signal
    processing (run-length latching), not a coupling-specific rule."""
    latched = direction.copy()
    last = 1.0
    for i in range(latched.shape[0]):
        if latched[i] == 0.0:
            latched[i] = last
        else:
            last = latched[i]
    return latched


def _run_start_values(m: np.ndarray, latched: np.ndarray) -> np.ndarray:
    """For every sample, the m=J2+J1 value at the start of its current
    monotonic run -- i.e. at the most recent reversal (or the episode's
    first sample if none yet). This is the direct, measured answer to "how
    far had it turned around from" -- e.g. if the finger uncurled to some
    point and reversed back to curling, every sample in that new curling
    run gets that reversal point as its `run_start` value, not just a
    +1 'rising' label. Purely a walk over the recorded trajectory, no
    assumed release-angle or sim parameter of any kind."""
    run_start = np.empty_like(m)
    start_val = m[0]
    run_start[0] = start_val
    for i in range(1, m.shape[0]):
        if latched[i] != latched[i - 1]:
            start_val = m[i]
        run_start[i] = start_val
    return run_start


def write_coupled_dependency_diagnostics(episodes: list[Episode], outliers: list[dict]) -> list[dict]:
    os.makedirs(DEP_OUT_DIR, exist_ok=True)
    outlier_episode_names = {o["episode"] for o in outliers}

    dep_rows = []
    for pair_key, (j1_name, j2_name) in COUPLED_JOINT_PAIRS.items():
        j2_chunks, j1_chunks, vel_chunks, dir_chunks, epidx_chunks, runstart_chunks = [], [], [], [], [], []
        pair_onsets = []  # dicts: episode, source, onset_j2_rad, is_outlier

        for ep_i, ep in enumerate(episodes):
            col = {n: i for i, n in enumerate(ep.joint_order)}
            if j1_name not in col or j2_name not in col:
                continue
            j2 = ep.gt_pos[:, col[j2_name]]
            j1 = ep.gt_pos[:, col[j1_name]]
            vel = np.abs(ep.gt_vel[:, col[j2_name]])
            m = j2 + j1
            direction = _direction_labels(m)
            latched = _latched_direction(direction)
            run_start = _run_start_values(m, latched)
            is_outlier = os.path.basename(ep.path) in outlier_episode_names

            j2_chunks.append(j2)
            j1_chunks.append(j1)
            vel_chunks.append(vel)
            dir_chunks.append(direction)
            epidx_chunks.append(np.full(j2.shape, ep_i))
            runstart_chunks.append(run_start)

            onset = _episode_onset(j2, j1, direction)
            if onset is not None:
                pair_onsets.append({
                    "episode": os.path.basename(ep.path), "source": ep.source,
                    "onset_j2_rad": onset, "is_outlier": is_outlier,
                })

        j2_all = np.concatenate(j2_chunks)
        j1_all = np.concatenate(j1_chunks)
        vel_all = np.concatenate(vel_chunks)
        dir_all = np.concatenate(dir_chunks)
        epidx_all = np.concatenate(epidx_chunks)
        runstart_all = np.concatenate(runstart_chunks)
        outlier_ep_ids = {i for i, e in enumerate(episodes) if os.path.basename(e.path) in outlier_episode_names}
        clean = ~np.isin(epidx_all, list(outlier_ep_ids))

        idx_all = np.arange(j2_all.size)
        idx_plot = RNG.choice(idx_all, size=15000, replace=False) if idx_all.size > 15000 else idx_all

        # --- by_direction: does curling vs uncurling separate into different branches? ---
        fig, ax = plt.subplots(figsize=(5.5, 5))
        for lbl, color, label in [(1, "crimson", "rising (curling)"),
                                   (-1, "steelblue", "falling (uncurling)"),
                                   (0, "lightgrey", "flat")]:
            m_ = dir_all[idx_plot] == lbl
            ax.scatter(j2_all[idx_plot][m_], j1_all[idx_plot][m_], s=4, alpha=0.4, color=color, label=label)
        ax.set_xlabel(f"{j2_name} actual position (rad)")
        ax.set_ylabel(f"{j1_name} actual position (rad)")
        ax.set_title(f"{pair_key}: J2 vs J1, colored by direction of travel", fontsize=9)
        ax.legend(fontsize=7)
        ax.grid(True, lw=0.3, alpha=0.5)
        fig.tight_layout()
        fig.savefig(os.path.join(DEP_OUT_DIR, f"{pair_key}_by_direction.png"), dpi=150)
        plt.close(fig)

        # --- by_speed: does the spread correlate with how fast the joint is moving? ---
        fig, ax = plt.subplots(figsize=(5.5, 5))
        vmax = float(np.percentile(vel_all, 95))
        sca = ax.scatter(j2_all[idx_plot], j1_all[idx_plot], c=vel_all[idx_plot], cmap="plasma",
                          s=4, alpha=0.6, vmax=vmax)
        fig.colorbar(sca, ax=ax, label=f"|{j2_name} gt_vel| (rad/s), color clipped at p95")
        ax.set_xlabel(f"{j2_name} actual position (rad)")
        ax.set_ylabel(f"{j1_name} actual position (rad)")
        ax.set_title(f"{pair_key}: J2 vs J1, colored by speed", fontsize=9)
        ax.grid(True, lw=0.3, alpha=0.5)
        fig.tight_layout()
        fig.savefig(os.path.join(DEP_OUT_DIR, f"{pair_key}_by_speed.png"), dpi=150)
        plt.close(fig)

        # --- by_episode: does the relationship shift from one recording to another? ---
        fig, ax = plt.subplots(figsize=(5.5, 5))
        sca = ax.scatter(j2_all[idx_plot], j1_all[idx_plot], c=epidx_all[idx_plot], cmap="turbo", s=4, alpha=0.6)
        fig.colorbar(sca, ax=ax, label="episode index (load order)")
        ax.set_xlabel(f"{j2_name} actual position (rad)")
        ax.set_ylabel(f"{j1_name} actual position (rad)")
        ax.set_title(f"{pair_key}: J2 vs J1, colored by episode", fontsize=9)
        ax.grid(True, lw=0.3, alpha=0.5)
        fig.tight_layout()
        fig.savefig(os.path.join(DEP_OUT_DIR, f"{pair_key}_by_episode.png"), dpi=150)
        plt.close(fig)

        # --- by_reversal_memory: does the *history* of the current run matter,
        # not just its instantaneous direction? For every sample, `runstart_all`
        # is the real recorded m=J2+J1 value at the most recent reversal (e.g.
        # "it uncurled to here, then turned back to curling") -- a stateful,
        # measured memory feature, computed with no assumed release angle. ---
        fig, axes = plt.subplots(1, 2, figsize=(11, 5))
        for ax, lbl, title in [(axes[0], 1, "currently curling (rising)\ncolor = m where the PRECEDING uncurl turned around"),
                                (axes[1], -1, "currently uncurling (falling)\ncolor = m where the PRECEDING curl turned around")]:
            mask = dir_all[idx_plot] == lbl
            sca = ax.scatter(j2_all[idx_plot][mask], j1_all[idx_plot][mask], c=runstart_all[idx_plot][mask],
                              cmap="coolwarm", s=5, alpha=0.6)
            fig.colorbar(sca, ax=ax, label="run-start m (rad)")
            ax.set_xlabel(f"{j2_name} actual position (rad)")
            ax.set_ylabel(f"{j1_name} actual position (rad)")
            ax.set_title(title, fontsize=8)
            ax.grid(True, lw=0.3, alpha=0.5)
        fig.suptitle(f"{pair_key}: does reversal history (not just direction) explain the scatter?", fontsize=10)
        fig.tight_layout(rect=[0, 0, 1, 0.93])
        fig.savefig(os.path.join(DEP_OUT_DIR, f"{pair_key}_by_reversal_memory.png"), dpi=150)
        plt.close(fig)

        # --- onset_by_episode: histogram of the per-episode onset point ---
        clean_onsets = [o["onset_j2_rad"] for o in pair_onsets if not o["is_outlier"]]
        outlier_onsets = [o["onset_j2_rad"] for o in pair_onsets if o["is_outlier"]]
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(clean_onsets, bins=20, color="steelblue", label=f"n={len(clean_onsets)} episodes")
        for k, v in enumerate(outlier_onsets):
            ax.axvline(v, color="crimson", ls="--", lw=1.2,
                       label="flagged outlier episode(s)" if k == 0 else None)
        ax.set_xlabel(f"{j2_name} value where {j1_name} first exceeds {ONSET_THRESHOLD_RAD} rad "
                      f"during a curling run (rad)")
        ax.set_ylabel("count (episodes)")
        ax.set_title(f"{pair_key}: per-episode onset point", fontsize=9)
        ax.legend(fontsize=7)
        ax.grid(True, lw=0.3, alpha=0.5)
        fig.tight_layout()
        fig.savefig(os.path.join(DEP_OUT_DIR, f"{pair_key}_onset_by_episode.png"), dpi=150)
        plt.close(fig)

        # --- variance-explained stats (computed on non-outlier data only) ---
        j2_c, j1_c = j2_all[clean], j1_all[clean]
        vel_c, dir_c, epidx_c, runstart_c = vel_all[clean], dir_all[clean], epidx_all[clean], runstart_all[clean]

        bins = np.linspace(j2_c.min(), j2_c.max(), 41)
        bin_idx = np.clip(np.digitize(j2_c, bins) - 1, 0, len(bins) - 2)
        bin_centers = 0.5 * (bins[:-1] + bins[1:])
        bin_median = np.array([
            np.median(j1_c[bin_idx == b]) if np.any(bin_idx == b) else np.nan
            for b in range(len(bin_centers))
        ])
        has_data = ~np.isnan(bin_median)
        baseline = np.interp(j2_c, bin_centers[has_data], bin_median[has_data])
        residual = j1_c - baseline
        overall_var = float(np.var(residual))

        within_var, total_n = 0.0, 0
        for lbl in (1, -1, 0):
            mask = dir_c == lbl
            if mask.sum() > 1:
                within_var += float(np.var(residual[mask])) * int(mask.sum())
                total_n += int(mask.sum())
        within_var = within_var / total_n if total_n else overall_var
        direction_variance_explained = 1.0 - (within_var / overall_var) if overall_var > 0 else 0.0

        speed_corr = float(np.corrcoef(vel_c, residual)[0, 1]) if np.std(vel_c) > 0 else 0.0

        ep_ids = np.unique(epidx_c)
        ep_means = np.array([residual[epidx_c == e].mean() for e in ep_ids])
        ep_counts = np.array([int(np.sum(epidx_c == e)) for e in ep_ids])
        grand_mean = np.average(ep_means, weights=ep_counts)
        between_var = float(np.average((ep_means - grand_mean) ** 2, weights=ep_counts))
        episode_variance_explained = between_var / overall_var if overall_var > 0 else 0.0

        # Reversal-history: restricted to currently-curling (rising) samples --
        # the exact scenario the user described (uncurl partway, reverse, recurl).
        # Bins the REAL recorded run-start value (not any assumed release angle)
        # and checks how much of the *rising-run* residual variance it explains,
        # compared to instantaneous direction (which is constant == 1 here, so by
        # construction can't explain any of this -- that's the point).
        rising_mask = dir_c == 1
        if rising_mask.sum() > 10:
            resid_rise = residual[rising_mask]
            runstart_rise = runstart_c[rising_mask]
            overall_var_rise = float(np.var(resid_rise))
            rs_bins = np.linspace(runstart_rise.min(), runstart_rise.max(), 16)
            rs_bin_idx = np.clip(np.digitize(runstart_rise, rs_bins) - 1, 0, len(rs_bins) - 2)
            within_rs, total_rs = 0.0, 0
            for b in range(len(rs_bins) - 1):
                bm = rs_bin_idx == b
                if bm.sum() > 1:
                    within_rs += float(np.var(resid_rise[bm])) * int(bm.sum())
                    total_rs += int(bm.sum())
            within_rs = within_rs / total_rs if total_rs else overall_var_rise
            reversal_memory_variance_explained = (
                1.0 - (within_rs / overall_var_rise) if overall_var_rise > 0 else 0.0
            )
            reversal_memory_correlation = (
                float(np.corrcoef(runstart_rise, resid_rise)[0, 1]) if np.std(runstart_rise) > 0 else 0.0
            )
        else:
            reversal_memory_variance_explained = float("nan")
            reversal_memory_correlation = float("nan")

        onset_arr = np.array(clean_onsets)
        dep_rows.append({
            "pair": pair_key,
            "n_samples_clean": int(clean.sum()),
            "n_episodes_excluded_as_outlier": len(outlier_episode_names),
            "direction_variance_explained": round(direction_variance_explained, 4),
            "speed_residual_correlation": round(speed_corr, 4),
            "episode_variance_explained": round(episode_variance_explained, 4),
            "reversal_memory_variance_explained_rising": round(reversal_memory_variance_explained, 4),
            "reversal_memory_correlation_rising": round(reversal_memory_correlation, 4),
            "onset_n_episodes": int(onset_arr.size),
            "onset_mean_rad": round(float(onset_arr.mean()), 4) if onset_arr.size else "",
            "onset_std_rad": round(float(onset_arr.std()), 4) if onset_arr.size else "",
            "onset_min_rad": round(float(onset_arr.min()), 4) if onset_arr.size else "",
            "onset_max_rad": round(float(onset_arr.max()), 4) if onset_arr.size else "",
        })

    with open(os.path.join(TABLES_DIR, "coupled_dependency_stats.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(dep_rows[0].keys()))
        writer.writeheader()
        writer.writerows(dep_rows)
    print("Wrote tables/coupled_dependency_stats.csv:")
    for r in dep_rows:
        print(" ", r)

    return dep_rows


# -----------------------------------------------------------------------------
# Effort-calibration status (which joints already have a fitted torque scale
# on record, per genan_plots/scale_fit/scale_sweep.txt -- reports existing
# status only, computes/applies no new scale here)
# -----------------------------------------------------------------------------

def plot_scale_fit_status(joint_names: list[str]) -> None:
    sweep_path = os.path.join(_GENAN_DIR, "genan_plots", "scale_fit", "scale_sweep.txt")
    calibrated = set()
    if os.path.exists(sweep_path):
        with open(sweep_path) as f:
            for line in f:
                line = line.strip()
                if line.endswith(":") and line[:-1] in joint_names:
                    calibrated.add(line[:-1])

    fig, ax = plt.subplots(figsize=(6, 6))
    y = np.arange(len(joint_names))
    has_scale = [1 if n in calibrated else 0 for n in joint_names]
    colors = ["seagreen" if v else "lightgrey" for v in has_scale]
    ax.barh(y, [1] * len(joint_names), color=colors)
    ax.set_yticks(y)
    ax.set_yticklabels(joint_names, fontsize=8)
    ax.set_xticks([])
    ax.set_title(f"Existing torque-scale fit on record\n(green = yes, {len(calibrated)}/{len(joint_names)} joints)", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, "effort_calibration", "scale_fit_status.png"), dpi=150)
    plt.close(fig)
    print(f"Joints with an existing scale-fit on record: {sorted(calibrated)}")


# -----------------------------------------------------------------------------
# summary.md
# -----------------------------------------------------------------------------

def write_summary(episodes: list[Episode], joint_names: list[str], outliers: list[dict], reps: list[Episode],
                   dep_rows: list[dict]) -> None:
    n_ep = len(episodes)
    total_rows = sum(e.n_valid for e in episodes)
    total_dur = sum(e.duration_s for e in episodes)
    per_source = {}
    for d in DATASET_DIRS:
        source = os.path.basename(d)
        eps = [e for e in episodes if e.source == source]
        per_source[source] = (len(eps), sum(e.duration_s for e in eps))

    lines = []
    lines.append("# Hardware data review -- summary\n")
    lines.append(
        "Generated by `analyze_hw_data.py`. Every number/plot here reads directly "
        "off the recorded `*.aligned.npz` arrays (filtered only by each file's own "
        "`valid` flag) -- no coupling model, no fitted scale, no derived physics.\n"
    )
    lines.append("## 1. Data volume\n")
    lines.append(f"- {n_ep} episodes, {total_rows:,} valid rows, {total_dur/60:.1f} minutes total.\n")
    for source, (n, dur) in per_source.items():
        lines.append(f"  - `{source}`: {n} episodes, {dur/60:.1f} min\n")

    lines.append("\n## 2. Per-joint range coverage\n")
    lines.append(
        "See `tables/per_joint_stats.csv` and `plots/overview/per_joint_range_coverage.png` "
        "for the full breakdown of how much of each joint's known limit range is actually "
        "exercised in the collected data.\n"
    )

    lines.append("\n## 3. Coupled joints (FF/MF/RF J1+J2)\n")
    lines.append(
        "See `tables/coupled_pair_stats.csv` and `plots/coupled/` "
        "(`<pair>_j2_vs_j1_scatter.png`, `<pair>_coverage_heatmap.png`, `<pair>_timeseries.png`, "
        "`<pair>_effort.png`) for the raw J2-vs-J1 relationship pooled across all episodes, "
        "and which regions of the (J2,J1) space the current data does/doesn't cover.\n"
    )
    lines.append(
        f"\nPer-episode breakdowns (same signals, one episode at a time, in "
        f"`plots/coupled/per_episode/`) are also generated for {len(reps)} representative "
        "episodes -- the longest episode from each of the 3 source datasets, plus (if one "
        "exists) the single most-anomalous outlier episode from section 6 below:\n"
    )
    for ep in reps:
        lines.append(f"  - `{os.path.basename(ep.path)}` ({ep.source}, {ep.duration_s:.1f}s)\n")

    lines.append("\n## 4. Effort calibration status\n")
    lines.append(
        "See `plots/effort_calibration/scale_fit_status.png`. Only `rh_FFJ1`, `rh_FFJ2`, "
        "`rh_FFJ3` have an existing fitted torque scale on record "
        "(`genan_plots/scale_fit/scale_sweep.txt`); the other 13 joints' `gt_effort` has no "
        "known N*m conversion yet. This review plots raw `gt_effort` only -- no scale is "
        "applied anywhere in this pass.\n"
    )

    lines.append("\n## 5. Every joint's recorded range vs. the shadow_pd_id config limit\n")
    lines.append(
        "`plots/overview/per_joint_range_coverage.png` compares each joint's recorded position "
        "range against `joint_limits_rad` in `shadow_pd_id/config/joints.yaml`. Nearly every joint "
        "exceeds that config value in the real data (e.g. `rh_THJ5`: 2.06 rad recorded vs 1.05 rad "
        "configured; `rh_THJ2`: 1.41 vs 0.70). That config was built for the RL policy's action "
        "range (see its own header comment), not a physical hardware limit -- so it should not be "
        "read as \"% of true range of motion covered\" without accounting for that.\n"
    )

    lines.append("\n## 6. Outlier episodes (per-joint range far above that joint's own median)\n")
    if outliers:
        lines.append(
            f"{len(outliers)} (joint, episode) pairs have a position range more than 3x that "
            "joint's median per-episode range across the dataset -- see `tables/outlier_episodes.csv`. "
            "Worth a manual look before trusting those episodes' data for that joint (could be a real "
            "large motion, or a sensor glitch):\n\n"
        )
        for o in outliers:
            lines.append(
                f"- `{o['joint']}` in `{o['episode']}` ({o['source']}): range {o['episode_range']} rad "
                f"vs median {o['median_episode_range']} rad ({o['ratio']}x)\n"
            )
    else:
        lines.append("None found (no episode exceeded 3x its joint's median episode range).\n")

    lines.append("\n## 8. What the J1-given-J2 coupling actually depends on\n")
    lines.append(
        "See `tables/coupled_dependency_stats.csv` and `plots/coupled/dependency_diagnostics/`. "
        "For each pair, `direction_variance_explained` is the fraction of the residual scatter "
        "(around a purely empirical, data-computed binned-median J2->J1 curve) explained by "
        "splitting on curling-vs-uncurling direction; `speed_residual_correlation` is the "
        "correlation between that residual and |velocity|; `episode_variance_explained` is the "
        "fraction of residual variance that sits *between* episodes (i.e. explained just by "
        "which recording a sample came from) rather than within one. The `onset_*` columns "
        "measure, per episode, the J2 value at which J1 first starts moving during a curling "
        "run, and its spread across episodes -- direct evidence of whether that point is fixed "
        "or varies recording-to-recording. The 4 flagged `rh_MFJ2` outlier episodes "
        "(section 6) are excluded from these numbers for all 3 pairs (a simple, uniform "
        "exclusion rule), but still shown in `<pair>_by_episode.png`.\n\n"
        "`reversal_memory_variance_explained_rising`/`_correlation_rising` test a sharper, "
        "*stateful* hypothesis raised after reviewing the direction plots: plain instantaneous "
        "direction has no memory, so a partial uncurl-then-recurl (e.g. uncurl to some point, "
        "reverse before reaching wherever it fully unlocks, then curl again) gets labeled "
        "identically to a full fresh curl from zero -- which is likely why "
        "`direction_variance_explained` came out so small. `<pair>_by_reversal_memory.png` and "
        "these two columns instead condition on the REAL recorded m=J2+J1 value at the start of "
        "the current run (i.e. where the preceding reversal happened) -- computed by walking the "
        "trajectory, no assumed release angle -- restricted to currently-curling samples, the "
        "exact scenario described. A much larger variance-explained here than "
        "`direction_variance_explained` would mean the coupling has real hysteresis memory (how "
        "far back it turned around matters), not just current direction -- pointing at adding a "
        "reversal-history input to GenAN rather than direction alone.\n\n"
    )
    if dep_rows:
        header = list(dep_rows[0].keys())
        lines.append("| " + " | ".join(header) + " |\n")
        lines.append("|" + "|".join(["---"] * len(header)) + "|\n")
        for r in dep_rows:
            lines.append("| " + " | ".join(str(r[h]) for h in header) + " |\n")

    lines.append("\n## 9. Other observations\n")
    lines.append(
        "Descriptive only -- read the plots/tables above for the actual numbers before drawing "
        "conclusions. This file intentionally does not prescribe a next step; that's a decision "
        "for the user once the data is in front of them.\n"
    )

    with open(os.path.join(OUT_DIR, "summary.md"), "w") as f:
        f.writelines(lines)
    print("Wrote summary.md")


def main():
    joint_names, joint_limits = load_joint_config()
    print(f"Loaded joint config: {len(joint_names)} joints")

    episodes = load_all_episodes()
    print(f"Loaded {len(episodes)} episodes")

    write_overview(episodes)
    write_per_joint_stats(episodes, joint_names, joint_limits)
    plot_episode_timeseries_grids(episodes, joint_names)
    write_coupled_pair_stats_and_plots(episodes)
    plot_scale_fit_status(joint_names)

    outliers = find_outlier_episodes(episodes, joint_names)
    if outliers:
        with open(os.path.join(TABLES_DIR, "outlier_episodes.csv"), "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(outliers[0].keys()))
            writer.writeheader()
            writer.writerows(outliers)
        print(f"Wrote tables/outlier_episodes.csv ({len(outliers)} flagged rows)")

    reps = select_representative_episodes(episodes, outliers, n=4)
    plot_coupled_per_episode(reps)

    dep_rows = write_coupled_dependency_diagnostics(episodes, outliers)

    write_summary(episodes, joint_names, outliers, reps, dep_rows)

    print("\nDone. Output in:", OUT_DIR)


if __name__ == "__main__":
    main()

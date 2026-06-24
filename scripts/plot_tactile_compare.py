"""Compare tactile signals between a sim rollout and a hardware rollout.

Loads two .npz recordings — one from record_tactile_sim.py (sim) and one from
my_policy_node.py (hardware, tactile_hw_*.npz) — and produces time-series
overlays. The runs are not time-synced, so both time axes start at t=0; the
comparison is qualitative (activation patterns / frequencies), not step-for-step.

Fingers are aligned by NAME (ff/mf/rf/th), not column index, so it doesn't
matter that sim stores body names like 'rh_ffdistal' while hardware stores 'ff'.

Usage:
    python plot_tactile_compare.py --sim tactile_sim.npz --hw tactile_hw_20260623_101500.npz
    python plot_tactile_compare.py --sim tactile_sim.npz --hw tactile_sim.npz   # sanity (overlays coincide)
"""

import argparse

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# Canonical finger tokens, in sim sensor order (no little finger in sim).
CANON_ORDER = ["ff", "mf", "rf", "th"]
_ALL_TOKENS = ["ff", "mf", "rf", "th", "lf"]


def canon_finger(name):
    """Map any finger label ('rh_ffdistal', 'ff', ...) to a canonical token."""
    n = str(name).lower()
    for tok in _ALL_TOKENS:
        if tok in n:
            return tok
    return n


def finger_columns(finger_names):
    """{canonical token -> column index} for a recording's finger_names array."""
    return {canon_finger(nm): j for j, nm in enumerate(finger_names)}


def time_axis(data):
    """Seconds-from-start for a recording (uses timestamps if present, else rl_dt)."""
    if "timestamps" in data.files and data["timestamps"].size > 0:
        ts = data["timestamps"].astype(np.float64)
        return ts - ts[0]
    T = data["tactile_binary"].shape[0]
    return np.arange(T) * float(data["rl_dt"])


def activation_fraction(binary_col):
    return float(np.mean(binary_col)) if binary_col.size else 0.0


def contact_events(binary_col):
    """Count rising edges (0 -> 1 transitions) = number of distinct contacts."""
    if binary_col.size < 2:
        return 0
    b = (binary_col > 0.5).astype(np.int8)
    return int(np.sum((b[1:] == 1) & (b[:-1] == 0)))


def main():
    parser = argparse.ArgumentParser(description="Compare sim vs hardware tactile recordings.")
    parser.add_argument("--sim", required=True, help="Sim recording (.npz from record_tactile_sim.py)")
    parser.add_argument("--hw", required=True, help="Hardware recording (.npz from my_policy_node.py)")
    parser.add_argument("--out", default="tactile_compare.pdf", help="Output figure (PDF or PNG).")
    args = parser.parse_args()

    sim = np.load(args.sim, allow_pickle=True)
    hw = np.load(args.hw, allow_pickle=True)

    sim_t = time_axis(sim)
    hw_t = time_axis(hw)
    sim_bin = sim["tactile_binary"]   # [T, n_fingers]
    hw_bin = hw["tactile_binary"]     # [T, 4]
    sim_cols = finger_columns(sim["finger_names"])
    hw_cols = finger_columns(hw["finger_names"])

    common = [f for f in CANON_ORDER if f in sim_cols and f in hw_cols]
    if not common:
        raise ValueError(
            f"No common fingers. sim={list(sim_cols)} hw={list(hw_cols)}"
        )

    # -------------------------------------------------------------------------
    # Figure 1: per-finger (4-dim) binary overlay, sim vs hardware
    # -------------------------------------------------------------------------
    n = len(common)
    fig1, axes = plt.subplots(n, 1, figsize=(11, 2.2 * n), sharex=True)
    if n == 1:
        axes = [axes]
    fig1.suptitle("Per-finger tactile contact: sim vs hardware", fontsize=13, fontweight="bold")

    for ax, fng in zip(axes, common):
        s = sim_bin[:, sim_cols[fng]]
        h = hw_bin[:, hw_cols[fng]]
        # small vertical offsets so overlapping 0/1 traces stay visible
        ax.step(sim_t, s + 0.00, where="post", color="steelblue", lw=1.3, label="sim")
        ax.step(hw_t, h + 0.05, where="post", color="darkorange", lw=1.3, label="hardware")
        ax.set_ylim(-0.15, 1.25)
        ax.set_yticks([0, 1])
        ax.set_ylabel("contact", fontsize=8)
        ax.grid(True, lw=0.3, alpha=0.5)
        ax.set_title(
            f"{fng}   |   activation: sim={activation_fraction(s):.2f}  hw={activation_fraction(h):.2f}",
            fontsize=9, pad=2,
        )
    axes[-1].set_xlabel("time from start (s)", fontsize=9)
    fig1.legend(
        handles=[
            Line2D([0], [0], color="steelblue", lw=1.5, label="sim"),
            Line2D([0], [0], color="darkorange", lw=1.5, label="hardware"),
        ],
        loc="upper right", fontsize=9,
    )
    fig1.tight_layout(rect=[0, 0, 1, 0.96])

    # -------------------------------------------------------------------------
    # Figure 2: raw signals (sim continuous norms | hardware 80-taxel heatmap)
    # -------------------------------------------------------------------------
    has_sim_norm = "tactile_norm" in sim.files
    has_hw_raw = "tactile_raw80" in hw.files
    fig2 = None
    if has_sim_norm or has_hw_raw:
        fig2, ax2 = plt.subplots(1, 2, figsize=(14, 5))
        fig2.suptitle("Raw tactile signals", fontsize=13, fontweight="bold")

        # Left: sim continuous per-finger force norm
        if has_sim_norm:
            norm = sim["tactile_norm"]
            for fng in common:
                ax2[0].plot(sim_t, norm[:, sim_cols[fng]], lw=1.0, label=fng)
            ax2[0].set_title("sim: per-finger contact-force norm", fontsize=10)
            ax2[0].set_xlabel("time from start (s)", fontsize=9)
            ax2[0].set_ylabel("force norm", fontsize=9)
            ax2[0].legend(fontsize=8)
            ax2[0].grid(True, lw=0.3, alpha=0.5)
        else:
            ax2[0].set_visible(False)

        # Right: hardware 80-taxel activity heatmap (taxel index x time)
        if has_hw_raw:
            raw80 = hw["tactile_raw80"]  # [T, 80]
            im = ax2[1].imshow(
                raw80.T, aspect="auto", origin="lower", interpolation="nearest",
                extent=[hw_t[0], hw_t[-1], 0, raw80.shape[1]], cmap="magma",
            )
            # finger-block boundaries (every 16 taxels)
            block = 16
            for b in range(block, raw80.shape[1], block):
                ax2[1].axhline(b, color="cyan", lw=0.5, alpha=0.6)
            ax2[1].set_title("hardware: 80 taxels (blocks: ff/mf/rf/lf/th)", fontsize=10)
            ax2[1].set_xlabel("time from start (s)", fontsize=9)
            ax2[1].set_ylabel("taxel index", fontsize=9)
            fig2.colorbar(im, ax=ax2[1], fraction=0.046, pad=0.04)
        else:
            ax2[1].set_visible(False)

        fig2.tight_layout(rect=[0, 0, 1, 0.96])

    # -------------------------------------------------------------------------
    # Save
    # -------------------------------------------------------------------------
    out = args.out
    stem = out.rsplit(".", 1)[0]
    ext = out.rsplit(".", 1)[1] if "." in out else "pdf"
    out1 = f"{stem}_perfinger.{ext}"
    fig1.savefig(out1, dpi=150, bbox_inches="tight")
    saved = [out1]
    if fig2 is not None:
        out2 = f"{stem}_raw.{ext}"
        fig2.savefig(out2, dpi=150, bbox_inches="tight")
        saved.append(out2)
    print("Saved:\n  " + "\n  ".join(saved))

    # -------------------------------------------------------------------------
    # Summary table
    # -------------------------------------------------------------------------
    print("\n--- Per-finger tactile summary (sim vs hardware) ---")
    print(f"{'finger':>8} | {'sim act%':>9} {'hw act%':>9} | {'sim events':>11} {'hw events':>10}")
    print("-" * 56)
    for fng in common:
        s = sim_bin[:, sim_cols[fng]]
        h = hw_bin[:, hw_cols[fng]]
        print(f"{fng:>8} | {activation_fraction(s) * 100:>8.1f}% {activation_fraction(h) * 100:>8.1f}% "
              f"| {contact_events(s):>11d} {contact_events(h):>10d}")
    print(f"\nsim: {sim_bin.shape[0]} steps over {sim_t[-1]:.1f}s   "
          f"hw: {hw_bin.shape[0]} steps over {hw_t[-1]:.1f}s")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Smooth real ShadowLite trajectory recordings + build GenAN delta-histories.

Isaac-free (pure numpy/scipy/torch), same convention as train_genan.py. Feeds
`compute_dynamics.py` and the Position-loss path in train_genan.py/
train_genan_single.py: this script's job is purely kinematic (smoothing +
differentiating recorded position), compute_dynamics.py's job is purely
dynamic (querying M(q)/C(q,qdot)/G(q) from the real robot model at the
kinematic states this script produces).

Loads the same AlignedTrajectoryDataset train_genan.py uses (reusing its
segment-boundary detection and coupled-joint command-splitting rather than
re-parsing raw .aligned.npz files), then per trajectory segment (smoothing
must not cross a segment boundary -- two recordings stitched back to back
have no continuous derivative at the seam) runs scipy.signal.savgol_filter
three times: deriv=0 (smoothed position), deriv=1 (velocity), deriv=2
(acceleration) -- all ANALYTIC derivatives of the fitted local polynomial,
not finite-differenced, matching AlignedTrajectoryDataset's own
`_finite_diff_velocity` in spirit but not implementation (that method stays
untouched; this is a separate, opt-in kinematic source used only by the
Position-loss training path).

Delta-histories (`history.py`'s existing `build_delta_history`, reused
unmodified) are also built and cached here for convenience/inspection, even
though train_genan.py already builds them on-the-fly from any (T, D) array +
dataset -- this script's cache is the more accurate (savgol-smoothed) input,
so downstream scripts should read q_hist/u_hist from here rather than
rebuilding from raw AlignedTrajectoryDataset.q_meas.

Usage:
    python preprocess.py
    python preprocess.py --window_length 15 --polyorder 3 --out cache/run.npz
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import torch
from scipy.signal import savgol_filter

from config_utils import load_config
from dataset_loader import AlignedTrajectoryDataset
from history import build_delta_history
from joint_config import load_joint_config

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_CONFIG = os.path.join(_THIS_DIR, "agents", "shadowlite", "default.yaml")
_DEFAULT_OUT = os.path.join(_THIS_DIR, "cache", "smoothed.npz")


def smooth_segment(
    q_seg: np.ndarray, dt: float, window_length: int, polyorder: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Savgol-smooth one (T, D) segment along axis 0.

    Returns (q_smooth, q_dot, q_ddot), each (T, D). `window_length` is
    clamped down to the largest valid odd value <= segment length -- short
    segments can't support the configured window. If even that clamped
    window is too short to support `polyorder` (segment shorter than
    polyorder+1 rows), falls back to the raw signal for position and
    np.gradient for derivatives rather than crashing on a handful of short
    segments -- flagged via a printed warning, not silently ignored.
    """
    t = q_seg.shape[0]
    wl = min(window_length, t)
    if wl % 2 == 0:
        wl -= 1
    if wl < polyorder + 1 or wl < 1:
        print(
            f"[WARN] segment length {t} too short for window_length={window_length}, "
            f"polyorder={polyorder} (clamped window={wl}); falling back to raw position "
            "+ np.gradient for this segment."
        )
        q_smooth = q_seg.copy()
        q_dot = np.gradient(q_seg, dt, axis=0)
        q_ddot = np.gradient(q_dot, dt, axis=0)
        return q_smooth, q_dot, q_ddot

    q_smooth = savgol_filter(q_seg, wl, polyorder, deriv=0, delta=dt, axis=0)
    q_dot = savgol_filter(q_seg, wl, polyorder, deriv=1, delta=dt, axis=0)
    q_ddot = savgol_filter(q_seg, wl, polyorder, deriv=2, delta=dt, axis=0)
    return q_smooth, q_dot, q_ddot


def preprocess(
    dataset: AlignedTrajectoryDataset,
    window_length: int = 11,
    polyorder: int = 3,
    history_len: int = 3,
    stride: int = 1,
) -> dict[str, np.ndarray]:
    """Smooth `dataset.q_meas` per segment and build delta-histories.

    Returns a dict of numpy arrays, all row-aligned to `dataset`'s own global
    indexing (same rows as `dataset.q_meas`/`q_cmd`/`q_torque`), ready to
    save directly to one .npz and to be indexed identically by
    `compute_dynamics.py` and the training scripts.
    """
    q_meas_np = dataset.q_meas.cpu().numpy()
    q_smooth = np.zeros_like(q_meas_np)
    q_dot = np.zeros_like(q_meas_np)
    q_ddot = np.zeros_like(q_meas_np)

    for s, e in zip(dataset.traj_starts.tolist(), dataset.traj_ends.tolist()):
        seg_smooth, seg_dot, seg_ddot = smooth_segment(q_meas_np[s : e + 1], dataset.rl_dt, window_length, polyorder)
        q_smooth[s : e + 1] = seg_smooth
        q_dot[s : e + 1] = seg_dot
        q_ddot[s : e + 1] = seg_ddot

    t_all = torch.arange(dataset.num_steps, dtype=torch.long)
    q_hist = build_delta_history(torch.as_tensor(q_smooth, dtype=torch.float32), t_all, history_len, stride, dataset)
    u_hist = build_delta_history(dataset.q_cmd, t_all, history_len, stride, dataset)

    return {
        "q_meas_smooth": q_smooth,
        "q_dot": q_dot,
        "q_ddot": q_ddot,
        "q_hist": q_hist.numpy(),
        "u_hist": u_hist.numpy(),
        "traj_starts": dataset.traj_starts.cpu().numpy(),
        "traj_ends": dataset.traj_ends.cpu().numpy(),
        "rl_dt": np.array(dataset.rl_dt, dtype=np.float64),
    }


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Smooth ShadowLite recordings + build GenAN delta-histories.")
    parser.add_argument("--config", type=str, default=_DEFAULT_CONFIG, help="Base agent yaml (dataset/genan sections).")
    parser.add_argument("--agent_cfg", type=str, default=None, help="Optional yaml merged OVER --config.")
    parser.add_argument(
        "--dataset", type=str, action="append", default=None,
        help="Override dataset.paths (repeatable) -- directories, glob patterns, or explicit files.",
    )
    parser.add_argument("--joints_yaml", type=str, default=None, help="Override path to joints.yaml.")
    parser.add_argument("--min_horizon", type=int, default=None)
    parser.add_argument("--history_len", type=int, default=None, help="Default: genan.history_len from --config.")
    parser.add_argument("--stride", type=int, default=None, help="Default: genan.stride from --config.")
    parser.add_argument("--window_length", type=int, default=11, help="savgol_filter window length (rows).")
    parser.add_argument("--polyorder", type=int, default=3, help="savgol_filter polynomial order.")
    parser.add_argument("--out", type=str, default=_DEFAULT_OUT, help="Output .npz cache path.")
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    cfg = load_config(args.config, args.agent_cfg)
    g = cfg["genan"]
    history_len = args.history_len if args.history_len is not None else g["history_len"]
    stride = args.stride if args.stride is not None else g["stride"]

    dataset_paths = args.dataset if args.dataset is not None else cfg["dataset"]["paths"]
    min_horizon = args.min_horizon if args.min_horizon is not None else cfg["dataset"]["min_horizon"]
    joint_names, joint_upper_limits = load_joint_config(args.joints_yaml)
    dataset = AlignedTrajectoryDataset(
        paths=dataset_paths, joint_names=joint_names, device="cpu",
        joint_upper_limits=joint_upper_limits, min_horizon=min_horizon,
    )
    print(
        f"[INFO] Loaded {len(dataset.paths)} file(s), {dataset.num_steps} steps, "
        f"{dataset.traj_starts.shape[0]} trajectory segment(s)."
    )

    out = preprocess(
        dataset, window_length=args.window_length, polyorder=args.polyorder,
        history_len=history_len, stride=stride,
    )
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    np.savez(args.out, joint_names=np.array(joint_names), **out)
    print(
        f"[INFO] Saved smoothed kinematics + delta-histories to {args.out} "
        f"({out['q_meas_smooth'].shape[0]} rows, history dims q={out['q_hist'].shape[1]} u={out['u_hist'].shape[1]})."
    )


if __name__ == "__main__":
    main()

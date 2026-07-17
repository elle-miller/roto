#!/usr/bin/env python3
"""Plot calibrated recorded torque (gt_effort_raw / scale) vs the known PD
baseline torque, for one joint -- a pure calibration diagnostic, no trained
GenAN network involved. Same comparison genan_plots/scale_fit's earlier
sim_vs_real_best_fit.png made for rh_FFJ3 (best scale ~11.7), generalized to
take an arbitrary joint name + scale so it works for any per-joint scale-fit
result (e.g. rh_FFJ2: scale=18.9965, rh_FFJ1: scale=33.7469).

`pd_baseline_torque` (losses.py) is the deterministic Kp*(q_cmd-q_meas) -
Kd*qdot_meas torque the identified PD controller would apply, using
`pd_gains.load_pd_gains`'s per-joint Kp/Kd (shadow_pd_id's identified
values). A good `scale` is one where `gt_effort_raw / scale` tracks that
known baseline closely -- confirming per-joint calibration, independent of
`losses.HARDWARE_EFFORT_TO_NM`'s single global scale (30.0) assumption.

Usage:
    python plot_scale_fit.py --joint rh_FFJ2 --scale 18.9965 --dataset dirA --traj_idx 46
    python plot_scale_fit.py --joint rh_FFJ1 --scale 33.7469 --dataset dirA --all_segments
"""

from __future__ import annotations

import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from dataset_loader import AlignedTrajectoryDataset
from joint_config import load_joint_config
from losses import pd_baseline_torque
from pd_gains import load_pd_gains


def plot_segment(
    dataset: AlignedTrajectoryDataset, joint_idx: int, joint_name: str, scale: float,
    kp: float, kd: float, seg_idx: int, out_dir: str,
) -> str:
    t_start, t_end = int(dataset.traj_starts[seg_idx]), int(dataset.traj_ends[seg_idx])
    t = torch.arange(t_start, t_end + 1)
    t_c = dataset.clamp(t)

    label_raw = dataset.q_torque[t_c][:, joint_idx]
    q_cmd = dataset.q_cmd[t_c][:, joint_idx]
    q_meas = dataset.q_meas[t_c][:, joint_idx]
    qdot_meas = dataset.q_meas_vel[t_c][:, joint_idx]

    calibrated = label_raw / scale
    tau_pd = pd_baseline_torque(q_cmd, q_meas, qdot_meas, kp, kd)

    mse = torch.mean((calibrated - tau_pd) ** 2).item()
    print(f"[{joint_name} segment {seg_idx}] t=[{t_start},{t_end}] scale={scale:g} "
          f"MSE(calibrated vs pd_baseline)={mse:.6f}")

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(t.numpy(), calibrated.numpy(), label=f"recorded / scale={scale:g}", linewidth=1.4)
    ax.plot(t.numpy(), tau_pd.numpy(), "--", label="pd_baseline_torque (known Kp/Kd)", linewidth=1.4)
    ax.axhline(0.0, color="gray", linewidth=0.5)
    ax.set_xlabel("timestep")
    ax.set_ylabel("torque (N*m)")
    ax.set_title(f"{joint_name} -- segment {seg_idx}, scale={scale:g}, MSE={mse:.4f}")
    ax.legend()
    fig.tight_layout()

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{joint_name}_scale_fit_segment_{seg_idx}.png")
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot calibrated recorded torque vs PD baseline for one joint.")
    parser.add_argument("--joint", type=str, required=True, help="Joint name, e.g. rh_FFJ2.")
    parser.add_argument("--scale", type=float, required=True, help="Best-fit calibration scale for this joint.")
    parser.add_argument(
        "--dataset", type=str, action="append", required=True,
        help="Aligned dataset directory/glob/file (repeatable).",
    )
    parser.add_argument("--joints_yaml", type=str, default=None)
    parser.add_argument("--min_horizon", type=int, default=1)
    parser.add_argument("--traj_idx", type=int, default=0, help="Which trajectory segment to plot.")
    parser.add_argument("--all_segments", action="store_true", help="Plot every segment instead of just --traj_idx.")
    parser.add_argument("--out_dir", type=str, default="genan_plots/scale_fit")
    args = parser.parse_args()

    joint_names, joint_upper_limits = load_joint_config(args.joints_yaml)
    dataset = AlignedTrajectoryDataset(
        paths=args.dataset, joint_names=joint_names, device="cpu",
        joint_upper_limits=joint_upper_limits, min_horizon=args.min_horizon,
    )
    joint_idx = joint_names.index(args.joint)
    kp, kd = load_pd_gains(args.joint)
    print(f"[INFO] {args.joint}: kp={kp:.4f} kd={kd:.4f} scale={args.scale:g}")

    segments = range(dataset.traj_starts.shape[0]) if args.all_segments else [args.traj_idx]
    for seg_idx in segments:
        out_path = plot_segment(dataset, joint_idx, args.joint, args.scale, kp, kd, seg_idx, args.out_dir)
        print(f"[INFO] Saved {out_path}")


if __name__ == "__main__":
    main()

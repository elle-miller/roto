#!/usr/bin/env python3
"""Plot predicted vs recorded torque for ONE joint's trajectory, from a
checkpoint saved by train_genan_single_joint.py.

No Isaac dependency, no simulator rollout -- this reads the recorded
trajectory data directly and compares the model's prediction against the
recorded q_torque (gt_effort) column for that joint, timestep by timestep,
for one contiguous trajectory segment at a time. This is a much cheaper
sanity check than play_genan.py's Isaac rollout: it tells you whether the
network learned to imitate the label at all, before you invest in wiring up
a sim evaluation.

Remember (DESIGN.md Decision 1): q_torque is uncalibrated. `ensemble(x)`
already de-standardizes back into the same (uncalibrated) scale the
recorded label lives in -- that is what gets plotted below. Judge SHAPE
agreement between the two curves, not absolute magnitude; the number itself
was never verified to be true N*m.

Usage:
    python plot_single_joint_trajectory.py --checkpoint out/genan_joint_ffj1.pt --dataset dirA --traj_idx 0
    python plot_single_joint_trajectory.py --checkpoint out/genan_joint_ffj1.pt --dataset dirA --all_segments
"""

from __future__ import annotations

import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from dataset_loader import AlignedTrajectoryDataset
from history import build_delta_history
from joint_config import load_joint_config
from model import GenANEnsemble


def load_ensemble(checkpoint_path: str) -> tuple[GenANEnsemble, dict]:
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    ensemble = GenANEnsemble(ckpt["input_dim"], ckpt["num_joints"], ensemble_size=ckpt["ensemble_size"])
    ensemble.load_state_dict(ckpt["ensemble_state_dict"])
    ensemble.eval()
    return ensemble, ckpt


def build_single_joint_input(
    dataset: AlignedTrajectoryDataset, t: torch.Tensor, history_len: int, stride: int
) -> torch.Tensor:
    """Same full-multi-joint history the model was trained on -- the label
    column narrowing happened at TRAINING time, not here; inference always
    needs the full 16-joint input.
    """
    q_hist = build_delta_history(dataset.q_meas, t, history_len, stride, dataset)
    u_hist = build_delta_history(dataset.q_cmd, t, history_len, stride, dataset)
    return torch.cat([q_hist, u_hist], dim=-1)


def plot_segment(
    dataset: AlignedTrajectoryDataset, ensemble: GenANEnsemble, ckpt: dict, seg_idx: int, out_dir: str
) -> str:
    joint_idx, joint_name = ckpt["joint_idx"], ckpt["joint_name"]
    t_start, t_end = int(dataset.traj_starts[seg_idx]), int(dataset.traj_ends[seg_idx])
    t = torch.arange(t_start, t_end + 1)

    x = build_single_joint_input(dataset, t, ckpt["history_len"], ckpt["stride"])
    label = dataset.q_torque[dataset.clamp(t)][:, joint_idx]

    with torch.no_grad():
        preds = ensemble(x)  # (ensemble_size, len(t), 1)
    mean_pred = preds.mean(dim=0).squeeze(-1)
    std_pred = preds.std(dim=0).squeeze(-1)

    mse = torch.mean((mean_pred - label) ** 2).item()
    print(f"[segment {seg_idx}] t=[{t_start},{t_end}] raw-scale MSE={mse:.6f}")

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(t.numpy(), label.numpy(), label="recorded (gt_effort)", linewidth=1.4)
    ax.plot(t.numpy(), mean_pred.numpy(), label="predicted (ensemble mean)", linewidth=1.4)
    if ensemble.ensemble_size > 1:
        ax.fill_between(
            t.numpy(), (mean_pred - std_pred).numpy(), (mean_pred + std_pred).numpy(),
            alpha=0.2, label="ensemble ±1 std",
        )
    ax.set_xlabel("timestep")
    ax.set_ylabel("torque (uncalibrated scale)")
    ax.set_title(f"joint {joint_idx} ({joint_name}) -- segment {seg_idx}, MSE={mse:.4f}")
    ax.legend()
    fig.tight_layout()

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{joint_name}_segment_{seg_idx}.png")
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot predicted vs recorded torque for one joint.")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument(
        "--dataset", type=str, action="append", required=True,
        help="Dataset paths -- must match what the checkpoint was trained on (repeatable).",
    )
    parser.add_argument("--joints_yaml", type=str, default=None)
    parser.add_argument("--min_horizon", type=int, default=1)
    parser.add_argument("--traj_idx", type=int, default=0, help="Which trajectory segment to plot.")
    parser.add_argument("--all_segments", action="store_true", help="Plot every segment instead of just --traj_idx.")
    parser.add_argument("--out_dir", type=str, default="genan_plots")
    args = parser.parse_args()

    joint_names, joint_upper_limits = load_joint_config(args.joints_yaml)
    dataset = AlignedTrajectoryDataset(
        paths=args.dataset, joint_names=joint_names, device="cpu",
        joint_upper_limits=joint_upper_limits, min_horizon=args.min_horizon,
    )
    ensemble, ckpt = load_ensemble(args.checkpoint)
    print(f"[INFO] Loaded checkpoint for joint {ckpt['joint_idx']} ({ckpt['joint_name']}), "
          f"ensemble_size={ckpt['ensemble_size']}, best_val_loss={ckpt['best_val_loss']:.6f}")

    segments = range(dataset.traj_starts.shape[0]) if args.all_segments else [args.traj_idx]
    for seg_idx in segments:
        out_path = plot_segment(dataset, ensemble, ckpt, seg_idx, args.out_dir)
        print(f"[INFO] Saved {out_path}")


if __name__ == "__main__":
    main()
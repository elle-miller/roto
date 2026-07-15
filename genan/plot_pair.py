#!/usr/bin/env python3
"""Plot predicted (two-share sum) vs recorded torque for a tendon-coupled
J1/J2 mimic pair's trajectory, from a checkpoint saved by train_genan_pair.py.

No Isaac dependency, no simulator rollout -- same style as plot_single.py.
Plots in the normalized [-1,1] space the loss actually compares
(losses.coupled_pair_activity_loss): the two individual tanh-bounded shares,
their sum, the real shared gt_effort target (either joint's q_torque column
-- verified bit-identical, see train_genan_pair.py), AND the activity-
weighted pseudo-labels each share was actually trained against
(target_a/target_b, see losses.coupled_pair_activity_weights) -- so it's
visible at a glance whether each share tracks its OWN target, not just
whether their sum fits the overall shape.

Usage:
    python plot_pair.py --checkpoint checkpoints/genan_pair_rh_FFJ1_rh_FFJ2.pt \\
        --dataset dirA --traj_idx 46
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
from losses import coupled_pair_activity_weights
from model import GenANEnsemble
from train_genan_pair import build_activity_inputs


def load_ensemble(checkpoint_path: str) -> tuple[GenANEnsemble, dict]:
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    torque_range = ckpt["torque_range"]
    ensemble = GenANEnsemble(
        ckpt["input_dim"], ckpt["num_joints"], ensemble_size=ckpt["ensemble_size"],
        bounded_output=True, torque_range=torque_range,
    )
    ensemble.load_state_dict(ckpt["ensemble_state_dict"])
    ensemble.eval()
    return ensemble, ckpt


def build_pair_input(
    dataset: AlignedTrajectoryDataset, t: torch.Tensor, history_len: int, stride: int
) -> torch.Tensor:
    """Same full-multi-joint history the model was trained on."""
    q_hist = build_delta_history(dataset.q_meas, t, history_len, stride, dataset)
    u_hist = build_delta_history(dataset.q_cmd, t, history_len, stride, dataset)
    return torch.cat([q_hist, u_hist], dim=-1)


def plot_segment(
    dataset: AlignedTrajectoryDataset, ensemble: GenANEnsemble, ckpt: dict, seg_idx: int, out_dir: str
) -> str:
    joint_a_idx, joint_b_idx = ckpt["joint_pair_idx"]
    joint_a_name, joint_b_name = ckpt["joint_pair_names"]
    torque_range = ckpt["torque_range"]
    t_start, t_end = int(dataset.traj_starts[seg_idx]), int(dataset.traj_ends[seg_idx])
    t = torch.arange(t_start, t_end + 1)

    x = build_pair_input(dataset, t, ckpt["history_len"], ckpt["stride"])
    t_c = dataset.clamp(t)
    label_raw = dataset.q_torque[t_c][:, joint_a_idx]  # == joint_b's column, verified in train_genan_pair.py
    label = (label_raw / torque_range).clamp(-1.0, 1.0)

    activity_window = ckpt.get("activity_window", ckpt["history_len"] * ckpt["stride"])  # old checkpoints lack this key
    q_a_now, q_a_past, q_b_now, q_b_past = build_activity_inputs(dataset, t, activity_window, joint_a_idx, joint_b_idx)
    activity_a, activity_b = coupled_pair_activity_weights(q_a_now, q_a_past, q_b_now, q_b_past)
    target_a = (activity_a.squeeze(-1) * label)
    target_b = (activity_b.squeeze(-1) * label)

    with torch.no_grad():
        preds_norm = ensemble.forward_standardized(x)  # (ensemble_size, len(t), 2), each column in (-1,1)
    mean_shares = preds_norm.mean(dim=0)  # (len(t), 2)
    share_a, share_b = mean_shares[:, 0], mean_shares[:, 1]
    pred_sum = share_a + share_b

    mse = torch.mean((pred_sum - label) ** 2).item()
    mse_a = torch.mean((share_a - target_a) ** 2).item()
    mse_b = torch.mean((share_b - target_b) ** 2).item()
    sign_agree_frac = ((share_a.sign() == share_b.sign())).float().mean().item()
    print(f"[segment {seg_idx}] t=[{t_start},{t_end}] MSE(sum vs target)={mse:.6f} "
          f"MSE(share_a vs target_a)={mse_a:.6f} MSE(share_b vs target_b)={mse_b:.6f} "
          f"sign_agreement={sign_agree_frac * 100:.1f}%")

    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(t.numpy(), label.numpy(), "k--", label="recorded (target, shared gt_effort)", linewidth=1.4)
    ax.plot(t.numpy(), pred_sum.numpy(), label="predicted sum (share_a + share_b)", linewidth=1.6)
    ax.plot(t.numpy(), share_a.numpy(), label=f"share_a ({joint_a_name})", linewidth=1.0, alpha=0.7)
    ax.plot(t.numpy(), share_b.numpy(), label=f"share_b ({joint_b_name})", linewidth=1.0, alpha=0.7)
    ax.plot(t.numpy(), target_a.numpy(), ":", label=f"target_a ({joint_a_name}, activity-weighted)", linewidth=0.9, alpha=0.6)
    ax.plot(t.numpy(), target_b.numpy(), ":", label=f"target_b ({joint_b_name}, activity-weighted)", linewidth=0.9, alpha=0.6)
    ax.axhline(0.0, color="gray", linewidth=0.5)
    ax.set_ylim(-2.05, 2.05)
    ax.set_xlabel("timestep")
    ax.set_ylabel(f"normalized torque (label/torque_range={torque_range:g})")
    ax.set_title(
        f"{joint_a_name}/{joint_b_name} -- segment {seg_idx}, MSE(sum)={mse:.4f}, "
        f"sign agreement={sign_agree_frac * 100:.1f}%"
    )
    ax.legend(fontsize=8)
    fig.tight_layout()

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{joint_a_name}_{joint_b_name}_segment_{seg_idx}.png")
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot predicted (two-share sum) vs recorded torque for a mimic pair.")
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
    joint_a_name, joint_b_name = ckpt["joint_pair_names"]
    print(f"[INFO] Loaded checkpoint for pair {joint_a_name}/{joint_b_name}, "
          f"ensemble_size={ckpt['ensemble_size']}, best_val_loss={ckpt['best_val_loss']:.6f}")

    segments = range(dataset.traj_starts.shape[0]) if args.all_segments else [args.traj_idx]
    for seg_idx in segments:
        out_path = plot_segment(dataset, ensemble, ckpt, seg_idx, args.out_dir)
        print(f"[INFO] Saved {out_path}")


if __name__ == "__main__":
    main()

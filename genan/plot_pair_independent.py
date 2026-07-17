#!/usr/bin/env python3
"""Post-hoc verification + plot for a tendon-coupled J1/J2 mimic pair trained
as TWO INDEPENDENT single-output networks (`train_genan_pair_independent.py`),
instead of one shared-trunk two-head network (`plot_pair.py`).

No Isaac dependency, no simulator rollout -- same style as plot_pair.py, but
loads TWO checkpoints (`plot_single.py`'s `load_ensemble` pattern, since
these are single-joint-schema checkpoints) instead of one. The two networks
were trained with NO shared loss/gradient at all (see
`train_genan_pair_independent.py`'s module docstring) -- this script is the
ONLY place the sum-matching property (`pred_a + pred_b ~= label_norm`) gets
checked, and it's purely a post-hoc read, not something enforced during
training. A large MSE(sum) here means the two independently-trained networks
disagree on the real split -- a real modeling problem to flag BEFORE trusting
independent deployment scales (`fit_torque_scale.py`) for each, not something
those scales should be used to paper over.

Usage:
    python plot_pair_independent.py --checkpoint_a checkpoints/..._pairindep_rh_FFJ1.pt \\
        --checkpoint_b checkpoints/..._pairindep_rh_FFJ2.pt --dataset dirA --traj_idx 46
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
from plot_single import load_ensemble
from train_genan_pair import build_activity_inputs


def build_pair_input(
    dataset: AlignedTrajectoryDataset, t: torch.Tensor, history_len: int, stride: int
) -> torch.Tensor:
    """Same full-multi-joint history both models were trained on."""
    q_hist = build_delta_history(dataset.q_meas, t, history_len, stride, dataset)
    u_hist = build_delta_history(dataset.q_cmd, t, history_len, stride, dataset)
    return torch.cat([q_hist, u_hist], dim=-1)


def plot_one_joint(
    t: torch.Tensor, label: torch.Tensor, target: torch.Tensor, pred: torch.Tensor,
    joint_name: str, torque_range: float, seg_idx: int, out_dir: str,
) -> str:
    """ONE network's own prediction against ITS OWN activity-weighted
    target, in its own figure -- no clutter from the other joint's curves.
    Also shows the raw shared label for context (dashed, faint), since a
    low-activity stretch can make `pred` look like it's "missing" the label
    when it's correctly tracking a near-zero target instead.
    """
    mse = torch.mean((pred - target) ** 2).item()
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(t.numpy(), label.numpy(), "k--", label="recorded (shared gt_effort)", linewidth=1.0, alpha=0.4)
    ax.plot(t.numpy(), target.numpy(), ":", label=f"{joint_name} activity-weighted target", linewidth=1.3)
    ax.plot(t.numpy(), pred.numpy(), label=f"{joint_name} predicted (independent model)", linewidth=1.4)
    ax.axhline(0.0, color="gray", linewidth=0.5)
    ax.set_ylim(-1.05, 1.05)
    ax.set_xlabel("timestep")
    ax.set_ylabel(f"normalized torque (label/torque_range={torque_range:g})")
    ax.set_title(f"{joint_name} (independent model) -- segment {seg_idx}, MSE(own target)={mse:.4f}")
    ax.legend(fontsize=8)
    fig.tight_layout()

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{joint_name}_independent_segment_{seg_idx}.png")
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    return out_path


def plot_segment(
    dataset: AlignedTrajectoryDataset,
    ensemble_a: GenANEnsemble, ckpt_a: dict,
    ensemble_b: GenANEnsemble, ckpt_b: dict,
    seg_idx: int, out_dir: str, max_steps: int | None = None,
) -> list[str]:
    joint_a_idx, joint_a_name = ckpt_a["joint_idx"], ckpt_a["joint_name"]
    joint_b_idx, joint_b_name = ckpt_b["joint_idx"], ckpt_b["joint_name"]
    torque_range_a, torque_range_b = ckpt_a["torque_range"], ckpt_b["torque_range"]
    if torque_range_a != torque_range_b:
        raise ValueError(
            f"checkpoint_a/checkpoint_b have different torque_range ({torque_range_a} vs {torque_range_b}) -- "
            "both must share the same normalization to compare/sum their outputs meaningfully."
        )
    torque_range = torque_range_a

    t_start, t_end = int(dataset.traj_starts[seg_idx]), int(dataset.traj_ends[seg_idx])
    if max_steps is not None:
        t_end = min(t_end, t_start + max_steps - 1)
    t = torch.arange(t_start, t_end + 1)

    x_a = build_pair_input(dataset, t, ckpt_a["history_len"], ckpt_a["stride"])
    x_b = build_pair_input(dataset, t, ckpt_b["history_len"], ckpt_b["stride"])
    t_c = dataset.clamp(t)
    label_raw = dataset.q_torque[t_c][:, joint_a_idx]  # == joint_b's column, verified at training time
    label = (label_raw / torque_range).clamp(-1.0, 1.0)

    activity_window = ckpt_a.get("activity_window", ckpt_a["history_len"] * ckpt_a["stride"])
    q_a_now, q_a_future, q_b_now, q_b_future = build_activity_inputs(dataset, t, activity_window, joint_a_idx, joint_b_idx)
    activity_a, activity_b = coupled_pair_activity_weights(q_a_now, q_a_future, q_b_now, q_b_future)
    target_a = activity_a.squeeze(-1) * label
    target_b = activity_b.squeeze(-1) * label

    with torch.no_grad():
        pred_a = ensemble_a.forward_standardized(x_a).mean(dim=0).squeeze(-1)  # (len(t),), in (-1,1)
        pred_b = ensemble_b.forward_standardized(x_b).mean(dim=0).squeeze(-1)
    pred_sum = pred_a + pred_b

    mse_sum = torch.mean((pred_sum - label) ** 2).item()
    mse_a = torch.mean((pred_a - target_a) ** 2).item()
    mse_b = torch.mean((pred_b - target_b) ** 2).item()
    corr_sum = torch.corrcoef(torch.stack([pred_sum, label]))[0, 1].item()
    sign_agree_frac = (pred_a.sign() == pred_b.sign()).float().mean().item()
    print(f"[segment {seg_idx}] t=[{t_start},{t_end}] MSE(sum vs label)={mse_sum:.6f} corr(sum,label)={corr_sum:.4f} "
          f"MSE({joint_a_name} vs its target)={mse_a:.6f} MSE({joint_b_name} vs its target)={mse_b:.6f} "
          f"sign_agreement={sign_agree_frac * 100:.1f}%")

    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(t.numpy(), label.numpy(), "k--", label="recorded (target, shared gt_effort)", linewidth=1.4)
    ax.plot(t.numpy(), pred_sum.numpy(), label=f"predicted sum ({joint_a_name} + {joint_b_name})", linewidth=1.6)
    ax.plot(t.numpy(), pred_a.numpy(), label=f"{joint_a_name} (independent model)", linewidth=1.0, alpha=0.7)
    ax.plot(t.numpy(), pred_b.numpy(), label=f"{joint_b_name} (independent model)", linewidth=1.0, alpha=0.7)
    ax.plot(t.numpy(), target_a.numpy(), ":", label=f"target_a ({joint_a_name}, activity-weighted)", linewidth=0.9, alpha=0.6)
    ax.plot(t.numpy(), target_b.numpy(), ":", label=f"target_b ({joint_b_name}, activity-weighted)", linewidth=0.9, alpha=0.6)
    ax.axhline(0.0, color="gray", linewidth=0.5)
    ax.set_ylim(-2.05, 2.05)
    ax.set_xlabel("timestep")
    ax.set_ylabel(f"normalized torque (label/torque_range={torque_range:g})")
    ax.set_title(
        f"{joint_a_name}/{joint_b_name} (INDEPENDENT models) -- segment {seg_idx}, MSE(sum)={mse_sum:.4f}, "
        f"sign agreement={sign_agree_frac * 100:.1f}%"
    )
    ax.legend(fontsize=8)
    fig.tight_layout()

    os.makedirs(out_dir, exist_ok=True)
    combined_path = os.path.join(out_dir, f"{joint_a_name}_{joint_b_name}_independent_segment_{seg_idx}.png")
    fig.savefig(combined_path, dpi=140)
    plt.close(fig)

    path_a = plot_one_joint(t, label, target_a, pred_a, joint_a_name, torque_range, seg_idx, out_dir)
    path_b = plot_one_joint(t, label, target_b, pred_b, joint_b_name, torque_range, seg_idx, out_dir)
    return [combined_path, path_a, path_b]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Post-hoc sum-check + plot for two INDEPENDENTLY trained mimic-pair torque models."
    )
    parser.add_argument("--checkpoint_a", type=str, required=True)
    parser.add_argument("--checkpoint_b", type=str, required=True)
    parser.add_argument(
        "--dataset", type=str, action="append", required=True,
        help="Dataset paths -- must match what the checkpoints were trained on (repeatable).",
    )
    parser.add_argument("--joints_yaml", type=str, default=None)
    parser.add_argument("--min_horizon", type=int, default=1)
    parser.add_argument("--traj_idx", type=int, default=0, help="Which trajectory segment to plot.")
    parser.add_argument("--all_segments", action="store_true", help="Plot every segment instead of just --traj_idx.")
    parser.add_argument("--max_steps", type=int, default=None, help="Zoom in on only the first N timesteps.")
    parser.add_argument("--out_dir", type=str, default="genan_plots")
    args = parser.parse_args()

    joint_names, joint_upper_limits = load_joint_config(args.joints_yaml)
    dataset = AlignedTrajectoryDataset(
        paths=args.dataset, joint_names=joint_names, device="cpu",
        joint_upper_limits=joint_upper_limits, min_horizon=args.min_horizon,
    )
    ensemble_a, ckpt_a = load_ensemble(args.checkpoint_a)
    ensemble_b, ckpt_b = load_ensemble(args.checkpoint_b)
    print(f"[INFO] Loaded {ckpt_a['joint_name']} (best_val_loss={ckpt_a['best_val_loss']:.6f}) and "
          f"{ckpt_b['joint_name']} (best_val_loss={ckpt_b['best_val_loss']:.6f}) -- both "
          f"independently_trained_pair={ckpt_a.get('independently_trained_pair', False)}/"
          f"{ckpt_b.get('independently_trained_pair', False)}")

    segments = range(dataset.traj_starts.shape[0]) if args.all_segments else [args.traj_idx]
    for seg_idx in segments:
        out_paths = plot_segment(dataset, ensemble_a, ckpt_a, ensemble_b, ckpt_b, seg_idx, args.out_dir, args.max_steps)
        for out_path in out_paths:
            print(f"[INFO] Saved {out_path}")


if __name__ == "__main__":
    main()

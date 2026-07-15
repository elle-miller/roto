#!/usr/bin/env python3
"""Train a GenAN dual-output torque model for ONE tendon-coupled J1/J2 mimic
pair (e.g. rh_FFJ1/rh_FFJ2), for fast per-pair debugging.

Diagnostic variant of train_genan.py, mirroring train_genan_single.py's
structure. The input is IDENTICAL to the full model: delta-histories of
q_meas/q_cmd across all 16 joints (DESIGN.md Decision 2). Only the ensemble's
output head narrows -- to TWO columns, one "share" per joint in the pair.

Why two outputs summed, not one: ShadowLite's FF/MF/RF J1/J2 pairs are
tendon-coupled on real hardware -- ONE motor drives both DOFs, so `gt_effort`
is recorded IDENTICALLY for both joints (verified empirically: 100%
bit-identical across the whole real dataset for all three pairs -- there is
no separate combined-actuator effort signal anywhere in the raw recordings,
see roto/roto/tasks/uan_shadowlite/dataset.py's module docstring). But in
sim, J1 and J2 are two fully independent PhysX-actuated DOFs
(`convert_mimic_joints_to_normal_joints: false`, coupling enforced only in
Python at the position/command level) -- so a sim rollout needs two separate
per-joint torques. This script predicts two INDEPENDENTLY tanh-bounded
(-1,1) "shares" (`GenANEnsemble(num_joints=2, bounded_output=True)` --
unmodified model.py architecture, see model.py's own docstring), supervised
against an ACTIVITY-WEIGHTED per-share pseudo-label
(`losses.coupled_pair_activity_loss`/`coupled_pair_activity_weights`): the
one shared `gt_effort` label is split between the two shares in proportion
to how much each joint actually displaced over a lookback window (NOT
single-step velocity -- `dataset.q_meas_vel` for J1/J2 is motor-level, not a
faithful per-joint signal, same issue as `gt_effort` itself; a windowed
position diff is trustworthy since `q_meas` position IS independently
faithful per-DOF, see DESIGN.md). This directly attributes torque to
whichever joint is actually moving (locked joint at a hard stop -> ~0 of the
shared torque credited to it). Per user decision, a hinge-style direction
penalty (`--direction_penalty_weight`, `losses.coupled_pair_hinge_direction_
loss`) is ALSO kept as an extra safety net on top of the activity loss (not
a sum-matching term -- the activity-weighted targets already subsume that,
`target_a + target_b == label_norm` exactly) -- see `losses.coupled_pair_loss`.

Always uses the fixed min-max torque_range/tanh-bounded scheme (no
RunningStandardScaler fallback -- this mode has no non-bounded variant, see
losses.coupled_pair_activity_loss). No residual/HARDWARE_EFFORT_TO_NM
calibration, no Position loss -- both out of scope for this script (see
train_genan_single.py for those).

Checkpoints from this script are pair-specific: they are NOT compatible with
play_genan.py's Isaac rollout (which expects a 16-joint torque vector), nor
with plot_single.py/train_genan_single.py's single-joint checkpoints. Use
plot_pair.py to inspect this script's output.

Usage:
    python train_genan_pair.py --joint_a rh_FFJ1 --joint_b rh_FFJ2 --torque_range 900
    python train_genan_pair.py --joint_a rh_FFJ1 --joint_b rh_FFJ2 --torque_range 900 \\
        --activity_window 16 --device cuda:0
"""

from __future__ import annotations

import argparse
import os

import torch

from config_utils import load_config
from dataset_loader import AlignedTrajectoryDataset
from history import build_delta_history
from joint_config import load_joint_config
from losses import (
    coupled_pair_activity_loss_terms,
    coupled_pair_activity_weights,
    coupled_pair_hinge_direction_loss,
    coupled_pair_loss,
)
from model import GenANEnsemble
from train_genan_single import resolve_joint_idx, split_segments

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_CONFIG = os.path.join(_THIS_DIR, "agents", "shadowlite", "default.yaml")


def build_inputs_and_labels(
    dataset: AlignedTrajectoryDataset, t: torch.Tensor, history_len: int, stride: int, joint_idx: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Same full-multi-joint input as train_genan_single.py's
    build_inputs_and_labels. Label is ONE column (either joint in the pair --
    they're bit-identical, see module docstring), not two: the two-column
    SPLIT is the network's job, not the label's.
    """
    q_hist = build_delta_history(dataset.q_meas, t, history_len, stride, dataset)
    u_hist = build_delta_history(dataset.q_cmd, t, history_len, stride, dataset)
    raw_input = torch.cat([q_hist, u_hist], dim=-1)
    t_c = dataset.clamp(t)
    torque_label = dataset.q_torque[t_c][:, joint_idx : joint_idx + 1]
    return raw_input, torque_label


def build_activity_inputs(
    dataset: AlignedTrajectoryDataset, t: torch.Tensor, window: int, joint_a_idx: int, joint_b_idx: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """`(q_a_now, q_a_past, q_b_now, q_b_past)` for `losses.
    coupled_pair_activity_weights` -- `past` is `window` steps back, clipped
    to THIS ROW'S OWN trajectory segment start (`dataset.segment_start`, same
    per-row boundary bound `history.build_delta_history` uses) so it never
    silently reads into a different, unrelated episode that happens to sit
    right before this one in the concatenated arrays. At/near a segment's own
    start, `past` collapses toward `now` (zero displacement on both sides),
    which is the correct fallback -- there's no real motion history to read
    yet, so `coupled_pair_activity_weights`'s `eps` stabilizer default the
    split to ~50/50.
    """
    t_c = dataset.clamp(t)
    seg_start = dataset.segment_start(t_c)
    t_past = dataset.clamp(torch.maximum(t_c - window, seg_start))
    q_a_now = dataset.q_meas[t_c][:, joint_a_idx : joint_a_idx + 1]
    q_a_past = dataset.q_meas[t_past][:, joint_a_idx : joint_a_idx + 1]
    q_b_now = dataset.q_meas[t_c][:, joint_b_idx : joint_b_idx + 1]
    q_b_past = dataset.q_meas[t_past][:, joint_b_idx : joint_b_idx + 1]
    return q_a_now, q_a_past, q_b_now, q_b_past


def train(
    dataset: AlignedTrajectoryDataset,
    joint_a_idx: int,
    joint_b_idx: int,
    torque_range: float,
    history_len: int = 3,
    stride: int = 1,
    ensemble_size: int = 5,
    epochs: int = 150,
    batch_size: int = 4096,
    lr: float = 1e-4,
    val_frac: float = 0.2,
    patience: int = 10,
    seed: int = 0,
    activity_window: int | None = None,
    direction_penalty_weight: float = 0.0,
    device: str = "cpu",
    lr_decay: bool = False,
) -> tuple[GenANEnsemble, dict]:
    """`torque_range`: required (not optional, unlike train_genan_single.py --
    this mode has no non-bounded fallback). `activity_window`: lookback (in
    steps) for `losses.coupled_pair_activity_weights`'s windowed displacement
    -- defaults to `history_len * stride` (see module docstring: reuses the
    same "how far back is history" span the network's own input already
    looks at, rather than a second independent hyperparameter).
    `direction_penalty_weight`: weight for the hinge direction-agreement
    safety net on top of the activity loss (see `losses.coupled_pair_loss`).

    Label is read from `joint_a_idx`'s own q_torque column -- verified
    bit-identical to `joint_b_idx`'s across the whole real dataset (see
    module docstring), so either would do.
    """
    if activity_window is None:
        activity_window = history_len * stride
    train_t, val_t = split_segments(dataset, val_frac=val_frac, seed=seed)
    if train_t.numel() == 0 or val_t.numel() == 0:
        raise ValueError(
            f"Need at least one trajectory in each split (train={train_t.numel()}, val={val_t.numel()})."
        )

    x_train, y_train = build_inputs_and_labels(dataset, train_t, history_len, stride, joint_a_idx)
    x_val, y_val = build_inputs_and_labels(dataset, val_t, history_len, stride, joint_a_idx)
    x_train, y_train = x_train.to(device), y_train.to(device)
    x_val, y_val = x_val.to(device), y_val.to(device)

    # Activity weights for the FULL val set, computed once (network-independent,
    # unlike train's per-step random batches -- see the training loop below).
    val_q_a_now, val_q_a_past, val_q_b_now, val_q_b_past = build_activity_inputs(
        dataset, val_t, activity_window, joint_a_idx, joint_b_idx
    )
    val_activity_a, val_activity_b = coupled_pair_activity_weights(val_q_a_now, val_q_a_past, val_q_b_now, val_q_b_past)
    val_activity_a, val_activity_b = val_activity_a.to(device), val_activity_b.to(device)

    input_dim = x_train.shape[1]
    ensemble = GenANEnsemble(
        input_dim, num_joints=2, ensemble_size=ensemble_size, seed=seed,
        bounded_output=True, torque_range=torque_range,
    )
    ensemble.to(device)
    ensemble.fit_scalers(x_train, y_train.repeat(1, 2))  # label_scaler unused in this mode, but fit_scalers needs matching width

    optimizers = [torch.optim.Adam(m.parameters(), lr=lr) for m in ensemble.members]
    generators = [torch.Generator().manual_seed(seed + 1000 + i) for i in range(ensemble_size)]
    schedulers = (
        [torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=lr * 0.01) for opt in optimizers]
        if lr_decay else None
    )

    best_val_loss = float("inf")
    best_state = None
    epochs_since_improvement = 0
    history_log = {"train_loss": [], "val_loss": []}

    n_train = x_train.shape[0]
    steps_per_epoch = max(1, n_train // batch_size)

    for epoch in range(epochs):
        epoch_losses = []
        for _ in range(steps_per_epoch):
            step_losses = []
            for member, opt, gen in zip(ensemble.members, optimizers, generators):
                idx = torch.randint(0, n_train, (min(batch_size, n_train),), generator=gen)  # CPU, for determinism
                idx_dev = idx.to(device)
                x = ensemble.input_scaler(x_train[idx_dev], train=False)
                pred_std = member(x)  # (batch, 2), each column tanh-bounded (-1,1)

                t_batch = train_t[idx]  # CPU: dataset is CPU-resident
                q_a_now, q_a_past, q_b_now, q_b_past = build_activity_inputs(
                    dataset, t_batch, activity_window, joint_a_idx, joint_b_idx
                )
                activity_a, activity_b = coupled_pair_activity_weights(q_a_now, q_a_past, q_b_now, q_b_past)
                activity_a, activity_b = activity_a.to(device), activity_b.to(device)
                loss = coupled_pair_loss(
                    pred_std, y_train[idx_dev], torque_range, activity_a, activity_b, direction_penalty_weight
                )

                opt.zero_grad()
                loss.backward()
                opt.step()
                step_losses.append(loss.item())
            if step_losses:
                epoch_losses.append(sum(step_losses) / len(step_losses))

        with torch.no_grad():
            preds_std_val = ensemble.forward_standardized(x_val)  # (ensemble_size, batch, 2)
            val_loss_t = coupled_pair_loss(
                preds_std_val, y_val, torque_range, val_activity_a, val_activity_b, direction_penalty_weight
            )
            val_mse_a, val_mse_b = coupled_pair_activity_loss_terms(
                preds_std_val, y_val, torque_range, val_activity_a, val_activity_b
            )
            val_hinge = coupled_pair_hinge_direction_loss(preds_std_val)
            val_loss = val_loss_t.item()

        train_loss = sum(epoch_losses) / len(epoch_losses)
        history_log["train_loss"].append(train_loss)
        history_log["val_loss"].append(val_loss)
        print(f"[epoch {epoch:4d}] train_loss={train_loss:.6f} val_loss={val_loss:.6f} "
              f"(val_mse_a={val_mse_a.item():.6f} val_mse_b={val_mse_b.item():.6f} val_hinge={val_hinge.item():.6f})")

        if schedulers is not None:
            for sch in schedulers:
                sch.step()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in ensemble.state_dict().items()}
            epochs_since_improvement = 0
        else:
            epochs_since_improvement += 1
            if epochs_since_improvement >= patience:
                print(f"[INFO] Early stopping at epoch {epoch} (no improvement for {patience} epochs).")
                break

    if best_state is not None:
        ensemble.load_state_dict(best_state)
    ensemble.to("cpu")  # see train_genan.py's train() -- callers always get a CPU-resident ensemble
    history_log["best_val_loss"] = best_val_loss
    return ensemble, history_log


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a mimic-pair dual-output GenAN torque model for ShadowLite.")
    parser.add_argument("--config", type=str, default=_DEFAULT_CONFIG, help="Base agent yaml (dataset/genan sections).")
    parser.add_argument("--agent_cfg", type=str, default=None, help="Optional yaml merged OVER --config.")
    parser.add_argument(
        "--dataset", type=str, action="append", default=None,
        help="Override dataset.paths (repeatable) -- directories, glob patterns, or explicit files.",
    )
    parser.add_argument("--joints_yaml", type=str, default=None, help="Override path to joints.yaml.")
    parser.add_argument("--joint_a", type=str, required=True, help="First joint in the pair (e.g. 'rh_FFJ1').")
    parser.add_argument("--joint_b", type=str, required=True, help="Second joint in the pair (e.g. 'rh_FFJ2').")
    parser.add_argument("--history_len", type=int, default=None)
    parser.add_argument("--stride", type=int, default=None)
    parser.add_argument("--ensemble_size", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--val_frac", type=float, default=None)
    parser.add_argument("--patience", type=int, default=None)
    parser.add_argument("--min_horizon", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--checkpoint", type=str, default=None, help="Override output checkpoint path.")
    parser.add_argument("--torque_range", type=float, required=True,
                         help="Fixed min-max torque normalization range (e.g. 900.0), REQUIRED -- bounds each "
                              "of the two output shares to (-1,1) via tanh; their sum is trained against "
                              "label/torque_range. No non-bounded fallback in this mode.")
    parser.add_argument("--activity_window", type=int, default=None,
                         help="Lookback (in steps) for the activity-weighted per-share pseudo-label split "
                              "(losses.coupled_pair_activity_weights). Default: history_len * stride.")
    parser.add_argument("--direction_penalty_weight", type=float, default=0.0,
                         help="Weight for the hinge direction-agreement penalty, ADDED ON TOP of the activity "
                              "loss (not a sum-matching term -- see losses.coupled_pair_loss). Default 0.0 "
                              "(off); per user decision this is a CLI-configurable safety net, not a fixed value.")
    parser.add_argument("--device", type=str, default="cpu",
                         help="Training device, e.g. 'cpu' (default) or 'cuda:0'. Dataset loading stays on "
                              "CPU regardless; only the network/training tensors move.")
    parser.add_argument("--lr_decay", action="store_true", default=None,
                         help="Cosine-anneal lr from its initial value down to lr*0.01 over `epochs`. "
                              "Default: off (constant lr).")
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    cfg = load_config(args.config, args.agent_cfg)
    g = cfg["genan"]
    overrides = {
        "history_len": args.history_len, "stride": args.stride, "ensemble_size": args.ensemble_size,
        "epochs": args.epochs, "batch_size": args.batch_size, "lr": args.lr, "val_frac": args.val_frac,
        "patience": args.patience,
    }
    for key, value in overrides.items():
        if value is not None:
            g[key] = value
    seed = args.seed if args.seed is not None else cfg["seed"]

    dataset_paths = args.dataset if args.dataset is not None else cfg["dataset"]["paths"]
    min_horizon = args.min_horizon if args.min_horizon is not None else cfg["dataset"]["min_horizon"]
    joint_names, joint_upper_limits = load_joint_config(args.joints_yaml)
    dataset = AlignedTrajectoryDataset(
        paths=dataset_paths, joint_names=joint_names, device="cpu",
        joint_upper_limits=joint_upper_limits, min_horizon=min_horizon,
    )
    joint_a_idx = resolve_joint_idx(args.joint_a, joint_names)
    joint_b_idx = resolve_joint_idx(args.joint_b, joint_names)
    joint_a_name, joint_b_name = joint_names[joint_a_idx], joint_names[joint_b_idx]

    # Sanity-check the core assumption this whole training mode depends on --
    # see module docstring / losses.coupled_pair_activity_loss.
    diff = (dataset.q_torque[:, joint_a_idx] - dataset.q_torque[:, joint_b_idx]).abs()
    if diff.max().item() > 1e-3:
        raise ValueError(
            f"{joint_a_name}/{joint_b_name} q_torque columns are NOT bit-identical (max abs diff="
            f"{diff.max().item():.4f}) -- this training mode assumes a shared tendon-coupled label "
            f"(verified for FF/MF/RF pairs; if this pair genuinely differs, coupled_pair_activity_loss's "
            f"design doesn't apply as-is)."
        )
    print(f"[INFO] Loaded {len(dataset.paths)} file(s), {dataset.num_steps} steps, "
          f"{dataset.traj_starts.shape[0]} trajectory segment(s). Training pair {joint_a_name}/{joint_b_name} "
          f"(q_torque max abs diff={diff.max().item():.2e}, confirmed shared label).")

    checkpoint_path = args.checkpoint or os.path.join(
        cfg["experiment"]["checkpoint_dir"], f"{cfg['experiment']['experiment_name']}_pair_{joint_a_name}_{joint_b_name}.pt"
    )
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)

    ensemble, history_log = train(
        dataset, joint_a_idx=joint_a_idx, joint_b_idx=joint_b_idx, torque_range=args.torque_range,
        history_len=g["history_len"], stride=g["stride"], ensemble_size=g["ensemble_size"],
        epochs=g["epochs"], batch_size=g["batch_size"], lr=g["lr"],
        val_frac=g["val_frac"], patience=g["patience"], seed=seed,
        activity_window=args.activity_window,
        direction_penalty_weight=args.direction_penalty_weight,
        device=args.device,
        lr_decay=args.lr_decay if args.lr_decay is not None else g.get("lr_decay", False),
    )

    resolved_activity_window = args.activity_window if args.activity_window is not None else g["history_len"] * g["stride"]
    torch.save(
        {
            "ensemble_state_dict": ensemble.state_dict(),
            "input_dim": ensemble.members[0].trunk[0].in_features,
            "num_joints": 2,
            "ensemble_size": ensemble.ensemble_size,
            "history_len": g["history_len"],
            "stride": g["stride"],
            "joint_names": joint_names,  # full 16-joint order -- needed to rebuild the input at inference time
            "joint_pair_idx": (joint_a_idx, joint_b_idx),
            "joint_pair_names": (joint_a_name, joint_b_name),
            "coupled_pair": True,
            "torque_range": args.torque_range,
            "bounded_output": True,
            "activity_window": resolved_activity_window,
            "direction_penalty_weight": args.direction_penalty_weight,
            "best_val_loss": history_log["best_val_loss"],
        },
        checkpoint_path,
    )
    print(f"[INFO] Saved checkpoint to {checkpoint_path} (best val_loss={history_log['best_val_loss']:.6f}).")


if __name__ == "__main__":
    main()

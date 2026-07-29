#!/usr/bin/env python3
"""Train a residual position-command network for all 16 joints (coupled
pairs included), replacing the abandoned torque-scaling approach
(fit_torque_scale.py) entirely -- no Kp-zeroing, no per-joint torque
injection, no _asymmetric_backlash.

Deployed command: `q_cmd_final = q_cmd_heuristic + Delta_q_cmd`, fed through
the SAME real, fixed Kp/Kd via set_joint_position_target (play_qcmd_residual.py).
Since `tau = Kp*(q_cmd_heuristic + Delta_q_cmd - q_meas) - Kd*qdot
          = tau_PD_baseline + Kp*Delta_q_cmd`,
a residual q_cmd IS a residual torque / Kp -- but expressed as a position
offset it composes with PD instead of fighting/replacing it.

Training target (position-only, no torque, no M/C/G dynamics model -- see
command_lag.py's module docstring and this session's finding that the offline
M/C/G cache is currently unreliable):

    Delta_q_cmd_target_j(t) = q_meas_j(t + L_j) - q_cmd_heuristic_j(t)

`L_j` is each joint's own command-to-position lag (command_lag.py, run once
offline). One network sees ALL 16 joints' history (build_delta_history,
unchanged from train_genan.py) and outputs all 16 joints' Delta_q_cmd --
coupling between a driver/mimic pair is learned internally from their shared
input history and jointly-real targets, not hand-engineered.

Isaac-free (pure torch), CPU-only, mirrors train_genan.py's structure.

Usage:
    python train_qcmd_residual.py --epochs 100 --max_delta_deg 20
"""

from __future__ import annotations

import argparse
import os

import torch
import torch.nn.functional as F
import yaml

from command_lag import compute_joint_lags
from config_utils import load_config
from dataset_loader import AlignedTrajectoryDataset
from history import build_delta_history
from joint_config import load_joint_config
from model import GenANEnsemble

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_CONFIG = os.path.join(_THIS_DIR, "agents", "shadowlite", "default.yaml")
_DEFAULT_LAGS = os.path.join(_THIS_DIR, "agents", "shadowlite", "command_lags.yaml")


def split_segments(dataset: AlignedTrajectoryDataset, val_frac: float = 0.2, seed: int = 0):
    """Identical trajectory-level split to train_genan.py's -- duplicated per
    this repo's established convention (see train_genan_single.py's
    split_segments docstring) so this script has no cross-file dependency.
    """
    n_seg = dataset.traj_starts.shape[0]
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n_seg, generator=g)
    n_val = max(1, round(n_seg * val_frac)) if n_seg > 1 else 0
    val_segs, train_segs = perm[:n_val], perm[n_val:]

    def _indices_for(segs):
        chunks = [torch.arange(int(dataset.traj_starts[s]), int(dataset.traj_ends[s]) + 1) for s in segs.tolist()]
        return torch.cat(chunks) if chunks else torch.empty(0, dtype=torch.long)

    return _indices_for(train_segs), _indices_for(val_segs)


def build_targets(dataset: AlignedTrajectoryDataset, t: torch.Tensor, lags: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Delta_q_cmd_target(t) = q_meas(t + L_j) - q_cmd(t), per joint, plus a
    valid_mask (False wherever t+L_j spills past t's own segment end -- same
    boundary-respecting logic build_delta_history uses for the past side,
    mirrored here for the future side via dataset.segment_end).
    """
    t_c = dataset.clamp(t)
    seg_end = dataset.segment_end(t_c)  # (N,) this row's own segment's last valid index
    t_future = t_c.unsqueeze(-1) + lags.unsqueeze(0)  # (N, num_joints)
    valid = t_future <= seg_end.unsqueeze(-1)  # (N, num_joints)
    t_future_c = dataset.clamp(t_future.reshape(-1)).reshape(t_future.shape)

    # Each joint j needs its OWN future index (t_future_c[:, j]) applied to its OWN
    # column j -- not every joint's column at every other joint's future index -- so
    # this is a per-column gather, done as a cheap loop (num_joints=16, once per call).
    q_meas_at_lag = torch.stack(
        [dataset.q_meas[t_future_c[:, j], j] for j in range(dataset.num_joints)], dim=-1
    )
    target = q_meas_at_lag - dataset.q_cmd[t_c]
    return target, valid


def build_input(dataset: AlignedTrajectoryDataset, t: torch.Tensor, history_len: int, stride: int) -> torch.Tensor:
    q_hist = build_delta_history(dataset.q_meas, t, history_len, stride, dataset)
    u_hist = build_delta_history(dataset.q_cmd, t, history_len, stride, dataset)
    return torch.cat([q_hist, u_hist], dim=-1)


def train(
    dataset: AlignedTrajectoryDataset,
    lags: torch.Tensor,
    max_delta: float,
    history_len: int = 4,
    stride: int = 2,
    ensemble_size: int = 5,
    epochs: int = 150,
    batch_size: int = 4096,
    lr: float = 1e-3,
    val_frac: float = 0.2,
    patience: int = 10,
    seed: int = 0,
) -> tuple[GenANEnsemble, dict]:
    train_t, val_t = split_segments(dataset, val_frac=val_frac, seed=seed)
    if train_t.numel() == 0 or val_t.numel() == 0:
        raise ValueError(f"Need at least one trajectory in each split (train={train_t.numel()}, val={val_t.numel()}).")

    x_train = build_input(dataset, train_t, history_len, stride)
    x_val = build_input(dataset, val_t, history_len, stride)
    y_train, valid_train = build_targets(dataset, train_t, lags)
    y_val, valid_val = build_targets(dataset, val_t, lags)

    input_dim = x_train.shape[1]
    num_joints = dataset.num_joints
    ensemble = GenANEnsemble(input_dim, num_joints, ensemble_size=ensemble_size, seed=seed,
                              bounded_output=True, torque_range=1.0)
    ensemble.input_scaler(x_train, train=True)

    optimizers = [torch.optim.Adam(m.parameters(), lr=lr) for m in ensemble.members]
    generators = [torch.Generator().manual_seed(seed + 1000 + i) for i in range(ensemble_size)]

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
                idx = torch.randint(0, n_train, (min(batch_size, n_train),), generator=gen)
                x = ensemble.input_scaler(x_train[idx], train=False)
                raw = member(x)  # (batch, num_joints) tanh in (-1,1)
                pred_delta = raw * max_delta
                mask = valid_train[idx]
                if not mask.any():
                    continue
                loss = F.mse_loss(pred_delta[mask], y_train[idx][mask])
                opt.zero_grad()
                loss.backward()
                opt.step()
                step_losses.append(loss.item())
            if step_losses:
                epoch_losses.append(sum(step_losses) / len(step_losses))

        with torch.no_grad():
            val_losses = []
            for member in ensemble.members:
                x = ensemble.input_scaler(x_val, train=False)
                raw = member(x)
                pred_delta = raw * max_delta
                val_losses.append(F.mse_loss(pred_delta[valid_val], y_val[valid_val]).item())
            val_loss = sum(val_losses) / len(val_losses)

        train_loss = sum(epoch_losses) / len(epoch_losses) if epoch_losses else float("nan")
        history_log["train_loss"].append(train_loss)
        history_log["val_loss"].append(val_loss)
        print(f"[epoch {epoch:4d}] train_mse={train_loss:.8f} val_mse={val_loss:.8f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in ensemble.state_dict().items()}
            epochs_since_improvement = 0
        else:
            epochs_since_improvement += 1
            if epochs_since_improvement >= patience:
                print(f"[INFO] Early stopping at epoch {epoch}.")
                break

    if best_state is not None:
        ensemble.load_state_dict(best_state)
    history_log["best_val_loss"] = best_val_loss
    return ensemble, history_log


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a residual-q_cmd GenAN model for ShadowLite.")
    parser.add_argument("--config", type=str, default=_DEFAULT_CONFIG)
    parser.add_argument("--dataset", type=str, action="append", default=None)
    parser.add_argument("--lags", type=str, default=_DEFAULT_LAGS, help="command_lag.py's output yaml.")
    parser.add_argument("--max_delta_deg", type=float, default=20.0, help="Bound on |Delta_q_cmd| per joint.")
    parser.add_argument("--history_len", type=int, default=4)
    parser.add_argument("--stride", type=int, default=2)
    parser.add_argument("--ensemble_size", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch_size", type=int, default=4096)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--val_frac", type=float, default=0.2)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--checkpoint", type=str, default=None)
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    dataset_paths = args.dataset if args.dataset is not None else cfg["dataset"]["paths"]
    joint_names, joint_upper_limits = load_joint_config()
    dataset = AlignedTrajectoryDataset(
        paths=dataset_paths, joint_names=joint_names, device="cpu",
        joint_upper_limits=joint_upper_limits, min_horizon=cfg["dataset"].get("min_horizon", 1),
    )
    print(f"[INFO] {dataset.num_steps} rows, {dataset.traj_starts.shape[0]} segments")

    if os.path.exists(args.lags):
        with open(args.lags) as f:
            lags_dict = yaml.safe_load(f)["lags"]
    else:
        print("[WARN] No command_lags.yaml found -- computing lags now (run command_lag.py separately to cache this).")
        lags_dict = compute_joint_lags(dataset, joint_names)
    lags = torch.tensor([lags_dict[n] for n in joint_names], dtype=torch.long)
    print(f"[INFO] Per-joint lags (steps): {dict(zip(joint_names, lags.tolist()))}")

    max_delta = args.max_delta_deg * 3.14159265 / 180.0

    ensemble, history_log = train(
        dataset, lags, max_delta,
        history_len=args.history_len, stride=args.stride, ensemble_size=args.ensemble_size,
        epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
        val_frac=args.val_frac, patience=args.patience, seed=args.seed,
    )

    checkpoint_path = args.checkpoint or os.path.join(
        cfg["experiment"]["checkpoint_dir"], "qcmd_residual_default.pt"
    )
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    torch.save({
        "ensemble_state_dict": ensemble.state_dict(),
        "input_dim": ensemble.members[0].trunk[0].in_features,
        "num_joints": dataset.num_joints,
        "ensemble_size": ensemble.ensemble_size,
        "history_len": args.history_len,
        "stride": args.stride,
        "joint_names": joint_names,
        "lags": lags_dict,
        "max_delta_rad": max_delta,
        "best_val_loss": history_log["best_val_loss"],
    }, checkpoint_path)
    print(f"[INFO] Saved checkpoint to {checkpoint_path} (best val_loss={history_log['best_val_loss']:.8f})")


if __name__ == "__main__":
    main()

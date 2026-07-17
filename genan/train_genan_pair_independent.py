#!/usr/bin/env python3
"""Train TWO fully independent GenAN single-output torque models for a
tendon-coupled J1/J2 mimic pair (e.g. rh_FFJ1/rh_FFJ2), instead of ONE
shared-trunk two-head network (`train_genan_pair.py`).

Phase 7 (see roto's own planning notes, not reproduced here) established
that rh_FFJ1/rh_FFJ2 must share ONE deployment-time scale under
`train_genan_pair.py`'s design: its two output "shares" come from ONE
network's shared trunk, trained JOINTLY via `losses.coupled_pair_activity_loss`
so `share_a + share_b` reconstructs the one real shared `gt_effort` signal --
independently rescaling two outputs that came from one shared loss/trunk
would distort the ratio the network learned between them.

This script removes that constraint at its root: TWO fully separate
`GenANEnsemble(num_joints=1)` models, each with its OWN optimizer(s), each
trained ONLY against its OWN activity-weighted share
(`losses.single_share_activity_loss`) -- no combined loss, no shared trunk,
no cross-network gradient at all. The sum-matching property
(`pred_a + pred_b ~= label_norm`) is then only a POST-HOC verification check
on the two independently-trained models (see `plot_pair.py`'s existing
sum-check, or a smoke test at the bottom of this run), not something enforced
during training. Because nothing here ties the two networks' output
magnitudes together, each one can legitimately get its own independently-fit
deployment scale later (`fit_torque_scale.py`) -- there's no jointly-learned
ratio a mismatched scale could distort, only each network's own
training-time calibration to correct for.

Activity weights (`losses.coupled_pair_activity_weights`, reused unchanged)
still need BOTH joints' `q_meas` history to compute "how much did THIS joint
move relative to the other" -- that cross-joint READ is still required, it's
only the two networks' GRADIENTS that are now fully decoupled.

Checkpoints from this script are saved in the SAME single-joint schema
`train_genan_single.py` uses (`num_joints=1`, `joint_idx`, `joint_name`,
`single_joint: True`, `bounded_output`/`torque_range`) -- deliberately, so
`fit_torque_scale.py`'s existing `load_ffj3_ensemble` loader works on them
UNCHANGED (it was already joint-agnostic despite the name). An extra
`pair_partner_joint`/`activity_window` field records which OTHER joint this
one was trained alongside, for traceability only (not read by any loader).

Usage:
    python train_genan_pair_independent.py --joint_a rh_FFJ1 --joint_b rh_FFJ2 --torque_range 900
"""

from __future__ import annotations

import argparse
import os

import torch

from config_utils import load_config
from dataset_loader import AlignedTrajectoryDataset
from history import build_delta_history
from joint_config import load_joint_config
from losses import coupled_pair_activity_weights, single_share_activity_loss
from model import GenANEnsemble
from train_genan_pair import build_activity_inputs
from train_genan_single import resolve_joint_idx, split_segments

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_CONFIG = os.path.join(_THIS_DIR, "agents", "shadowlite", "default.yaml")


def build_inputs_and_labels(
    dataset: AlignedTrajectoryDataset, t: torch.Tensor, history_len: int, stride: int, joint_idx: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Identical to train_genan_pair.py's build_inputs_and_labels -- the
    label is the ONE shared `gt_effort` column (either joint in the pair
    works, they're bit-identical), the input is the full 16-joint history.
    """
    q_hist = build_delta_history(dataset.q_meas, t, history_len, stride, dataset)
    u_hist = build_delta_history(dataset.q_cmd, t, history_len, stride, dataset)
    raw_input = torch.cat([q_hist, u_hist], dim=-1)
    t_c = dataset.clamp(t)
    torque_label = dataset.q_torque[t_c][:, joint_idx : joint_idx + 1]
    return raw_input, torque_label


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
    device: str = "cpu",
    lr_decay: bool = False,
) -> tuple[GenANEnsemble, GenANEnsemble, dict]:
    """Trains TWO independent `GenANEnsemble(num_joints=1)` models -- one for
    `joint_a_idx`, one for `joint_b_idx` -- each with its own optimizer(s),
    each stepped from `losses.single_share_activity_loss` computed against
    ONLY its own activity-weighted share. No shared loss/backward pass ties
    them together (see module docstring and `losses.single_share_activity_loss`'s
    own docstring for why this matters for downstream independent
    scale-fitting). Returns `(ensemble_a, ensemble_b, history_log)`.

    Early stopping/checkpoint selection uses the SUM `val_loss_a + val_loss_b`
    as one combined criterion -- still saves each ensemble's OWN best state at
    that shared best epoch (not two independently-chosen best epochs), so the
    reported val loss stays simple to reason about and the two checkpoints
    stay temporally paired to a single training snapshot.
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

    val_q_a_now, val_q_a_past, val_q_b_now, val_q_b_past = build_activity_inputs(
        dataset, val_t, activity_window, joint_a_idx, joint_b_idx
    )
    val_activity_a, val_activity_b = coupled_pair_activity_weights(val_q_a_now, val_q_a_past, val_q_b_now, val_q_b_past)
    val_activity_a, val_activity_b = val_activity_a.to(device), val_activity_b.to(device)

    input_dim = x_train.shape[1]
    ensemble_a = GenANEnsemble(
        input_dim, num_joints=1, ensemble_size=ensemble_size, seed=seed,
        bounded_output=True, torque_range=torque_range,
    )
    ensemble_b = GenANEnsemble(
        input_dim, num_joints=1, ensemble_size=ensemble_size, seed=seed + 500,  # different seed -- fully independent init
        bounded_output=True, torque_range=torque_range,
    )
    ensemble_a.to(device)
    ensemble_b.to(device)
    ensemble_a.fit_scalers(x_train, y_train)
    ensemble_b.fit_scalers(x_train, y_train)

    optimizers_a = [torch.optim.Adam(m.parameters(), lr=lr) for m in ensemble_a.members]
    optimizers_b = [torch.optim.Adam(m.parameters(), lr=lr) for m in ensemble_b.members]
    generators = [torch.Generator().manual_seed(seed + 1000 + i) for i in range(ensemble_size)]
    schedulers_a = (
        [torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=lr * 0.01) for opt in optimizers_a]
        if lr_decay else None
    )
    schedulers_b = (
        [torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=lr * 0.01) for opt in optimizers_b]
        if lr_decay else None
    )

    best_val_loss = float("inf")
    best_state_a = best_state_b = None
    epochs_since_improvement = 0
    history_log = {"train_loss": [], "val_loss": []}

    n_train = x_train.shape[0]
    steps_per_epoch = max(1, n_train // batch_size)

    for epoch in range(epochs):
        epoch_losses = []
        for _ in range(steps_per_epoch):
            step_losses = []
            for member_a, opt_a, member_b, opt_b, gen in zip(
                ensemble_a.members, optimizers_a, ensemble_b.members, optimizers_b, generators
            ):
                idx = torch.randint(0, n_train, (min(batch_size, n_train),), generator=gen)  # CPU, for determinism
                idx_dev = idx.to(device)
                # Both ensembles were fit_scalers'd on the SAME x_train, so their
                # input_scaler stats are numerically identical -- using ensemble_a's
                # here vs ensemble_b's would give the same result; kept explicit
                # per-ensemble below for clarity that each network owns its input.
                x_a = ensemble_a.input_scaler(x_train[idx_dev], train=False)
                x_b = ensemble_b.input_scaler(x_train[idx_dev], train=False)

                t_batch = train_t[idx]  # CPU: dataset is CPU-resident
                q_a_now, q_a_past, q_b_now, q_b_past = build_activity_inputs(
                    dataset, t_batch, activity_window, joint_a_idx, joint_b_idx
                )
                activity_a, activity_b = coupled_pair_activity_weights(q_a_now, q_a_past, q_b_now, q_b_past)
                activity_a, activity_b = activity_a.to(device), activity_b.to(device)
                label = y_train[idx_dev]

                pred_a = member_a(x_a)  # (batch, 1)
                loss_a = single_share_activity_loss(pred_a, label, torque_range, activity_a)
                opt_a.zero_grad()
                loss_a.backward()
                opt_a.step()

                pred_b = member_b(x_b)  # (batch, 1) -- fully independent forward/backward from pred_a
                loss_b = single_share_activity_loss(pred_b, label, torque_range, activity_b)
                opt_b.zero_grad()
                loss_b.backward()
                opt_b.step()

                step_losses.append(loss_a.item() + loss_b.item())
            if step_losses:
                epoch_losses.append(sum(step_losses) / len(step_losses))

        with torch.no_grad():
            preds_std_val_a = ensemble_a.forward_standardized(x_val)  # (ensemble_size, batch, 1)
            preds_std_val_b = ensemble_b.forward_standardized(x_val)
            val_loss_a = single_share_activity_loss(preds_std_val_a, y_val, torque_range, val_activity_a).item()
            val_loss_b = single_share_activity_loss(preds_std_val_b, y_val, torque_range, val_activity_b).item()
            val_loss = val_loss_a + val_loss_b

        train_loss = sum(epoch_losses) / len(epoch_losses)
        history_log["train_loss"].append(train_loss)
        history_log["val_loss"].append(val_loss)
        print(f"[epoch {epoch:4d}] train_loss={train_loss:.6f} val_loss={val_loss:.6f} "
              f"(val_loss_a={val_loss_a:.6f} val_loss_b={val_loss_b:.6f})")

        if schedulers_a is not None:
            for sch in schedulers_a:
                sch.step()
        if schedulers_b is not None:
            for sch in schedulers_b:
                sch.step()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state_a = {k: v.clone() for k, v in ensemble_a.state_dict().items()}
            best_state_b = {k: v.clone() for k, v in ensemble_b.state_dict().items()}
            epochs_since_improvement = 0
        else:
            epochs_since_improvement += 1
            if epochs_since_improvement >= patience:
                print(f"[INFO] Early stopping at epoch {epoch} (no improvement for {patience} epochs).")
                break

    if best_state_a is not None:
        ensemble_a.load_state_dict(best_state_a)
        ensemble_b.load_state_dict(best_state_b)
    ensemble_a.to("cpu")  # see train_genan.py's train() -- callers always get CPU-resident ensembles
    ensemble_b.to("cpu")
    history_log["best_val_loss"] = best_val_loss
    return ensemble_a, ensemble_b, history_log


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train TWO independent single-output GenAN torque models for a tendon-coupled mimic pair."
    )
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
    parser.add_argument("--checkpoint_a", type=str, default=None, help="Override joint_a's output checkpoint path.")
    parser.add_argument("--checkpoint_b", type=str, default=None, help="Override joint_b's output checkpoint path.")
    parser.add_argument("--torque_range", type=float, required=True,
                         help="Fixed min-max torque normalization range (e.g. 900.0), REQUIRED -- bounds each "
                              "network's output to (-1,1) via tanh; each is trained against its OWN "
                              "activity-weighted share of label/torque_range. No non-bounded fallback.")
    parser.add_argument("--activity_window", type=int, default=None,
                         help="Lookback (in steps) for the activity-weighted per-share pseudo-label split "
                              "(losses.coupled_pair_activity_weights). Default: history_len * stride.")
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
    # see module docstring / losses.coupled_pair_activity_weights.
    diff = (dataset.q_torque[:, joint_a_idx] - dataset.q_torque[:, joint_b_idx]).abs()
    if diff.max().item() > 1e-3:
        raise ValueError(
            f"{joint_a_name}/{joint_b_name} q_torque columns are NOT bit-identical (max abs diff="
            f"{diff.max().item():.4f}) -- this training mode assumes a shared tendon-coupled label "
            f"(verified for FF/MF/RF pairs; if this pair genuinely differs, the activity-weighted design "
            f"doesn't apply as-is)."
        )
    print(f"[INFO] Loaded {len(dataset.paths)} file(s), {dataset.num_steps} steps, "
          f"{dataset.traj_starts.shape[0]} trajectory segment(s). Training INDEPENDENT models for "
          f"{joint_a_name}/{joint_b_name} (q_torque max abs diff={diff.max().item():.2e}, confirmed shared label).")

    checkpoint_path_a = args.checkpoint_a or os.path.join(
        cfg["experiment"]["checkpoint_dir"], f"{cfg['experiment']['experiment_name']}_pairindep_{joint_a_name}.pt"
    )
    checkpoint_path_b = args.checkpoint_b or os.path.join(
        cfg["experiment"]["checkpoint_dir"], f"{cfg['experiment']['experiment_name']}_pairindep_{joint_b_name}.pt"
    )
    os.makedirs(os.path.dirname(checkpoint_path_a), exist_ok=True)
    os.makedirs(os.path.dirname(checkpoint_path_b), exist_ok=True)

    ensemble_a, ensemble_b, history_log = train(
        dataset, joint_a_idx=joint_a_idx, joint_b_idx=joint_b_idx, torque_range=args.torque_range,
        history_len=g["history_len"], stride=g["stride"], ensemble_size=g["ensemble_size"],
        epochs=g["epochs"], batch_size=g["batch_size"], lr=g["lr"],
        val_frac=g["val_frac"], patience=g["patience"], seed=seed,
        activity_window=args.activity_window,
        device=args.device,
        lr_decay=args.lr_decay if args.lr_decay is not None else g.get("lr_decay", False),
    )

    resolved_activity_window = args.activity_window if args.activity_window is not None else g["history_len"] * g["stride"]

    def _save(ensemble: GenANEnsemble, joint_idx: int, joint_name: str, partner_name: str, path: str) -> None:
        torch.save(
            {
                "ensemble_state_dict": ensemble.state_dict(),
                "input_dim": ensemble.members[0].trunk[0].in_features,
                "num_joints": 1,
                "ensemble_size": ensemble.ensemble_size,
                "history_len": g["history_len"],
                "stride": g["stride"],
                "joint_names": joint_names,  # full 16-joint order -- needed to rebuild the input at inference time
                "joint_idx": joint_idx,      # which column of q_torque this model predicts
                "joint_name": joint_name,
                "single_joint": True,
                "torque_range": args.torque_range,
                "bounded_output": True,
                # Traceability only -- NOT read by load_ffj3_ensemble/rollout(), which are joint-agnostic
                # single-joint loaders. Records which OTHER joint this one was trained alongside and with
                # what activity-weighting, since that's what its own label was derived from.
                "pair_partner_joint": partner_name,
                "activity_window": resolved_activity_window,
                "independently_trained_pair": True,
                "best_val_loss": history_log["best_val_loss"],
            },
            path,
        )
        print(f"[INFO] Saved checkpoint to {path}.")

    _save(ensemble_a, joint_a_idx, joint_a_name, joint_b_name, checkpoint_path_a)
    _save(ensemble_b, joint_b_idx, joint_b_name, joint_a_name, checkpoint_path_b)
    print(f"[INFO] Done (best combined val_loss={history_log['best_val_loss']:.6f}).")


if __name__ == "__main__":
    main()

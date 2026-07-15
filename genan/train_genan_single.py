#!/usr/bin/env python3
"""Train a GenAN torque-loss model for ONE joint, for fast per-joint debugging.

Diagnostic variant of train_genan.py, not a replacement for it. The input is
IDENTICAL to the full model: delta-histories of q_meas/q_cmd across all 16
joints (see DESIGN.md Decision 2 -- a coupled joint's own history only makes
sense alongside its partner's, so isolating the input to one joint would
throw away exactly the signal that matters for coupled joints). Only the
ensemble's output head and the Torque loss narrow to a single joint, via
`--joint`.

This lets you iterate on one joint at a time: check whether the basic
history -> torque mapping is learnable at all for that joint, before paying
for a full 16-joint training run.

Isaac-free, same convention as train_genan.py. Reads the SAME
roto/genan/agents/shadowlite/default.yaml (dataset + genan sections).

Like train_genan.py, also supports an OPTIONAL Position loss
(`genan.position_loss_weight` / `--position_loss_weight`, default 0.0 =
disabled) -- but ISOLATED to the one tested joint: the tested joint gets this
script's own single-joint predicted torque, every OTHER joint gets its real/
target torque (`tau_target`, from compute_dynamics.py) rather than any
prediction, so any resulting position error is attributable only to the
tested joint ("rest kept still"). Still integrates the FULL 16x16 coupled
dynamics step (`losses.predict_next_position`) -- not a reduced single-joint
inertia, since ShadowLite's joints are physically coupled through the hand's
rigid-body structure even though the tendon-pair coupling itself is
software-only (see DESIGN.md) -- only the resulting loss narrows to the
tested joint's own position column.

Checkpoints from this script are single-joint: they are NOT compatible with
play_genan.py's Isaac rollout, which expects a 16-joint torque vector from
`set_joint_effort_target`. Use plot_single_joint_trajectory.py to inspect
this script's output instead -- no simulator required.

Usage:
    python train_genan_single_joint.py --joint ffj1
    python train_genan_single_joint.py --joint 3 --epochs 300 --ensemble_size 3
    python train_genan_single_joint.py --joint ffj1 --position_loss_weight 0.1 \\
        --preprocess_cache cache/smoothed.npz --dynamics_cache cache/dynamics.npz
"""

from __future__ import annotations

import argparse
import os

import torch
import torch.nn.functional as F

from config_utils import load_config
from dataset_loader import AlignedTrajectoryDataset
from dynamics_cache import DynamicsCache
from history import build_delta_history
from joint_config import load_joint_config
from losses import (
    HARDWARE_EFFORT_TO_NM,
    pd_baseline_torque,
    predict_next_position,
    torque_direction_loss,
    torque_loss,
    torque_minmax_loss,
)
from model import GenANEnsemble
from pd_gains import load_pd_gains

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_CONFIG = os.path.join(_THIS_DIR, "agents", "shadowlite", "default.yaml")


def resolve_joint_idx(joint_arg: str, joint_names: list[str]) -> int:
    """Accept either a joint name (e.g. 'ffj1') or a raw integer index."""
    try:
        idx = int(joint_arg)
    except ValueError:
        if joint_arg not in joint_names:
            raise ValueError(f"Unknown joint name {joint_arg!r}. Known joints: {joint_names}")
        return joint_names.index(joint_arg)
    if not (0 <= idx < len(joint_names)):
        raise ValueError(f"Joint index {idx} out of range for {len(joint_names)} joints.")
    return idx


def split_segments(
    dataset: AlignedTrajectoryDataset, val_frac: float = 0.2, seed: int = 0
) -> tuple[torch.Tensor, torch.Tensor]:
    """Identical logic to train_genan.py's split_segments (trajectory-level
    split, see that file's docstring for why) -- duplicated rather than
    imported so this script has no dependency on train_genan.py and stays
    independently runnable/editable.
    """
    n_seg = dataset.traj_starts.shape[0]
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n_seg, generator=g)
    n_val = max(1, round(n_seg * val_frac)) if n_seg > 1 else 0
    val_segs, train_segs = perm[:n_val], perm[n_val:]

    def _indices_for(segs: torch.Tensor) -> torch.Tensor:
        chunks = [
            torch.arange(int(dataset.traj_starts[s]), int(dataset.traj_ends[s]) + 1) for s in segs.tolist()
        ]
        return torch.cat(chunks) if chunks else torch.empty(0, dtype=torch.long)

    return _indices_for(train_segs), _indices_for(val_segs)


def build_inputs_and_labels(
    dataset: AlignedTrajectoryDataset, t: torch.Tensor, history_len: int, stride: int, joint_idx: int,
    residual_kp: float | None = None, residual_kd: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Same full-multi-joint input as train_genan.py's build_inputs_and_labels
    -- only the label column narrows to `joint_idx`.

    `residual_kp`/`residual_kd`: if both set, the label becomes the RESIDUAL
    `gt_effort/HARDWARE_EFFORT_TO_NM - pd_baseline_torque(...)` (both sides in
    N*m -- see `losses.HARDWARE_EFFORT_TO_NM`/`pd_baseline_torque`) instead of
    the raw recorded torque -- the network then only has to learn what the
    known, fixed-gain PD term doesn't already explain, in N*m.
    """
    q_hist = build_delta_history(dataset.q_meas, t, history_len, stride, dataset)
    u_hist = build_delta_history(dataset.q_cmd, t, history_len, stride, dataset)
    raw_input = torch.cat([q_hist, u_hist], dim=-1)
    t_c = dataset.clamp(t)
    torque_label = dataset.q_torque[t_c][:, joint_idx : joint_idx + 1]
    if residual_kp is not None:
        q_cmd = dataset.q_cmd[t_c][:, joint_idx : joint_idx + 1]
        q_meas = dataset.q_meas[t_c][:, joint_idx : joint_idx + 1]
        qdot_meas = dataset.q_meas_vel[t_c][:, joint_idx : joint_idx + 1]
        label_calibrated_nm = torque_label / HARDWARE_EFFORT_TO_NM
        torque_label = label_calibrated_nm - pd_baseline_torque(q_cmd, q_meas, qdot_meas, residual_kp, residual_kd)
    return raw_input, torque_label


def train(
    dataset: AlignedTrajectoryDataset,
    joint_idx: int,
    history_len: int = 3,
    stride: int = 1,
    ensemble_size: int = 5,
    epochs: int = 150,
    batch_size: int = 4096,
    lr: float = 1e-4,
    val_frac: float = 0.2,
    patience: int = 10,
    seed: int = 0,
    torque_loss_weight: float = 1.0,
    torque_loss_direction: bool = False,
    position_loss_weight: float = 0.0,
    dyn_cache: DynamicsCache | None = None,
    device: str = "cpu",
    lr_decay: bool = False,
    torque_range: float | None = None,
    residual_kp: float | None = None,
    residual_kd: float | None = None,
) -> tuple[GenANEnsemble, dict]:
    """`torque_loss_weight`/`position_loss_weight`/`dyn_cache`: see module
    docstring's "isolated single-joint Position loss" section, and
    train_genan.py's `train()` docstring for the weight semantics (either
    term skipped entirely, not just zero-weighted, when its weight is
    `<= 0.0`). Defaults reproduce the exact previous Torque-loss-only
    behavior bit-for-bit. Set `torque_loss_weight=0.0` with
    `position_loss_weight > 0.0` for PURE Position-loss training.

    `torque_loss_direction`: if True, use `torque_direction_loss` (cosine
    similarity, calibration-free) instead of `torque_loss` (magnitude MSE)
    for the torque term. Default `False` reproduces prior behavior exactly.

    `device`: see train_genan.py's `train()` docstring -- same pattern,
    dataset/dyn_cache stay CPU-resident, only training tensors move.

    `lr_decay`: see train_genan.py's `train()` docstring -- same
    CosineAnnealingLR(T_max=epochs, eta_min=lr*0.01) schedule, per member.

    `torque_range`: see train_genan.py's `train()` docstring -- same fixed
    min-max normalization + tanh-bounded output, in place of the default
    RunningStandardScaler-based standardization. Overrides
    `torque_loss_direction` when set (always uses `torque_minmax_loss` for
    the torque term in this mode).

    `residual_kp`/`residual_kd`: if both set, the Torque loss trains against
    the RESIDUAL `gt_effort - pd_baseline_torque(...)` (see
    `losses.pd_baseline_torque`, `pd_gains.load_pd_gains`) instead of the raw
    recorded torque -- the network's own predicted value (whatever
    `torque_range`/`ensemble.forward()` de-normalizes it to) is then this
    RESIDUAL, not the absolute torque; callers that need absolute torque
    (e.g. Position loss's dynamics step, or plotting against raw `gt_effort`)
    must add `pd_baseline_torque(...)` back explicitly (see plot_single.py).
    Mutually exclusive with `position_loss_weight > 0` for now -- the
    position-loss dynamics step needs absolute torque and isn't residual-
    aware yet (out of scope for this mode's first pass).
    """
    train_t, val_t = split_segments(dataset, val_frac=val_frac, seed=seed)
    if train_t.numel() == 0 or val_t.numel() == 0:
        raise ValueError(
            f"Need at least one trajectory in each split (train={train_t.numel()}, val={val_t.numel()})."
        )
    if position_loss_weight > 0.0 and dyn_cache is None:
        raise ValueError("position_loss_weight > 0 requires a DynamicsCache (dyn_cache).")
    if torque_loss_weight <= 0.0 and position_loss_weight <= 0.0:
        raise ValueError("At least one of torque_loss_weight/position_loss_weight must be > 0.")
    if (residual_kp is not None) and position_loss_weight > 0.0:
        raise ValueError(
            "residual_kp/residual_kd (residual-torque training) is not yet supported together with "
            "position_loss_weight > 0 -- the Position loss's dynamics step needs absolute torque."
        )
    torque_loss_fn = torque_direction_loss if torque_loss_direction else torque_loss

    x_train, y_train = build_inputs_and_labels(
        dataset, train_t, history_len, stride, joint_idx, residual_kp=residual_kp, residual_kd=residual_kd
    )
    x_val, y_val = build_inputs_and_labels(
        dataset, val_t, history_len, stride, joint_idx, residual_kp=residual_kp, residual_kd=residual_kd
    )
    x_train, y_train = x_train.to(device), y_train.to(device)
    x_val, y_val = x_val.to(device), y_val.to(device)

    input_dim = x_train.shape[1]
    ensemble = GenANEnsemble(
        input_dim, num_joints=1, ensemble_size=ensemble_size, seed=seed,
        bounded_output=(torque_range is not None), torque_range=torque_range,
    )
    ensemble.to(device)
    ensemble.fit_scalers(x_train, y_train)

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
                pred_std = member(x)
                loss = None

                if torque_loss_weight > 0.0:
                    if torque_range is not None:
                        loss = torque_loss_weight * torque_minmax_loss(pred_std, y_train[idx_dev], torque_range)
                    else:
                        y_std = ensemble.label_scaler(y_train[idx_dev], train=False)
                        loss = torque_loss_weight * torque_loss_fn(pred_std, y_std)

                if position_loss_weight > 0.0:
                    t_batch = train_t[idx]  # CPU: dataset/dyn_cache are CPU-resident
                    tau_target, m_inv, C, G, q_t, qdot_t, q_next, valid = dyn_cache.position_targets(dataset, t_batch)
                    valid = valid.to(device)
                    tau_target = tau_target.to(device)
                    m_inv, C, G = m_inv.to(device), C.to(device), G.to(device)
                    q_t, qdot_t, q_next = q_t.to(device), qdot_t.to(device), q_next.to(device)
                    if valid.any():
                        # Differentiable physical-torque prediction for the ONE
                        # tested joint -- explicit no_grad=False, see
                        # train_genan.py's module docstring for why.
                        tau_pred_physical = ensemble.label_scaler(pred_std, train=False, inverse=True, no_grad=False)
                        # "Rest kept still": every OTHER joint gets its real/
                        # target torque, only joint_idx gets this script's own
                        # prediction substituted in -- see module docstring.
                        tau_full = tau_target.clone()
                        tau_full[:, joint_idx] = tau_pred_physical.squeeze(-1)
                        q_next_pred = predict_next_position(tau_full, m_inv, C, G, q_t, qdot_t, dataset.rl_dt)
                        pos_loss = F.mse_loss(q_next_pred[valid, joint_idx], q_next[valid, joint_idx])
                        pos_term = position_loss_weight * pos_loss
                        loss = pos_term if loss is None else loss + pos_term

                if loss is None:
                    continue  # all-boundary batch AND torque_loss_weight==0 -- vanishingly rare

                opt.zero_grad()
                loss.backward()
                opt.step()
                step_losses.append(loss.item())
            if step_losses:
                epoch_losses.append(sum(step_losses) / len(step_losses))

        with torch.no_grad():
            preds_std_val = ensemble.forward_standardized(x_val)
            val_loss_t = None

            if torque_loss_weight > 0.0:
                if torque_range is not None:
                    val_loss_t = torque_loss_weight * torque_minmax_loss(preds_std_val, y_val, torque_range)
                else:
                    y_std_val = ensemble.label_scaler(y_val, train=False)
                    val_loss_t = torque_loss_weight * torque_loss_fn(preds_std_val, y_std_val)

            if position_loss_weight > 0.0:
                tau_target, m_inv, C, G, q_t, qdot_t, q_next, valid = dyn_cache.position_targets(dataset, val_t)
                valid = valid.to(device)
                tau_target = tau_target.to(device)
                m_inv, C, G = m_inv.to(device), C.to(device), G.to(device)
                q_t, qdot_t, q_next = q_t.to(device), qdot_t.to(device), q_next.to(device)
                if valid.any():
                    pred_std_mean_val = preds_std_val.mean(dim=0)
                    tau_pred_physical_val = ensemble.label_scaler(pred_std_mean_val, train=False, inverse=True)
                    tau_full_val = tau_target.clone()
                    tau_full_val[:, joint_idx] = tau_pred_physical_val.squeeze(-1)
                    q_next_pred_val = predict_next_position(tau_full_val, m_inv, C, G, q_t, qdot_t, dataset.rl_dt)
                    val_pos_loss = F.mse_loss(q_next_pred_val[valid, joint_idx], q_next[valid, joint_idx])
                    val_pos_term = position_loss_weight * val_pos_loss
                    val_loss_t = val_pos_term if val_loss_t is None else val_loss_t + val_pos_term
            val_loss = val_loss_t.item() if val_loss_t is not None else float("nan")

        train_loss = sum(epoch_losses) / len(epoch_losses)
        history_log["train_loss"].append(train_loss)
        history_log["val_loss"].append(val_loss)
        print(f"[epoch {epoch:4d}] train_loss={train_loss:.6f} val_loss={val_loss:.6f}")

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
    parser = argparse.ArgumentParser(description="Train a single-joint GenAN torque model for ShadowLite.")
    parser.add_argument("--config", type=str, default=_DEFAULT_CONFIG, help="Base agent yaml (dataset/genan sections).")
    parser.add_argument("--agent_cfg", type=str, default=None, help="Optional yaml merged OVER --config.")
    parser.add_argument(
        "--dataset", type=str, action="append", default=None,
        help="Override dataset.paths (repeatable) -- directories, glob patterns, or explicit files.",
    )
    parser.add_argument("--joints_yaml", type=str, default=None, help="Override path to joints.yaml.")
    parser.add_argument("--joint", type=str, required=True, help="Joint name (e.g. 'ffj1') or index (e.g. '3').")
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
    parser.add_argument("--torque_loss_weight", type=float, default=None,
                         help="Weight for the Torque loss term (default 1.0). Set to 0.0 (together with "
                              "--position_loss_weight > 0) for PURE Position-loss training.")
    parser.add_argument("--torque_loss_direction", action="store_true", default=None,
                         help="Use direction-only (cosine similarity) torque loss instead of magnitude MSE. "
                              "Calibration-free, like uan_shadowlite/reward.py's torque_sign term. Default: off.")
    parser.add_argument("--torque_range", type=float, default=None,
                         help="Fixed min-max torque normalization range (e.g. 900.0): bounds the network's "
                              "output to (-1,1) via tanh and trains with plain MSE against label/torque_range, "
                              "instead of the default RunningStandardScaler-based standardization. Default: "
                              "unset (prior behavior).")
    parser.add_argument("--residual_torque", action="store_true", default=False,
                         help="Train against the RESIDUAL torque (gt_effort minus the identified-PD baseline "
                              "torque, using this joint's fixed Kp/Kd from shadow_pd_id/pd_gains.py) instead of "
                              "raw gt_effort. Not yet supported together with --position_loss_weight > 0. "
                              "Default: off (prior behavior).")
    parser.add_argument("--position_loss_weight", type=float, default=None,
                         help="Weight for the isolated single-joint Position loss (default 0.0 = disabled). "
                              "Requires --preprocess_cache/--dynamics_cache (or the matching yaml keys) when > 0.")
    parser.add_argument("--preprocess_cache", type=str, default=None, help="preprocess.py's output .npz.")
    parser.add_argument("--dynamics_cache", type=str, default=None, help="compute_dynamics.py's output .npz.")
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
    joint_idx = resolve_joint_idx(args.joint, joint_names)
    joint_name = joint_names[joint_idx]
    print(f"[INFO] Loaded {len(dataset.paths)} file(s), {dataset.num_steps} steps, "
          f"{dataset.traj_starts.shape[0]} trajectory segment(s). Training joint {joint_idx} ({joint_name}).")

    checkpoint_path = args.checkpoint or os.path.join(
        cfg["experiment"]["checkpoint_dir"], f"{cfg['experiment']['experiment_name']}_joint_{joint_name}.pt"
    )
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)

    torque_loss_weight = args.torque_loss_weight if args.torque_loss_weight is not None else g.get("torque_loss_weight", 1.0)
    torque_loss_direction = args.torque_loss_direction if args.torque_loss_direction is not None else g.get("torque_loss_direction", False)
    torque_range = args.torque_range if args.torque_range is not None else g.get("torque_range", None)
    lr_decay = args.lr_decay if args.lr_decay is not None else g.get("lr_decay", False)
    residual_torque = args.residual_torque or g.get("residual_torque", False)
    residual_kp, residual_kd = (load_pd_gains(joint_name) if residual_torque else (None, None))
    if residual_torque:
        print(f"[INFO] Residual-torque training for {joint_name}: kp={residual_kp:.4f}, kd={residual_kd:.4f} "
              f"(from shadow_pd_id, see pd_gains.py).")
    position_loss_weight = args.position_loss_weight if args.position_loss_weight is not None else g.get("position_loss_weight", 0.0)
    dyn_cache = None
    if position_loss_weight > 0.0:
        preprocess_cache = args.preprocess_cache or g.get("preprocess_cache")
        dynamics_cache_path = args.dynamics_cache or g.get("dynamics_cache")
        if not preprocess_cache or not dynamics_cache_path:
            raise ValueError(
                "position_loss_weight > 0 requires both --preprocess_cache and --dynamics_cache "
                "(or genan.preprocess_cache/genan.dynamics_cache in the yaml)."
            )
        dyn_cache = DynamicsCache(preprocess_cache, dynamics_cache_path)
        print(f"[INFO] Loaded DynamicsCache ({dyn_cache.num_rows} rows) for isolated single-joint "
              f"Position loss (weight={position_loss_weight}).")

    ensemble, history_log = train(
        dataset, joint_idx=joint_idx,
        history_len=g["history_len"], stride=g["stride"], ensemble_size=g["ensemble_size"],
        epochs=g["epochs"], batch_size=g["batch_size"], lr=g["lr"],
        val_frac=g["val_frac"], patience=g["patience"], seed=seed,
        torque_loss_weight=torque_loss_weight,
        torque_loss_direction=torque_loss_direction,
        torque_range=torque_range,
        position_loss_weight=position_loss_weight, dyn_cache=dyn_cache,
        device=args.device,
        lr_decay=lr_decay,
        residual_kp=residual_kp,
        residual_kd=residual_kd,
    )

    torch.save(
        {
            "ensemble_state_dict": ensemble.state_dict(),
            "input_dim": ensemble.members[0].trunk[0].in_features,
            "num_joints": 1,
            "ensemble_size": ensemble.ensemble_size,
            "history_len": g["history_len"],
            "stride": g["stride"],
            "joint_names": joint_names,   # full 16-joint order -- needed to rebuild the input at inference time
            "joint_idx": joint_idx,       # which column of q_torque this model predicts
            "joint_name": joint_name,
            "single_joint": True,
            "best_val_loss": history_log["best_val_loss"],
            "torque_range": torque_range,
            "bounded_output": torque_range is not None,
            "residual_torque": residual_torque,
            "residual_kp": residual_kp,
            "residual_kd": residual_kd,
        },
        checkpoint_path,
    )
    print(f"[INFO] Saved checkpoint to {checkpoint_path} (best val_loss={history_log['best_val_loss']:.6f}).")


if __name__ == "__main__":
    main()
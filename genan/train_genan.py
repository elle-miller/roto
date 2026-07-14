#!/usr/bin/env python3
"""Train a GenAN ensemble for ShadowLite from real trajectory data.

Isaac-free (no `isaaclab` import anywhere in this file or in anything it
imports) -- run with any environment that has torch+numpy+pyyaml, not
necessarily the one Isaac Lab is installed in (mirrors the convention set by
`roto/roto/tasks/uan_shadowlite/dataset.py`). See `DESIGN.md` for the
calibration caveat on the resulting torque scale.

Trains a Torque loss (standardized MSE against `q_torque`/`gt_effort`)
always, and OPTIONALLY (`genan.position_loss_weight > 0` in the yaml) an
additional Position loss on top: a closed-form, differentiable one-step
semi-implicit-Euler dynamics prediction (using `M_inv`/`C`/`G` precomputed
offline by `roto/scripts/compute_dynamics.py` via Isaac's PhysX tensor API,
treated as CONSTANTS here -- see `losses.py`'s `position_loss`) compared via
MSE against the real recorded next position. No RL, no rollout, no live
Isaac/PhysX call anywhere in this file -- the one non-differentiable
simulator query already happened, once per data point, offline, in
`compute_dynamics.py`. Default `position_loss_weight` is `0.0` (inert), so
existing Torque-loss-only runs/checkpoints are unaffected.

IMPORTANT gradient-correctness note: `GenANEnsemble.forward()` (model.py)
silently breaks gradients -- its final `label_scaler(..., inverse=True)` call
defaults to `no_grad=True` (`RunningStandardScaler.forward`), so the
de-standardized (physical-torque) output has no `grad_fn`. The Position loss
needs physical-unit predicted torque (to be dimensionally consistent with
`M_inv`/`dt**2`), so this file calls
`ensemble.label_scaler(pred_std, train=False, inverse=True, no_grad=False)`
explicitly wherever a differentiable physical-torque value is needed, rather
than `ensemble.forward()`.

Config is read from `roto/genan/agents/shadowlite/default.yaml` (`dataset` +
`genan` sections), mirroring `train_uan.py`'s `--config`/`--agent_cfg`/
`--dataset` conventions -- see that yaml's own header comment for why it
omits the RL-only sections (`encoder`/`policy`/`value`/`agent`) UAN's yaml has.

Usage:
    python train_genan.py
    python train_genan.py --config my_variant.yaml --checkpoint out.pt
    python train_genan.py --dataset dirA --dataset dirB --epochs 200
    python train_genan.py --position_loss_weight 0.1 \\
        --preprocess_cache cache/smoothed.npz --dynamics_cache cache/dynamics.npz
"""

from __future__ import annotations

import argparse
import os

import torch

from config_utils import load_config
from dataset_loader import AlignedTrajectoryDataset
from dynamics_cache import DynamicsCache
from history import build_delta_history
from joint_config import load_joint_config
from losses import position_loss, torque_direction_loss, torque_loss
from model import GenANEnsemble

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_CONFIG = os.path.join(_THIS_DIR, "agents", "shadowlite", "default.yaml")


def split_segments(
    dataset: AlignedTrajectoryDataset, val_frac: float = 0.2, seed: int = 0
) -> tuple[torch.Tensor, torch.Tensor]:
    """Split at the TRAJECTORY (segment) level, not the step level.

    Splitting at the step level would put training and validation rows from
    the same trajectory a few timesteps apart -- near-duplicates given the
    temporal correlation within a trajectory -- which would make validation
    loss an overly optimistic estimate of generalization (see the paper's
    Appendix A.2, and this repo's own `AlignedTrajectoryDataset` docstring
    convention of segmenting by trajectory for exactly this reason).
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
    dataset: AlignedTrajectoryDataset, t: torch.Tensor, history_len: int, stride: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build (raw_input, torque_label) for every time index in `t`.

    `raw_input` concatenates the delta-histories of `q_meas` (paper's `q`)
    and `q_cmd` (paper's control signal `u`; ShadowLite's PD target is the
    natural analogue -- see DESIGN.md Decision 2).
    """
    q_hist = build_delta_history(dataset.q_meas, t, history_len, stride, dataset)
    u_hist = build_delta_history(dataset.q_cmd, t, history_len, stride, dataset)
    raw_input = torch.cat([q_hist, u_hist], dim=-1)
    torque_label = dataset.q_torque[dataset.clamp(t)]
    return raw_input, torque_label


def train(
    dataset: AlignedTrajectoryDataset,
    history_len: int = 3,
    stride: int = 1,
    ensemble_size: int = 5,
    epochs: int = 150,
    batch_size: int = 4096,
    lr: float = 1e-4,
    val_frac: float = 0.2,
    patience: int = 10,
    seed: int = 0,
    trial=None,
    torque_loss_weight: float = 1.0,
    torque_loss_direction: bool = False,
    position_loss_weight: float = 0.0,
    dyn_cache: DynamicsCache | None = None,
    device: str = "cpu",
    lr_decay: bool = False,
) -> tuple[GenANEnsemble, dict]:
    """Train a GenAN ensemble. `trial`, if given, is an `optuna.Trial`-like
    object (duck-typed: only `.report(value, step)` and `.should_prune()` are
    used) -- reported every epoch and pruned via `optuna.TrialPruned` for
    `sweep_genan.py`'s Optuna search. `optuna` itself is only imported lazily,
    inside the `trial is not None` branch, so plain training never needs it
    installed.

    `torque_loss_weight`/`position_loss_weight`/`dyn_cache`: each member's
    total loss is `torque_loss_weight * torque_loss + position_loss_weight *
    position_loss` -- either term is skipped entirely (not just zero-weighted)
    when its weight is `<= 0.0`. Defaults (`torque_loss_weight=1.0`,
    `position_loss_weight=0.0`) reproduce the exact previous Torque-loss-only
    behavior bit-for-bit. Set `torque_loss_weight=0.0` with
    `position_loss_weight > 0.0` for PURE Position-loss training (the paper's
    other loss variant, per DESIGN.md) -- `dyn_cache` (a `DynamicsCache`, see
    dynamics_cache.py) is required whenever `position_loss_weight > 0.0`.
    Raises if both weights are `<= 0.0` (nothing left to train against).

    `torque_loss_direction`: if True, use `torque_direction_loss` (cosine
    similarity -- calibration-free, direction-only) in place of `torque_loss`
    (magnitude-sensitive MSE) for the torque term, wherever it's active.
    Default `False` reproduces prior behavior exactly.

    `device`: everything the network trains against (`x_train`/`y_train`/
    `x_val`/`y_val`, the ensemble itself, and the position-loss tensors
    pulled from `dyn_cache`) is moved to this device -- `dataset`/`dyn_cache`
    themselves stay on CPU (index/RNG bookkeeping, cheap either way), only
    the actual training tensors move. Default `"cpu"` matches all prior
    behavior exactly; batch-index sampling still uses a CPU `torch.Generator`
    for determinism (see `check` in `phase_a_tests.py`), then `.to(device)`s
    the sampled indices before using them to index GPU tensors.

    `lr_decay`: if True, each member's optimizer gets a
    `CosineAnnealingLR(opt, T_max=epochs, eta_min=lr*0.01)` schedule, stepped
    once per epoch (decaying from `lr` down to `lr*0.01` by the final epoch,
    matched to the fixed `epochs` budget). Default `False` reproduces prior
    behavior exactly (constant `lr` for the whole run).
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
    torque_loss_fn = torque_direction_loss if torque_loss_direction else torque_loss

    x_train, y_train = build_inputs_and_labels(dataset, train_t, history_len, stride)
    x_val, y_val = build_inputs_and_labels(dataset, val_t, history_len, stride)
    x_train, y_train = x_train.to(device), y_train.to(device)
    x_val, y_val = x_val.to(device), y_val.to(device)

    input_dim = x_train.shape[1]
    ensemble = GenANEnsemble(input_dim, dataset.num_joints, ensemble_size=ensemble_size, seed=seed)
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
                idx = torch.randint(0, n_train, (batch_size,), generator=gen)  # CPU, for determinism
                idx_dev = idx.to(device)
                x = ensemble.input_scaler(x_train[idx_dev], train=False)
                pred_std = member(x)
                loss = None

                if torque_loss_weight > 0.0:
                    y_std = ensemble.label_scaler(y_train[idx_dev], train=False)
                    loss = torque_loss_weight * torque_loss_fn(pred_std, y_std)

                if position_loss_weight > 0.0:
                    t_batch = train_t[idx]  # CPU: dataset/dyn_cache are CPU-resident
                    _, m_inv, C, G, q_t, qdot_t, q_next, valid = dyn_cache.position_targets(dataset, t_batch)
                    valid = valid.to(device)
                    m_inv, C, G = m_inv.to(device), C.to(device), G.to(device)
                    q_t, qdot_t, q_next = q_t.to(device), qdot_t.to(device), q_next.to(device)
                    if valid.any():
                        # Differentiable physical-torque prediction -- explicit
                        # no_grad=False is required, see module docstring.
                        tau_pred_physical = ensemble.label_scaler(pred_std, train=False, inverse=True, no_grad=False)
                        pos_loss = position_loss(
                            tau_pred_physical[valid], m_inv[valid], C[valid], G[valid],
                            q_t[valid], qdot_t[valid], q_next[valid], dataset.rl_dt,
                        )
                        pos_term = position_loss_weight * pos_loss
                        loss = pos_term if loss is None else loss + pos_term

                if loss is None:
                    # Only reachable if this batch's `valid` mask was all-False
                    # (every sampled row a segment boundary) AND torque_loss_weight
                    # is 0 -- vanishingly rare, but skip the step rather than crash.
                    continue

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
                y_std_val = ensemble.label_scaler(y_val, train=False)
                val_loss_t = torque_loss_weight * torque_loss_fn(preds_std_val, y_std_val)

            if position_loss_weight > 0.0:
                _, m_inv, C, G, q_t, qdot_t, q_next, valid = dyn_cache.position_targets(dataset, val_t)
                valid = valid.to(device)
                m_inv, C, G = m_inv.to(device), C.to(device), G.to(device)
                q_t, qdot_t, q_next = q_t.to(device), qdot_t.to(device), q_next.to(device)
                if valid.any():
                    # Ensemble-MEAN prediction for validation-time position loss
                    # (a monitoring/early-stopping signal, not a training
                    # gradient) -- avoids broadcasting the full per-member
                    # ensemble dimension through the (N, 16, 16) m_inv batch.
                    pred_std_mean_val = preds_std_val.mean(dim=0)
                    tau_pred_physical_val = ensemble.label_scaler(pred_std_mean_val, train=False, inverse=True)
                    val_pos_loss = position_loss(
                        tau_pred_physical_val[valid], m_inv[valid], C[valid], G[valid],
                        q_t[valid], qdot_t[valid], q_next[valid], dataset.rl_dt,
                    )
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

        if trial is not None:
            import optuna

            trial.report(val_loss, epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()

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
    # Return a CPU-resident ensemble regardless of `device` -- callers (main()'s
    # torch.save, sweep_genan.py, etc.) have always assumed a CPU ensemble; only
    # the training loop itself needs the GPU tensors.
    ensemble.to("cpu")
    history_log["best_val_loss"] = best_val_loss
    return ensemble, history_log


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a GenAN torque-loss ensemble for ShadowLite.")
    parser.add_argument("--config", type=str, default=_DEFAULT_CONFIG, help="Base agent yaml (dataset/genan sections).")
    parser.add_argument("--agent_cfg", type=str, default=None, help="Optional yaml merged OVER --config.")
    parser.add_argument(
        "--dataset", type=str, action="append", default=None,
        help="Override dataset.paths (repeatable) -- directories, glob patterns, or explicit files.",
    )
    parser.add_argument("--joints_yaml", type=str, default=None, help="Override path to joints.yaml.")
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
    parser.add_argument("--position_loss_weight", type=float, default=None,
                         help="Weight for the Position loss term (default 0.0 = disabled). Requires "
                              "--preprocess_cache/--dynamics_cache (or the matching yaml keys) when > 0.")
    parser.add_argument("--preprocess_cache", type=str, default=None, help="preprocess.py's output .npz.")
    parser.add_argument("--dynamics_cache", type=str, default=None, help="compute_dynamics.py's output .npz.")
    parser.add_argument("--device", type=str, default="cpu",
                         help="Training device, e.g. 'cpu' (default) or 'cuda:0'. Dataset loading stays on "
                              "CPU regardless; only the network/training tensors move.")
    parser.add_argument("--lr_decay", action="store_true", default=None,
                         help="Cosine-anneal lr from its initial value down to lr*0.01 over `epochs`. "
                              "Default: off (constant lr).")
    return parser


def load_dataset_from_cfg(
    cfg: dict, dataset_override: list[str] | None, joints_yaml: str | None, min_horizon_override: int | None = None
):
    dataset_paths = dataset_override if dataset_override is not None else cfg["dataset"]["paths"]
    min_horizon = min_horizon_override if min_horizon_override is not None else cfg["dataset"]["min_horizon"]
    joint_names, joint_upper_limits = load_joint_config(joints_yaml)
    dataset = AlignedTrajectoryDataset(
        paths=dataset_paths,
        joint_names=joint_names,
        device="cpu",
        joint_upper_limits=joint_upper_limits,
        min_horizon=min_horizon,
    )
    return dataset, joint_names


def main() -> None:
    args = build_argparser().parse_args()
    cfg = load_config(args.config, args.agent_cfg)
    g = cfg["genan"]
    # CLI flags (all default None) override the yaml only when explicitly passed.
    overrides = {
        "history_len": args.history_len, "stride": args.stride, "ensemble_size": args.ensemble_size,
        "epochs": args.epochs, "batch_size": args.batch_size, "lr": args.lr, "val_frac": args.val_frac,
        "patience": args.patience,
    }
    for key, value in overrides.items():
        if value is not None:
            g[key] = value
    seed = args.seed if args.seed is not None else cfg["seed"]
    checkpoint_path = args.checkpoint or os.path.join(
        cfg["experiment"]["checkpoint_dir"], f"{cfg['experiment']['experiment_name']}.pt"
    )
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)

    dataset, joint_names = load_dataset_from_cfg(cfg, args.dataset, args.joints_yaml, args.min_horizon)
    print(f"[INFO] Loaded {len(dataset.paths)} file(s), {dataset.num_steps} steps, "
          f"{dataset.traj_starts.shape[0]} trajectory segment(s).")

    torque_loss_weight = args.torque_loss_weight if args.torque_loss_weight is not None else g.get("torque_loss_weight", 1.0)
    torque_loss_direction = args.torque_loss_direction if args.torque_loss_direction is not None else g.get("torque_loss_direction", False)
    lr_decay = args.lr_decay if args.lr_decay is not None else g.get("lr_decay", False)
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
        print(f"[INFO] Loaded DynamicsCache ({dyn_cache.num_rows} rows) for Position loss "
              f"(weight={position_loss_weight}).")

    ensemble, history_log = train(
        dataset,
        history_len=g["history_len"],
        stride=g["stride"],
        ensemble_size=g["ensemble_size"],
        epochs=g["epochs"],
        batch_size=g["batch_size"],
        lr=g["lr"],
        val_frac=g["val_frac"],
        patience=g["patience"],
        seed=seed,
        torque_loss_weight=torque_loss_weight,
        torque_loss_direction=torque_loss_direction,
        position_loss_weight=position_loss_weight,
        dyn_cache=dyn_cache,
        device=args.device,
        lr_decay=lr_decay,
    )

    torch.save(
        {
            "ensemble_state_dict": ensemble.state_dict(),
            "input_dim": ensemble.members[0].trunk[0].in_features,
            "num_joints": ensemble.num_joints,
            "ensemble_size": ensemble.ensemble_size,
            "history_len": g["history_len"],
            "stride": g["stride"],
            "joint_names": joint_names,
            "best_val_loss": history_log["best_val_loss"],
        },
        checkpoint_path,
    )
    print(f"[INFO] Saved checkpoint to {checkpoint_path} (best val_loss={history_log['best_val_loss']:.6f}).")


if __name__ == "__main__":
    main()

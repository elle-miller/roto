#!/usr/bin/env python3
"""Fit a deployment-time torque scale via single-step, ground-truth-history
POSITION MATCHING, Isaac-free -- an alternative to `scripts/fit_torque_scale.py`'s
full free-running Isaac Sim rollout.

For each sampled REAL timestep `t` (many at once, not one long rollout), this
uses the REAL recorded history/state to build the network's input (NOT any
evolving sim state -- fully teacher-forced), predicts `scale * <raw tanh
output>`, adds it to the identified PD's own contribution
(`losses.pd_baseline_torque`, same Kp/Kd `fit_torque_scale.py`/
`shadow_hand_lite.py` use) to form an ABSOLUTE one-step torque estimate for
the tested joint, substitutes THAT into the otherwise-real torque vector
(every other joint keeps its own real/target torque -- "rest kept still",
the same convention `train_genan_single.py`'s isolated single-joint Position
loss already uses), and checks whether the resulting ONE-STEP predicted next
position (closed-form semi-implicit-Euler, `losses.predict_next_position`,
using `compute_dynamics.py`'s precomputed M_inv/C/G via `DynamicsCache`)
matches the REAL next position -- averaged over MANY such single-step
samples at once. No rollout, no compounding error, no Isaac Sim boot.

Requires the SAME preprocess/dynamics cache `train_genan_single.py --position_loss_weight`
uses (defaults to `genan/cache/smoothed.npz`/`dynamics.npz`, already present
in this repo).

Usage:
    python fit_scale_position.py --checkpoint checkpoints/genan_default_pairindep_rh_FFJ1.pt \\
        --joint rh_FFJ1 --dataset /path/to/aligned/dir --bounds 0 30
"""

from __future__ import annotations

import argparse
import os

import torch
from scipy.optimize import minimize_scalar

from dataset_loader import AlignedTrajectoryDataset
from dynamics_cache import DynamicsCache
from history import build_delta_history
from joint_config import load_joint_config
from losses import pd_baseline_torque, predict_next_position
from model import GenANEnsemble
from pd_gains import load_pd_gains
from train_genan_single import resolve_joint_idx

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_PREPROCESS_CACHE = os.path.join(_THIS_DIR, "cache", "smoothed.npz")
_DEFAULT_DYNAMICS_CACHE = os.path.join(_THIS_DIR, "cache", "dynamics.npz")


def load_ensemble(checkpoint_path: str) -> tuple[GenANEnsemble, dict]:
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    torque_range = ckpt["torque_range"]
    ensemble = GenANEnsemble(
        ckpt["input_dim"], ckpt["num_joints"], ensemble_size=ckpt["ensemble_size"],
        bounded_output=True, torque_range=torque_range,
    )
    ensemble.load_state_dict(ckpt["ensemble_state_dict"])
    ensemble.eval()
    return ensemble, ckpt


def build_input(dataset: AlignedTrajectoryDataset, t: torch.Tensor, history_len: int, stride: int) -> torch.Tensor:
    """Same full-multi-joint history the model was trained on -- built from
    REAL recorded q_meas/q_cmd throughout, never any sim state.
    """
    q_hist = build_delta_history(dataset.q_meas, t, history_len, stride, dataset)
    u_hist = build_delta_history(dataset.q_cmd, t, history_len, stride, dataset)
    return torch.cat([q_hist, u_hist], dim=-1)


def position_match_score(
    ensemble: GenANEnsemble, ckpt: dict, dataset: AlignedTrajectoryDataset, dyn_cache: DynamicsCache,
    t: torch.Tensor, joint_idx: int, kp: float, kd: float, scale: float,
) -> tuple[float, int]:
    """Mean one-step position-prediction MSE for `joint_idx`, at `scale`,
    over sampled real timesteps `t`. Returns (mse, num_valid_rows_used).
    """
    x = build_input(dataset, t, ckpt["history_len"], ckpt["stride"])
    with torch.no_grad():
        raw_pred = ensemble.forward_standardized(x).mean(dim=0)[:, 0]  # (N,), raw tanh output in (-1,1)

    tau_target, m_inv, C, G, q_t, qdot_t, q_next, valid = dyn_cache.position_targets(dataset, t)
    t_c = dataset.clamp(t)
    q_cmd_j = dataset.q_cmd[t_c][:, joint_idx]
    q_meas_j = dataset.q_meas[t_c][:, joint_idx]
    qdot_meas_j = dataset.q_meas_vel[t_c][:, joint_idx]
    tau_pd_j = pd_baseline_torque(q_cmd_j, q_meas_j, qdot_meas_j, kp, kd)

    tau_full = tau_target.clone()
    tau_full[:, joint_idx] = tau_pd_j + scale * raw_pred  # PD + injected residual -- same mechanism as fit_torque_scale.py

    q_next_pred = predict_next_position(tau_full, m_inv, C, G, q_t, qdot_t, dataset.rl_dt)
    err = (q_next_pred[valid, joint_idx] - q_next[valid, joint_idx]) ** 2
    return err.mean().item(), int(valid.sum().item())


def main() -> None:
    parser = argparse.ArgumentParser(description="Fit deployment-time torque scale via Isaac-free position matching.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Single-joint checkpoint (e.g. genan_default_pairindep_rh_FFJ1.pt).")
    parser.add_argument("--joint", type=str, required=True, help="Joint name (e.g. rh_FFJ1) -- must match the checkpoint.")
    parser.add_argument("--dataset", type=str, action="append", required=True, help="Dataset paths (repeatable).")
    parser.add_argument("--joints_yaml", type=str, default=None)
    parser.add_argument("--min_horizon", type=int, default=1)
    parser.add_argument("--preprocess_cache", type=str, default=_DEFAULT_PREPROCESS_CACHE)
    parser.add_argument("--dynamics_cache", type=str, default=_DEFAULT_DYNAMICS_CACHE)
    parser.add_argument(
        "--bounds", type=float, nargs=2, default=[0.0, 30.0],
        help="Search bounds for scale. Default matches shadow_hand_lite.py's effort_limit_sim=30.0 N*m -- "
             "see fit_torque_scale.py's module docstring for why values much above this are already implausible.",
    )
    parser.add_argument(
        "--sample_stride", type=int, default=1,
        help="Use every Nth row (speedup knob) -- default 1 uses every valid row in the dataset.",
    )
    args = parser.parse_args()

    joint_names, joint_upper_limits = load_joint_config(args.joints_yaml)
    dataset = AlignedTrajectoryDataset(
        paths=args.dataset, joint_names=joint_names, device="cpu",
        joint_upper_limits=joint_upper_limits, min_horizon=args.min_horizon,
    )
    dyn_cache = DynamicsCache(args.preprocess_cache, args.dynamics_cache)
    ensemble, ckpt = load_ensemble(args.checkpoint)
    joint_idx = resolve_joint_idx(args.joint, joint_names)
    kp, kd = load_pd_gains(args.joint)
    print(f"[INFO] Loaded checkpoint for {args.joint} (idx {joint_idx}), kp={kp:.4f} kd={kd:.4f}, "
          f"best_val_loss={ckpt['best_val_loss']:.6f}")

    n = min(dataset.num_steps, dyn_cache.num_rows)
    t_all = torch.arange(0, n, args.sample_stride)
    print(f"[INFO] {t_all.numel()} sampled timesteps (stride={args.sample_stride}, out of {n} total rows).")

    bounds = tuple(args.bounds)
    evals = []

    def objective(s: float) -> float:
        score, n_valid = position_match_score(ensemble, ckpt, dataset, dyn_cache, t_all, joint_idx, kp, kd, s)
        evals.append((s, score))
        print(f"  scale={s:8.4f}  position_mse={score:.8f}  (n_valid={n_valid})")
        return score

    result = minimize_scalar(objective, bounds=bounds, method="bounded")
    print(f"\n[RESULT] {args.joint}: best scale={result.x:.4f} (position_mse={result.fun:.8f}, {len(evals)} evals)")

    # Reference points for context.
    zero_score, _ = position_match_score(ensemble, ckpt, dataset, dyn_cache, t_all, joint_idx, kp, kd, 0.0)
    one_score, _ = position_match_score(ensemble, ckpt, dataset, dyn_cache, t_all, joint_idx, kp, kd, 1.0)
    print(f"[RESULT] Reference -- scale=0 (PD only) position_mse: {zero_score:.8f}")
    print(f"[RESULT] Reference -- scale=1 (unscaled) position_mse: {one_score:.8f}")


if __name__ == "__main__":
    main()

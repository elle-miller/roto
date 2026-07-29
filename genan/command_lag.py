"""Per-joint command-to-position lag, used to build the residual-q_cmd
training target (train_qcmd_residual.py): `L_j` is the number of control
steps such that `q_cmd_j(t)` best explains `q_meas_j(t + L_j)` -- i.e. "how
long after a command does the joint actually get there," measured directly
from real recorded data (no simulator, no dynamics model).

Isaac-free (pure torch/numpy), matching the rest of roto/genan/'s convention.
Prototyped as /tmp/qcmd_theory_test/phase_a_tracking_window.py's lag-scan
during this session's investigation; this is the productionized version,
extended to all 16 joints (that prototype only covered the 10 independently-
driven joints) and made segment-boundary-aware (the prototype didn't need to
be, since it only reported aggregate stats, not per-row training targets).
"""

from __future__ import annotations

import torch


def _lag_rmse(q_meas: torch.Tensor, q_cmd: torch.Tensor, dataset, joint_idx: int, k: int) -> float:
    """RMSE(q_meas_j(t), q_cmd_j(t-k)) over all rows where t-k stays inside
    row t's own segment (see build_delta_history's docstring for why this
    per-row segment bound matters, not just a global clamp).
    """
    t = torch.arange(dataset.num_steps)
    seg_start = dataset.segment_start(t)
    valid = (t - k) >= seg_start
    if not valid.any():
        return float("inf")
    t_valid = t[valid]
    err = q_meas[t_valid, joint_idx] - q_cmd[t_valid - k, joint_idx]
    return float(torch.sqrt(torch.mean(err ** 2)))


def compute_joint_lags(dataset, joint_names: list[str], max_lag: int = 15) -> dict[str, int]:
    """Return {joint_name: best_lag_steps}, one independent 1-D scan per
    joint (argmin RMSE over k in [0, max_lag]). A single global integer per
    joint -- not learned, not per-timestep -- matching Phase A's own
    methodology. For joints with a poor/flat error-vs-k curve (expected for
    the J1 mimic joints per this session's Phase C finding -- most command
    jumps never settle), the argmin is still well-defined but is a weaker
    single-delay approximation of what's actually a stateful, hysteresis-
    driven response; `lag_curve_flatness` below exists to flag exactly that.
    """
    lags = {}
    for j, name in enumerate(joint_names):
        rmses = [_lag_rmse(dataset.q_meas, dataset.q_cmd, dataset, j, k) for k in range(max_lag + 1)]
        best_k = int(torch.argmin(torch.tensor(rmses)))
        lags[name] = best_k
    return lags


def lag_curve_flatness(dataset, joint_names: list[str], lags: dict[str, int], max_lag: int = 15) -> dict[str, float]:
    """(rmse_at_worst_k - rmse_at_best_k) / rmse_at_best_k, per joint -- large
    means the lag scan found a clear, confident minimum (a well-behaved
    single-delay joint); near 0 means every k scores about the same (the
    single-fixed-delay model doesn't really fit this joint's dynamics, e.g.
    J1's backlash-driven response -- a signal to treat that joint's L_j with
    more skepticism, not a hard error).
    """
    flatness = {}
    for j, name in enumerate(joint_names):
        rmses = [_lag_rmse(dataset.q_meas, dataset.q_cmd, dataset, j, k) for k in range(max_lag + 1)]
        best, worst = min(rmses), max(rmses)
        flatness[name] = (worst - best) / best if best > 0 else float("inf")
    return flatness


if __name__ == "__main__":
    import argparse
    import os

    import yaml

    from dataset_loader import AlignedTrajectoryDataset
    from joint_config import load_joint_config

    _THIS_DIR = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser(description="Compute per-joint command-to-position lag from real data.")
    parser.add_argument("--config", type=str, default=os.path.join(_THIS_DIR, "agents", "shadowlite", "default.yaml"))
    parser.add_argument("--dataset", type=str, action="append", default=None)
    parser.add_argument("--max_lag", type=int, default=15)
    parser.add_argument("--out", type=str, default=os.path.join(_THIS_DIR, "agents", "shadowlite", "command_lags.yaml"))
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    dataset_paths = args.dataset if args.dataset is not None else cfg["dataset"]["paths"]
    joint_names, joint_upper_limits = load_joint_config()
    dataset = AlignedTrajectoryDataset(
        paths=dataset_paths, joint_names=joint_names, device="cpu",
        joint_upper_limits=joint_upper_limits, min_horizon=cfg["dataset"].get("min_horizon", 1),
    )
    print(f"[INFO] {dataset.num_steps} rows, {len(dataset.traj_starts)} segments")

    lags = compute_joint_lags(dataset, joint_names, args.max_lag)
    flatness = lag_curve_flatness(dataset, joint_names, lags, args.max_lag)

    print(f"{'joint':<10s} {'lag_steps':>9s} {'lag_ms':>7s} {'flatness':>9s}")
    for name in joint_names:
        ms = lags[name] * dataset.rl_dt * 1000
        flag = "  <- flat (weak single-delay fit)" if flatness[name] < 0.3 else ""
        print(f"{name:<10s} {lags[name]:9d} {ms:7.1f} {flatness[name]:9.3f}{flag}")

    with open(args.out, "w") as f:
        yaml.safe_dump({"lags": lags, "flatness": flatness, "max_lag": args.max_lag}, f, sort_keys=False)
    print(f"\n[INFO] Saved to {args.out}")

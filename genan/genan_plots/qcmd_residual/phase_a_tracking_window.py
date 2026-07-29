#!/usr/bin/env python3
"""Phase A: empirically characterize PD command-tracking quality on the 10
INDEPENDENTLY-driven joints (real per-joint q_cmd exists, 1:1 with the
recorded `action`). This calibrates the assumption behind the "learn q_cmd"
theory: how close does q_meas actually sit to q_cmd, and after a step change
in q_cmd, how many timesteps until tracking is "pitch perfect"?
"""
import os
import sys

_GENAN_DIR = "/home/ayush/icra/roto/genan"
_ROTO_ROOT = "/home/ayush/icra/roto"
sys.path.insert(0, _GENAN_DIR)
sys.path.insert(0, _ROTO_ROOT)

import numpy as np
import torch
import yaml

from dataset_loader import AlignedTrajectoryDataset, COUPLED_JOINT_PAIRS
from joint_config import load_joint_config

with open(os.path.join(_GENAN_DIR, "agents", "shadowlite", "default.yaml")) as f:
    cfg = yaml.safe_load(f)

joint_names, joint_upper_limits = load_joint_config()
dataset = AlignedTrajectoryDataset(
    paths=cfg["dataset"]["paths"], joint_names=joint_names, device="cpu",
    joint_upper_limits=joint_upper_limits, min_horizon=cfg["dataset"].get("min_horizon", 1),
)
print(f"[INFO] Loaded dataset: {dataset.num_steps} rows, {len(dataset.traj_starts)} segments, dt={dataset.rl_dt:.5f}s")

coupled_names = {n for pair in COUPLED_JOINT_PAIRS.values() for n in pair}
independent_idx = [i for i, n in enumerate(joint_names) if n not in coupled_names]
independent_names = [joint_names[i] for i in independent_idx]
print(f"[INFO] Independent (ground-truth-command) joints: {independent_names}")

q_cmd = dataset.q_cmd.numpy()
q_meas = dataset.q_meas.numpy()

print("\n=== Global tracking error (q_meas - q_cmd), independent joints only ===")
print(f"{'joint':<10s} {'rmse_rad':>10s} {'rmse_deg':>10s} {'p50_deg':>9s} {'p90_deg':>9s} {'p99_deg':>9s} "
      f"{'frac<1deg':>10s} {'frac<2deg':>10s} {'frac<5deg':>10s}")
for i, name in zip(independent_idx, independent_names):
    e = np.abs(q_meas[:, i] - q_cmd[:, i])
    e_deg = np.degrees(e)
    rmse = np.sqrt(np.mean(e ** 2))
    print(f"{name:<10s} {rmse:10.5f} {np.degrees(rmse):10.3f} {np.percentile(e_deg,50):9.3f} "
          f"{np.percentile(e_deg,90):9.3f} {np.percentile(e_deg,99):9.3f} "
          f"{np.mean(e_deg<1.0)*100:9.2f}% {np.mean(e_deg<2.0)*100:9.2f}% {np.mean(e_deg<5.0)*100:9.2f}%")

print("\n=== Lag scan: does shifting q_cmd back by k steps reduce error? (best-fit pure delay) ===")
print(f"{'joint':<10s} {'best_k':>7s} {'rmse@best_k_deg':>16s} {'rmse@k=0_deg':>13s}")
max_lag = 15
for i, name in zip(independent_idx, independent_names):
    best_k, best_rmse = 0, None
    rmse_at_0 = None
    for k in range(0, max_lag + 1):
        if k == 0:
            e = q_meas[:, i] - q_cmd[:, i]
        else:
            e = q_meas[k:, i] - q_cmd[:-k, i]
        rmse = np.sqrt(np.mean(e ** 2))
        if k == 0:
            rmse_at_0 = rmse
        if best_rmse is None or rmse < best_rmse:
            best_rmse, best_k = rmse, k
    print(f"{name:<10s} {best_k:7d} {np.degrees(best_rmse):16.3f} {np.degrees(rmse_at_0):13.3f}")

print("\n=== Step-response settling window: after a command jump > threshold, how many "
      "steps until |error| < 2deg and stays there for >=5 consecutive steps? ===")
jump_threshold_deg = 5.0
settle_threshold_deg = 2.0
hold_steps = 5
for i, name in zip(independent_idx, independent_names):
    cmd = q_cmd[:, i]
    meas = q_meas[:, i]
    dcmd = np.abs(np.diff(cmd, prepend=cmd[0]))
    jump_idx = np.nonzero(np.degrees(dcmd) > jump_threshold_deg)[0]
    # de-duplicate jumps that are within hold_steps of each other (same event)
    jump_idx = jump_idx[np.concatenate(([True], np.diff(jump_idx) > hold_steps))] if len(jump_idx) else jump_idx
    settle_times = []
    for j in jump_idx:
        err_deg = np.degrees(np.abs(meas[j:j + 200] - cmd[j:j + 200]))
        below = err_deg < settle_threshold_deg
        settled_at = None
        for t in range(len(below) - hold_steps):
            if below[t:t + hold_steps].all():
                settled_at = t
                break
        if settled_at is not None:
            settle_times.append(settled_at)
    if settle_times:
        st = np.array(settle_times)
        print(f"{name:<10s} n_jumps={len(jump_idx):4d} n_settled={len(st):4d} "
              f"settle_steps median={np.median(st):5.1f} p90={np.percentile(st,90):5.1f} "
              f"({np.median(st)*dataset.rl_dt*1000:.0f}ms median)")
    else:
        print(f"{name:<10s} n_jumps={len(jump_idx):4d}  no settle events found")

#!/usr/bin/env python3
"""Sanity check: (1) reproduce POSITION_LOSS_PROGRESS.md's claim that feeding
the REAL tau_target through predict_next_position reproduces real q_next to
~1e-4 rad. (2) Quantify how sensitive that same one-step prediction is to a
small per-joint torque perturbation on FFJ1/FFJ2 specifically, to explain why
train_qcmd_pair.py's losses are enormous.
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

from dataset_loader import AlignedTrajectoryDataset
from dynamics_cache import DynamicsCache
from joint_config import load_joint_config
from losses import predict_next_position

with open(os.path.join(_GENAN_DIR, "agents", "shadowlite", "default.yaml")) as f:
    cfg = yaml.safe_load(f)
joint_names, joint_upper_limits = load_joint_config()
dataset = AlignedTrajectoryDataset(
    paths=cfg["dataset"]["paths"], joint_names=joint_names, device="cpu",
    joint_upper_limits=joint_upper_limits, min_horizon=cfg["dataset"].get("min_horizon", 1),
)
dyn_cache = DynamicsCache(os.path.join(_GENAN_DIR, "cache", "smoothed.npz"), os.path.join(_GENAN_DIR, "cache", "dynamics.npz"))

idx_a, idx_b = joint_names.index("rh_FFJ1"), joint_names.index("rh_FFJ2")
t = torch.arange(0, 5000)
tau_target, m_inv, C, G, q_t, qdot_t, q_next, valid = dyn_cache.position_targets(dataset, t)
tau_target, m_inv, C, G, q_t, qdot_t, q_next = (
    x[valid] for x in (tau_target, m_inv, C, G, q_t, qdot_t, q_next)
)
print(f"[INFO] {tau_target.shape[0]} valid rows in this slice, dt={dataset.rl_dt:.6f}")

# (1) exact real tau_target -> should reproduce real q_next almost perfectly
q_next_pred = predict_next_position(tau_target, m_inv, C, G, q_t, qdot_t, dataset.rl_dt)
err_all = (q_next_pred - q_next).abs()
print(f"[CHECK 1] ALL-16-joint exact tau_target: max err={err_all.max().item():.6f} rad, "
      f"mean err={err_all.mean().item():.6f} rad (expect ~1e-4, per POSITION_LOSS_PROGRESS.md)")
print(f"[CHECK 1] FFJ1 max err={err_all[:, idx_a].max().item():.6f}  FFJ2 max err={err_all[:, idx_b].max().item():.6f}")

# (2) perturb ONLY FFJ1's torque by a small amount, everything else exact -- how much does
# FFJ1's OWN predicted next-position (and FFJ2's, via coupling) move?
print(f"\n[CHECK 2] M_inv diagonal magnitude at FFJ1/FFJ2:")
print(f"  M_inv[FFJ1,FFJ1] mean={m_inv[:, idx_a, idx_a].mean().item():.2f} "
      f"M_inv[FFJ2,FFJ2] mean={m_inv[:, idx_b, idx_b].mean().item():.2f} "
      f"M_inv[FFJ1,FFJ2] mean={m_inv[:, idx_a, idx_b].mean().item():.2f}")

for dtau in [0.001, 0.01, 0.1, 1.0]:
    tau_pert = tau_target.clone()
    tau_pert[:, idx_a] += dtau
    q_next_pert = predict_next_position(tau_pert, m_inv, C, G, q_t, qdot_t, dataset.rl_dt)
    d_qa = (q_next_pert[:, idx_a] - q_next[:, idx_a]).abs().mean().item()
    d_qb = (q_next_pert[:, idx_b] - q_next[:, idx_b]).abs().mean().item()
    print(f"  dtau_FFJ1={dtau:6.3f} Nm -> mean |dq_FFJ1_pred|={d_qa:.5f} rad ({np.degrees(d_qa):.2f} deg)  "
          f"mean |dq_FFJ2_pred| (cross-coupled)={d_qb:.5f} rad ({np.degrees(d_qb):.2f} deg)   "
          f"[one {dataset.rl_dt*1000:.1f}ms step]")

print(f"\n[INFO] tau_target stats: FFJ1 |tau| mean={tau_target[:,idx_a].abs().mean().item():.4f} "
      f"max={tau_target[:,idx_a].abs().max().item():.4f}   "
      f"FFJ2 |tau| mean={tau_target[:,idx_b].abs().mean().item():.4f} max={tau_target[:,idx_b].abs().max().item():.4f}")

#!/usr/bin/env python3
"""v2: same theory test as train_qcmd_pair.py, but reparameterized to fix the
numerical pathology found in v1 -- FFJ1/FFJ2's identified inertia is so tiny
(M_inv diagonal ~15,000-44,000) that even a ~0.01 Nm torque error blows the
one-step semi-implicit-Euler position prediction up by several degrees (see
sanity_check.py). Fix: predict a SMALL BOUNDED DELTA around q_meas (not an
unconstrained absolute position over the full joint range), so the induced
Kp*delta torque magnitude starts near the real ~0.002 Nm scale instead of
spanning the full multi-Nm range a naive tanh-to-[lo,hi] mapping allows. Also
adds gradient clipping and a lower LR since the loss surface here is steep.
"""
import os
import sys

_GENAN_DIR = "/home/ayush/icra/roto/genan"
_ROTO_ROOT = "/home/ayush/icra/roto"
sys.path.insert(0, _GENAN_DIR)
sys.path.insert(0, _ROTO_ROOT)

import numpy as np
import torch
import torch.nn.functional as F
import yaml

from dataset_loader import AlignedTrajectoryDataset
from dynamics_cache import DynamicsCache
from history import build_delta_history
from joint_config import load_joint_config
from losses import pd_baseline_torque, predict_next_position
from model import GenANEnsemble
from pd_gains import load_pd_gains
from train_genan_single import split_segments

torch.manual_seed(0)

PAIR = ("rh_FFJ1", "rh_FFJ2")
EPOCHS = 60
BATCH_SIZE = 4096
LR = 1e-4
ENSEMBLE_SIZE = 3
HISTORY_LEN, STRIDE = 4, 2
MAX_DELTA_RAD = 0.05  # ~2.9 deg -- bounds |q_cmd_pred - q_meas|, see module docstring
GRAD_CLIP = 1.0

with open(os.path.join(_GENAN_DIR, "agents", "shadowlite", "default.yaml")) as f:
    cfg = yaml.safe_load(f)
joint_names, joint_upper_limits = load_joint_config()

dataset = AlignedTrajectoryDataset(
    paths=cfg["dataset"]["paths"], joint_names=joint_names, device="cpu",
    joint_upper_limits=joint_upper_limits, min_horizon=cfg["dataset"].get("min_horizon", 1),
)
dyn_cache = DynamicsCache(os.path.join(_GENAN_DIR, "cache", "smoothed.npz"), os.path.join(_GENAN_DIR, "cache", "dynamics.npz"))
print(f"[INFO] Dataset: {dataset.num_steps} rows. DynamicsCache: {dyn_cache.num_rows} rows")

idx_a, idx_b = joint_names.index(PAIR[0]), joint_names.index(PAIR[1])
kp_a, kd_a = load_pd_gains(PAIR[0])
kp_b, kd_b = load_pd_gains(PAIR[1])
print(f"[INFO] {PAIR[0]}: idx={idx_a} kp={kp_a:.4f} kd={kd_a:.4f}   {PAIR[1]}: idx={idx_b} kp={kp_b:.4f} kd={kd_b:.4f}")
print(f"[INFO] max_delta={MAX_DELTA_RAD} rad ({np.degrees(MAX_DELTA_RAD):.2f} deg) -> "
      f"max induced |tau|~{MAX_DELTA_RAD*kp_a:.4f} Nm (real tau_target scale ~0.002-0.02 Nm)")

train_t, val_t = split_segments(dataset, val_frac=0.2, seed=0)


def build_input(t: torch.Tensor) -> torch.Tensor:
    q_hist = build_delta_history(dataset.q_meas, t, HISTORY_LEN, STRIDE, dataset)
    u_hist = build_delta_history(dataset.q_cmd, t, HISTORY_LEN, STRIDE, dataset)
    return torch.cat([q_hist, u_hist], dim=-1)


x_train = build_input(train_t)
x_val = build_input(val_t)
input_dim = x_train.shape[1]

ensemble = GenANEnsemble(input_dim, num_joints=2, ensemble_size=ENSEMBLE_SIZE, seed=0, bounded_output=True, torque_range=1.0)
ensemble.input_scaler(x_train, train=True)

optimizers = [torch.optim.Adam(m.parameters(), lr=LR) for m in ensemble.members]
generators = [torch.Generator().manual_seed(1000 + i) for i in range(ENSEMBLE_SIZE)]


def qcmd_from_raw(raw_tanh: torch.Tensor, q_meas_ab: torch.Tensor) -> torch.Tensor:
    """raw_tanh: (...,2) in (-1,1) -> q_meas_ab + bounded delta in [-MAX_DELTA_RAD, MAX_DELTA_RAD]."""
    return q_meas_ab + raw_tanh * MAX_DELTA_RAD


def position_loss_for_qcmd(q_cmd_ab: torch.Tensor, t_batch: torch.Tensor):
    tau_target, m_inv, C, G, q_t, qdot_t, q_next, valid = dyn_cache.position_targets(dataset, t_batch)
    if not valid.any():
        return None, None
    tau_a = pd_baseline_torque(q_cmd_ab[:, 0:1], q_t[:, idx_a:idx_a + 1], qdot_t[:, idx_a:idx_a + 1], kp_a, kd_a)
    tau_b = pd_baseline_torque(q_cmd_ab[:, 1:2], q_t[:, idx_b:idx_b + 1], qdot_t[:, idx_b:idx_b + 1], kp_b, kd_b)
    tau_full = tau_target.clone()
    tau_full[:, idx_a] = tau_a.squeeze(-1)
    tau_full[:, idx_b] = tau_b.squeeze(-1)
    q_next_pred = predict_next_position(tau_full, m_inv, C, G, q_t, qdot_t, dataset.rl_dt)
    err = q_next_pred[valid][:, [idx_a, idx_b]] - q_next[valid][:, [idx_a, idx_b]]
    return F.mse_loss(err, torch.zeros_like(err)), q_t


# baseline (heuristic q_cmd, unchanged)
with torch.no_grad():
    t_c_val = dataset.clamp(val_t)
    q_cmd_heuristic_val = dataset.q_cmd[t_c_val][:, [idx_a, idx_b]]
    baseline_loss, _ = position_loss_for_qcmd(q_cmd_heuristic_val, val_t)
    print(f"[BASELINE] heuristic q_cmd: val MSE={baseline_loss.item():.8f} RMSE_deg={np.degrees(np.sqrt(baseline_loss.item())):.4f}")

    # reference: q_cmd_pred == q_meas exactly (zero delta) -- i.e. "PD holds position, no correction"
    _, q_t_ref = position_loss_for_qcmd(q_cmd_heuristic_val, val_t)  # just to get q_t aligned
    tau_target, m_inv, C, G, q_t, qdot_t, q_next, valid = dyn_cache.position_targets(dataset, val_t)
    q_cmd_zero_delta = q_t[:, [idx_a, idx_b]]
    zero_delta_loss, _ = position_loss_for_qcmd(q_cmd_zero_delta, val_t)
    print(f"[REFERENCE] q_cmd=q_meas (zero correction): val MSE={zero_delta_loss.item():.8f} "
          f"RMSE_deg={np.degrees(np.sqrt(zero_delta_loss.item())):.4f}")

best_val = float("inf")
n_train = x_train.shape[0]
steps_per_epoch = max(1, n_train // BATCH_SIZE)

for epoch in range(EPOCHS):
    epoch_losses = []
    for _ in range(steps_per_epoch):
        step_losses = []
        for member, opt, gen in zip(ensemble.members, optimizers, generators):
            idx = torch.randint(0, n_train, (min(BATCH_SIZE, n_train),), generator=gen)
            x = ensemble.input_scaler(x_train[idx], train=False)
            raw = member(x)
            t_c = dataset.clamp(train_t[idx])
            q_meas_ab = dataset.q_meas[t_c][:, [idx_a, idx_b]]
            q_cmd_pred = qcmd_from_raw(raw, q_meas_ab)
            loss, _ = position_loss_for_qcmd(q_cmd_pred, train_t[idx])
            if loss is None:
                continue
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(member.parameters(), GRAD_CLIP)
            opt.step()
            step_losses.append(loss.item())
        if step_losses:
            epoch_losses.append(sum(step_losses) / len(step_losses))

    with torch.no_grad():
        val_losses = []
        t_c_val = dataset.clamp(val_t)
        q_meas_ab_val = dataset.q_meas[t_c_val][:, [idx_a, idx_b]]
        for member in ensemble.members:
            x = ensemble.input_scaler(x_val, train=False)
            raw = member(x)
            q_cmd_pred = qcmd_from_raw(raw, q_meas_ab_val)
            vloss, _ = position_loss_for_qcmd(q_cmd_pred, val_t)
            val_losses.append(vloss.item())
        val_loss = float(np.mean(val_losses))
    train_loss = float(np.mean(epoch_losses)) if epoch_losses else float("nan")
    print(f"[epoch {epoch:3d}] train_mse={train_loss:.8f} val_mse={val_loss:.8f} val_rmse_deg={np.degrees(np.sqrt(val_loss)):.4f}")
    if val_loss < best_val:
        best_val = val_loss

print(f"\n[RESULT] Best val position MSE (learned, bounded-delta q_cmd) = {best_val:.8f} (RMSE={np.degrees(np.sqrt(best_val)):.4f} deg)")
print(f"[RESULT] Baseline (heuristic q_cmd)                            = {baseline_loss.item():.8f} (RMSE={np.degrees(np.sqrt(baseline_loss.item())):.4f} deg)")
print(f"[RESULT] Reference (zero-correction, q_cmd=q_meas)             = {zero_delta_loss.item():.8f} (RMSE={np.degrees(np.sqrt(zero_delta_loss.item())):.4f} deg)")

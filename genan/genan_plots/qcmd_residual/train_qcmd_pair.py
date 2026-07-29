#!/usr/bin/env python3
"""PROTOTYPE / theory test: instead of learning per-joint TORQUE for the
rh_FFJ1/rh_FFJ2 coupled pair, learn a corrected per-joint POSITION COMMAND
(q_cmd) for each, keep the real identified Kp/Kd fixed, derive
tau = Kp*(q_cmd_pred - q_meas) - Kd*qdot_meas (pd_baseline_torque, unchanged
from the rest of the codebase), and train via the EXISTING Position loss
(predict_next_position/DynamicsCache) so the label is real future q_meas,
not a hand-built q_cmd label (none exists for this pair on hardware).

Ablation baseline computed alongside: what position-loss MSE would the
CURRENT heuristic-split dataset.q_cmd get if run through the exact same
known-Kp/Kd + real dynamics pipeline, with no learning at all? This isolates
whether a LEARNED q_cmd beats the existing geometric split, holding the
control law (PD) and the dynamics model fixed.
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
EPOCHS = 40
BATCH_SIZE = 4096
LR = 1e-3
ENSEMBLE_SIZE = 3
HISTORY_LEN, STRIDE = 4, 2

with open(os.path.join(_GENAN_DIR, "agents", "shadowlite", "default.yaml")) as f:
    cfg = yaml.safe_load(f)

joint_names, joint_upper_limits = load_joint_config()
joint_lower_limits = {}
import yaml as _yaml
with open(os.path.join(_ROTO_ROOT, "shadow_pd_id", "config", "joints.yaml")) as f:
    _jy = _yaml.safe_load(f)
    joint_lower_limits = {k: float(v["lower"]) for k, v in _jy["joint_limits_rad"].items()}

dataset = AlignedTrajectoryDataset(
    paths=cfg["dataset"]["paths"], joint_names=joint_names, device="cpu",
    joint_upper_limits=joint_upper_limits, min_horizon=cfg["dataset"].get("min_horizon", 1),
)
print(f"[INFO] Dataset: {dataset.num_steps} rows, {len(dataset.traj_starts)} segments")

dyn_cache = DynamicsCache(
    os.path.join(_GENAN_DIR, "cache", "smoothed.npz"),
    os.path.join(_GENAN_DIR, "cache", "dynamics.npz"),
)
print(f"[INFO] DynamicsCache: {dyn_cache.num_rows} rows")

idx_a, idx_b = joint_names.index(PAIR[0]), joint_names.index(PAIR[1])
lo_a, hi_a = joint_lower_limits[PAIR[0]], joint_upper_limits[PAIR[0]]
lo_b, hi_b = joint_lower_limits[PAIR[1]], joint_upper_limits[PAIR[1]]
kp_a, kd_a = load_pd_gains(PAIR[0])
kp_b, kd_b = load_pd_gains(PAIR[1])
print(f"[INFO] {PAIR[0]}: idx={idx_a} range=[{lo_a:.4f},{hi_a:.4f}] kp={kp_a:.4f} kd={kd_a:.4f}")
print(f"[INFO] {PAIR[1]}: idx={idx_b} range=[{lo_b:.4f},{hi_b:.4f}] kp={kp_b:.4f} kd={kd_b:.4f}")

train_t, val_t = split_segments(dataset, val_frac=0.2, seed=0)
print(f"[INFO] train rows={train_t.numel()} val rows={val_t.numel()}")


def build_input(t: torch.Tensor) -> torch.Tensor:
    q_hist = build_delta_history(dataset.q_meas, t, HISTORY_LEN, STRIDE, dataset)
    u_hist = build_delta_history(dataset.q_cmd, t, HISTORY_LEN, STRIDE, dataset)
    return torch.cat([q_hist, u_hist], dim=-1)


x_train = build_input(train_t)
x_val = build_input(val_t)
input_dim = x_train.shape[1]

ensemble = GenANEnsemble(input_dim, num_joints=2, ensemble_size=ENSEMBLE_SIZE, seed=0, bounded_output=True, torque_range=1.0)
ensemble.input_scaler(x_train, train=True)  # fit input scaler only -- no torque label to fit label_scaler against

optimizers = [torch.optim.Adam(m.parameters(), lr=LR) for m in ensemble.members]
generators = [torch.Generator().manual_seed(1000 + i) for i in range(ENSEMBLE_SIZE)]


def qcmd_from_raw(raw_tanh: torch.Tensor) -> torch.Tensor:
    """raw_tanh: (..., 2) in (-1,1) -> (..., 2) affine-mapped to each joint's own [lo,hi]."""
    a = lo_a + (raw_tanh[..., 0:1] + 1.0) * 0.5 * (hi_a - lo_a)
    b = lo_b + (raw_tanh[..., 1:2] + 1.0) * 0.5 * (hi_b - lo_b)
    return torch.cat([a, b], dim=-1)


def position_loss_for_qcmd(q_cmd_ab: torch.Tensor, t_batch: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """q_cmd_ab: (batch,2) predicted commands for (a,b). Returns (loss, valid_count)."""
    tau_target, m_inv, C, G, q_t, qdot_t, q_next, valid = dyn_cache.position_targets(dataset, t_batch)
    if not valid.any():
        return None, 0
    tau_a = pd_baseline_torque(q_cmd_ab[:, 0:1], q_t[:, idx_a:idx_a + 1], qdot_t[:, idx_a:idx_a + 1], kp_a, kd_a)
    tau_b = pd_baseline_torque(q_cmd_ab[:, 1:2], q_t[:, idx_b:idx_b + 1], qdot_t[:, idx_b:idx_b + 1], kp_b, kd_b)
    tau_full = tau_target.clone()
    tau_full[:, idx_a] = tau_a.squeeze(-1)
    tau_full[:, idx_b] = tau_b.squeeze(-1)
    q_next_pred = predict_next_position(tau_full, m_inv, C, G, q_t, qdot_t, dataset.rl_dt)
    err = q_next_pred[valid][:, [idx_a, idx_b]] - q_next[valid][:, [idx_a, idx_b]]
    return F.mse_loss(err, torch.zeros_like(err)), int(valid.sum())


# --- Baseline: current heuristic-split dataset.q_cmd, run through the SAME known PD + dynamics, no learning ---
with torch.no_grad():
    t_c_val = dataset.clamp(val_t)
    q_cmd_heuristic_val = dataset.q_cmd[t_c_val][:, [idx_a, idx_b]]
    baseline_loss, baseline_n = position_loss_for_qcmd(q_cmd_heuristic_val, val_t)
    print(f"[BASELINE] heuristic-split q_cmd -> known PD -> real dynamics: "
          f"val position MSE={baseline_loss.item():.8f}  RMSE_deg={np.degrees(np.sqrt(baseline_loss.item())):.4f}  n={baseline_n}")

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
            raw = member(x)  # (batch,2) in (-1,1)
            q_cmd_pred = qcmd_from_raw(raw)
            loss, n_valid = position_loss_for_qcmd(q_cmd_pred, train_t[idx])
            if loss is None:
                continue
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
            q_cmd_pred = qcmd_from_raw(raw)
            vloss, _ = position_loss_for_qcmd(q_cmd_pred, val_t)
            val_losses.append(vloss.item())
        val_loss = float(np.mean(val_losses))
    train_loss = float(np.mean(epoch_losses)) if epoch_losses else float("nan")
    print(f"[epoch {epoch:3d}] train_mse={train_loss:.8f} val_mse={val_loss:.8f} val_rmse_deg={np.degrees(np.sqrt(val_loss)):.4f}")
    if val_loss < best_val:
        best_val = val_loss
        best_state = {k: v.clone() for k, v in ensemble.state_dict().items()}

print(f"\n[RESULT] Best val position MSE (learned q_cmd) = {best_val:.8f}  (RMSE={np.degrees(np.sqrt(best_val)):.4f} deg)")
print(f"[RESULT] Baseline val position MSE (heuristic q_cmd) = {baseline_loss.item():.8f}  (RMSE={np.degrees(np.sqrt(baseline_loss.item())):.4f} deg)")
improvement = (1 - best_val / baseline_loss.item()) * 100
print(f"[RESULT] Improvement over heuristic baseline: {improvement:.2f}%")

ensemble.load_state_dict(best_state)
torch.save({"ensemble_state_dict": ensemble.state_dict(), "input_dim": input_dim,
            "history_len": HISTORY_LEN, "stride": STRIDE, "pair": PAIR,
            "idx_a": idx_a, "idx_b": idx_b, "lo_a": lo_a, "hi_a": hi_a, "lo_b": lo_b, "hi_b": hi_b,
            "kp_a": kp_a, "kd_a": kd_a, "kp_b": kp_b, "kd_b": kd_b}, "/tmp/qcmd_theory_test/qcmd_pair_ffj1_ffj2.pt")
print("[INFO] Saved /tmp/qcmd_theory_test/qcmd_pair_ffj1_ffj2.pt")

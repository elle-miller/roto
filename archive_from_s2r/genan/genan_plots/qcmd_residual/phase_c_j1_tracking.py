#!/usr/bin/env python3
"""How well does the CURRENT (fixed) heuristic q_cmd, together with the
identified/aliased Kp,Kd, actually describe where J1 (the coupled mimic
joint) really was -- using the same direct q_cmd-vs-q_meas comparison Phase A
used for the independent joints (the most trustworthy measure available,
since it's real recorded data end to end, no M/C/G model involved).

Important framing difference from Phase A: for INDEPENDENT joints, q_cmd was
a real transmitted setpoint and q_meas was the real closed-loop response, so
a gap there means "PD control lag." For J1, there is no real per-joint
setpoint on hardware at all (one motor/tendon drives J1+J2 together) -- q_cmd
here is a construct (dataset.py's _split_coupled_command, a static two-
segment law). So a gap here means "the heuristic's implied per-joint target
doesn't match how the real tendon actually moved J1" -- i.e. it's measuring
the heuristic's own fidelity to the real backlash mechanism, not PD lag.

Also compares against the OLD (pre-fix, 90deg split-point) q_cmd, and against
J2 (the driver) and the independent-joint numbers from Phase A, for context.
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
from pd_gains import load_pd_gains

with open(os.path.join(_GENAN_DIR, "agents", "shadowlite", "default.yaml")) as f:
    cfg = yaml.safe_load(f)

joint_names, joint_upper_limits_fixed = load_joint_config()  # NOW includes the fix

# Rebuild the OLD (pre-fix) limits by hand for comparison: same as fixed
# except FFJ2/MFJ2/RFJ2 use joints.yaml's original 1.5708 rad proxy range.
joint_upper_limits_old = dict(joint_upper_limits_fixed)
for n in ["rh_FFJ2", "rh_MFJ2", "rh_RFJ2"]:
    joint_upper_limits_old[n] = 1.5708

dataset_fixed = AlignedTrajectoryDataset(
    paths=cfg["dataset"]["paths"], joint_names=joint_names, device="cpu",
    joint_upper_limits=joint_upper_limits_fixed, min_horizon=cfg["dataset"].get("min_horizon", 1),
)
dataset_old = AlignedTrajectoryDataset(
    paths=cfg["dataset"]["paths"], joint_names=joint_names, device="cpu",
    joint_upper_limits=joint_upper_limits_old, min_horizon=cfg["dataset"].get("min_horizon", 1),
)
print(f"[INFO] {dataset_fixed.num_steps} rows, {len(dataset_fixed.traj_starts)} segments, dt={dataset_fixed.rl_dt:.5f}s")

q_meas = dataset_fixed.q_meas.numpy()  # identical between the two datasets -- q_meas doesn't depend on the split
q_cmd_fixed = dataset_fixed.q_cmd.numpy()
q_cmd_old = dataset_old.q_cmd.numpy()


def report(name_list, q_cmd, label):
    print(f"\n=== {label} ===")
    print(f"{'joint':<10s} {'rmse_deg':>9s} {'p50_deg':>8s} {'p90_deg':>8s} {'p99_deg':>8s} "
          f"{'frac<1':>7s} {'frac<2':>7s} {'frac<5':>7s} {'frac<10':>8s}")
    for name in name_list:
        i = joint_names.index(name)
        e_deg = np.degrees(np.abs(q_meas[:, i] - q_cmd[:, i]))
        rmse = np.sqrt(np.mean(e_deg ** 2))
        print(f"{name:<10s} {rmse:9.3f} {np.percentile(e_deg,50):8.3f} {np.percentile(e_deg,90):8.3f} "
              f"{np.percentile(e_deg,99):8.3f} {np.mean(e_deg<1)*100:6.1f}% {np.mean(e_deg<2)*100:6.1f}% "
              f"{np.mean(e_deg<5)*100:6.1f}% {np.mean(e_deg<10)*100:7.1f}%")


j1_names = [pair[0] for pair in COUPLED_JOINT_PAIRS.values()]
j2_names = [pair[1] for pair in COUPLED_JOINT_PAIRS.values()]

report(j1_names, q_cmd_old, "J1 (mimic) -- OLD/buggy 90deg split q_cmd")
report(j1_names, q_cmd_fixed, "J1 (mimic) -- FIXED 100deg split q_cmd")
report(j2_names, q_cmd_fixed, "J2 (driver) -- FIXED split q_cmd (driver's own split point unaffected by the J2-upper fix itself, since combined<=j2_upper is a direct clip either way)")

# Step-response settling window for J1, fixed split -- same methodology as Phase A
print("\n=== J1 settling window after a >5deg q_cmd jump (fixed split) ===")
jump_threshold_deg, settle_threshold_deg, hold_steps = 5.0, 2.0, 5
for name in j1_names:
    i = joint_names.index(name)
    cmd, meas = q_cmd_fixed[:, i], q_meas[:, i]
    dcmd = np.abs(np.diff(cmd, prepend=cmd[0]))
    jump_idx = np.nonzero(np.degrees(dcmd) > jump_threshold_deg)[0]
    jump_idx = jump_idx[np.concatenate(([True], np.diff(jump_idx) > hold_steps))] if len(jump_idx) else jump_idx
    settle_times = []
    for j in jump_idx:
        err_deg = np.degrees(np.abs(meas[j:j + 200] - cmd[j:j + 200]))
        below = err_deg < settle_threshold_deg
        for t in range(len(below) - hold_steps):
            if below[t:t + hold_steps].all():
                settle_times.append(t)
                break
    if settle_times:
        st = np.array(settle_times)
        print(f"{name:<10s} n_jumps={len(jump_idx):4d} n_settled={len(st):4d} "
              f"settle_steps median={np.median(st):5.1f} p90={np.percentile(st,90):5.1f} "
              f"never-settled_frac={1 - len(st)/max(len(jump_idx),1):.2%}")
    else:
        print(f"{name:<10s} n_jumps={len(jump_idx):4d}  no settle events found")

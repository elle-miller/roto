#!/usr/bin/env python3
"""Load, clean, and split collected Shadow Hand Lite trajectory logs.

WHY THIS FILE EXISTS: the optimizer (Step 4) compares simulated joint
position against the REAL measured joint position, sample by sample. If the
real data is noisy, unevenly sampled, or the train/held-out split leaks,
every later step silently inherits that problem. This is the one place all
of that gets handled, once, carefully.

Tasks performed here (see plan Step 1):
  1. Discover collected logs (from collect_traj_hw.py or collect_traj_sim.py,
     both save the same schema) under a directory.
  2. Read each log's OWN `meta_type` / `meta_held_out` fields (not the
     filename or folder) to classify it -- see DECISIONS.md for why this is
     more robust than relying on directory layout.
  3. Zero-phase low-pass filter the measured position, so noise doesn't get
     mistaken for real dynamics by the optimizer.
  4. Return per-joint {train: {...}, held_out: {...}} datasets, keeping the
     two firmly separate.

KNOWN LIMITATION (flagged, not silently ignored): collect_traj_hw.py's `t`
array is a NOMINAL step count (`step / RL_HZ`), not a measured wall-clock
timestamp -- rospy.Rate.sleep() jitter is not currently recorded. So the
"resampling" here is really just interpolation onto the nominal grid (a
no-op if there was no jitter) plus filtering; it cannot correct for real
timing jitter because that information isn't in the log. If validation
later shows timing jitter matters, collect_traj_hw.py should be extended to
log rospy.get_time() per sample -- that's a hardware-script change, not
something to paper over here.
"""

from __future__ import annotations

import glob
import os

import numpy as np
import yaml
from scipy.signal import butter, filtfilt

_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_HERE)
_DEFAULT_CONFIG = os.path.join(_PROJECT_ROOT, "config", "joints.yaml")


def load_joint_config(path: str = _DEFAULT_CONFIG) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def find_logs(log_dir: str) -> list[str]:
    """All collected-log .npz files under log_dir (recursive)."""
    return sorted(glob.glob(os.path.join(log_dir, "**", "*.npz"), recursive=True))


def load_log(path: str) -> dict:
    """Read one collected log into a plain dict, tagged with its own type/held_out.

    `q` is the measured position of the ONE joint this log excited (indexed
    out of the full 13-joint `actual_pos` array using the log's own
    joint_idx) -- that's the column the rest of this project cares about.
    """
    d = np.load(path, allow_pickle=True)
    joint_idx = int(d["joint_idx"])
    joint_name = str(d["joint_name"])
    traj_type = str(d["meta_type"]) if "meta_type" in d else "unknown"
    held_out = bool(d["meta_held_out"]) if "meta_held_out" in d else False

    t = np.asarray(d["t"], dtype=np.float64)
    cmd = np.asarray(d["cmd"], dtype=np.float64)
    actual_pos = np.asarray(d["actual_pos"], dtype=np.float64)  # (N, 13)
    actual_vel = np.asarray(d["actual_vel"], dtype=np.float64)  # (N, 13)

    if actual_pos.shape[0] != len(t):
        raise ValueError(
            f"{path}: t has {len(t)} samples but actual_pos has {actual_pos.shape[0]} -- "
            "logging bug upstream, not something to silently truncate/pad here."
        )

    return dict(
        path=path,
        joint_idx=joint_idx,
        joint_name=joint_name,
        traj_type=traj_type,
        held_out=held_out,
        t=t,
        cmd=cmd,
        q_raw=actual_pos[:, joint_idx],
        qdot_raw=actual_vel[:, joint_idx],
        actual_pos_all=actual_pos,  # kept in case cross-joint leakage needs checking
    )


def resample_to_nominal_grid(log: dict, control_rate_hz: float) -> dict:
    """Interpolate onto an exactly-even grid at 1/control_rate_hz.

    This is a no-op in the common case (the hw/sim scripts already sample
    once per control tick), but guards against the rare dropped/duplicated
    sample without silently misaligning cmd and q. See module docstring for
    why this does NOT correct for real timing jitter (that data isn't
    recorded upstream).
    """
    t, cmd, q, qdot = log["t"], log["cmd"], log["q_raw"], log["qdot_raw"]
    n = len(t)
    dt = 1.0 / control_rate_hz
    t_grid = np.arange(n) * dt

    out = dict(log)
    out["t"] = t_grid
    out["cmd"] = np.interp(t_grid, t, cmd)
    out["q_raw"] = np.interp(t_grid, t, q)
    out["qdot_raw"] = np.interp(t_grid, t, qdot)
    return out


def filter_position(
    q: np.ndarray,
    control_rate_hz: float,
    cutoff_hz: float = 10.0,
    order: int = 4,
) -> np.ndarray:
    """Zero-phase Butterworth low-pass on measured position.

    cutoff_hz=10 is a conservative default: the fastest excitation content we
    generate is the chirp's f1=2.5 Hz (see make_trajectories.py), so 10 Hz
    leaves a wide margin above real signal content while still cutting
    sensor/encoder noise, which is typically much higher frequency. This is a
    placeholder until real hardware data lets us look at an actual noise
    spectrum -- if real encoder noise turns out to have energy below 10 Hz,
    this needs to be lowered, and if the cutoff turns out to visibly distort
    the chirp's fastest cycles, it needs to be raised. Use filtfilt (not
    lfilter) specifically because it's zero-phase -- a phase-shifting filter
    would offset q_filt from cmd in time, which would corrupt every
    downstream tracking-error computation.
    """
    nyquist = control_rate_hz / 2.0
    if cutoff_hz >= nyquist:
        raise ValueError(f"cutoff_hz={cutoff_hz} must be below Nyquist ({nyquist} Hz).")
    b, a = butter(order, cutoff_hz / nyquist, btype="low")
    return filtfilt(b, a, q)


def load_dataset(log_dir: str, cfg: dict | None = None, cutoff_hz: float = 10.0) -> dict:
    """Load every log under log_dir into {joint_name: {"train": [...], "held_out": [...]}}.

    Each entry in the train/held_out lists is a dict with t, cmd, q_raw,
    q_filt, qdot_raw, traj_type -- everything Step 2's sim_rollout.py and
    Step 4's optimizer need, with train and held-out kept in physically
    separate lists so it's structurally hard to accidentally mix them.
    """
    if cfg is None:
        cfg = load_joint_config()
    control_rate_hz = cfg["control_rate_hz"]

    dataset: dict = {name: {"train": [], "held_out": []} for name in cfg["policy_joint_order"]}

    for path in find_logs(log_dir):
        log = load_log(path)
        log = resample_to_nominal_grid(log, control_rate_hz)
        log["q_filt"] = filter_position(log["q_raw"], control_rate_hz, cutoff_hz=cutoff_hz)

        bucket = "held_out" if log["held_out"] else "train"
        dataset[log["joint_name"]][bucket].append(log)

    return dataset


def summarize(dataset: dict) -> None:
    for joint_name, buckets in dataset.items():
        n_train = len(buckets["train"])
        n_ho = len(buckets["held_out"])
        types = sorted({log["traj_type"] for log in buckets["train"] + buckets["held_out"]})
        print(f"  {joint_name:10s}  train={n_train:2d}  held_out={n_ho:2d}  types={types}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log_dir", type=str, required=True)
    parser.add_argument("--cutoff_hz", type=float, default=10.0)
    args = parser.parse_args()

    ds = load_dataset(args.log_dir, cutoff_hz=args.cutoff_hz)
    print(f"Loaded dataset from {args.log_dir}:")
    summarize(ds)

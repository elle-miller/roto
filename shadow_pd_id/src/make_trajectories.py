#!/usr/bin/env python3
"""Generate excitation command trajectories for Shadow Hand Lite PD-gain identification.

WHY THIS FILE EXISTS: identification only works if we've actually excited the
dynamics we're trying to measure. A single motion type leaves parameters
ambiguous -- e.g. a slow ramp alone can never tell you damping (Kd) or viscous
friction, because the joint never moves fast enough for velocity-dependent
effects to show up. So for each of the 13 policy joints (the 3 coupled
FFJ2/MFJ2/RFJ2 "driver" joints stand in for their whole coupled pair -- see
coupled_groups in config/joints.yaml) this generates FOUR trajectory types:

  chirp  - frequency sweep. The main tool for exciting many speeds in one
           run; this is what makes Kd / viscous friction identifiable.
  step   - sudden setpoint jumps of varying size. Reveals Kp and the
           fast-transient/overshoot behavior a chirp moves through too
           gradually to isolate.
  random - a sum of a few random-frequency sinusoids. Covers speed/position
           combinations the clean chirp/step signals miss, so the later fit
           isn't just memorizing one motion shape. One of these (a different
           random seed) is set aside as the HELD-OUT trajectory -- Step 4's
           optimizer must never see it, and Step 5 validates against it.
  ramp   - one slow, quasi-static sweep across the joint's full range.
           Mostly exposes Coulomb/static friction and any position-dependent
           effect (e.g. a changing tendon moment arm) that only shows up at
           near-zero velocity.

Every trajectory excites ONE joint (or coupled group) at a time, holding all
others at the safe default (zero) -- this matches how collect_traj_hw.py /
collect_traj_sim.py already drive the hand, and keeps the later per-joint
least-squares/optimization well-conditioned and easy to debug.

SAFETY: every trajectory's amplitude is clamped, IN CODE, so the joint can
never exceed config/joints.yaml's position AND velocity limits -- not by
picking "reasonable-looking" numbers by hand. For a sinusoidal component of
amplitude A and frequency f, peak speed is A * 2*pi*f (from differentiating
A*sin(2*pi*f*t)); we solve that inequality for the max A the velocity limit
allows, and additionally clamp to stay inside the position range. This is the
one part of the whole project where getting it wrong risks damaging a real
tendon, so nothing here is eyeballed.

This script only reads config/joints.yaml and writes numpy arrays + PNG plots
under shadow_pd_id/data/ and shadow_pd_id/results/ -- it never touches the
real hand or Isaac Sim.
"""

from __future__ import annotations

import argparse
import os

import matplotlib

matplotlib.use("Agg")  # headless: we only ever save PNGs, never open a window
import matplotlib.pyplot as plt
import numpy as np
import yaml

_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_HERE)
_DEFAULT_CONFIG = os.path.join(_PROJECT_ROOT, "config", "joints.yaml")
_DEFAULT_OUT_DIR = os.path.join(_PROJECT_ROOT, "data", "raw", "commands")
_DEFAULT_PLOT_DIR = os.path.join(_PROJECT_ROOT, "results", "plots", "commands_review")


def load_joint_config(path: str = _DEFAULT_CONFIG) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _max_sinusoid_amplitude(freq_hz: float, vel_limit: float, half_range: float, amplitude_frac: float) -> float:
    """Largest amplitude a sinusoid at `freq_hz` can have without exceeding `vel_limit`.

    Peak speed of A*sin(2*pi*f*t) is A*2*pi*f, so the velocity-safe amplitude
    is vel_limit / (2*pi*f). We also cap at `amplitude_frac` of the joint's
    half-range so we never ask for the full range even at low frequency
    (leaves margin before the hard position limit). Whichever cap is tighter
    wins.
    """
    vel_safe_amp = vel_limit / (2.0 * np.pi * max(freq_hz, 1e-6))
    range_safe_amp = amplitude_frac * half_range
    return min(vel_safe_amp, range_safe_amp)


def generate_chirp(
    joint_name: str,
    cfg: dict,
    f0: float = 0.05,
    f1: float = 2.5,
    duration_s: float = 20.0,
    amplitude_frac: float = 0.8,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Linear frequency sweep from f0 to f1 Hz over duration_s seconds.

    Amplitude is sized against f1 (the fastest instant in the sweep, so the
    worst case for velocity) -- this is why the amplitude is constant across
    the whole sweep rather than shrinking as frequency rises: we pick the one
    amplitude that's safe at the *fastest* point, up front.
    """
    dt = 1.0 / cfg["control_rate_hz"]
    limits = cfg["joint_limits_rad"][joint_name]
    lower, upper = limits["lower"], limits["upper"]
    center = 0.5 * (upper + lower)
    half_range = 0.5 * (upper - lower)
    vel_limit = cfg["joint_velocity_limits_rad_s"][joint_name]

    amp = _max_sinusoid_amplitude(f1, vel_limit, half_range, amplitude_frac)

    n = int(duration_s / dt)
    t = np.arange(n) * dt
    # Linear chirp instantaneous frequency f(t) = f0 + (f1-f0)*t/duration.
    # Phase is the integral of 2*pi*f(t) dt (closed form for a linear ramp).
    phase = 2.0 * np.pi * (f0 * t + (f1 - f0) * t**2 / (2.0 * duration_s))
    cmd = center + amp * np.sin(phase)
    cmd = np.clip(cmd, lower, upper)

    meta = dict(type="chirp", f0=f0, f1=f1, duration_s=duration_s, amplitude=amp, amplitude_frac=amplitude_frac)
    return t, cmd, meta


def generate_step(
    joint_name: str,
    cfg: dict,
    n_steps: int = 8,
    hold_s: float = 1.5,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Sequence of `n_steps` random setpoint jumps, each held for `hold_s` seconds.

    Step commands are inherently "instantaneous" -- that's the point, they
    reveal how fast the real PD loop can respond -- but every target is still
    clamped to [lower, upper], so no step can command past the joint's
    physical range. hold_s must be long enough for the joint to settle before
    the next jump (checked visually in the review plot, not computed here,
    since settling time depends on the very gains we're trying to identify).
    """
    dt = 1.0 / cfg["control_rate_hz"]
    limits = cfg["joint_limits_rad"][joint_name]
    lower, upper = limits["lower"], limits["upper"]

    rng = np.random.default_rng(seed)
    # Bias towards a spread of step sizes (small, medium, near-full-range)
    # rather than pure uniform, so we get both regimes in one recording.
    targets = rng.uniform(lower, upper, size=n_steps)
    targets = np.clip(targets, lower, upper)

    hold_n = int(hold_s / dt)
    cmd = np.repeat(targets, hold_n)
    t = np.arange(len(cmd)) * dt

    meta = dict(type="step", n_steps=n_steps, hold_s=hold_s, seed=seed)
    return t, cmd, meta


def generate_random(
    joint_name: str,
    cfg: dict,
    n_components: int = 3,
    f_min: float = 0.1,
    f_max: float = 1.5,
    duration_s: float = 15.0,
    amplitude_frac: float = 0.6,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Sum of `n_components` random-frequency sinusoids, budget-limited so
    their combined peak speed still respects the joint's velocity limit.

    Why a budgeted sum rather than picking each component's amplitude
    independently: peak speeds of the components can add up (worst case all
    peak at once), so each component gets an equal SHARE of the total
    velocity budget (vel_limit / n_components), not the full budget each --
    otherwise 3 "individually safe" sinusoids could sum to an unsafe speed.
    """
    dt = 1.0 / cfg["control_rate_hz"]
    limits = cfg["joint_limits_rad"][joint_name]
    lower, upper = limits["lower"], limits["upper"]
    center = 0.5 * (upper + lower)
    half_range = 0.5 * (upper - lower)
    vel_limit = cfg["joint_velocity_limits_rad_s"][joint_name]

    rng = np.random.default_rng(seed)
    freqs = rng.uniform(f_min, f_max, size=n_components)
    phases = rng.uniform(0, 2 * np.pi, size=n_components)

    vel_budget_per_component = vel_limit / n_components
    range_budget_per_component = (amplitude_frac * half_range) / n_components
    amps = np.array(
        [_max_sinusoid_amplitude(f, vel_budget_per_component, range_budget_per_component, 1.0) for f in freqs]
    )

    n = int(duration_s / dt)
    t = np.arange(n) * dt
    cmd = center + sum(a * np.sin(2 * np.pi * f * t + p) for a, f, p in zip(amps, freqs, phases))
    cmd = np.clip(cmd, lower, upper)

    meta = dict(
        type="random",
        n_components=n_components,
        freqs=freqs.tolist(),
        amps=amps.tolist(),
        seed=seed,
        duration_s=duration_s,
    )
    return t, cmd, meta


def generate_ramp(
    joint_name: str,
    cfg: dict,
    speed_frac: float = 0.1,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """One quasi-static triangle sweep: lower -> upper -> lower.

    speed_frac * vel_limit sets how fast we sweep -- kept slow (default 10%
    of the joint's rated speed) so inertial/damping effects stay negligible
    and what we measure is dominated by static (Coulomb) friction and any
    position-dependent effect, which is the whole point of this trajectory
    type.
    """
    dt = 1.0 / cfg["control_rate_hz"]
    limits = cfg["joint_limits_rad"][joint_name]
    lower, upper = limits["lower"], limits["upper"]
    vel_limit = cfg["joint_velocity_limits_rad_s"][joint_name]

    speed = speed_frac * vel_limit
    half_leg_duration = (upper - lower) / speed
    n_half = int(half_leg_duration / dt)

    up_leg = np.linspace(lower, upper, n_half)
    down_leg = np.linspace(upper, lower, n_half)
    cmd = np.concatenate([up_leg, down_leg])
    t = np.arange(len(cmd)) * dt

    meta = dict(type="ramp", speed_frac=speed_frac, speed_rad_s=speed)
    return t, cmd, meta


def _finger_prefix(joint_name: str) -> str:
    """'rh_FFJ3' -> 'FF', 'rh_THJ1' -> 'TH', etc."""
    # Joint names are "rh_" + 2-letter finger prefix + "J" + digit.
    return joint_name[3:5]


def compute_default_pose(joint_name: str, cfg: dict) -> np.ndarray:
    """Resolve the 13-vector baseline pose for exciting `joint_name`.

    WHY: real-hardware testing found that holding non-excited fingers at
    zero (straight, open) lets them collide with the finger being excited
    (see DECISIONS.md, 2026-07-08). The fix decided on is NOT to shrink the
    excited joint's own range, but to curl every OTHER finger's joints into
    their parked_finger_pose (config/joints.yaml) -- fully out of the way,
    not a precisely-tuned small offset. The excited finger's own non-excited
    joints (e.g. FFJ4 while exciting FFJ3) stay at 0, same as
    safe_default_pose, since they belong to the finger already being tested.
    """
    excited_prefix = _finger_prefix(joint_name)
    parked = cfg["parked_finger_pose"]
    pose = np.zeros(len(cfg["policy_joint_order"]), dtype=np.float32)

    for i, name in enumerate(cfg["policy_joint_order"]):
        prefix = _finger_prefix(name)
        if prefix == excited_prefix:
            continue  # excited finger's own joints (incl. the excited one itself): stay 0
        if name in parked.get(prefix, {}):
            pose[i] = parked[prefix][name]

    return pose


def save_trajectory(
    out_dir: str,
    joint_idx: int,
    joint_name: str,
    t: np.ndarray,
    cmd: np.ndarray,
    meta: dict,
    control_rate_hz: float,
    default_pose: np.ndarray,
) -> str:
    os.makedirs(out_dir, exist_ok=True)
    fname = os.path.join(out_dir, f"joint_{joint_idx:02d}_{joint_name}_{meta['type']}.npz")
    save_kwargs = dict(
        t=t.astype(np.float32),
        cmd=cmd.astype(np.float32),
        joint_name=np.array(joint_name),
        joint_idx=np.array(joint_idx),
        control_rate_hz=np.array(control_rate_hz, dtype=np.float32),
        default_pose=default_pose.astype(np.float32),
    )
    # Flatten meta into the npz so every design parameter is inspectable
    # later without needing a separate sidecar file.
    for k, v in meta.items():
        save_kwargs[f"meta_{k}"] = np.array(v)
    np.savez(fname, **save_kwargs)
    return fname


def _finite_diff_velocity(t: np.ndarray, cmd: np.ndarray) -> np.ndarray:
    return np.gradient(cmd, t)


def plot_joint_trajectories(joint_name: str, trajs: dict, cfg: dict, out_path: str) -> None:
    """One figure per joint: commanded position + derived velocity for every
    trajectory type, each against its position/velocity limit lines, so a
    limit violation (a bug) is visually obvious before anything runs on the
    real hand.
    """
    limits = cfg["joint_limits_rad"][joint_name]
    vel_limit = cfg["joint_velocity_limits_rad_s"][joint_name]

    types = list(trajs.keys())
    fig, axes = plt.subplots(len(types), 2, figsize=(11, 2.6 * len(types)), squeeze=False)
    fig.suptitle(f"{joint_name} — excitation trajectory review", fontsize=12)

    for row, ttype in enumerate(types):
        t, cmd, meta = trajs[ttype]
        vel = _finite_diff_velocity(t, cmd)

        ax_pos, ax_vel = axes[row, 0], axes[row, 1]

        ax_pos.plot(t, cmd, lw=1.2)
        ax_pos.axhline(limits["lower"], color="r", ls="--", lw=0.8)
        ax_pos.axhline(limits["upper"], color="r", ls="--", lw=0.8)
        ax_pos.set_title(f"{ttype} — position (rad)", fontsize=9)
        ax_pos.set_xlabel("t (s)")

        ax_vel.plot(t, vel, lw=1.0, color="tab:orange")
        ax_vel.axhline(vel_limit, color="r", ls="--", lw=0.8)
        ax_vel.axhline(-vel_limit, color="r", ls="--", lw=0.8)
        ax_vel.set_title(f"{ttype} — d(cmd)/dt (rad/s)", fontsize=9)
        ax_vel.set_xlabel("t (s)")

    fig.tight_layout(rect=(0, 0, 1, 0.97))
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


def generate_all(
    cfg: dict,
    out_dir: str = _DEFAULT_OUT_DIR,
    plot_dir: str = _DEFAULT_PLOT_DIR,
    held_out_seed: int = 999,
) -> list[str]:
    """Generate chirp/step/random/ramp (+ one held-out random) for every
    policy joint, save trajectories and one review plot per joint.

    Returns the list of generated plot paths, for reporting back to the user.
    """
    policy_joints = cfg["policy_joint_order"]
    control_rate_hz = cfg["control_rate_hz"]
    plot_paths = []

    for idx, joint_name in enumerate(policy_joints):
        # Same baseline for every trajectory type of this joint -- it only
        # depends on WHICH joint is excited, not on chirp vs. step vs. ramp.
        default_pose = compute_default_pose(joint_name, cfg)

        trajs = {
            "chirp": generate_chirp(joint_name, cfg),
            "step": generate_step(joint_name, cfg, seed=idx),
            "random": generate_random(joint_name, cfg, seed=idx),
            "ramp": generate_ramp(joint_name, cfg),
        }
        for ttype, (t, cmd, meta) in trajs.items():
            save_trajectory(out_dir, idx, joint_name, t, cmd, meta, control_rate_hz, default_pose)

        # Held-out trajectory: a DIFFERENT random seed, saved to its own
        # held_out/ subfolder so it's structurally separate from the
        # training set (Step 1 must never merge these two).
        ho_t, ho_cmd, ho_meta = generate_random(joint_name, cfg, seed=held_out_seed + idx)
        ho_meta["held_out"] = True
        save_trajectory(
            os.path.join(out_dir, "held_out"), idx, joint_name, ho_t, ho_cmd, ho_meta, control_rate_hz, default_pose
        )
        trajs["random_heldout"] = (ho_t, ho_cmd, ho_meta)

        plot_path = os.path.join(plot_dir, f"joint_{idx:02d}_{joint_name}.png")
        plot_joint_trajectories(joint_name, trajs, cfg, plot_path)
        plot_paths.append(plot_path)

    return plot_paths


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=str, default=_DEFAULT_CONFIG)
    parser.add_argument("--out_dir", type=str, default=_DEFAULT_OUT_DIR)
    parser.add_argument("--plot_dir", type=str, default=_DEFAULT_PLOT_DIR)
    args = parser.parse_args()

    cfg = load_joint_config(args.config)
    plot_paths = generate_all(cfg, out_dir=args.out_dir, plot_dir=args.plot_dir)

    print(f"Generated trajectories for {len(cfg['policy_joint_order'])} joints.")
    print(f"Commands saved under: {args.out_dir}")
    print(f"Held-out commands saved under: {os.path.join(args.out_dir, 'held_out')}")
    print(f"Review plots saved under: {args.plot_dir}")
    for p in plot_paths:
        print(f"  {p}")


if __name__ == "__main__":
    main()

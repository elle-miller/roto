#!/usr/bin/env python3
"""Score how well a simulated trajectory matches a real one.

WHY THIS FILE EXISTS: Step 4's optimizer needs a single number to minimize.
This turns (simulated position, real position) into that number, plus a
breakdown so we can tell WHICH part of the fit is bad, not just that it is.

Two things baked in here are specific to how this project's data actually
looks (not generic system-ID boilerplate -- see config/optim.yaml for the
full reasoning):

  1. Every recorded trajectory in this project has a real startup transient
     (the hand settles to zero, then the recording starts mid-jump to the
     excitation's first commanded value) -- so the first `warmup_samples`
     are down-weighted by default, not just "optionally" as the general plan
     put it.
  2. A candidate gain set can make the sim go unstable (e.g. Kp too high for
     the solver -> NaN/Inf position). That must turn into a large FINITE
     penalty, not a NaN loss -- NaN breaks comparison/ranking in most
     optimizers (including CMA-ES), silently corrupting the search rather
     than just making that one candidate look bad.
"""

from __future__ import annotations

import os

import numpy as np
import yaml

_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_HERE)
_DEFAULT_OPTIM_CONFIG = os.path.join(_PROJECT_ROOT, "config", "optim.yaml")


def load_loss_config(path: str = _DEFAULT_OPTIM_CONFIG) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)["loss"]


def compute_loss(
    sim_q: np.ndarray,
    real_q: np.ndarray,
    control_rate_hz: float,
    velocity_weight: float = 0.1,
    warmup_samples: int = 30,
    warmup_weight: float = 0.1,
    unstable_penalty: float = 1.0e6,
) -> tuple[float, dict]:
    """Compare a simulated position trace against a real one, sample-for-sample.

    Both arrays must be the same length and represent the same joint driven
    by the same command trajectory at the same rate -- sim_rollout.py's
    `.rollout()` already returns its output at `control_rate_hz` for exactly
    this reason (so no resampling is needed here).

    Returns (total_loss, sub_losses) where sub_losses has enough detail to
    tell position error apart from velocity error, and to see whether this
    candidate was numerically unstable.
    """
    sim_q = np.asarray(sim_q, dtype=np.float64)
    real_q = np.asarray(real_q, dtype=np.float64)
    if sim_q.shape != real_q.shape:
        raise ValueError(f"shape mismatch: sim_q {sim_q.shape} vs real_q {real_q.shape}")

    if not (np.all(np.isfinite(sim_q)) and np.all(np.isfinite(real_q))):
        return float(unstable_penalty), dict(
            pos_mse=float("nan"), vel_mse=float("nan"), total=float(unstable_penalty), unstable=True
        )

    n = len(sim_q)
    weights = np.ones(n)
    weights[: min(warmup_samples, n)] = warmup_weight

    pos_err = sim_q - real_q
    pos_mse = float(np.average(pos_err**2, weights=weights))

    dt = 1.0 / control_rate_hz
    sim_v = np.gradient(sim_q, dt)
    real_v = np.gradient(real_q, dt)
    vel_err = sim_v - real_v
    vel_mse = float(np.average(vel_err**2, weights=weights))

    total = pos_mse + velocity_weight * vel_mse

    return total, dict(pos_mse=pos_mse, vel_mse=vel_mse, total=total, unstable=False)


if __name__ == "__main__":
    # Minimal self-test with synthetic data (no sim/hardware needed): a
    # perfect match must score ~0, a perturbed signal must score higher, and
    # a NaN-contaminated signal must return the finite unstable_penalty, not
    # NaN itself.
    cfg = load_loss_config()
    t = np.linspace(0, 5, 300)
    real = np.sin(2 * np.pi * 0.5 * t)

    identical_loss, subs = compute_loss(real.copy(), real, control_rate_hz=60, **cfg)
    print(f"identical signal loss = {identical_loss:.6f} (should be ~0)  subs={subs}")
    assert identical_loss < 1e-8

    perturbed = real + 0.05 * np.sin(2 * np.pi * 3.0 * t)
    perturbed_loss, subs = compute_loss(perturbed, real, control_rate_hz=60, **cfg)
    print(f"perturbed signal loss = {perturbed_loss:.6f} (should be > 0)  subs={subs}")
    assert perturbed_loss > identical_loss

    unstable = real.copy()
    unstable[100] = np.nan
    unstable_loss, subs = compute_loss(unstable, real, control_rate_hz=60, **cfg)
    print(f"NaN-contaminated loss  = {unstable_loss:.1f} (should equal unstable_penalty)  subs={subs}")
    assert unstable_loss == cfg["unstable_penalty"] and subs["unstable"]

    print("All loss.py self-tests passed.")

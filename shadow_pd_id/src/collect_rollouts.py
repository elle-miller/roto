#!/usr/bin/env python3
"""Batch-simulate candidate PD gains for one or all joints, saving every result to disk.

WHY THIS FILE EXISTS, AND WHY IT LOOKS LIKE THIS: the original plan (see the
project root plan doc) had the optimizer call the simulator live, iteration by
iteration (propose gains -> simulate -> get loss -> propose next). We hit a
reproducible, unexplained multi-minute stall inside Isaac Sim on this specific
machine/asset (see DECISIONS.md) that makes a tight live loop impractical --
if any single call can silently hang for minutes, an adaptive optimizer
sitting on top of it is very hard to reason about or debug.

So this project splits that loop into two independent phases:
  1. THIS FILE: sample a batch of candidate {Kp, Kd, Fc} up front (Latin
     Hypercube, not adaptive), simulate each one against REAL command
     trajectories, and write each result to disk the moment it's computed.
  2. src/optimize.py: reads whatever has been collected (possibly from
     multiple interrupted runs of this file) and picks/refines the best.

FITTING AGAINST ALL EXCITATION TYPES, NOT JUST ONE: a single trajectory type
under-constrains the fit -- a chirp alone is great for Kd (it explores many
speeds) but spends little time near zero velocity, so it barely constrains
Fc (Coulomb friction); a ramp is the opposite. So for each joint, EVERY
available training trajectory type (chirp/step/ramp/random -- whichever
files exist; the held-out random is never used here) is rolled out under the
same candidate gains, and the combined loss (mean across types) is what's
actually ranked. Per-type sub-losses are still saved so a bad fit can be
traced back to which motion type it fails on.

ALL 13 JOINTS IN ONE ISAAC SIM BOOT: loading the asset costs about a minute
and a fresh SimRolloutEngine per invocation would pay that cost 13 times.
Instead this script defaults to looping over every policy joint within a
SINGLE persistent engine (see sim_rollout.py's own docstring for why a
persistent instance matters). Pass --joint_idx to do just one joint instead
(useful for re-running a single joint after tuning search bounds).

Resilience is the main design constraint here, not elegance:
  - Every candidate's result is saved IMMEDIATELY after that candidate's
    rollout(s) finish -- not batched and saved at the end -- so a stall or
    kill partway through never loses completed work.
  - Already-collected candidates (by index) are skipped on a re-run, and a
    joint whose output dir already has n_candidates results is skipped
    entirely -- so this script can be killed and restarted as many times as
    the sim's stalls require, across a single joint or the whole 13-joint
    run, without redoing finished work.
  - Progress is printed (unbuffered) BEFORE each candidate's rollout starts,
    not after, so whoever is watching the log always knows which candidate
    (and which trajectory type within it) is currently in flight if
    something does stall.
"""

from __future__ import annotations

import argparse
import glob
import os
import sys

import numpy as np
import yaml
from isaaclab.app import AppLauncher

_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _HERE)

TRAJ_TYPES = ["chirp", "step", "ramp", "random"]  # held-out "random" is a separate file, never used here

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--hw_dir", type=str, default=os.path.join(_PROJECT_ROOT, "data", "raw", "hw"),
                     help="Directory of collected TRAINING logs (joint_XX_name_TYPE.npz). "
                          "Held-out files (data/raw/hw/held_out/) are never read here.")
parser.add_argument("--joint_idx", type=int, default=None,
                     help="Process only this policy joint index (0-12). Default: all 13.")
parser.add_argument("--out_dir", type=str, default=None,
                     help="Override results/rollouts/<joint_name>/ for a single-joint run. "
                          "Ignored when processing all joints (each joint gets its own dir).")
parser.add_argument("--n_candidates", type=int, default=None, help="Overrides config/optim.yaml's search.n_candidates.")
parser.add_argument("--seed", type=int, default=None, help="Overrides config/optim.yaml's search.sampling_seed.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from scipy.stats import qmc  # noqa: E402

from load_data import load_joint_config  # noqa: E402
from loss import compute_loss, load_loss_config  # noqa: E402
from sim_rollout import SimRolloutEngine  # noqa: E402


def sample_candidates(bounds: dict, n: int, seed: int) -> np.ndarray:
    """Latin Hypercube samples over [kp, kd, fc], scaled to `bounds`.

    LHS (not plain uniform random) guarantees even coverage of each
    parameter's 1D range even at modest N -- important because N is small by
    necessity (each sample costs a full Isaac Sim rollout); plain random
    sampling can leave gaps in a range purely by chance at this budget.
    """
    sampler = qmc.LatinHypercube(d=3, seed=seed)
    unit_samples = sampler.random(n=n)
    lows = np.array([bounds["kp"]["min"], bounds["kd"]["min"], bounds["fc"]["min"]])
    highs = np.array([bounds["kp"]["max"], bounds["kd"]["max"], bounds["fc"]["max"]])
    return qmc.scale(unit_samples, lows, highs)


def find_reference_files(hw_dir: str, joint_idx: int, joint_name: str) -> dict:
    """Every available TRAINING trajectory-type file for one joint. {type: path}."""
    found = {}
    for t in TRAJ_TYPES:
        p = os.path.join(hw_dir, f"joint_{joint_idx:02d}_{joint_name}_{t}.npz")
        if os.path.exists(p):
            found[t] = p
    return found


def load_references(paths_by_type: dict) -> tuple[dict, np.ndarray | None]:
    """Load {type: {cmd, real_q}} plus the shared default_pose (asserted consistent
    across types -- it should only depend on which joint is excited, never on the
    excitation type; a mismatch means something is wrong upstream, not a value to
    silently average over)."""
    refs = {}
    default_pose = None
    for t, p in paths_by_type.items():
        d = np.load(p, allow_pickle=True)
        joint_idx = int(d["joint_idx"])
        pose = np.asarray(d["default_pose"], dtype=np.float32) if "default_pose" in d else None
        if default_pose is None:
            default_pose = pose
        elif pose is not None and default_pose is not None and not np.allclose(pose, default_pose, atol=1e-4):
            raise ValueError(
                f"{p}: default_pose does not match the other trajectory types for this joint -- "
                "this should be impossible since it only depends on which joint is excited, not "
                "excitation type. Don't average over this silently; something upstream is wrong."
            )
        refs[t] = dict(cmd=d["cmd"], real_q=d["actual_pos"][:, joint_idx])
    return refs, default_pose


def collect_for_joint(engine, joint_cfg, loss_cfg, search_cfg, joint_idx, hw_dir, out_dir, n_candidates, seed):
    joint_name = joint_cfg["policy_joint_order"][joint_idx]
    paths_by_type = find_reference_files(hw_dir, joint_idx, joint_name)
    missing = [t for t in TRAJ_TYPES if t not in paths_by_type]
    if not paths_by_type:
        print(f"[collect_rollouts] joint {joint_idx} ({joint_name}): NO training files found in {hw_dir}, skipping.",
              flush=True)
        return
    if missing:
        print(f"[collect_rollouts] joint {joint_idx} ({joint_name}): missing types {missing} -- "
              f"fitting against {sorted(paths_by_type)} only.", flush=True)

    refs, default_pose = load_references(paths_by_type)

    os.makedirs(out_dir, exist_ok=True)
    candidates = sample_candidates(search_cfg["bounds"], n_candidates, seed)

    n_existing = len(glob.glob(os.path.join(out_dir, "candidate_*.npz")))
    if n_existing >= n_candidates:
        print(f"[collect_rollouts] joint {joint_idx} ({joint_name}): {n_existing} candidates already collected "
              f">= requested {n_candidates}, skipping joint entirely.", flush=True)
        return

    print(f"[collect_rollouts] joint {joint_idx} ({joint_name}), types={sorted(refs)}, "
          f"{n_candidates} candidates -> {out_dir}", flush=True)

    n_done = 0
    n_skipped = 0
    for i, (kp, kd, fc) in enumerate(candidates):
        out_path = os.path.join(out_dir, f"candidate_{i:03d}.npz")
        if os.path.exists(out_path):
            n_skipped += 1
            continue

        print(f"[collect_rollouts] {joint_name} candidate {i}/{n_candidates}: "
              f"Kp={kp:.4f} Kd={kd:.4f} Fc={fc:.4f} ...", flush=True)

        engine.set_gains(joint_name, kp=kp, kd=kd, fc=fc)

        per_type_loss = {}
        per_type_subs = {}
        any_unstable = False
        for t, ref in refs.items():
            engine.reset(default_pose)
            sim_q = engine.rollout(joint_idx, ref["cmd"], default_pose)
            loss_t, subs = compute_loss(sim_q, ref["real_q"], joint_cfg["control_rate_hz"], **loss_cfg)
            per_type_loss[t] = loss_t
            per_type_subs[t] = subs
            any_unstable = any_unstable or subs["unstable"]
            print(f"[collect_rollouts]     {t:6s} -> loss={loss_t:.6f} "
                  f"(pos={subs['pos_mse']:.6f} vel={subs['vel_mse']:.6f})", flush=True)

        combined_loss = float(np.mean(list(per_type_loss.values())))

        save_kwargs = dict(
            kp=np.array(kp), kd=np.array(kd), fc=np.array(fc),
            loss_total=np.array(combined_loss),
            unstable=np.array(any_unstable),
            joint_idx=np.array(joint_idx),
            joint_name=np.array(joint_name),
            types_used=np.array(sorted(refs)),
        )
        for t, loss_t in per_type_loss.items():
            save_kwargs[f"loss_{t}"] = np.array(loss_t)
        np.savez(out_path, **save_kwargs)

        print(f"[collect_rollouts]   -> combined loss={combined_loss:.6f}  saved {out_path}", flush=True)
        n_done += 1

    print(f"[collect_rollouts] joint {joint_idx} ({joint_name}) done. "
          f"{n_done} new, {n_skipped} already-collected (skipped).", flush=True)


def main():
    joint_cfg = load_joint_config()
    loss_cfg = load_loss_config()

    with open(os.path.join(_PROJECT_ROOT, "config", "optim.yaml"), encoding="utf-8") as f:
        search_cfg = yaml.safe_load(f)["search"]

    n_candidates = args_cli.n_candidates or search_cfg["n_candidates"]
    seed = args_cli.seed if args_cli.seed is not None else search_cfg["sampling_seed"]

    joint_indices = [args_cli.joint_idx] if args_cli.joint_idx is not None else list(range(13))

    engine = SimRolloutEngine(joint_cfg, simulation_app=simulation_app)

    for joint_idx in joint_indices:
        joint_name = joint_cfg["policy_joint_order"][joint_idx]
        out_dir = args_cli.out_dir if (args_cli.out_dir and args_cli.joint_idx is not None) else \
            os.path.join(_PROJECT_ROOT, "results", "rollouts", joint_name)
        collect_for_joint(engine, joint_cfg, loss_cfg, search_cfg, joint_idx,
                           args_cli.hw_dir, out_dir, n_candidates, seed)

    engine.close()
    print("[collect_rollouts] All requested joints done.", flush=True)


if __name__ == "__main__":
    main()

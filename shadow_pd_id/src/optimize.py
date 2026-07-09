#!/usr/bin/env python3
"""Select the best PD gains from candidates already simulated by collect_rollouts.py.

WHY THIS FILE LOOKS THE WAY IT DOES: this project does NOT run a live,
adaptive optimizer against the simulator (see collect_rollouts.py's docstring
and DECISIONS.md for why -- a reproducible Isaac Sim stall makes a tight
propose/simulate/feedback loop impractical on this machine). Instead,
collect_rollouts.py already simulated a batch of candidates and saved each
one's loss to disk; this file just reads that batch and picks the best.

This means there is no "convergence over iterations" the way a live
optimizer would produce -- there's no sequence, just an unordered batch. The
closest equivalent, and what this file plots instead, is the SORTED loss
across the whole batch: it shows the shape of the loss landscape this
particular sample of gains explored (how many candidates were close to the
best vs. how much of the space was clearly bad), which is the honest
substitute for a convergence curve when the search isn't sequential.

Does not require Isaac Sim -- this only reads the .npz files
collect_rollouts.py already wrote.
"""

from __future__ import annotations

import argparse
import glob
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml

_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_HERE)


def load_candidates(rollout_dir: str) -> dict:
    """Read every candidate_*.npz in rollout_dir into flat arrays.

    `loss` is the COMBINED loss across every excitation type the candidate
    was fit against (see collect_rollouts.py's module docstring for why
    fitting spans chirp/step/ramp/random rather than just one type); the
    per-type breakdown (`loss_by_type`) is kept too so a bad fit can be
    traced back to which motion type it fails on, not just that it's bad.
    """
    paths = sorted(glob.glob(os.path.join(rollout_dir, "candidate_*.npz")))
    if not paths:
        raise FileNotFoundError(
            f"No candidate_*.npz found in {rollout_dir} -- run collect_rollouts.py first."
        )

    kp, kd, fc, loss, unstable = [], [], [], [], []
    loss_by_type: dict[str, list] = {}
    joint_name = None
    types_used = None
    for p in paths:
        d = np.load(p, allow_pickle=True)
        kp.append(float(d["kp"]))
        kd.append(float(d["kd"]))
        fc.append(float(d["fc"]))
        loss.append(float(d["loss_total"]))
        unstable.append(bool(d["unstable"]))
        joint_name = str(d["joint_name"])
        if "types_used" in d:
            types_used = [str(t) for t in d["types_used"]]
            for t in types_used:
                key = f"loss_{t}"
                if key in d:
                    loss_by_type.setdefault(t, []).append(float(d[key]))

    return dict(
        joint_name=joint_name,
        kp=np.array(kp), kd=np.array(kd), fc=np.array(fc),
        loss=np.array(loss), unstable=np.array(unstable),
        loss_by_type={t: np.array(v) for t, v in loss_by_type.items()},
        types_used=types_used,
        n=len(paths),
    )


def select_best(data: dict) -> dict:
    """Best = lowest loss among STABLE candidates only.

    An unstable candidate's loss is the large `unstable_penalty` placeholder
    (see loss.py), not a real fit quality number -- including them in the
    ranking would be comparing real fit errors against an arbitrary penalty
    constant, which isn't a meaningful comparison even though the penalty is
    deliberately large enough that it wouldn't normally be picked anyway.
    """
    stable_mask = ~data["unstable"]
    if not stable_mask.any():
        raise RuntimeError(
            "Every collected candidate was unstable -- the search bounds in "
            "config/optim.yaml are probably too aggressive (Kp too high). "
            "Narrow them and re-run collect_rollouts.py."
        )
    stable_idx = np.where(stable_mask)[0]
    best_local = np.argmin(data["loss"][stable_idx])
    best_idx = stable_idx[best_local]
    loss_by_type_at_best = {t: float(v[best_idx]) for t, v in data["loss_by_type"].items()}
    return dict(
        idx=int(best_idx),
        kp=float(data["kp"][best_idx]),
        kd=float(data["kd"][best_idx]),
        fc=float(data["fc"][best_idx]),
        loss=float(data["loss"][best_idx]),
        loss_by_type=loss_by_type_at_best,
        n_stable=int(stable_mask.sum()),
        n_total=data["n"],
    )


def plot_loss_landscape(data: dict, best: dict, out_path: str) -> None:
    fig, axes = plt.subplots(1, 4, figsize=(16, 3.5))

    stable = ~data["unstable"]
    for ax, key, label in zip(axes[:3], ["kp", "kd", "fc"], ["Kp (N*m/rad)", "Kd (N*m*s/rad)", "Fc (N*m)"]):
        ax.scatter(data[key][stable], data["loss"][stable], s=18, alpha=0.6, label="stable")
        if (~stable).any():
            ax.scatter(data[key][~stable], data["loss"][~stable], s=18, alpha=0.4, color="red", marker="x", label="unstable")
        ax.scatter([data[key][best["idx"]]], [data["loss"][best["idx"]]], s=90, facecolors="none",
                   edgecolors="black", linewidths=1.5, label="best")
        ax.set_xlabel(label)
        ax.set_ylabel("loss")
        ax.set_yscale("log")
        ax.legend(fontsize=7)

    # Sorted-loss "landscape" plot -- the honest substitute for a convergence
    # curve when the search is a batch, not a sequence (see module docstring).
    ax = axes[3]
    sorted_loss = np.sort(data["loss"][stable])
    ax.plot(sorted_loss, marker="o", ms=3)
    ax.set_yscale("log")
    ax.set_xlabel("candidate rank (best to worst)")
    ax.set_ylabel("loss")
    ax.set_title("sorted loss across batch")

    fig.suptitle(f"{data['joint_name']} — {data['n']} candidates ({best['n_stable']} stable)")
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


def save_best_gains(joint_name: str, best: dict, types_used: list, out_path: str) -> None:
    payload = dict(
        joint_name=joint_name,
        kp=best["kp"],
        kd=best["kd"],
        fc=best["fc"],
        fv=0.0,  # intentionally fixed -- see DECISIONS.md
        loss=best["loss"],
        loss_by_type={t: float(v) for t, v in best["loss_by_type"].items()},
        n_candidates_evaluated=best["n_total"],
        n_stable=best["n_stable"],
        types_used=types_used,
        note=(
            "Selected by src/optimize.py from a batch sampled by collect_rollouts.py "
            "(offline collect-then-select design, not a live adaptive optimizer -- see DECISIONS.md)."
        ),
    )
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(payload, f, sort_keys=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rollout_dir", type=str, required=True,
                         help="Directory of candidate_*.npz from collect_rollouts.py, e.g. results/rollouts/rh_FFJ4")
    args = parser.parse_args()

    data = load_candidates(args.rollout_dir)
    best = select_best(data)

    print(f"Loaded {data['n']} candidates for {data['joint_name']} ({best['n_stable']} stable), "
          f"fit against types={data['types_used']}.")
    print(f"Best: Kp={best['kp']:.4f}  Kd={best['kd']:.4f}  Fc={best['fc']:.4f}  "
          f"combined loss={best['loss']:.6f}  per-type={best['loss_by_type']}")

    plot_path = os.path.join(_PROJECT_ROOT, "results", "plots", f"{data['joint_name']}_loss_landscape.png")
    plot_loss_landscape(data, best, plot_path)
    print(f"Saved {plot_path}")

    params_path = os.path.join(_PROJECT_ROOT, "results", "params", f"{data['joint_name']}_gains.yaml")
    save_best_gains(data["joint_name"], best, data["types_used"], params_path)
    print(f"Saved {params_path}")


if __name__ == "__main__":
    main()

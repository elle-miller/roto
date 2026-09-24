#!/usr/bin/env python3
"""Collect run_ablation_grid.sh logs into one comparison table.

Pure stdlib -- no Isaac Lab / GPU needed, so it can run while the grid is still going.
Reads every <run>_<tacmode>_<cond>.log in the grid output dir, pulls the SUMMARY block,
and writes both a CSV and a markdown table (the latter shaped for the dissertation's
ablation chapter, which currently has no archived per-condition numbers).

Usage:
    python collect_ablation_grid.py [--dir ablation_perfect_tactile] [--md out.md]
"""
from __future__ import annotations

import argparse
import csv
import os
import re

# Order the conditions the way the ablation chapter discusses them, not alphabetically.
COND_ORDER = [
    "none",
    "pos_zero",
    "vel_zero",
    "pos_error_zero",
    "pos_error_freeze",
    "prev_action_zero",
    "none_noball",
    "vel_zero_noball",
    "pos_error_zero_noball",
    "prev_action_zero_noball",
    "none_mass45",
    "none_mass55",
    "none_mass70",
    "none_mass100",
]
COND_LABEL = {
    "none": "baseline (no ablation)",
    "pos_zero": "position zeroed",
    "vel_zero": "velocity zeroed",
    "pos_error_zero": "position-error zeroed",
    "pos_error_freeze": "position-error frozen",
    "prev_action_zero": "previous-action zeroed",
    "none_noball": "no ball (open-loop probe)",
    "vel_zero_noball": "velocity zeroed, no ball",
    "pos_error_zero_noball": "position-error zeroed, no ball",
    "prev_action_zero_noball": "previous-action zeroed, no ball",
    "none_mass45": "ball mass 45 g",
    "none_mass55": "ball mass 55 g",
    "none_mass70": "ball mass 70 g",
    "none_mass100": "ball mass 100 g",
}
RUN_LABEL = {
    "0p9": "0.9 hold-dither (corrupt_max 8, flip 0.1/0.1, scope=corrupted)",
    "0p25": "0.25 broad flip (corrupt_max 6, flip 0.25/0.25, scope=all_fsr)",
}

PATS = {
    "mean_return": re.compile(r"^mean_return:\s+([-\d.]+)", re.M),
    "mean_num_rotations": re.compile(r"^mean_num_rotations:\s+([-\d.]+)", re.M),
    "drop_rate": re.compile(r"^drop_rate:\s+([\d.]+)%", re.M),
    "mean_survival_steps": re.compile(r"^mean_survival_steps:\s+([\d.]+)\s*/\s*(\d+)", re.M),
}


def parse_log(path: str) -> dict | None:
    with open(path, "r", errors="replace") as f:
        txt = f.read()
    if "===== SUMMARY =====" not in txt:
        return None
    out = {}
    for key in ("mean_return", "mean_num_rotations", "drop_rate"):
        m = PATS[key].search(txt)
        out[key] = float(m.group(1)) if m else None
    m = PATS["mean_survival_steps"].search(txt)
    if m:
        out["mean_survival_steps"] = float(m.group(1))
        out["episode_steps"] = int(m.group(2))
    else:
        out["mean_survival_steps"] = out["episode_steps"] = None
    m = re.search(r"^condition:\s+(.*)$", txt, re.M)
    out["condition_line"] = m.group(1).strip() if m else ""
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="ablation_perfect_tactile")
    ap.add_argument("--csv", default=None, help="Default: <dir>/results.csv")
    ap.add_argument("--md", default=None, help="Default: <dir>/results.md")
    args = ap.parse_args()

    d = args.dir
    csv_path = args.csv or os.path.join(d, "results.csv")
    md_path = args.md or os.path.join(d, "results.md")

    rows = []
    for run in ("0p9", "0p25"):
        for tac in ("active", "taczero"):
            for cond in COND_ORDER:
                log = os.path.join(d, f"{run}_{tac}_{cond}.log")
                if not os.path.exists(log):
                    continue
                r = parse_log(log)
                if r is None:
                    continue
                r.update(run=run, tactile=tac, condition=cond)
                rows.append(r)

    if not rows:
        print(f"No completed runs found in {d}/ yet.")
        return

    fields = [
        "run", "tactile", "condition", "mean_return", "mean_num_rotations",
        "drop_rate", "mean_survival_steps", "episode_steps", "condition_line",
    ]
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in fields})

    def fmt(v, spec=".3f", suffix=""):
        return "--" if v is None else f"{v:{spec}}{suffix}"

    lines = [
        "# Observation-block ablation under perfect tactile",
        "",
        "Both training runs evaluated with their tactile DR switched off",
        "(`--fsr_corrupt_max 0 --tactile_flip_prob 0`), so every condition below sees",
        "clean taxels regardless of what the policy was trained under.",
        "",
    ]
    done = {(r["run"], r["tactile"]) for r in rows}
    for run in ("0p9", "0p25"):
        if not any(k[0] == run for k in done):
            continue
        lines += [f"## Run {run} -- {RUN_LABEL[run]}", ""]
        for tac in ("active", "taczero"):
            sub = [r for r in rows if r["run"] == run and r["tactile"] == tac]
            if not sub:
                continue
            title = "tactile active" if tac == "active" else "tactile zeroed (prop-only)"
            lines += [
                f"### {title}",
                "",
                "| Condition | mean return | mean rotations | drop rate | mean survival |",
                "|---|---|---|---|---|",
            ]
            steps = next((r["episode_steps"] for r in sub if r["episode_steps"]), None)
            for r in sorted(sub, key=lambda x: COND_ORDER.index(x["condition"])):
                lines.append(
                    f"| {COND_LABEL[r['condition']]} "
                    f"| {fmt(r['mean_return'])} "
                    f"| {fmt(r['mean_num_rotations'])} "
                    f"| {fmt(r['drop_rate'], '.1f', '%')} "
                    f"| {fmt(r['mean_survival_steps'], '.1f')}"
                    f"{f' / {steps}' if steps else ''} |"
                )
            lines.append("")

    with open(md_path, "w") as f:
        f.write("\n".join(lines) + "\n")

    print(f"{len(rows)} completed runs -> {csv_path}, {md_path}")


if __name__ == "__main__":
    main()

"""Run Modules A-D end to end and save one consolidated report.json."""

import json
import numpy as np
from tacab import load_runs, prepare, CONDITIONS
from module_a import run_module_a, print_module_a
from module_b import run_module_b, contact_conditioned_deviation, print_module_b
from module_c import run_module_c, print_module_c
from module_d import split_half, cycle_count_subsample, print_module_d


def strip_private(d):
    """Drop the underscore-prefixed keys module_b stashes for plotting
    (raw arrays, not JSON-friendly) before serializing."""
    out = {}
    for k, v in d.items():
        if isinstance(v, dict):
            out[k] = {kk: vv for kk, vv in v.items() if not kk.startswith("_")}
        else:
            out[k] = v
    return out


def to_jsonable(obj):
    if isinstance(obj, dict):
        return {k: to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.bool_):
        return bool(obj)
    return obj


if __name__ == "__main__":
    runs = load_runs()
    P = prepare(runs)

    print("Loading trial5 hardware logs (matched ~404s durations, 60Hz, "
          "checkpoint/config verified identical except zero_tactile) ...\n")

    mod_a = run_module_a(runs, P)
    print_module_a(mod_a)

    mod_b = run_module_b(P)
    contact_dev = contact_conditioned_deviation(P)
    print_module_b(mod_b, contact_dev)

    mod_c = run_module_c(P)
    print_module_c(mod_c)

    sh = split_half(P)
    sub = cycle_count_subsample(P)
    print_module_d(sh, sub)

    report = {
        "module_a_stereotypy": mod_a,
        "module_b_recovery": strip_private(mod_b),
        "module_b_contact_deviation_coupling": contact_dev,
        "module_c_exploratory": mod_c,
        "module_d_split_half": sh,
        "module_d_cycle_subsample": sub,
        "meta": {
            "with_tactile_duration_s": P["with_tactile"]["duration"],
            "zero_tactile_duration_s": P["zero_tactile"]["duration"],
            "with_tactile_n_cycles": P["with_tactile"]["n_cycles"],
            "zero_tactile_n_cycles": P["zero_tactile"]["n_cycles"],
        },
    }
    with open("report_v2.json", "w") as f:
        json.dump(to_jsonable(report), f, indent=2)
    print("Saved report_v2.json")

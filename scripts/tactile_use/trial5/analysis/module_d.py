"""Module D -- controls, because this is n=1 run per condition.

Split-half within each run gives the within-condition variability floor that
any between-condition claim has to clear. Cycle-count subsampling checks
whether the headline numbers are an artifact of with_tactile having fewer,
longer cycles (296) than zero_tactile (359).
"""

import numpy as np
from tacab import load_runs, prepare, MASTER_IDX, CONDITIONS, cliffs_delta
from module_a import mean_pairwise_corr, period_stats
from module_b import detect_disruptions

rng = np.random.default_rng(42)


def split_half(P):
    """First half vs second half of each run, on the two Module-A headline
    metrics (period CV, pairwise cycle correlation) and the Module-B
    disruption rate."""
    out = {}
    for name in CONDITIONS:
        p = P[name]
        bounds, t, ens = p["bounds"], p["t"], p["ens"]
        n_cyc = len(ens)
        half = n_cyc // 2
        halves = {}
        for label, sl in [("first_half", slice(0, half)), ("second_half", slice(half, n_cyc))]:
            b_sl = bounds[sl.start:sl.stop + 1]
            per = np.diff(t[b_sl])
            pw = mean_pairwise_corr(ens[sl, :, MASTER_IDX])
            s_idx, e_idx = b_sl[0], b_sl[-1]
            # local mean cycle from THIS half's own cycles, not the whole-run
            # ensemble -- the correct "nominal" reference for a within-half test
            local_mean_cycle = ens[sl].mean(axis=0)
            det = detect_disruptions(
                p["t"][s_idx:e_idx], p["phase"][s_idx:e_idx], p["phase_vel"][s_idx:e_idx],
                local_mean_cycle, p["q"][s_idx:e_idx], p["cmd"][s_idx:e_idx], p["tac_real"][s_idx:e_idx])
            dur_min = (t[e_idx] - t[s_idx]) / 60.0
            halves[label] = {
                "period_cv": float(per.std() / per.mean()),
                "pairwise_corr": float(pw),
                "disruption_rate_per_min": det["n_consensus"] / dur_min,
            }
        out[name] = halves
    return out


def cycle_count_subsample(P, n_target=None, n_boot=300):
    """Repeatedly subsample the run with MORE cycles down to the other run's
    count, and check whether period-CV / pairwise-corr are stable -- rather
    than an artifact of one condition having more cycles to average over.
    (with_tactile=296 cycles, zero_tactile=359 here, so this subsamples
    zero_tactile down to 296; written generically in case that ever flips.)
    """
    counts = {name: len(P[name]["ens"]) for name in CONDITIONS}
    larger_name = max(counts, key=counts.get)
    smaller_name = min(counts, key=counts.get)
    a_cycles = P[larger_name]["ens"][:, :, MASTER_IDX]
    a_bounds = P[larger_name]["bounds"]
    a_t = P[larger_name]["t"]
    a_periods = np.diff(a_t[a_bounds])
    n_full = len(a_cycles)
    n_target = n_target or counts[smaller_name]

    boot_cv, boot_corr = [], []
    for _ in range(n_boot):
        idx = rng.choice(n_full, size=n_target, replace=False)
        boot_cv.append(a_periods[idx].std() / a_periods[idx].mean())
        boot_corr.append(mean_pairwise_corr(a_cycles[idx]))
    return {
        "larger_run": larger_name, "smaller_run": smaller_name,
        "n_full": n_full, "n_target": n_target,
        "period_cv_full": float(a_periods.std() / a_periods.mean()),
        "period_cv_subsampled_mean": float(np.mean(boot_cv)), "period_cv_subsampled_ci95": np.percentile(boot_cv, [2.5, 97.5]).tolist(),
        "pairwise_corr_full": float(mean_pairwise_corr(a_cycles)),
        "pairwise_corr_subsampled_mean": float(np.mean(boot_corr)), "pairwise_corr_subsampled_ci95": np.percentile(boot_corr, [2.5, 97.5]).tolist(),
    }


def print_module_d(sh, sub):
    print("=" * 78)
    print("MODULE D: CONTROLS -- n=1 run per condition")
    print("=" * 78)
    print("-- Split-half within each run (the honest variability floor) --")
    for name in CONDITIONS:
        h = sh[name]
        print(f"[{name:12s}]")
        for metric in ["period_cv", "pairwise_corr", "disruption_rate_per_min"]:
            a, b = h["first_half"][metric], h["second_half"][metric]
            print(f"    {metric:26s}  first_half={a:.3f}  second_half={b:.3f}  |within-run diff|={abs(a-b):.3f}")

    print("\nBetween-condition differences, for comparison against the within-run diffs above:")
    print(f"  period_cv:        with_tac={period_via_p(sh, 'with_tactile'):.3f}  zero_tac={period_via_p(sh, 'zero_tactile'):.3f}")
    print()

    print(f"-- Cycle-count subsampling: is the {sub['larger_run']} result an artifact of having")
    print(f"   more cycles ({sub['n_full']}) than {sub['smaller_run']} ({sub['n_target']} -> subsampled to match)? --")
    print(f"  period CV:      full(n={sub['n_full']})={sub['period_cv_full']:.3f}   "
          f"subsampled(n={sub['n_target']}) mean={sub['period_cv_subsampled_mean']:.3f}  "
          f"95% range={sub['period_cv_subsampled_ci95']}")
    print(f"  pairwise corr:  full(n={sub['n_full']})={sub['pairwise_corr_full']:.3f}   "
          f"subsampled(n={sub['n_target']}) mean={sub['pairwise_corr_subsampled_mean']:.3f}  "
          f"95% range={sub['pairwise_corr_subsampled_ci95']}")
    print()


def period_via_p(sh, name):
    return (sh[name]["first_half"]["period_cv"] + sh[name]["second_half"]["period_cv"]) / 2


if __name__ == "__main__":
    runs = load_runs()
    P = prepare(runs)
    sh = split_half(P)
    sub = cycle_count_subsample(P)
    print_module_d(sh, sub)

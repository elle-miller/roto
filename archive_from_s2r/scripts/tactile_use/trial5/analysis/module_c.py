"""Module C -- 'sometimes gets a lot of rotations' (exploratory, no counts).

Without rotation numbers this can only test the STRUCTURE the claim implies:
that the zeroed policy is bimodal (clean cycling sometimes, degenerate
otherwise) rather than uniformly mediocre. Both pieces here are proxies,
never a rotation count, and are labelled as such throughout.
"""

import numpy as np
from tacab import (
    load_runs, prepare, MASTER_IDX, FINGER_TAC_CH, CONDITIONS, FS,
)
from module_a import mean_pairwise_corr

BLOCK_S = 30.0


def windowed_regime_stats(p, block_s=BLOCK_S):
    """Per-block (default 30s) amplitude, period and within-block cycle
    self-similarity, to see whether a run alternates between regimes or
    stays steady."""
    t, bounds, q = p["t"], p["bounds"], p["q"]
    ens = p["ens"]
    cycle_t = t[bounds[:-1]]  # onset time of each cycle, for block assignment
    periods = np.diff(t[bounds])
    n_blocks = int(np.ceil((t[-1] - t[0]) / block_s))
    rows = []
    for b in range(n_blocks):
        lo, hi = t[0] + b * block_s, t[0] + (b + 1) * block_s
        sel = (cycle_t >= lo) & (cycle_t < hi)
        idx = np.where(sel)[0]
        if len(idx) < 3:
            continue
        amp = np.ptp(ens[idx, :, MASTER_IDX], axis=1)
        pw = mean_pairwise_corr(ens[idx, :, MASTER_IDX])
        rows.append({
            "t_start": float(lo - t[0]), "n_cycles": int(len(idx)),
            "mean_period_s": float(periods[idx].mean()),
            "mean_amp_deg": float(np.degrees(amp.mean())),
            "within_block_pairwise_corr": float(pw) if not np.isnan(pw) else None,
        })
    return rows


def travelling_wave(p):
    """Per master-cycle, the ORDER in which FF/MF/RF first make contact.
    A consistent sequential order is a plausible signature of a coordinated,
    ball-advancing cycle -- reported as a proxy, never as a rotation count."""
    bounds, tac_real, t = p["bounds"], p["tac_real"], p["t"]
    finger_any = {f: tac_real[:, ch].sum(axis=1) > 0 for f, ch in FINGER_TAC_CH.items()}
    orders = []
    for i in range(len(bounds) - 1):
        s, e = bounds[i], bounds[i + 1]
        onsets = {}
        for f, sig in finger_any.items():
            seg = sig[s:e]
            on = np.where(seg)[0]
            onsets[f] = t[s + on[0]] - t[s] if len(on) else None
        present = {f: v for f, v in onsets.items() if v is not None}
        if len(present) >= 2:
            order = tuple(sorted(present, key=present.get))
        else:
            order = None
        orders.append(order)
    return orders


def run_module_c(P):
    results = {}
    for name in CONDITIONS:
        p = P[name]
        blocks = windowed_regime_stats(p)
        amps = [b["mean_amp_deg"] for b in blocks]
        pers = [b["mean_period_s"] for b in blocks]
        corrs = [b["within_block_pairwise_corr"] for b in blocks if b["within_block_pairwise_corr"] is not None]

        orders = travelling_wave(p)
        valid = [o for o in orders if o is not None]
        n_valid = len(valid)
        from collections import Counter
        order_counts = Counter(valid)
        top_order, top_n = (order_counts.most_common(1)[0] if order_counts else (None, 0))

        results[name] = {
            "blocks": blocks,
            "amp_cv_across_blocks": float(np.std(amps) / np.mean(amps)) if amps else None,
            "period_cv_across_blocks": float(np.std(pers) / np.mean(pers)) if pers else None,
            "within_block_corr_range": [float(min(corrs)), float(max(corrs))] if corrs else None,
            "n_cycles_with_multi_finger_contact": n_valid,
            "n_cycles_total": len(orders),
            "dominant_order": top_order, "dominant_order_frac": (top_n / n_valid) if n_valid else None,
            "order_distribution": {str(k): v for k, v in order_counts.items()},
        }
    return results


def print_module_c(results):
    print("=" * 78)
    print("MODULE C: exploratory -- structure consistent with 'sometimes many rotations'?")
    print("(no rotation counts exist yet; everything here is a proxy, stated as such)")
    print("=" * 78)

    print("\n-- Regime stability across the run (30s blocks) --")
    for name in results:
        r = results[name]
        print(f"[{name:12s}] amplitude CV across blocks={r['amp_cv_across_blocks']:.3f}  "
              f"period CV across blocks={r['period_cv_across_blocks']:.3f}  "
              f"within-block self-similarity range={r['within_block_corr_range']}")
        for b in r["blocks"]:
            print(f"    t={b['t_start']:6.1f}s  n_cyc={b['n_cycles']:3d}  "
                  f"period={b['mean_period_s']:.3f}s  amp={b['mean_amp_deg']:5.2f}deg  "
                  f"self-sim={b['within_block_pairwise_corr']}")

    print("\n-- Contact travelling-wave (FF/MF/RF onset order per cycle) --")
    print("(proxy for a coordinated, ball-advancing cycle -- NOT a rotation count)")
    for name in results:
        r = results[name]
        print(f"[{name:12s}] cycles with >=2 fingers contacting: {r['n_cycles_with_multi_finger_contact']}"
              f" / {r['n_cycles_total']}")
        print(f"    dominant order={r['dominant_order']}  "
              f"frac={r['dominant_order_frac']:.3f}" if r['dominant_order'] else "    no dominant order")
        print(f"    full distribution: {r['order_distribution']}")
    print()


if __name__ == "__main__":
    runs = load_runs()
    P = prepare(runs)
    results = run_module_c(P)
    print_module_c(results)

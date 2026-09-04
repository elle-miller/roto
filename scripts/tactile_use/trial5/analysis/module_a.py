"""Module A -- stereotypy: is the zeroed policy a rigid limit cycle?

Every sub-metric predicts the same sign under H1 (zeroed = more stereotyped):
higher pairwise-cycle correlation, higher variance-explained-by-mean-cycle,
higher single-clock-reconstruction R^2, tighter Poincare cloud, higher
return-map contraction, higher RQA determinism/laminarity, lower sample
entropy, and a smaller (less positive / more negative) Lyapunov exponent.
"""

import numpy as np
from tacab import (
    load_runs, prepare, POLICY_JOINTS, MASTER_IDX, MASTER_JOINT, CURL_IDX,
    FS, CONDITIONS, cliffs_delta, permutation_test, bh_fdr,
)
from nonlinear import sample_entropy, embed, first_min_ami, rqa_metrics, rosenstein_lyapunov


def normalized_dispersion(ens):
    """Cycle-to-cycle std at each phase, normalized by that joint's own
    run-wide amplitude (p5-p95), averaged over phase. One value per joint."""
    disp = ens.std(axis=0)                       # (n_phase, n_joints)
    amp = np.percentile(ens, 95, axis=(0, 1)) - np.percentile(ens, 5, axis=(0, 1))
    return disp.mean(axis=0) / (amp + 1e-12)      # (n_joints,)


def mean_pairwise_corr(ens_1d):
    """ens_1d: (n_cycles, n_phase) for one joint. Mean off-diagonal correlation."""
    n = len(ens_1d)
    if n < 2:
        return np.nan
    C = np.corrcoef(ens_1d)
    iu = np.triu_indices(n, k=1)
    return float(np.mean(C[iu]))


def variance_explained_by_mean(ens_1d):
    """R^2 of the ensemble mean cycle as a predictor of each individual cycle."""
    mean_cycle = ens_1d.mean(axis=0)
    ss_res = np.sum((ens_1d - mean_cycle[None, :]) ** 2)
    ss_tot = np.sum((ens_1d - ens_1d.mean()) ** 2)
    return float(1.0 - ss_res / ss_tot)


def clock_reconstruction_r2(q_col, t, phase, f0, mean_cycle_1d):
    """Simplest possible open-loop model: a constant-frequency clock (f0,
    starting at the run's own initial phase) replaying the fixed mean-cycle
    waveform. Compared against the REAL signal at REAL timestamps -- so any
    genuine period jitter (not just shape variation) shows up as error too.
    """
    phase0 = phase[0]
    recon_phase = phase0 + 2 * np.pi * f0 * (t - t[0])
    frac = (recon_phase % (2 * np.pi)) / (2 * np.pi)
    grid = np.linspace(0.0, 1.0, len(mean_cycle_1d))
    recon = np.interp(frac, grid, mean_cycle_1d)
    ss_res = np.sum((q_col - recon) ** 2)
    ss_tot = np.sum((q_col - q_col.mean()) ** 2)
    return float(1.0 - ss_res / ss_tot), recon


def local_clock_reconstruction_r2(q_col, t, phase, bounds, n_phase, window_cycles=8):
    """Same idea as clock_reconstruction_r2, but re-fit f0 and the mean
    waveform locally every `window_cycles` cycles rather than once for the
    whole ~400s run. A single global clock necessarily decorrelates over
    hundreds of cycles even for a genuinely stereotyped rhythm, because
    small per-cycle period error compounds into full-cycle phase drift --
    that is a property of the test, not evidence against stereotypy. This
    local version asks the fairer, shorter-horizon question: does a fixed
    clock + fixed waveform fit well over a handful of consecutive cycles?
    Returns the list of per-window R^2 values.
    """
    grid = np.linspace(0.0, 1.0, n_phase)
    r2s = []
    n_windows = (len(bounds) - 1) // window_cycles
    for w in range(n_windows):
        c0, c1 = w * window_cycles, (w + 1) * window_cycles
        s, e = bounds[c0], bounds[c1]
        if e - s < 10:
            continue
        seg_q, seg_t, seg_phase = q_col[s:e], t[s:e], phase[s:e]
        # local mean waveform: resample each of this window's cycles and average
        cyc_list = []
        for i in range(c0, c1):
            cs, ce = bounds[i], bounds[i + 1]
            if ce - cs < 3:
                continue
            src = np.linspace(0.0, 1.0, ce - cs)
            cyc_list.append(np.interp(grid, src, q_col[cs:ce]))
        if len(cyc_list) < 2:
            continue
        local_mean = np.mean(cyc_list, axis=0)
        local_f0 = (window_cycles) / (seg_t[-1] - seg_t[0])  # cycles / duration
        recon_phase = seg_phase[0] + 2 * np.pi * local_f0 * (seg_t - seg_t[0])
        frac = (recon_phase % (2 * np.pi)) / (2 * np.pi)
        recon = np.interp(frac, grid, local_mean)
        ss_res = np.sum((seg_q - recon) ** 2)
        ss_tot = np.sum((seg_q - seg_q.mean()) ** 2)
        if ss_tot > 0:
            r2s.append(1.0 - ss_res / ss_tot)
    return r2s


def poincare_points(q, bounds):
    """Full 13-d joint state sampled once per cycle, at the cycle boundary
    (a fixed phase). Returns (n_cycles, n_joints)."""
    return q[bounds[:-1]]


def return_map_slope(points):
    """Distance-from-mean at cycle n vs n+1; OLS slope = contraction rate.
    slope < 1 => deviations shrink cycle-to-cycle (self-correcting).
    slope >= 1 => deviations persist or grow (marginal / unstable)."""
    mean_pt = points.mean(axis=0)
    dev = np.linalg.norm(points - mean_pt, axis=1)
    a, b = dev[:-1], dev[1:]
    if len(a) < 3 or a.std() == 0:
        return np.nan, np.nan
    slope, intercept = np.polyfit(a, b, 1)
    r = np.corrcoef(a, b)[0, 1]
    return float(slope), float(r)


def period_stats(t, bounds, window_cycles=8):
    """Whole-run and within-window period CV. Unconfounded by amplitude or
    shape -- pure timing regularity, from the master-rhythm cycle boundaries
    (shared across all three curl joints by construction, see tacab.prepare).
    """
    per = np.diff(t[bounds])
    n_win = len(per) // window_cycles
    within_cv = [per[w * window_cycles:(w + 1) * window_cycles].std()
                 / per[w * window_cycles:(w + 1) * window_cycles].mean()
                 for w in range(n_win)]
    return {
        "periods_s": per.tolist(),
        "mean_s": float(per.mean()), "std_s": float(per.std()),
        "whole_run_cv": float(per.std() / per.mean()),
        "within_window_cv_mean": float(np.mean(within_cv)),
        "within_window_cv_median": float(np.median(within_cv)),
        "n_windows": n_win,
    }


def run_module_a(runs, P):
    results = {}
    for name in CONDITIONS:
        p = P[name]
        ens = p["ens"]  # (n_cycles, n_phase, 13)
        q, t, phase, f0, bounds = p["q"], p["t"], p["phase"], p["f0"], p["bounds"]
        pstats = period_stats(t, bounds)

        disp = normalized_dispersion(ens)
        per_joint = {}
        for j, jn in enumerate(POLICY_JOINTS):
            ens_1d = ens[:, :, j]
            r2_mean = variance_explained_by_mean(ens_1d)
            pw_corr = mean_pairwise_corr(ens_1d)
            per_joint[jn] = {
                "normalized_dispersion": float(disp[j]),
                "pairwise_cycle_corr": pw_corr,
                "var_explained_by_mean_cycle": r2_mean,
            }

        # single-clock reconstruction, on MASTER joint + the 3 curl drivers
        recon_r2 = {}
        local_recon_r2 = {}
        for jn, idx in CURL_IDX.items():
            r2, _ = clock_reconstruction_r2(q[:, idx], t, phase, f0, p["mean_cycle"][:, idx])
            recon_r2[jn] = r2
            local_r2s = local_clock_reconstruction_r2(q[:, idx], t, phase, bounds, ens.shape[1])
            local_recon_r2[jn] = {"mean": float(np.mean(local_r2s)), "values": local_r2s}

        # Poincare section + return map, full 13-d state, z-scored per joint
        pts = poincare_points(q, bounds)
        z = (pts - pts.mean(axis=0)) / (pts.std(axis=0) + 1e-12)
        poincare_spread = float(np.mean(np.linalg.norm(z - z.mean(axis=0), axis=1)))
        rm_slope, rm_corr = return_map_slope(pts)

        # Nonlinear measures on MASTER joint raw (unresampled) trajectory
        x = q[:, MASTER_IDX]
        x_ds = x[::2]  # light downsample (30 Hz) to keep RQA/Lyapunov O(n^2) tractable
        fs_ds = FS / 2
        tau = first_min_ami(x_ds, max_lag=40)
        m_embed = 4
        emb = embed(x_ds, m_embed, tau)
        rqa = rqa_metrics(emb, eps_pct=10)
        samp_en = sample_entropy(x_ds, m=2, r=0.2 * x_ds.std())
        lyap, lyap_curve = rosenstein_lyapunov(x_ds, m_embed, tau, fs_ds, max_t=30)

        results[name] = {
            "period_stats": pstats,
            "per_joint": per_joint,
            "clock_reconstruction_r2": recon_r2,
            "local_clock_reconstruction_r2": local_recon_r2,
            "poincare_spread_zscore": poincare_spread,
            "return_map_slope": rm_slope,
            "return_map_corr": rm_corr,
            "rqa": rqa,
            "rqa_params": {"tau": int(tau), "m": m_embed, "fs_used": fs_ds},
            "sample_entropy": None if np.isnan(samp_en) else float(samp_en),
            "lyapunov_per_s": None if np.isnan(lyap) else float(lyap),
            "n_cycles": p["n_cycles"],
            "f0_hz": float(f0),
        }
    return results


def print_module_a(results):
    print("=" * 78)
    print("MODULE A: STEREOTYPY -- is the zeroed run a rigid limit cycle?")
    print("=" * 78)
    print(f"n_cycles: with_tactile={results['with_tactile']['n_cycles']}  "
          f"zero_tactile={results['zero_tactile']['n_cycles']}\n")

    print("-- Cycle PERIOD regularity (master rhythm, unconfounded by amplitude/shape) --")
    _, p_perm = permutation_test(results["with_tactile"]["period_stats"]["periods_s"],
                                  results["zero_tactile"]["period_stats"]["periods_s"])
    for name in CONDITIONS:
        s = results[name]["period_stats"]
        print(f"  [{name:12s}] mean period={s['mean_s']:.3f}s  whole-run CV={s['whole_run_cv']:.3f}  "
              f"within-8-cycle-window CV: mean={s['within_window_cv_mean']:.3f} median={s['within_window_cv_median']:.3f}")
    print(f"  (permutation p on mean period = {p_perm:.4f}; CV itself is a dispersion statistic, reported directly)")

    print("\n-- Single-clock reconstruction R^2, WHOLE RUN (one f0 + one waveform for ~400s) --")
    print("(negative = a flat line beats this model -- expected once cycle-to-cycle period")
    print(" jitter compounds into full-cycle phase drift over hundreds of cycles; this specific")
    print(" test is dominated by that compounding, not a clean stereotypy readout on its own)")
    for jn in CURL_IDX:
        a = results["with_tactile"]["clock_reconstruction_r2"][jn]
        b = results["zero_tactile"]["clock_reconstruction_r2"][jn]
        print(f"  {jn:8s}  with_tac={a:.3f}   zero_tac={b:.3f}   (zero - with = {b-a:+.3f})")

    print("\n-- Single-clock reconstruction R^2, LOCAL 8-cycle windows (fair short-horizon test) --")
    print("(re-fits f0 + mean waveform every 8 cycles; higher = more metronome-like over that horizon)")
    for jn in CURL_IDX:
        a = results["with_tactile"]["local_clock_reconstruction_r2"][jn]
        b = results["zero_tactile"]["local_clock_reconstruction_r2"][jn]
        _, p_perm = permutation_test(a["values"], b["values"])
        d = cliffs_delta(a["values"], b["values"])
        print(f"  {jn:8s}  with_tac={a['mean']:.3f} (n={len(a['values'])} windows)   "
              f"zero_tac={b['mean']:.3f} (n={len(b['values'])})   p_perm={p_perm:.4f}  Cliffs d={d:+.3f}")

    print("\n-- Poincare section (state once per cycle) + return map --")
    for name in CONDITIONS:
        r = results[name]
        print(f"  [{name:12s}] Poincare spread (z-score units) = {r['poincare_spread_zscore']:.3f}   "
              f"return-map slope = {r['return_map_slope']:.3f} (corr={r['return_map_corr']:.3f})")

    print("\n-- Nonlinear-dynamics estimates on MASTER joint (rh_MFJ2), 30Hz downsampled --")
    for name in CONDITIONS:
        r = results[name]
        rq = r["rqa"]
        print(f"  [{name:12s}] tau={r['rqa_params']['tau']} m={r['rqa_params']['m']}  "
              f"RQA: RR={rq['recurrence_rate']:.3f} DET={rq['determinism']:.3f} "
              f"LAM={rq['laminarity']:.3f} meanDiagLen={rq['mean_diag_len']:.2f}  "
              f"SampEn={r['sample_entropy']:.3f}  Lyapunov={r['lyapunov_per_s']:.3f}/s")

    print("\n-- Per-joint pairwise-cycle correlation & variance explained by mean cycle --")
    print(f"{'joint':>9s} | {'pw_corr with_tac':>17s} {'zero_tac':>9s} | "
          f"{'R2_mean with_tac':>17s} {'zero_tac':>9s} | {'norm_disp with_tac':>19s} {'zero_tac':>9s}")
    for jn in POLICY_JOINTS:
        a = results["with_tactile"]["per_joint"][jn]
        b = results["zero_tactile"]["per_joint"][jn]
        print(f"{jn:>9s} | {a['pairwise_cycle_corr']:17.3f} {b['pairwise_cycle_corr']:9.3f} | "
              f"{a['var_explained_by_mean_cycle']:17.3f} {b['var_explained_by_mean_cycle']:9.3f} | "
              f"{a['normalized_dispersion']:19.3f} {b['normalized_dispersion']:9.3f}")
    print()


if __name__ == "__main__":
    runs = load_runs()
    P = prepare(runs)
    results = run_module_a(runs, P)
    print_module_a(results)

"""Module B -- recovery: does the policy break its own rhythm to correct?

No ball state and no drop labels exist in this data, so "disruption" is
detected from the signals themselves, by four independent routes. Consensus
events (>=2 routes agreeing within +-0.5s) are then used for a peri-event
average of how far the trajectory departs from its own nominal cycle, and how
long it takes to return.

Prediction under H2 (tactile = recoverable, zeroed = runs straight through):
tactile shows LARGER peri-event departures that RESOLVE; zeroed shows smaller
departures, or departures that do not resolve back toward nominal. This is
stated as a two-sided prediction on purpose (see plan) -- tactile might show
*more* disruptions precisely because it attempts harder configurations.
"""

import numpy as np
from tacab import (
    load_runs, prepare, POLICY_JOINTS, MASTER_IDX, CURL_IDX, Q13_TO_PUB,
    LIVE_TAC_CH, FS, CONDITIONS, cliffs_delta, permutation_test,
)

MIN_EVENT_SAMPLES = 3          # >=50ms sustained, not a single-sample blip
CONSENSUS_WINDOW_S = 0.5
PERI_PRE_S, PERI_POST_S = 3.0, 5.0


def robust_z(x):
    med = np.median(x)
    mad = np.median(np.abs(x - med)) * 1.4826 + 1e-9
    return (x - med) / mad


def tracking_error_13(q, cmd):
    """cmd[:,pub_i] - q[:,q13_i] for every mapped joint, POLICY_JOINTS order
    restricted to the 13 slots that have a hardware measurement (Q13_TO_PUB
    already excludes the 3 unmeasured mimic joints)."""
    cols = [(pub_i, q13_i) for pub_i, q13_i in enumerate(Q13_TO_PUB) if q13_i is not None]
    err = np.zeros((len(q), len(cols)), dtype=np.float32)
    for k, (pub_i, q13_i) in enumerate(cols):
        err[:, k] = cmd[:, pub_i] - q[:, q13_i]
    return err


def cycle_prediction(phase, mean_cycle):
    """Predicted 13-d joint state at every timestep, from the phase-aligned
    ensemble mean cycle (built once from the WHOLE run -- this deliberately
    uses each run's own nominal cycle as the reference "normal" behaviour)."""
    n_phase = mean_cycle.shape[0]
    grid = np.linspace(0.0, 1.0, n_phase)
    frac = (phase % (2 * np.pi)) / (2 * np.pi)
    pred = np.stack([np.interp(frac, grid, mean_cycle[:, j]) for j in range(mean_cycle.shape[1])], axis=1)
    return pred


def find_events(flag, min_samples=MIN_EVENT_SAMPLES):
    """Contiguous True-runs in a boolean array -> list of (start, end) idx."""
    events = []
    in_run, start = False, 0
    for i, v in enumerate(flag):
        if v and not in_run:
            in_run, start = True, i
        elif not v and in_run:
            if i - start >= min_samples:
                events.append((start, i))
            in_run = False
    if in_run and len(flag) - start >= min_samples:
        events.append((start, len(flag)))
    return events


def detect_disruptions(t, phase, phase_vel, mean_cycle, q, cmd, tac_real):
    """All array args must already be time-aligned and the same length (a
    full run, or a matching slice of one -- see module_d.split_half, which
    slices t/phase/phase_vel alongside q/cmd/tac_real for exactly this)."""
    err = tracking_error_13(q, cmd)
    contact_total = tac_real[:, LIVE_TAC_CH].sum(axis=1)

    z13 = (q - q.mean(axis=0)) / (q.std(axis=0) + 1e-9)
    pred = cycle_prediction(phase, mean_cycle)
    pred_z = (pred - q.mean(axis=0)) / (q.std(axis=0) + 1e-9)
    cycle_dev = np.linalg.norm(z13 - pred_z, axis=1)

    nominal_phase_vel = np.median(np.abs(phase_vel))
    route_A = (phase_vel < 0.25 * nominal_phase_vel)                          # phase stall/reversal
    route_B = robust_z(np.abs(err).mean(axis=1)) > 3.5                        # tracking-error jamming
    # contact_total is a small discrete count (0-5 live channels); MAD-based
    # z-scoring saturates near its own ceiling, so use an absolute-count rule
    # instead: >3 of the 16 live channels active at once is already unusual
    # (median=1 in both runs; see per-run distribution check).
    route_C = contact_total > 3                                              # contact anomaly
    route_D = robust_z(cycle_dev) > 2.5                                      # off the nominal cycle

    routes = {"phase_stall": route_A, "tracking_jam": route_B,
              "contact_anomaly": route_C, "cycle_deviation": route_D}
    events = {name: find_events(flag) for name, flag in routes.items()}

    # consensus: any timestep flagged by >=2 routes within CONSENSUS_WINDOW_S
    win = int(CONSENSUS_WINDOW_S * FS)
    vote = np.zeros(len(t), dtype=np.int8)
    for flag in routes.values():
        idx = np.where(flag)[0]
        for i in idx:
            vote[max(0, i - win):i + win + 1] += 1
    consensus_flag = vote >= 2
    consensus_events = find_events(consensus_flag, min_samples=1)
    # keep only the ONSET of each consensus run (already merged by the window logic)
    onsets = [s for s, e in consensus_events]

    return {
        "routes": routes, "events": events, "cycle_dev": cycle_dev,
        "err_abs_mean": np.abs(err).mean(axis=1), "contact_total": contact_total,
        "consensus_onsets": onsets, "n_consensus": len(onsets),
    }


def peri_event(signal_arr, onsets, fs=FS, pre_s=PERI_PRE_S, post_s=PERI_POST_S):
    pre, post = int(pre_s * fs), int(post_s * fs)
    rows = []
    for o in onsets:
        if o - pre < 0 or o + post >= len(signal_arr):
            continue
        rows.append(signal_arr[o - pre:o + post])
    if not rows:
        return None
    arr = np.array(rows)
    tgrid = (np.arange(-pre, post)) / fs
    return tgrid, arr, arr.mean(axis=0), arr.std(axis=0) / np.sqrt(len(rows))


def recovery_time(cycle_dev, onsets, baseline, fs=FS, max_s=8.0, thresh_mult=1.2):
    """Samples from onset until cycle_dev first drops back under
    thresh_mult * baseline (baseline = that run's median cycle_dev)."""
    max_n = int(max_s * fs)
    times = []
    for o in onsets:
        seg = cycle_dev[o:o + max_n]
        below = np.where(seg < thresh_mult * baseline)[0]
        if len(below):
            times.append(below[0] / fs)
    return times


def peak_excess(cycle_dev, onsets, baseline, fs=FS, window_s=2.0):
    """How far above baseline the departure actually reaches -- a fast
    'recovery' under a lenient threshold can still hide a deep excursion;
    this is the size-of-disruption complement to recovery_time."""
    win = int(window_s * fs)
    vals = []
    for o in onsets:
        seg = cycle_dev[o:o + win]
        if len(seg):
            vals.append(float(seg.max() / baseline))
    return vals


def run_module_b(P):
    results = {}
    for name in CONDITIONS:
        p = P[name]
        det = detect_disruptions(p["t"], p["phase"], p["phase_vel"], p["mean_cycle"],
                                  p["q"], p["cmd"], p["tac_real"])
        dur_min = p["duration"] / 60.0
        baseline = np.median(det["cycle_dev"])

        peri_cycdev = peri_event(det["cycle_dev"], det["consensus_onsets"])
        actchange = np.concatenate([[0.0], np.abs(np.diff(p["act"], axis=0)).sum(axis=1)])
        peri_act = peri_event(actchange, det["consensus_onsets"])

        rec_times = recovery_time(det["cycle_dev"], det["consensus_onsets"], baseline)
        excess = peak_excess(det["cycle_dev"], det["consensus_onsets"], baseline)

        results[name] = {
            "n_consensus_events": det["n_consensus"],
            "events_per_route": {k: len(v) for k, v in det["events"].items()},
            "disruption_rate_per_min": det["n_consensus"] / dur_min,
            "baseline_cycle_dev": float(baseline),
            "recovery_times_s": rec_times,
            "mean_recovery_time_s": float(np.mean(rec_times)) if rec_times else None,
            "frac_events_recovered_within_8s": float(len(rec_times) / det["n_consensus"]) if det["n_consensus"] else None,
            "peak_excess_ratio": excess,
            "mean_peak_excess_ratio": float(np.mean(excess)) if excess else None,
            "_det": det, "_peri_cycdev": peri_cycdev, "_peri_act": peri_act,
            "_p": p,
        }
    return results


def contact_conditioned_deviation(P, rng=None):
    """Extends the earlier contact<->|d act| coupling test: does contact
    predict LEAVING THE NOMINAL CYCLE (not just changing action), and does
    that relationship exist only when tactile is on?"""
    rng = rng or np.random.default_rng(1)
    out = {}
    for name in CONDITIONS:
        p = P[name]
        det = detect_disruptions(p["t"], p["phase"], p["phase_vel"], p["mean_cycle"],
                                  p["q"], p["cmd"], p["tac_real"])
        contact = p["tac_real"][:, LIVE_TAC_CH].sum(axis=1).astype(np.float64)
        cycdev = det["cycle_dev"]

        max_lag = int(0.5 * FS)
        lags = np.arange(-max_lag, max_lag + 1)
        corrs = np.full(len(lags), np.nan)
        for i, k in enumerate(lags):
            if k >= 0:
                a, b = contact[:len(contact) - k], cycdev[k:]
            else:
                a, b = contact[-k:], cycdev[:len(cycdev) + k]
            m = min(len(a), len(b))
            if m < 20:
                continue
            corrs[i] = np.corrcoef(a[:m], b[:m])[0, 1]
        peak_i = int(np.nanargmax(np.abs(corrs)))

        n = len(contact)
        null_peaks = []
        for _ in range(150):
            shift = rng.integers(int(2 * FS), n - int(2 * FS))
            c_shift = np.roll(contact, shift)
            vals = []
            for k in (lags[peak_i],):
                if k >= 0:
                    a, b = c_shift[:len(c_shift) - k], cycdev[k:]
                else:
                    a, b = c_shift[-k:], cycdev[:len(cycdev) + k]
                m = min(len(a), len(b))
                vals.append(np.corrcoef(a[:m], b[:m])[0, 1] if m >= 20 else 0.0)
            null_peaks.append(abs(vals[0]))
        null_peaks = np.array(null_peaks)
        out[name] = {
            "lags_s": (lags / FS).tolist(), "corrs": corrs.tolist(),
            "peak_lag_s": float(lags[peak_i] / FS), "peak_corr": float(corrs[peak_i]),
            "null_p95": float(np.percentile(null_peaks, 95)),
            "p_value": float(np.mean(null_peaks >= abs(corrs[peak_i]))),
        }
    return out


def print_module_b(results, contact_dev):
    print("=" * 78)
    print("MODULE B: RECOVERY -- does the policy break its own rhythm to correct?")
    print("=" * 78)
    print("Disruptions detected from signals alone (no ball state, no drop labels):")
    print("  route A = phase stall/reversal, B = tracking-error jamming,")
    print("  route C = contact anomaly, D = departure from the nominal cycle.")
    print("  Consensus = flagged by >=2 routes within +-0.5s.\n")

    for name in CONDITIONS:
        r = results[name]
        print(f"[{name:12s}] route events: {r['events_per_route']}")
        print(f"  consensus events = {r['n_consensus_events']}  "
              f"({r['disruption_rate_per_min']:.2f} / min over {r['_p']['duration']/60:.2f} min)")
        if r["mean_recovery_time_s"] is not None:
            print(f"  recovery time (cycle-deviation back under 1.2x baseline, within 8s): "
                  f"mean={r['mean_recovery_time_s']:.2f}s  "
                  f"recovered-fraction={r['frac_events_recovered_within_8s']*100:.1f}%  "
                  f"(n={len(r['recovery_times_s'])} of {r['n_consensus_events']} events)")
            print(f"  peak excess (max deviation / baseline within 2s of onset): "
                  f"mean={r['mean_peak_excess_ratio']:.2f}x baseline")
        else:
            print("  no consensus events with a valid recovery window")
        print()

    a, b = results["with_tactile"]["recovery_times_s"], results["zero_tactile"]["recovery_times_s"]
    if len(a) >= 3 and len(b) >= 3:
        _, p_perm = permutation_test(a, b)
        d = cliffs_delta(a, b)
        print(f"Recovery-time comparison: with_tac mean={np.mean(a):.2f}s (n={len(a)})  "
              f"zero_tac mean={np.mean(b):.2f}s (n={len(b)})  p_perm={p_perm:.4f}  Cliffs d={d:+.3f}")
    else:
        print("Too few recovered events in one condition for a formal recovery-time comparison.")

    ea, eb = results["with_tactile"]["peak_excess_ratio"], results["zero_tactile"]["peak_excess_ratio"]
    if len(ea) >= 3 and len(eb) >= 3:
        _, p_perm = permutation_test(ea, eb)
        d = cliffs_delta(ea, eb)
        print(f"Peak-excess comparison: with_tac mean={np.mean(ea):.2f}x (n={len(ea)})  "
              f"zero_tac mean={np.mean(eb):.2f}x (n={len(eb)})  p_perm={p_perm:.4f}  Cliffs d={d:+.3f}\n")
    else:
        print("Too few events in one condition for a formal peak-excess comparison.\n")

    print("-- Contact -> departure-from-nominal-cycle coupling (extends the |d act| coupling test) --")
    for name in CONDITIONS:
        c = contact_dev[name]
        print(f"  [{name:12s}] peak corr={c['peak_corr']:+.4f} at lag={c['peak_lag_s']:+.3f}s  "
              f"null_p95={c['null_p95']:.4f}  p={c['p_value']:.4f}")
    print()


if __name__ == "__main__":
    runs = load_runs()
    P = prepare(runs)
    results = run_module_b(P)
    contact_dev = contact_conditioned_deviation(P)
    print_module_b(results, contact_dev)

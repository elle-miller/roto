"""Shared machinery for the trial5 tactile-ablation analysis.

Loads the two hardware runs, segments them into curl cycles by Hilbert phase,
and builds the phase-aligned cycle ensemble that Modules A-D all consume.

Hardware context (verified from the npz files themselves):
  - 13 policy joints in POLICY_JOINTS order, 60 Hz, radians
  - `tac`      = what the policy consumed (all-zero in the ablated run)
  - `tac_real` = real post-hysteresis contact, recorded in BOTH runs
  - no ball pose, no drop annotations, no rotation counts
"""

import numpy as np
from scipy import signal

DIR = "/home/ayush/icra/roto/scripts/tactile_use/trial5/logs"
FILES = {
    "with_tactile": f"{DIR}/hw_policy_log_legacy_withtac_vid.npz",
    "zero_tactile": f"{DIR}/hw_policy_log_legacy_notac_vid.npz",
}

POLICY_JOINTS = [
    "rh_FFJ4", "rh_MFJ4", "rh_RFJ4", "rh_THJ5",
    "rh_FFJ3", "rh_MFJ3", "rh_RFJ3", "rh_THJ4",
    "rh_FFJ2", "rh_MFJ2", "rh_RFJ2",
    "rh_THJ2", "rh_THJ1",
]
CURL_IDX = {"rh_FFJ2": 8, "rh_MFJ2": 9, "rh_RFJ2": 10}
MASTER_JOINT = "rh_MFJ2"          # cleanest curl driver: largest excursion,
MASTER_IDX = CURL_IDX[MASTER_JOINT]  # lowest harmonic ratio in both runs
Q13_TO_PUB = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, None, None, None, 11, 12]

# 24-d tactile vector: 12 FSR + 4 BioTac are live, the other 8 are structurally 0
FSR_CHANNELS = [10, 7, 4, 9, 5, 13, 2, 3, 8, 18, 12, 11]
FSR_NAMES = ["C0_thprox", "C1_ffprox", "C2_mfknuckle", "C3_rfprox", "C4_rfknuckle",
             "C5_rfmid", "C6_palm", "C7_ffknuckle", "C8_mfprox", "C9_thmiddle",
             "C10_mfmid", "C11_ffmid"]
BIOTAC_CH = [15, 16, 17, 22]
BIOTAC_NAMES = ["BT_ffdist", "BT_mfdist", "BT_rfdist", "BT_thdist"]
LIVE_TAC_CH = sorted(FSR_CHANNELS + BIOTAC_CH)

# per-finger tactile channels, for the contact travelling-wave analysis
FINGER_TAC_CH = {
    "FF": [7, 3, 11, 15],    # ffprox, ffknuckle, ffmid, BT_ffdist
    "MF": [4, 8, 12, 16],    # mfknuckle, mfprox, mfmid, BT_mfdist
    "RF": [9, 5, 13, 17],    # rfprox, rfknuckle, rfmid, BT_rfdist
}

FS = 60.0
N_PHASE = 100                     # resample grid for the cycle ensemble
COL = {"with_tactile": "#4c72b0", "zero_tactile": "#dd8452"}
LBL = {"with_tactile": "with tactile", "zero_tactile": "tactile zeroed"}
CONDITIONS = ["with_tactile", "zero_tactile"]


def load_runs(truncate_steps=None):
    """Load both runs. `truncate_steps` length-matches them (Module D control)."""
    runs = {}
    for name, fn in FILES.items():
        d = dict(np.load(fn, allow_pickle=True))
        if truncate_steps is not None:
            n = min(truncate_steps, len(d["t"]))
            for k, v in d.items():
                v = np.asarray(v)
                if v.ndim >= 1 and v.shape[0] == len(d["t"]):
                    d[k] = v[:n]
            d["t"] = np.asarray(d["t"])[:n]
        runs[name] = d
    return runs


def dominant_freq(x, fs=FS):
    """Welch peak with parabolic interpolation."""
    x = x - x.mean()
    freqs, psd = signal.welch(x, fs=fs, nperseg=min(512, len(x)))
    psd[0] = 0.0
    pk = int(np.argmax(psd))
    if 0 < pk < len(psd) - 1:
        y0, y1, y2 = psd[pk - 1], psd[pk], psd[pk + 1]
        den = y0 - 2 * y1 + y2
        delta = 0.5 * (y0 - y2) / den if den != 0 else 0.0
    else:
        delta = 0.0
    return float(freqs[pk] + delta * (freqs[1] - freqs[0]))


def hilbert_phase(x, f0=None, fs=FS, bw=0.6):
    """Band-pass around the dominant rhythm, then unwrapped analytic phase."""
    if f0 is None:
        f0 = dominant_freq(x, fs)
    lo, hi = max(0.05, f0 - bw), min(fs / 2 - 0.1, f0 + bw)
    sos = signal.butter(4, [lo, hi], btype="band", fs=fs, output="sos")
    xb = signal.sosfiltfilt(sos, x - x.mean())
    phase = np.unwrap(np.angle(signal.hilbert(xb)))
    return phase, xb, f0


def cycle_boundaries(phase):
    """Indices where the unwrapped phase passes successive multiples of 2*pi."""
    bounds = [0]
    target = phase[0] + 2 * np.pi
    for i in range(1, len(phase)):
        if phase[i] >= target:
            bounds.append(i)
            target += 2 * np.pi
    return np.array(bounds)


def build_ensemble(q, bounds, n_phase=N_PHASE):
    """Resample every cycle onto a common 0->2pi grid.

    Returns (n_cycles, n_phase, n_joints).
    """
    grid = np.linspace(0.0, 1.0, n_phase)
    cycles = []
    for i in range(len(bounds) - 1):
        s, e = bounds[i], bounds[i + 1]
        if e - s < 5:
            continue
        seg = q[s:e]
        src = np.linspace(0.0, 1.0, len(seg))
        cycles.append(np.stack([np.interp(grid, src, seg[:, j])
                                for j in range(seg.shape[1])], axis=1))
    return np.array(cycles)


def prepare(runs, n_phase=N_PHASE):
    """Segment both runs and build every derived signal the modules need."""
    out = {}
    for name, d in runs.items():
        q, t = d["q"], d["t"]
        phase, xb, f0 = hilbert_phase(q[:, MASTER_IDX])
        bounds = cycle_boundaries(phase)
        ens = build_ensemble(q, bounds, n_phase)
        dt = np.diff(t)
        out[name] = {
            "d": d, "q": q, "t": t, "act": d["act"], "cmd": d["cmd"],
            "tac_real": d["tac_real"],
            "phase": phase, "phase_bp": xb, "f0": f0,
            "bounds": bounds, "ens": ens,
            "mean_cycle": ens.mean(axis=0),
            "dt": dt,
            "qdot": np.diff(q, axis=0) / dt[:, None],
            "phase_vel": np.gradient(phase) * FS,     # rad/s
            "duration": float(t[-1] - t[0]),
            "n_cycles": len(ens),
        }
    return out


def cliffs_delta(a, b):
    a, b = np.asarray(a), np.asarray(b)
    gt = sum(np.sum(x > b) for x in a)
    lt = sum(np.sum(x < b) for x in a)
    return (gt - lt) / (len(a) * len(b))


def permutation_test(a, b, n_perm=10000, rng=None):
    rng = rng or np.random.default_rng(0)
    a, b = np.asarray(a), np.asarray(b)
    obs = a.mean() - b.mean()
    pooled = np.concatenate([a, b])
    na = len(a)
    cnt = 0
    for _ in range(n_perm):
        rng.shuffle(pooled)
        if abs(pooled[:na].mean() - pooled[na:].mean()) >= abs(obs):
            cnt += 1
    return float(obs), cnt / n_perm


def bh_fdr(pvals):
    """Benjamini-Hochberg step-up adjusted p-values."""
    p = np.asarray(pvals, dtype=float)
    m = len(p)
    order = np.argsort(p)
    adj = np.empty(m)
    prev = 1.0
    for rank, idx in enumerate(order[::-1]):
        i = m - rank
        prev = min(prev, p[idx] * m / i)
        adj[idx] = prev
    return adj

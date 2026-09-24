"""Small from-scratch implementations of standard nonlinear-dynamics measures.

No specialized package (nolds/pyrqa/antropy) is installed in this environment,
so these are direct, textbook implementations rather than reused libraries:
  - sample_entropy: Richman & Moorman (2000)
  - embed / rqa_metrics: standard time-delay embedding + recurrence quantification
  - rosenstein_lyapunov: Rosenstein, Collins & De Luca (1993)

Kept deliberately simple (no FFT-accelerated distance matrices, no fancy
neighbor search) since run lengths here (~300-400s @ 60Hz) are small enough
that O(n^2) is fine.
"""

import numpy as np


def sample_entropy(x, m=2, r=None):
    """SampEn(m, r): -log( A / B ), A/B = count of (m+1)/(m)-length matches
    within tolerance r, excluding self-matches. r defaults to 0.2*std(x).
    """
    x = np.asarray(x, dtype=float)
    n = len(x)
    if r is None:
        r = 0.2 * np.std(x)

    def _count(mm):
        templates = np.array([x[i:i + mm] for i in range(n - mm)])
        cnt = 0
        for i in range(len(templates)):
            d = np.max(np.abs(templates[i + 1:] - templates[i]), axis=1)
            cnt += np.sum(d <= r)
        return cnt

    B = _count(m)
    A = _count(m + 1)
    if B == 0 or A == 0:
        return np.nan
    return -np.log(A / B)


def embed(x, m, tau):
    """Time-delay embedding. Returns (n - (m-1)*tau, m)."""
    x = np.asarray(x, dtype=float)
    n = len(x) - (m - 1) * tau
    return np.array([x[i:i + (m - 1) * tau + 1:tau] for i in range(n)])


def first_min_ami(x, max_lag=60, bins=16):
    """First local minimum of the average mutual information, for tau selection.
    Falls back to the first zero-crossing of autocorrelation if AMI has no
    interior minimum within max_lag.
    """
    x = np.asarray(x, dtype=float)
    hist_range = (x.min(), x.max())

    def ami(lag):
        a, b = x[:-lag], x[lag:]
        c_xy, xe, ye = np.histogram2d(a, b, bins=bins, range=[hist_range, hist_range])
        pxy = c_xy / c_xy.sum()
        px = pxy.sum(axis=1, keepdims=True)
        py = pxy.sum(axis=0, keepdims=True)
        with np.errstate(divide="ignore", invalid="ignore"):
            terms = pxy * np.log(pxy / (px * py))
        return float(np.nansum(np.where(pxy > 0, terms, 0.0)))

    vals = [ami(lag) for lag in range(1, max_lag)]
    for i in range(1, len(vals) - 1):
        if vals[i] < vals[i - 1] and vals[i] < vals[i + 1]:
            return i + 1
    # fallback: first zero-crossing of autocorrelation
    xc = x - x.mean()
    ac = np.correlate(xc, xc, mode="full")[len(xc) - 1:]
    ac /= ac[0]
    zc = np.where(np.diff(np.sign(ac)) < 0)[0]
    return int(zc[0]) + 1 if len(zc) else 5


def rqa_metrics(emb, eps_pct=10, min_diag=2, min_vert=2, theiler=1):
    """Basic recurrence quantification on an embedded trajectory.

    eps_pct: recurrence threshold as a percentile of the pairwise-distance
    distribution (fixed recurrence-rate style threshold, not a fixed radius,
    so the two runs are compared at matched recurrence density).
    theiler: excludes points within this many samples of the diagonal to
    avoid trivial auto-recurrence from the embedding itself.

    Returns dict with recurrence_rate, determinism, laminarity, mean_diag_len.
    """
    n = len(emb)
    d = np.sqrt(((emb[:, None, :] - emb[None, :, :]) ** 2).sum(axis=2))
    iu = np.triu_indices(n, k=theiler + 1)
    eps = np.percentile(d[iu], eps_pct)
    R = (d <= eps).astype(np.int8)
    for k in range(-theiler, theiler + 1):
        np.fill_diagonal(R[max(0, k):, max(0, -k):], 0)

    # Denominator for DET/LAM must count the SAME point set the line-scans
    # traverse (the full symmetric matrix minus the Theiler band), not just
    # the upper triangle -- otherwise LAM (full-matrix column scan) is
    # compared against half the points and can exceed 1.
    n_recur_pts = int(R.sum())
    rr = n_recur_pts / (n * n - n)  # exclude the diagonal from the possible-pairs count

    def diag_lines(mat, min_len):
        total_pts, total_lines_len, n_lines = 0, 0, 0
        for offset in range(theiler + 1, n - 1):
            for diag in (np.diagonal(mat, offset=offset), np.diagonal(mat, offset=-offset)):
                run = 0
                for v in diag:
                    if v:
                        run += 1
                    else:
                        if run >= min_len:
                            total_pts += run
                            total_lines_len += run
                            n_lines += 1
                        run = 0
                if run >= min_len:
                    total_pts += run
                    total_lines_len += run
                    n_lines += 1
        return total_pts, (total_lines_len / n_lines if n_lines else 0.0)

    def vert_lines(mat, min_len):
        total_pts, n_lines = 0, 0
        for col in range(n):
            colvals = mat[:, col]
            run = 0
            for v in colvals:
                if v:
                    run += 1
                else:
                    if run >= min_len:
                        total_pts += run
                        n_lines += 1
                    run = 0
            if run >= min_len:
                total_pts += run
                n_lines += 1
        return total_pts

    diag_pts, mean_diag_len = diag_lines(R, min_diag)
    vert_pts = vert_lines(R, min_vert)

    det = diag_pts / n_recur_pts if n_recur_pts else np.nan
    lam = vert_pts / n_recur_pts if n_recur_pts else np.nan
    return {
        "recurrence_rate": float(rr),
        "determinism": float(det),
        "laminarity": float(lam),
        "mean_diag_len": float(mean_diag_len),
        "eps": float(eps),
        "n_points": int(n),
    }


def rosenstein_lyapunov(x, m, tau, fs, theiler=None, max_t=40, fit_range=None):
    """Largest Lyapunov exponent, Rosenstein et al. (1993), in nats/s.

    For each embedded point, find its nearest neighbor (excluding a Theiler
    window around it in time), track the log-divergence of the two
    trajectories for `max_t` samples, average across all reference points,
    then fit a line over `fit_range` samples (default: the whole curve) --
    the slope * fs is the exponent.
    """
    emb = embed(x, m, tau)
    n = len(emb)
    theiler = theiler or (2 * tau)
    d = np.sqrt(((emb[:, None, :] - emb[None, :, :]) ** 2).sum(axis=2))
    nn = np.full(n, -1, dtype=int)
    for i in range(n):
        row = d[i].copy()
        lo, hi = max(0, i - theiler), min(n, i + theiler + 1)
        row[lo:hi] = np.inf
        j = np.argmin(row)
        if np.isfinite(row[j]):
            nn[i] = j

    max_t = min(max_t, n - 1)
    log_div = np.full(max_t, np.nan)
    for k in range(max_t):
        vals = []
        for i in range(n - k):
            j = nn[i]
            if j < 0 or j + k >= n:
                continue
            dist = np.linalg.norm(emb[i + k] - emb[j + k])
            if dist > 0:
                vals.append(np.log(dist))
        if vals:
            log_div[k] = np.mean(vals)

    valid = ~np.isnan(log_div)
    ks = np.arange(max_t)[valid]
    ys = log_div[valid]
    if fit_range is not None:
        sel = (ks >= fit_range[0]) & (ks <= fit_range[1])
        ks, ys = ks[sel], ys[sel]
    if len(ks) < 3:
        return np.nan, (log_div, np.arange(max_t) / fs)
    slope, intercept = np.polyfit(ks, ys, 1)
    return float(slope * fs), (log_div, np.arange(max_t) / fs)

"""Temporal alignment between a sim trajectory and its hardware replay.

Hardware replays don't track the sim trajectory at a fixed 1:1 frame rate: the
real hand's PD/servo response is slower and more variable than sim's soft PD, so
the same nominal Baoding motion takes noticeably longer on hardware and the ratio
isn't constant across the trajectory (e.g. seed7: 600 sim frames @ 60Hz = 10s vs
1738 hw frames @ 60Hz = ~29s; hardware sits quasi-static for stretches where sim
is still moving). Frame index t means a different point in the motion in each
domain -- verified directly in an earlier session (hw t=300 was near-static while
sim t=300 was mid-motion, velocities ~0 vs ~2-6 rad/s).

This module finds, for every hardware frame, the sim frame at the SAME point in
the motion via classic DTW (dynamic time warping) on the ACHIEVED position
signal. The commanded signal was tried first and rejected: sim's raw policy
action is unbounded (see the encoder pipeline investigation), so sim's
commanded position is ~9x noisier frame-to-frame than its own achieved position
(measured: mean |diff| 0.27 vs 0.03 rad on FFJ2) while hardware's commanded and
achieved signals are both smooth (~0.01 rad) -- aligning on that mismatched
noise level produced a poor path (RMS 1.19 on a [-1,1] scale, barely better than
no alignment at all). Achieved position is smooth and physically comparable in
both domains (governed by real dynamics, not the raw noisy action), and is what
actually describes "where the hand is" at each instant. Both signals are
normalised with the SAME per-joint [lower, upper] limits (finetune_bc.
LOWER_LIMITS/UPPER_LIMITS, already fixed for the 3 coupled slots to [0, 1.745]
-- see finetune_bc.py) so no joint's larger range dominates the alignment cost.

No third-party DTW library is used (none installed in this env); this is a
plain O(T_sim * T_hw) numpy DP, ~0.5s for a 600x1738 pair -- fine as a one-off
per-seed precompute, not a training-loop operation.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
import finetune_bc as bc  # noqa: E402  (LOWER_LIMITS, UPPER_LIMITS, normalise, build_policy_column_map)


def dtw_align(sim_feat: np.ndarray, hw_feat: np.ndarray) -> np.ndarray:
    """Align hw_feat (T_hw, D) onto sim_feat (T_sim, D) via DTW (squared-L2 cost).

    Returns:
        hw_to_sim: (T_hw,) int64 array, monotonic non-decreasing. hw_to_sim[j] is
        the sim frame index at the same point in the motion as hw frame j.
    """
    T_sim, T_hw = sim_feat.shape[0], hw_feat.shape[0]

    diff = sim_feat[:, None, :] - hw_feat[None, :, :]
    cost = np.einsum("ijd,ijd->ij", diff, diff)  # (T_sim, T_hw)

    # D[i, j] = min cumulative cost aligning sim[:i] to hw[:j] (1-indexed DP table).
    D = np.full((T_sim + 1, T_hw + 1), np.inf, dtype=np.float64)
    D[0, 0] = 0.0
    for i in range(1, T_sim + 1):
        row, prev_row, c = D[i], D[i - 1], cost[i - 1]
        for j in range(1, T_hw + 1):
            row[j] = c[j - 1] + min(prev_row[j], row[j - 1], prev_row[j - 1])

    # Backtrack from (T_sim, T_hw) to (0, 0); record the sim index matched to
    # each hw index (last-write-wins per j, since the path is monotonic in j).
    hw_to_sim = np.zeros(T_hw, dtype=np.int64)
    i, j = T_sim, T_hw
    while i > 0 and j > 0:
        hw_to_sim[j - 1] = i - 1
        candidates = (D[i - 1, j], D[i, j - 1], D[i - 1, j - 1])
        step = int(np.argmin(candidates))
        if step == 0:
            i -= 1
        elif step == 1:
            j -= 1
        else:
            i -= 1
            j -= 1
    while j > 0:  # leftover hw prefix (only possible if i hit 0 first): pin to sim frame 0
        j -= 1
        hw_to_sim[j] = 0
    return hw_to_sim


def build_alignment_features(sim_npz, hw_npz) -> tuple[np.ndarray, np.ndarray]:
    """Extract the (T,13) normalised ACHIEVED-position signal from each domain,
    in CONTROL_JOINT_NAMES order, on the shared [lower, upper] scale.

    Deliberately achieved position, not commanded -- see the module docstring.
    """
    if "q13" in sim_npz.files:
        sim_pos = sim_npz["q13"].astype(np.float64)
    else:
        joints16 = list(sim_npz["joints"])
        cols = [joints16.index(name) for name in bc.CONTROL_JOINT_NAMES]
        sim_pos = sim_npz["q"][:, cols].astype(np.float64)

    actuator_order = list(hw_npz["actuator_order"])
    hw_cols = bc.build_policy_column_map(actuator_order)
    hw_pos = hw_npz["act_pos"][:, hw_cols].astype(np.float64)
    for i in bc.COUPLED_SLOTS:  # combined FFJ0/MFJ0/RFJ0 reading; clip like build_hw_frames
        hw_pos[:, i] = np.clip(hw_pos[:, i], bc.LOWER_LIMITS[i], bc.UPPER_LIMITS[i])

    sim_feat = bc.normalise(sim_pos, bc.LOWER_LIMITS, bc.UPPER_LIMITS)
    hw_feat = bc.normalise(hw_pos, bc.LOWER_LIMITS, bc.UPPER_LIMITS)
    return sim_feat, hw_feat


def align_seed(sim_npz, hw_npz) -> np.ndarray:
    """Convenience wrapper: build alignment features and DTW-align in one call.

    Returns hw_to_sim (T_hw,) int64, as in dtw_align.
    """
    sim_feat, hw_feat = build_alignment_features(sim_npz, hw_npz)
    return dtw_align(sim_feat, hw_feat)

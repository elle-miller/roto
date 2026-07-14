"""Delta-history construction for GenAN inputs.

Isaac-free (pure torch), matching the convention set by
`uan_shadowlite/dataset.py` and `features.py`.

Implements the paper's finding (their Appendix C) that dense, standardized
*delta* histories -- (x_t, x_{t-1} - x_t, ..., x_{t-H} - x_t) -- generalize
better than raw sparse-strided histories: consecutive raw values are too
similar for the network to make good use of, whereas differencing against
the current frame amplifies exactly the signal (recent change) that matters
for hysteresis. See DESIGN.md, Decision 3.
"""

from __future__ import annotations

import torch


def delta_history_indices(t: torch.Tensor, history_len: int, stride: int) -> torch.Tensor:
    """Return the `history_len + 1` time indices (t, t-stride, ..., t-H*stride)
    for each entry of `t`, as a (len(t), history_len + 1) tensor, most-recent
    first (column 0 == t itself).
    """
    if history_len < 0:
        raise ValueError(f"history_len must be >= 0, got {history_len}")
    if stride < 1:
        raise ValueError(f"stride must be >= 1, got {stride}")
    offsets = torch.arange(0, (history_len + 1) * stride, stride, device=t.device)
    return t.unsqueeze(-1) - offsets.unsqueeze(0)


def build_delta_history(
    x: torch.Tensor,
    t: torch.Tensor,
    history_len: int,
    stride: int,
    dataset,
) -> torch.Tensor:
    """Build a standardization-ready delta history for signal `x` at times `t`.

    Args:
        x: (T, D) full signal (e.g. dataset.q_meas or dataset.q_cmd).
        t: (N,) time indices to build histories for (the "current" step).
        history_len: H -- number of past frames beyond the current one.
        stride: step between consecutive history frames.
        dataset: the `AlignedTrajectoryDataset`/`TrajectoryDataset` `t` was
            drawn from -- needs `.clamp(idx)` (global bounds) AND
            `.segment_start(idx)` (per-row trajectory-start bound). Both are
            required, not just global clamping: a history reaching back past
            its OWN segment's start must be clipped to that segment's first
            frame, not allowed to silently read into the previous, unrelated
            trajectory file that happens to sit right before it in the
            concatenated arrays. Padding by repeating the first frame is
            still the desired behavior at a segment boundary -- it just has
            to be bounded per-row by that row's own segment, not globally.

    Returns:
        (N, (history_len + 1) * D) tensor: [x_t, x_{t-1} - x_t, ..., x_{t-H} - x_t],
        flattened in that (most-recent-delta-first) order.
    """
    t = dataset.clamp(t)
    seg_start = dataset.segment_start(t)  # (N,) -- this row's own trajectory start
    idx = delta_history_indices(t, history_len, stride)  # (N, history_len + 1)
    idx = torch.maximum(idx, seg_start.unsqueeze(-1))
    idx = dataset.clamp(idx.reshape(-1)).reshape(idx.shape)
    frames = x[idx]  # (N, history_len + 1, D)
    current = frames[:, :1]
    deltas = torch.cat([current, frames[:, 1:] - current], dim=1)
    return deltas.reshape(deltas.shape[0], -1)

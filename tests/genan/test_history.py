"""CPU-only unit tests for roto.genan.history.

No `isaaclab` import anywhere in this file or in `history.py` -- run with any
environment that has torch+numpy+pytest, matching the convention set by
`roto/tests/uan_shadowlite/test_dataset.py`.
"""

import os
import sys

import numpy as np
import torch

_GENAN_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "genan")
sys.path.insert(0, _GENAN_DIR)

from history import build_delta_history, delta_history_indices  # noqa: E402


class _FakeDataset:
    """Minimal `.clamp`/`.segment_start` stand-in with two segments: [0, 4] and [5, 9]."""

    def __init__(self):
        self.num_steps = 10
        self._segment_id = torch.tensor([0] * 5 + [1] * 5, dtype=torch.long)
        self._starts = torch.tensor([0, 5], dtype=torch.long)

    def clamp(self, t: torch.Tensor) -> torch.Tensor:
        return t.clamp(min=0, max=self.num_steps - 1)

    def segment_start(self, t: torch.Tensor) -> torch.Tensor:
        return self._starts[self._segment_id[self.clamp(t)]]


def test_delta_history_indices_shape_and_values():
    t = torch.tensor([10, 20])
    idx = delta_history_indices(t, history_len=3, stride=2)
    assert idx.shape == (2, 4)
    expected_row0 = torch.tensor([10, 8, 6, 4])
    assert torch.equal(idx[0], expected_row0)


def test_delta_history_indices_rejects_bad_args():
    t = torch.tensor([5])
    try:
        delta_history_indices(t, history_len=-1, stride=1)
        assert False, "expected ValueError"
    except ValueError:
        pass
    try:
        delta_history_indices(t, history_len=1, stride=0)
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_build_delta_history_current_frame_has_zero_delta_part():
    ds = _FakeDataset()
    x = torch.arange(10, dtype=torch.float32).unsqueeze(-1)  # (10, 1): x[i] = i
    t = torch.tensor([7])
    out = build_delta_history(x, t, history_len=2, stride=1, dataset=ds)
    # layout: [x_t, x_{t-1}-x_t, x_{t-2}-x_t] = [7, 6-7, 5-7] = [7, -1, -2]
    assert torch.allclose(out[0], torch.tensor([7.0, -1.0, -2.0]))


def test_build_delta_history_pads_at_segment_start_not_previous_segment():
    ds = _FakeDataset()
    x = torch.arange(10, dtype=torch.float32).unsqueeze(-1)
    # t=6 is the second frame of segment 1 (which starts at index 5). A naive
    # global clamp would read index 4 (x=4, end of segment 0) for the
    # "t-2" history slot; the correct, segment-aware padding must instead
    # repeat segment 1's own first frame (index 5, x=5).
    t = torch.tensor([6])
    out = build_delta_history(x, t, history_len=2, stride=1, dataset=ds)
    # x_t=6; x_{t-1}=5 (segment 1's own start, correctly reached); x_{t-2}
    # would naively be index 4 (segment 0) but must clamp to segment 1's
    # start (index 5) instead.
    assert torch.allclose(out[0], torch.tensor([6.0, 5.0 - 6.0, 5.0 - 6.0]))


def test_build_delta_history_batch_matches_per_row_reference():
    ds = _FakeDataset()
    rng = np.random.default_rng(0)
    x = torch.as_tensor(rng.normal(size=(10, 3)), dtype=torch.float32)
    t = torch.tensor([2, 6, 9])
    out = build_delta_history(x, t, history_len=2, stride=1, dataset=ds)
    assert out.shape == (3, 3 * 3)
    for row, ti in enumerate(t.tolist()):
        single = build_delta_history(x, torch.tensor([ti]), history_len=2, stride=1, dataset=ds)
        assert torch.allclose(out[row], single[0])

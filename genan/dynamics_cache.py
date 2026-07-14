"""Load preprocess.py + compute_dynamics.py output for GenAN's Position loss.

Isaac-free (pure numpy/torch) -- both cache files are already flat, precomputed
.npz arrays by the time anything here reads them; only PRODUCING
compute_dynamics.py's file needs Isaac Sim, not consuming it.
"""

from __future__ import annotations

import numpy as np
import torch


class DynamicsCache:
    """Row-aligned (same global index `t` as AlignedTrajectoryDataset) kinematic
    + dynamic quantities for the Position loss: `q`/`qdot` from preprocess.py,
    `M_inv`/`C`/`G`/`tau_target` from compute_dynamics.py.
    """

    def __init__(self, preprocess_path: str, dynamics_path: str) -> None:
        pre = np.load(preprocess_path)
        dyn = np.load(dynamics_path)
        self.q = torch.as_tensor(pre["q_meas_smooth"], dtype=torch.float32)
        self.qdot = torch.as_tensor(pre["q_dot"], dtype=torch.float32)
        self.m_inv = torch.as_tensor(dyn["M_inv"], dtype=torch.float32)
        self.C = torch.as_tensor(dyn["C"], dtype=torch.float32)
        self.G = torch.as_tensor(dyn["G"], dtype=torch.float32)
        self.tau_target = torch.as_tensor(dyn["tau_target"], dtype=torch.float32)

        n = min(self.q.shape[0], self.m_inv.shape[0])
        if self.q.shape[0] != self.m_inv.shape[0]:
            print(
                f"[WARN] DynamicsCache: preprocess.py has {self.q.shape[0]} rows but "
                f"compute_dynamics.py has {self.m_inv.shape[0]} rows (likely a partial "
                f"--limit run of one of the two); truncating both to {n} rows."
            )
        self.num_rows = n

    def position_targets(self, dataset, t: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """Return `(tau_target, m_inv, C, G, q_t, qdot_t, q_next, valid_mask)`
        for global time indices `t` (same indexing `dataset.q_torque[t]` etc.
        already use).

        `valid_mask` is False at each trajectory segment's final row (via
        `dataset.is_at_boundary(t)`) -- `t+1` there would read the FIRST row
        of the next segment/file, an unrelated trajectory, not the real
        "next" state. Callers must only average the position loss over rows
        where this mask is True.
        """
        t = t.clamp(max=self.num_rows - 1)
        t_next = (t + 1).clamp(max=self.num_rows - 1)
        valid_mask = ~dataset.is_at_boundary(t)
        return (
            self.tau_target[t],
            self.m_inv[t],
            self.C[t],
            self.G[t],
            self.q[t],
            self.qdot[t],
            self.q[t_next],
            valid_mask,
        )

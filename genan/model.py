"""GenAN network + ensemble.

Isaac-free (pure torch). Reuses the repo's own building blocks rather than
reimplementing them: `multimodal_rl.models.mlp.MLP` for the hidden stack
(the same builder `Encoder`/`GaussianPolicy` use) and
`multimodal_rl.models.running_standard_scaler.RunningStandardScaler` for
standardization (the same one `Encoder`'s `state_preprocessor` and PPO's
`value_preprocessor` already use). See DESIGN.md, Decision 2.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from multimodal_rl.models.mlp import MLP
from multimodal_rl.models.running_standard_scaler import RunningStandardScaler

# Table 1 of the paper: 2 hidden layers, 512 units, tanh.
HIDDEN_DIMS = [512, 512]
ACTIVATIONS = ["tanh", "tanh"]


class GenAN(nn.Module):
    """One actuator-network member: standardized-history -> raw torque.

    `input_dim` is the flattened, ALREADY-standardized-and-concatenated
    `(q_history, u_history)` vector -- standardization is the caller's job
    (see `GenANEnsemble`), so this class is a plain regressor and stays
    trivially testable in isolation.
    """

    def __init__(self, input_dim: int, num_joints: int) -> None:
        super().__init__()
        self.trunk = MLP(input_dim, HIDDEN_DIMS, ACTIVATIONS)
        self.head = nn.Linear(HIDDEN_DIMS[-1], num_joints)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.trunk(x))


class GenANEnsemble(nn.Module):
    """`N` independently-seeded `GenAN` members sharing one input scaler and
    one torque-label scaler (the input/label distributions are the same for
    every member -- only the network weights and training-data permutation
    differ per member, per DESIGN.md Decision 2).
    """

    def __init__(self, input_dim: int, num_joints: int, ensemble_size: int = 5, seed: int = 0) -> None:
        super().__init__()
        self.num_joints = num_joints
        self.ensemble_size = ensemble_size
        self.input_scaler = RunningStandardScaler(size=input_dim, device="cpu")
        self.label_scaler = RunningStandardScaler(size=num_joints, device="cpu")

        members = []
        for i in range(ensemble_size):
            gen = torch.Generator().manual_seed(seed + i)
            member = GenAN(input_dim, num_joints)
            with torch.no_grad():
                for p in member.parameters():
                    if p.dim() > 1:
                        nn.init.xavier_uniform_(p, generator=gen)
                    else:
                        nn.init.zeros_(p)
            members.append(member)
        self.members = nn.ModuleList(members)

    def forward(self, raw_input: torch.Tensor) -> torch.Tensor:
        """Return de-standardized torque predictions for every member.

        Returns: (ensemble_size, batch, num_joints).
        """
        x = self.input_scaler(raw_input, train=False)
        preds_std = torch.stack([member(x) for member in self.members], dim=0)
        return self.label_scaler(preds_std, train=False, inverse=True)

    def forward_standardized(self, raw_input: torch.Tensor) -> torch.Tensor:
        """Standardized-space predictions (what the torque loss trains against)."""
        x = self.input_scaler(raw_input, train=False)
        return torch.stack([member(x) for member in self.members], dim=0)

    def sample_member(self, raw_input: torch.Tensor, generator: torch.Generator | None = None) -> torch.Tensor:
        """One randomly-chosen member's de-standardized torque prediction per
        call, per the paper's per-step random-member rollout sampling.
        Returns: (batch, num_joints).
        """
        idx = torch.randint(0, self.ensemble_size, (1,), generator=generator).item()
        x = self.input_scaler(raw_input, train=False)
        pred_std = self.members[idx](x)
        return self.label_scaler(pred_std, train=False, inverse=True)

    def disagreement(self, raw_input: torch.Tensor) -> torch.Tensor:
        """Per-joint std across ensemble members (de-standardized torque units).
        Returns: (batch, num_joints).
        """
        preds = self.forward(raw_input)
        return preds.std(dim=0)

    def fit_scalers(self, raw_input: torch.Tensor, torque_label: torch.Tensor) -> None:
        """Fit both scalers in one pass over a (large) training batch."""
        self.input_scaler(raw_input, train=True)
        self.label_scaler(torque_label, train=True)

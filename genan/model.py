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

    `bounded_output`: if True, the output is passed through `tanh`, so it is
    architecturally guaranteed to stay in (-1, 1) -- paired with
    `losses.torque_minmax_loss`'s fixed min-max label normalization (see
    GenANEnsemble's `torque_range`), as an alternative to the default
    RunningStandardScaler-based (data-driven) standardization. Default
    `False` reproduces the exact prior unbounded-linear-output behavior.
    """

    def __init__(self, input_dim: int, num_joints: int, bounded_output: bool = False) -> None:
        super().__init__()
        self.trunk = MLP(input_dim, HIDDEN_DIMS, ACTIVATIONS)
        self.head = nn.Linear(HIDDEN_DIMS[-1], num_joints)
        self.bounded_output = bounded_output

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.head(self.trunk(x))
        return torch.tanh(out) if self.bounded_output else out


class GenANEnsemble(nn.Module):
    """`N` independently-seeded `GenAN` members sharing one input scaler and
    one torque-label scaler (the input/label distributions are the same for
    every member -- only the network weights and training-data permutation
    differ per member, per DESIGN.md Decision 2).

    `bounded_output`/`torque_range`: opt into the fixed min-max normalization
    scheme (see `GenAN.bounded_output`, `losses.torque_minmax_loss`) instead
    of the default data-driven `label_scaler` standardization. `label_scaler`
    is still constructed either way (harmless if unused) so `fit_scalers`
    doesn't need to branch. Default `torque_range=None` reproduces the exact
    prior behavior bit-for-bit.
    """

    def __init__(
        self, input_dim: int, num_joints: int, ensemble_size: int = 5, seed: int = 0,
        bounded_output: bool = False, torque_range: float | None = None,
    ) -> None:
        super().__init__()
        self.num_joints = num_joints
        self.ensemble_size = ensemble_size
        self.torque_range = torque_range
        self.input_scaler = RunningStandardScaler(size=input_dim, device="cpu")
        self.label_scaler = RunningStandardScaler(size=num_joints, device="cpu")

        members = []
        for i in range(ensemble_size):
            gen = torch.Generator().manual_seed(seed + i)
            member = GenAN(input_dim, num_joints, bounded_output=bounded_output)
            with torch.no_grad():
                for p in member.parameters():
                    if p.dim() > 1:
                        nn.init.xavier_uniform_(p, generator=gen)
                    else:
                        nn.init.zeros_(p)
            members.append(member)
        self.members = nn.ModuleList(members)

    def forward(self, raw_input: torch.Tensor) -> torch.Tensor:
        """Return de-standardized (physical-torque) predictions for every member.

        Returns: (ensemble_size, batch, num_joints). Branches on
        `self.torque_range`: fixed min-max de-normalization (`preds * torque_range`)
        if set, else the default `label_scaler`-based (data-driven) inverse --
        see class docstring.
        """
        preds_std = self.forward_standardized(raw_input)
        if self.torque_range is not None:
            return preds_std * self.torque_range
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
        if self.torque_range is not None:
            return pred_std * self.torque_range
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

"""Pure-tensor reward math for the UAN task.

Deliberately Isaac-free (pure torch), exactly like dataset.py/features.py -- so it can be
unit-tested on CPU without booting Isaac Sim. task.py imports this rather than defining it
inline.
"""

from __future__ import annotations

import torch


@torch.jit.script
def compute_uan_reward(
    q_real: torch.Tensor,
    q_sim: torch.Tensor,
    actions: torch.Tensor,
    last_actions: torch.Tensor,
    torque_sim: torch.Tensor,
    torque_real: torch.Tensor,
    survival: float,
    l1: float,
    exp_l2_loose: float,
    coef_loose: float,
    exp_l2: float,
    coef_l2: float,
    exp_l2_strict: float,
    coef_strict: float,
    exp_action_rate: float,
    coef_action_rate: float,
    torque_sign: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """UAN reward: L1 + three exponential-of-negative-L2 tracking bonuses at
    increasing sharpness + an action-rate smoothness bonus + survival, plus
    an optional calibration-free torque-sign-agreement term.

    The torque term compares sign(sim total torque) to sign(real uncalibrated
    effort) per joint, never magnitude -- this is invariant to any positive
    per-joint calibration scale factor (uncalibrated sensors report
    tau_raw = a_j * tau_true with unknown, possibly per-joint-different a_j;
    sign is invariant to a_j > 0). Weighted by `torque_sign` (0.0 = inert).
    """
    se = (q_real - q_sim).square()
    ae = (q_real - q_sim).abs()
    se_sum = se.sum(dim=1)
    ae_sum = ae.sum(dim=1)
    action_rate = torch.linalg.vector_norm(actions - last_actions, dim=1)
    sign_agree = (torch.sign(torque_sim) == torch.sign(torque_real)).float().mean(dim=1)

    reward = (
        survival
        + l1 * ae_sum
        + exp_l2_loose * torch.exp(-coef_loose * se_sum)
        + exp_l2 * torch.exp(-coef_l2 * se_sum)
        + exp_l2_strict * torch.exp(-coef_strict * se_sum)
        + exp_action_rate * torch.exp(-coef_action_rate * action_rate)
        + torque_sign * sign_agree
    )
    return reward, se_sum, ae_sum

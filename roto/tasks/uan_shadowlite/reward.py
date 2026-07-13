"""Pure-tensor reward and safety math for the UAN task.

Deliberately Isaac-free (pure torch), exactly like dataset.py/features.py --
so it can be unit-tested on CPU without booting Isaac Sim. task.py imports
these functions rather than defining them inline.
"""

from __future__ import annotations

import torch


@torch.jit.script
def soft_limit_avoidance(
    torque: torch.Tensor,
    pos: torch.Tensor,
    lower: torch.Tensor,
    upper: torch.Tensor,
    margin: float,
) -> torch.Tensor:
    """Smoothly scale outward-pushing torque to 0 within `margin` of a joint's limit.

    Exists because the 6 mechanically-coupled joints (task.py's coupled_local_idx) have
    PhysX's implicit-PD restoring force neutralized (their position target tracks
    current position, not a real setpoint -- see UANShadowLiteEnv._pre_physics_step),
    so nothing else pulls them back from a limit if the network's own torque pushes
    outward, and PhysX's own limit enforcement is a soft constraint under continuous
    torque, not guaranteed to hold on its own.

    `room_below`/`room_above` are 1.0 when at least `margin` radians clear of the
    respective bound, falling linearly to 0.0 exactly at the bound. Torque pushing
    *toward* the nearby limit (negative torque near the lower bound, positive torque
    near the upper bound) is scaled by the corresponding room fraction; torque pushing
    back toward the safe range is left completely untouched (scale factor 1.0),
    regardless of how close to a limit the joint currently is -- this only ever
    prevents the network from making an out-of-range excursion worse, never opposes a
    recovery.
    """
    room_below = torch.clamp((pos - lower) / margin, 0.0, 1.0)
    room_above = torch.clamp((upper - pos) / margin, 0.0, 1.0)
    scale = torch.where(torque < 0, room_below, room_above)
    return torque * scale


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

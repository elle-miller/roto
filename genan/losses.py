"""GenAN training loss.

Isaac-free (pure torch). See DESIGN.md, Decision 1: that decision's stated
reason for rejecting a Position loss ("no exposed M(q)") is superseded --
Isaac Lab's PhysX tensor API (`Articulation.root_physx_view`) does expose a
numeric M(q)/C(q,qdot)/G(q) query (see `roto/scripts/compute_dynamics.py`).
`position_loss` below consumes that query's *output*, precomputed offline and
loaded from disk as constants -- this file itself stays exactly as Isaac-free
as before.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def predict_next_position(
    tau_predicted_physical: torch.Tensor,
    m_inv: torch.Tensor,
    C: torch.Tensor,
    G: torch.Tensor,
    q_t: torch.Tensor,
    qdot_t: torch.Tensor,
    dt: float,
) -> torch.Tensor:
    """One closed-form semi-implicit-Euler dynamics step using `tau_predicted_physical`.

    `m_inv`/`C`/`G`/`q_t`/`qdot_t` are precomputed, loaded-from-disk CONSTANTS
    (`compute_dynamics.py`'s output; no `grad_fn` by construction -- they
    never touch autograd). `tau_predicted_physical` is GenAN's own
    DE-STANDARDIZED (physical N*m) predicted torque, the only tensor here
    that carries gradient -- see model.py's `GenANEnsemble` and the
    `no_grad=False` note on `RunningStandardScaler` in train_genan.py's
    docstring for why the caller must de-standardize explicitly with
    `no_grad=False` rather than calling `GenANEnsemble.forward()` (whose
    default `no_grad=True` scaler call silently breaks this gradient path).

    `tau_predicted_physical`/`C`/`G` are (..., num_joints); `m_inv` is
    (..., num_joints, num_joints); `q_t`/`qdot_t` are (..., num_joints).
    Leading dims (e.g. an ensemble dimension) broadcast through `m_inv`'s
    batched matmul as long as they match across arguments. Always uses the
    FULL num_joints-dim state -- ShadowLite's joints are physically coupled
    through the hand's rigid-body structure (shared links/inertia), so a
    reduced single-joint inertia would silently misrepresent that coupling
    even when only one joint's torque differs from its real/target value
    (see train_genan_single.py's isolated single-joint Position loss).
    """
    residual_torque = (tau_predicted_physical - C - G).unsqueeze(-1)
    q_ddot_pred = torch.matmul(m_inv, residual_torque).squeeze(-1)
    qdot_next_pred = qdot_t + dt * q_ddot_pred
    return q_t + dt * qdot_next_pred


def position_loss(
    tau_predicted_physical: torch.Tensor,
    m_inv: torch.Tensor,
    C: torch.Tensor,
    G: torch.Tensor,
    q_t: torch.Tensor,
    qdot_t: torch.Tensor,
    q_real_next: torch.Tensor,
    dt: float,
) -> torch.Tensor:
    """MSE between `predict_next_position(...)` and the real recorded next
    position -- a plain, non-differentiable-simulator, closed-form
    realization of the paper's Position loss (differentiate through one step
    to match *resulting position*): no RL, no rollout, no live PhysX call
    anywhere in this function -- the one non-differentiable simulator query
    already happened, once per data point, offline, in `compute_dynamics.py`.
    """
    q_next_pred = predict_next_position(tau_predicted_physical, m_inv, C, G, q_t, qdot_t, dt)
    return F.mse_loss(q_next_pred, q_real_next)


def torque_loss(pred_std: torch.Tensor, label_std: torch.Tensor) -> torch.Tensor:
    """Standardized-space MSE between predicted and labeled torque.

    Both arguments must already be in the SAME standardized space (i.e.
    `label_std` should come from `GenANEnsemble.label_scaler(label, train=False)`
    and `pred_std` from `GenANEnsemble.forward_standardized(...)`) -- comparing
    in standardized space is what lets this loss train against `q_torque`
    (`gt_effort`) despite it being uncalibrated: the network only ever has to
    match the *shape* of the labeled signal in its own learned units, not an
    absolute N*m scale (see DESIGN.md, Decision 1). Do not compare
    de-standardized (raw) torques with this function -- that would silently
    assume a calibration that doesn't exist.

    `pred_std` may carry a leading ensemble dimension (ensemble_size, batch,
    num_joints); `label_std` is broadcast against it.
    """
    return F.mse_loss(pred_std, label_std.expand_as(pred_std) if pred_std.dim() > label_std.dim() else label_std)


def torque_direction_loss(pred_std: torch.Tensor, label_std: torch.Tensor) -> torch.Tensor:
    """Direction-only (cosine-similarity) loss between predicted and labeled
    torque, in place of magnitude-sensitive MSE.

    Calibration-free by construction -- invariant to any positive scale
    factor on either side, for the same reason `uan_shadowlite/reward.py`'s
    `torque_sign` reward term compares sign, not magnitude (`gt_effort` is
    uncalibrated -- see DESIGN.md, Decision 1). Cosine similarity is used
    rather than a hard `sign()`/`==` comparison (as `reward.py` uses, fine
    there since it's never backpropagated through) because `sign()` has zero
    gradient almost everywhere -- unusable as a *training* loss. For a
    single-joint (1-dim) prediction this reduces EXACTLY to sign agreement:
    cosine similarity between two scalars is the product of their signs once
    normalized, so this one function correctly covers both train_genan.py's
    (num_joints-dim) and train_genan_single.py's (1-dim) case with no
    special-casing.

    `pred_std` may carry a leading ensemble dimension (ensemble_size, batch,
    num_joints); `label_std` is broadcast against it, matching `torque_loss`.
    Returns `1 - mean(cosine_similarity)` (0 = perfect direction agreement,
    2 = perfectly opposed), summed over the last (joint) dimension.
    """
    label = label_std.expand_as(pred_std) if pred_std.dim() > label_std.dim() else label_std
    cos_sim = F.cosine_similarity(pred_std, label, dim=-1, eps=1e-8)
    return (1.0 - cos_sim).mean()

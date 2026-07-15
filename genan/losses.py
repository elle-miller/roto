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


HARDWARE_EFFORT_TO_NM = 30.0
"""Calibration factor between `q_torque`/`gt_effort`'s raw, UNCALIBRATED
hardware units (DESIGN.md Decision 1) and real N*m, as used by
`shadow_hand_lite.py`'s ImplicitActuatorCfg (`effort_limit_sim=30.0`,
identical across all 16 joints) and by `pd_baseline_torque`'s Kp/Kd
(identified in that same sim, in N*m). By user decision: `gt_effort_raw /
HARDWARE_EFFORT_TO_NM` is assumed to land in the sim's N*m scale --
confirmed empirically for rh_FFJ3 (raw abs-max ~646 -> calibrated abs-max
~21.5, comfortably under the 30 N*m limit; residual against `tau_pd`,
abs-max ~24, then has abs-max ~41 with std ~7 -- a well-conditioned RESIDUAL,
unlike subtracting `tau_pd` directly from the raw, uncalibrated label).
Without this factor, `tau_pd` (bounded to +-30 N*m by construction) can only
ever explain a tiny slice of the raw label's +-900-ish swing -- not a
normalization bug, a genuine two-unit-system mismatch.
"""


def pd_baseline_torque(
    q_cmd: torch.Tensor, q_meas: torch.Tensor, qdot_meas: torch.Tensor, kp: float, kd: float,
) -> torch.Tensor:
    """`Kp*(q_cmd - q_meas) - Kd*qdot_meas` -- the deterministic torque (N*m)
    the identified PD controller (`kp`/`kd` from `pd_gains.load_pd_gains`,
    sourced from `shadow_pd_id`, matching `shadow_hand_lite.py`'s
    ImplicitActuatorCfg) would apply. Assumes zero commanded velocity: `q_cmd`
    is a position-only command (`AlignedTrajectoryDataset` has no
    `q_cmd_vel`), so there is no target-velocity term to add.

    Used to build a RESIDUAL torque training label
    (`gt_effort/HARDWARE_EFFORT_TO_NM - pd_baseline`, both sides now in N*m --
    see `HARDWARE_EFFORT_TO_NM`) -- per user decision, `kp`/`kd` are kept
    fixed at their identified values (not learned/adjusted), so the network
    only has to learn what the known linear PD term doesn't already explain
    (nonlinear friction, backlash, coupling), not reinvent a relationship
    that's already known.
    """
    return kp * (q_cmd - q_meas) - kd * qdot_meas


def torque_minmax_loss(pred_bounded: torch.Tensor, label_raw: torch.Tensor, torque_range: float = 900.0) -> torch.Tensor:
    """MSE in a FIXED min-max normalized space, in place of the
    `RunningStandardScaler`-based (data-driven mean/std) standardization
    `torque_loss` uses.

    `label_raw` is the RAW `q_torque`/`gt_effort` value (never touches
    `label_scaler`), linearly scaled to [-1,1] by a fixed, joint-independent
    `torque_range` (confirmed against real data: global |gt_effort| max is
    ~679, so the default 900.0 has headroom -- see
    `roto/genan/agents/shadowlite/default.yaml`'s `torque_range` comment).
    `pred_bounded` is expected to already be in (-1,1) by construction --
    i.e. from a `GenAN(bounded_output=True)` member (`tanh` output) -- so
    this is a fixed, symmetric, well-conditioned space on both sides, not
    requiring any data-driven fitting.

    `pred_bounded` may carry a leading ensemble dimension (ensemble_size,
    batch, num_joints); `label_raw` is broadcast against it, matching
    `torque_loss`'s convention.
    """
    label_norm = (label_raw / torque_range).clamp(-1.0, 1.0)
    label_norm = label_norm.expand_as(pred_bounded) if pred_bounded.dim() > label_norm.dim() else label_norm
    return F.mse_loss(pred_bounded, label_norm)


def coupled_pair_activity_weights(
    q_a_now: torch.Tensor, q_a_past: torch.Tensor, q_b_now: torch.Tensor, q_b_past: torch.Tensor, eps: float = 1e-3,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Soft per-joint "activity" weights for a tendon-coupled J1/J2 mimic
    pair, derived from WINDOWED position displacement (`q_*_now - q_*_past`,
    `past` some steps back -- NOT single-step velocity). Per user decision:
    `dataset.q_meas_vel` for J1/J2 is motor-level, not a faithful per-joint
    signal (same issue as `gt_effort` being duplicated across both columns --
    see `coupled_pair_activity_loss`), so a single-step finite difference
    doesn't reliably show which joint actually moved. Displacement over a
    longer window does: `q_meas` (position) itself IS asserted independently
    faithful per-DOF (DESIGN.md), so a windowed diff of it is a legitimate
    per-joint "how much did this joint move" signal even though the
    single-step derivative isn't trustworthy.

    Returns `(activity_a, activity_b)`, each `>= 0` and summing to exactly 1
    -- when one joint is locked (near-zero displacement) and the other moves,
    activity concentrates almost entirely on the moving joint; when both move
    (the hysteresis/backlash window between the two-segment law's handoff,
    see `roto_env.py`'s `_handle_coupled_joints`), it splits proportionally
    to how much each actually displaced.
    """
    disp_a = (q_a_now - q_a_past).abs()
    disp_b = (q_b_now - q_b_past).abs()
    total = disp_a + disp_b + eps
    return disp_a / total, disp_b / total


def coupled_pair_activity_loss_terms(
    pred_bounded: torch.Tensor, label_raw: torch.Tensor, torque_range: float,
    activity_a: torch.Tensor, activity_b: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """The two components `coupled_pair_activity_loss` sums, exposed
    separately for per-epoch logging (see that function's docstring for the
    full explanation). Returns `(mse_a, mse_b)`.
    """
    label_norm = (label_raw / torque_range).clamp(-1.0, 1.0)
    share_a, share_b = pred_bounded[..., 0:1], pred_bounded[..., 1:2]
    target_a = activity_a * label_norm
    target_b = activity_b * label_norm
    target_a = target_a.expand_as(share_a) if share_a.dim() > target_a.dim() else target_a
    target_b = target_b.expand_as(share_b) if share_b.dim() > target_b.dim() else target_b
    return F.mse_loss(share_a, target_a), F.mse_loss(share_b, target_b)


def coupled_pair_activity_loss(
    pred_bounded: torch.Tensor, label_raw: torch.Tensor, torque_range: float,
    activity_a: torch.Tensor, activity_b: torch.Tensor,
) -> torch.Tensor:
    """Loss for a tendon-coupled J1/J2 mimic pair (e.g. rh_FFJ1/rh_FFJ2): ONE
    real motor drives both DOFs on hardware, so `gt_effort` is recorded
    identically for both joints (verified empirically: 100% bit-identical
    across the whole real dataset for all three FF/MF/RF pairs) -- but sim
    actuates them as two independent DOFs (`convert_mimic_joints_to_normal_joints:
    false`), so training needs to predict two separate per-joint torque
    "shares" that reconstruct the one real shared signal when summed.

    Unlike a sum-only loss (which has no signal about WHICH joint should
    carry more of the shared torque at a given instant), this directly
    supervises each share against an activity-weighted pseudo-label:
    `target_a = activity_a * label_norm`, `target_b = activity_b *
    label_norm` (see `coupled_pair_activity_weights`) -- these sum to
    `label_norm` EXACTLY by construction, so fitting both shares well
    automatically satisfies the sum constraint too, and since `activity_a`/
    `activity_b` are both `>= 0`, `target_a`/`target_b` always carry the SAME
    sign as `label_norm` -- well-fit shares structurally can never end up
    opposite-signed, eliminating the degenerate-cancellation failure mode a
    separate direction-agreement penalty previously had to fight.

    `pred_bounded`: (..., 2), each of the 2 columns INDEPENDENTLY tanh-bounded
    to (-1,1) (from `GenAN(bounded_output=True, num_joints=2)` -- see
    `model.py`, no architecture change needed for this). `label_raw`: (..., 1),
    the single shared real `gt_effort` value (either joint's `q_torque`
    column -- they're identical, see above). `activity_a`/`activity_b`:
    (..., 1) each, from `coupled_pair_activity_weights`.

    `pred_bounded` may carry a leading ensemble dimension (ensemble_size,
    batch, 2); `label_raw`/`activity_a`/`activity_b` are broadcast against it,
    matching `torque_minmax_loss`'s convention.
    """
    mse_a, mse_b = coupled_pair_activity_loss_terms(pred_bounded, label_raw, torque_range, activity_a, activity_b)
    return mse_a + mse_b


def coupled_pair_hinge_direction_loss(pred_bounded: torch.Tensor) -> torch.Tensor:
    """Hinge-style direction penalty between the two shares: zero when they
    agree in sign (or either is ~0), grows linearly WITH the magnitude of
    disagreement when opposed. Kept as an EXTRA safety net alongside
    `coupled_pair_activity_loss` (per user decision) even though the
    activity-weighted targets already push both shares toward the same sign
    as the label in the well-fit limit -- this adds robustness early in
    training and in the ambiguous hysteresis window where `activity_a`/
    `activity_b` sit close to 0.5/0.5 and so provide only a weak same-sign
    pull on their own. NOT cosine similarity: cosine similarity for two
    1-dim "vectors" only sees SIGN (not magnitude of disagreement) and has a
    weak/unstable gradient near zero -- see this session's earlier finding
    that let a cosine-similarity version get stuck at near-total opposition
    (val_direction ~1.92/2.0) even at weight=0.1. `relu(-share_a*share_b)`
    doesn't have that failure mode.
    """
    share_a, share_b = pred_bounded[..., 0:1], pred_bounded[..., 1:2]
    return F.relu(-share_a * share_b).mean()


def coupled_pair_loss(
    pred_bounded: torch.Tensor, label_raw: torch.Tensor, torque_range: float,
    activity_a: torch.Tensor, activity_b: torch.Tensor, direction_penalty_weight: float = 0.0,
) -> torch.Tensor:
    """`coupled_pair_activity_loss(...) + direction_penalty_weight *
    coupled_pair_hinge_direction_loss(pred_bounded)` -- per user decision, NO
    separate sum-matching term (the activity-weighted per-share MSE already
    subsumes it: `target_a + target_b == label_norm` exactly by
    construction), just the per-share loss plus the hinge safety net.
    """
    activity_loss = coupled_pair_activity_loss(pred_bounded, label_raw, torque_range, activity_a, activity_b)
    hinge_loss = coupled_pair_hinge_direction_loss(pred_bounded)
    return activity_loss + direction_penalty_weight * hinge_loss

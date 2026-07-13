"""CPU-only unit tests for roto.tasks.uan_shadowlite.reward.

No Isaac Sim / isaaclab import anywhere in this file or in reward.py.
"""

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "roto", "tasks", "uan_shadowlite"))

from reward import compute_uan_reward, soft_limit_avoidance  # noqa: E402


# ---------------------------------------------------------------------------
# soft_limit_avoidance -- the joint-limit safety envelope
# ---------------------------------------------------------------------------


def test_full_torque_passes_through_in_the_middle_of_the_range():
    """Far from either limit, torque of either sign is completely unaffected."""
    torque = torch.tensor([-5.0, 5.0])
    pos = torch.tensor([0.0, 0.0])
    lower = torch.tensor([-1.0, -1.0])
    upper = torch.tensor([1.0, 1.0])
    out = soft_limit_avoidance(torque, pos, lower, upper, margin=0.1)
    assert torch.allclose(out, torque)


def test_outward_torque_is_zeroed_exactly_at_the_limit():
    """At the lower bound, negative (outward) torque -> 0. At the upper bound,
    positive (outward) torque -> 0."""
    torque = torch.tensor([-5.0, 5.0])
    pos = torch.tensor([-1.0, 1.0])  # exactly at lower / exactly at upper
    lower = torch.tensor([-1.0, -1.0])
    upper = torch.tensor([1.0, 1.0])
    out = soft_limit_avoidance(torque, pos, lower, upper, margin=0.1)
    assert torch.allclose(out, torch.zeros(2))


def test_inward_torque_is_never_touched_even_at_the_limit():
    """At the lower bound, POSITIVE (inward/recovery) torque must pass through
    unchanged -- the envelope never opposes a recovery back into range."""
    torque = torch.tensor([5.0, -5.0])
    pos = torch.tensor([-1.0, 1.0])  # at lower / at upper
    lower = torch.tensor([-1.0, -1.0])
    upper = torch.tensor([1.0, 1.0])
    out = soft_limit_avoidance(torque, pos, lower, upper, margin=0.1)
    assert torch.allclose(out, torque)


def test_scale_is_linear_within_the_margin_band():
    """Halfway through the margin band, outward torque is scaled by exactly 0.5."""
    margin = 0.2
    torque = torch.tensor([-4.0])
    lower = torch.tensor([-1.0])
    upper = torch.tensor([1.0])
    pos = lower + margin / 2  # halfway into the margin band from the lower bound
    out = soft_limit_avoidance(torque, pos, lower, upper, margin=margin)
    assert out.item() == pytest.approx(-2.0)


def test_scale_is_1_at_exactly_one_margin_away_from_the_limit():
    margin = 0.15
    torque = torch.tensor([-3.0])
    lower = torch.tensor([-1.0])
    upper = torch.tensor([1.0])
    pos = lower + margin
    out = soft_limit_avoidance(torque, pos, lower, upper, margin=margin)
    assert out.item() == pytest.approx(-3.0, abs=1e-5)


def test_beyond_the_limit_torque_stays_clamped_to_zero_not_negative():
    """If pos has somehow already gone past the bound, outward torque should still
    clamp to 0 (not flip sign / become "extra" outward), since room fraction is
    clamped to [0,1] rather than allowed to go negative."""
    torque = torch.tensor([-5.0])
    pos = torch.tensor([-1.5])  # already past the lower bound of -1.0
    lower = torch.tensor([-1.0])
    upper = torch.tensor([1.0])
    out = soft_limit_avoidance(torque, pos, lower, upper, margin=0.1)
    assert out.item() == pytest.approx(0.0)


def test_zero_torque_is_unaffected_regardless_of_position():
    torque = torch.tensor([0.0])
    pos = torch.tensor([-1.0])  # at the limit
    lower = torch.tensor([-1.0])
    upper = torch.tensor([1.0])
    out = soft_limit_avoidance(torque, pos, lower, upper, margin=0.1)
    assert out.item() == pytest.approx(0.0)


def test_batched_envs_and_joints():
    """Shape (num_envs, num_joints) works, not just 1D."""
    torque = torch.tensor([[-5.0, 5.0], [5.0, -5.0]])
    pos = torch.tensor([[-1.0, 1.0], [-1.0, 1.0]])
    lower = torch.tensor([-1.0, -1.0])
    upper = torch.tensor([1.0, 1.0])
    out = soft_limit_avoidance(torque, pos, lower, upper, margin=0.1)
    # row 0: both outward -> both zeroed. row 1: both inward -> both unchanged.
    assert torch.allclose(out[0], torch.zeros(2))
    assert torch.allclose(out[1], torque[1])


# ---------------------------------------------------------------------------
# compute_uan_reward -- sanity checks (extraction into reward.py shouldn't change behavior)
# ---------------------------------------------------------------------------


def _reward_kwargs(n=1, **overrides):
    # n=1 by default so a test overriding just one or two tensor kwargs (e.g. only
    # q_real, or only torque_sim/torque_real) can't silently end up with mismatched
    # batch dimensions across arguments -- pass n= explicitly for a batched test.
    defaults = dict(
        q_real=torch.zeros(n, 4),
        q_sim=torch.zeros(n, 4),
        actions=torch.zeros(n, 4),
        last_actions=torch.zeros(n, 4),
        torque_sim=torch.ones(n, 4),
        torque_real=torch.ones(n, 4),
        survival=0.0,
        l1=-1.5,
        exp_l2_loose=4.0,
        coef_loose=100.0,
        exp_l2=4.0,
        coef_l2=300.0,
        exp_l2_strict=5.0,
        coef_strict=1000.0,
        exp_action_rate=0.5,
        coef_action_rate=0.5,
        torque_sign=0.0,
    )
    defaults.update(overrides)
    return defaults


def test_perfect_tracking_gives_max_bounded_reward_terms():
    reward, se_sum, ae_sum = compute_uan_reward(**_reward_kwargs())
    assert se_sum.abs().max().item() == pytest.approx(0.0)
    assert ae_sum.abs().max().item() == pytest.approx(0.0)
    # l1 term is 0 (ae_sum=0); all three exp terms are at their max (exp(0)=1); action_rate=0.
    expected = 0.0 + 4.0 * 1.0 + 4.0 * 1.0 + 5.0 * 1.0 + 0.5 * 1.0
    assert reward[0].item() == pytest.approx(expected)


def test_torque_sign_term_is_inert_when_weight_is_zero():
    kwargs = _reward_kwargs(torque_sim=torch.ones(1, 4), torque_real=-torch.ones(1, 4), torque_sign=0.0)
    reward, _, _ = compute_uan_reward(**kwargs)
    kwargs2 = _reward_kwargs(torque_sim=torch.ones(1, 4), torque_real=torch.ones(1, 4), torque_sign=0.0)
    reward2, _, _ = compute_uan_reward(**kwargs2)
    # opposite sign_agree (0.0 vs 1.0) but torque_sign=0 weight -> identical reward
    assert torch.allclose(reward, reward2)


def test_torque_sign_term_rewards_sign_agreement_when_enabled():
    agree = _reward_kwargs(torque_sim=torch.ones(1, 4), torque_real=torch.ones(1, 4), torque_sign=1.0)
    disagree = _reward_kwargs(torque_sim=torch.ones(1, 4), torque_real=-torch.ones(1, 4), torque_sign=1.0)
    r_agree, _, _ = compute_uan_reward(**agree)
    r_disagree, _, _ = compute_uan_reward(**disagree)
    assert r_agree.item() > r_disagree.item()


def test_larger_error_gives_lower_reward():
    small_err = _reward_kwargs(q_real=torch.full((1, 4), 0.01))
    big_err = _reward_kwargs(q_real=torch.full((1, 4), 1.0))
    r_small, _, _ = compute_uan_reward(**small_err)
    r_big, _, _ = compute_uan_reward(**big_err)
    assert r_small.item() > r_big.item()

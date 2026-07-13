"""CPU-only unit tests for roto.tasks.uan_shadowlite.reward.

No Isaac Sim / isaaclab import anywhere in this file or in reward.py.
"""

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "roto", "tasks", "uan_shadowlite"))

from reward import compute_uan_reward  # noqa: E402


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

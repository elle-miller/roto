"""CPU-only unit tests for roto.genan.model.

No `isaaclab` import anywhere in this file, `model.py`, or anything it pulls
in from `multimodal_rl` (verified separately -- see DESIGN.md).
"""

import copy
import os
import sys

import torch

_GENAN_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "genan")
sys.path.insert(0, _GENAN_DIR)

from model import GenAN, GenANEnsemble  # noqa: E402

INPUT_DIM = 2 * 4 * 16  # 2 streams (q, u) * (history_len=3 + 1) frames * 16 joints
NUM_JOINTS = 16


def test_genan_forward_shape():
    net = GenAN(INPUT_DIM, NUM_JOINTS)
    x = torch.randn(5, INPUT_DIM)
    out = net(x)
    assert out.shape == (5, NUM_JOINTS)


def test_ensemble_forward_shape_and_scalers():
    ens = GenANEnsemble(INPUT_DIM, NUM_JOINTS, ensemble_size=5, seed=0)
    raw_input = torch.randn(200, INPUT_DIM) * 3 + 1
    torque_label = torch.randn(200, NUM_JOINTS) * 10 - 2
    ens.fit_scalers(raw_input, torque_label)

    preds = ens.forward(raw_input[:8])
    assert preds.shape == (5, 8, NUM_JOINTS)

    preds_std = ens.forward_standardized(raw_input[:8])
    assert preds_std.shape == (5, 8, NUM_JOINTS)


def test_sample_member_matches_exactly_one_member():
    ens = GenANEnsemble(INPUT_DIM, NUM_JOINTS, ensemble_size=5, seed=1)
    raw_input = torch.randn(50, INPUT_DIM)
    torque_label = torch.randn(50, NUM_JOINTS)
    ens.fit_scalers(raw_input, torque_label)

    x = torch.randn(4, INPUT_DIM)
    all_preds = ens.forward(x)  # (5, 4, num_joints)

    gen = torch.Generator().manual_seed(42)
    sampled = ens.sample_member(x, generator=gen)
    assert sampled.shape == (4, NUM_JOINTS)
    # Exactly one ensemble member's de-standardized output must match the sample.
    matches = [torch.allclose(sampled, all_preds[i], atol=1e-5) for i in range(5)]
    assert sum(matches) == 1


def test_disagreement_is_zero_when_members_share_weights():
    ens = GenANEnsemble(INPUT_DIM, NUM_JOINTS, ensemble_size=3, seed=2)
    # Force every member to share member 0's weights -- disagreement must vanish.
    ref_state = copy.deepcopy(ens.members[0].state_dict())
    for member in ens.members:
        member.load_state_dict(ref_state)

    raw_input = torch.randn(50, INPUT_DIM)
    torque_label = torch.randn(50, NUM_JOINTS)
    ens.fit_scalers(raw_input, torque_label)

    x = torch.randn(6, INPUT_DIM)
    disagreement = ens.disagreement(x)
    assert disagreement.shape == (6, NUM_JOINTS)
    assert torch.allclose(disagreement, torch.zeros_like(disagreement), atol=1e-5)


def test_ensemble_members_differ_with_different_seeds():
    ens = GenANEnsemble(INPUT_DIM, NUM_JOINTS, ensemble_size=5, seed=3)
    w0 = ens.members[0].head.weight
    w1 = ens.members[1].head.weight
    assert not torch.allclose(w0, w1)

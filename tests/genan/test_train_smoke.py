"""CPU-only smoke test for roto/genan/train_genan.py against a tiny synthetic
aligned dataset (same `.aligned.npz` format `test_dataset.py` synthesizes),
confirming the training loop actually reduces loss and that a saved
checkpoint round-trips to identical predictions.
"""

import os
import sys

import numpy as np
import torch

_GENAN_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "genan")
sys.path.insert(0, _GENAN_DIR)

from dataset_loader import AlignedTrajectoryDataset  # noqa: E402
from joint_config import load_joint_config  # noqa: E402
from model import GenANEnsemble  # noqa: E402
from train_genan import build_inputs_and_labels, train  # noqa: E402

JOINT_NAMES, JOINT_UPPER_LIMITS = load_joint_config()
# 13-entry actuator_order matching joints.yaml's policy_joint_order, but with
# the 3 coupled J2 drivers renamed to their shared "J0" motor channel, exactly
# like uan_shadowlite/tests/test_dataset.py's own ACTUATOR_ORDER_13.
ACTUATOR_ORDER_13 = [
    "rh_FFJ0", "rh_FFJ3", "rh_FFJ4",
    "rh_MFJ0", "rh_MFJ3", "rh_MFJ4",
    "rh_RFJ0", "rh_RFJ3", "rh_RFJ4",
    "rh_THJ1", "rh_THJ2", "rh_THJ4", "rh_THJ5",
]


def _write_synthetic_episode(path, n, seed, rate=60.0):
    rng = np.random.default_rng(seed)
    action = rng.uniform(-0.3, 0.3, (n, 13))
    gt_pos = rng.uniform(-0.3, 0.3, (n, 16)).astype(np.float64)
    gt_vel = rng.normal(0, 0.1, (n, 16))
    # Torque label is a smooth, learnable function of position + a small
    # per-joint bias so a trained model has something real to converge to
    # (pure noise would make "loss decreases" a coin flip).
    gt_effort = 5.0 * gt_pos + rng.normal(0, 0.05, (n, 16))
    gt_tactile = np.zeros((n, 64))
    valid = np.ones(n, dtype=bool)
    seg_id = np.zeros(n, dtype=np.int64)
    np.savez(
        path,
        t=np.arange(n) / rate, act_pos=action, act_err=action, act_vel=action, action=action,
        gt_pos=gt_pos, gt_vel=gt_vel, gt_effort=gt_effort, gt_tactile=gt_tactile,
        command=action, valid=valid, seg_id=seg_id,
        actuator_order=np.array(ACTUATOR_ORDER_13), joint_order=np.array(JOINT_NAMES),
        dataset_rate=np.float64(rate),
    )


def _make_dataset(tmp_path, num_episodes=6, n=200):
    for i in range(num_episodes):
        _write_synthetic_episode(tmp_path / f"ep{i}.aligned.npz", n=n, seed=i)
    return AlignedTrajectoryDataset(
        paths=str(tmp_path), joint_names=JOINT_NAMES, device="cpu",
        joint_upper_limits=JOINT_UPPER_LIMITS, min_horizon=1,
    )


def test_train_reduces_validation_loss(tmp_path):
    dataset = _make_dataset(tmp_path)
    _, history_log = train(
        dataset, history_len=2, stride=1, ensemble_size=2, epochs=30, batch_size=256,
        lr=1e-3, val_frac=0.3, patience=1000, seed=0,
    )
    assert history_log["val_loss"][-1] < history_log["val_loss"][0]
    assert history_log["best_val_loss"] < history_log["val_loss"][0]


def test_checkpoint_round_trip(tmp_path):
    dataset = _make_dataset(tmp_path)
    ensemble, _ = train(
        dataset, history_len=2, stride=1, ensemble_size=2, epochs=5, batch_size=256,
        lr=1e-3, val_frac=0.3, patience=1000, seed=0,
    )
    ensemble.eval()

    train_t, _ = None, None  # not needed; build a fixed probe input directly
    probe_t = dataset.clamp(torch.arange(0, 10))
    x_probe, _ = build_inputs_and_labels(dataset, probe_t, history_len=2, stride=1)
    with torch.no_grad():
        preds_before = ensemble.forward(x_probe)

    ckpt_path = tmp_path / "ckpt.pt"
    torch.save(
        {
            "ensemble_state_dict": ensemble.state_dict(),
            "input_dim": x_probe.shape[1],
            "num_joints": dataset.num_joints,
            "ensemble_size": ensemble.ensemble_size,
            "history_len": 2,
            "stride": 1,
            "joint_names": JOINT_NAMES,
        },
        ckpt_path,
    )

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    reloaded = GenANEnsemble(ckpt["input_dim"], ckpt["num_joints"], ensemble_size=ckpt["ensemble_size"])
    reloaded.load_state_dict(ckpt["ensemble_state_dict"])
    reloaded.eval()

    with torch.no_grad():
        preds_after = reloaded.forward(x_probe)

    assert torch.allclose(preds_before, preds_after, atol=1e-6)

"""CPU-only unit tests for roto.tasks.uan_shadowlite.dataset.

No Isaac Sim / isaaclab import anywhere in this file or in dataset.py --
run with any environment that has torch+numpy+pytest (e.g. the `icra`
conda env), not necessarily the one isaaclab is installed in.
"""

import os
import sys

import numpy as np
import pytest
import torch

# Import dataset.py directly as a standalone module (bypassing
# roto/roto/tasks/uan_shadowlite/__init__.py, which pulls in task.py ->
# isaaclab). dataset.py has zero internal package dependencies.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "roto", "tasks", "uan_shadowlite"))

from dataset import AlignedTrajectoryDataset, DatasetKeys, TrajectoryDataset  # noqa: E402

TARGET_NAMES_16 = [
    "rh_FFJ1", "rh_FFJ2", "rh_FFJ3", "rh_FFJ4",
    "rh_MFJ1", "rh_MFJ2", "rh_MFJ3", "rh_MFJ4",
    "rh_RFJ1", "rh_RFJ2", "rh_RFJ3", "rh_RFJ4",
    "rh_THJ1", "rh_THJ2", "rh_THJ4", "rh_THJ5",
]
ACTUATOR_ORDER_13 = [
    "rh_FFJ0", "rh_FFJ3", "rh_FFJ4",
    "rh_MFJ0", "rh_MFJ3", "rh_MFJ4",
    "rh_RFJ0", "rh_RFJ3", "rh_RFJ4",
    "rh_THJ1", "rh_THJ2", "rh_THJ4", "rh_THJ5",
]


def _write_aligned_npz(path, n, seg_id=None, valid=None, rate=60.0, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(n, dtype=np.float64) / rate
    action = rng.uniform(-1, 1, (n, 13))
    command = rng.uniform(-500, 500, (n, 13))
    act_pos = action + rng.normal(0, 0.01, (n, 13))
    act_err = rng.normal(0, 0.01, (n, 13))
    act_vel = rng.normal(0, 0.1, (n, 13))
    gt_pos = rng.uniform(-1, 1, (n, 16))
    gt_vel = rng.normal(0, 0.1, (n, 16))
    gt_effort = rng.uniform(-100, 100, (n, 16))
    gt_tactile = np.zeros((n, 64))
    if seg_id is None:
        seg_id = np.zeros(n, dtype=np.int64)
    if valid is None:
        valid = np.ones(n, dtype=bool)
    np.savez(
        path,
        t=t, act_pos=act_pos, act_err=act_err, act_vel=act_vel, action=action,
        gt_pos=gt_pos, gt_vel=gt_vel, gt_effort=gt_effort, gt_tactile=gt_tactile,
        command=command, valid=valid, seg_id=seg_id,
        actuator_order=np.array(ACTUATOR_ORDER_13), joint_order=np.array(TARGET_NAMES_16),
        dataset_rate=np.float64(rate),
    )
    return dict(action=action, gt_pos=gt_pos, gt_vel=gt_vel, gt_effort=gt_effort)


def test_aligned_directly_driven_joints_use_action_as_target(tmp_path):
    path = tmp_path / "ep1.aligned.npz"
    raw = _write_aligned_npz(path, 20)
    ds = AlignedTrajectoryDataset(str(path), TARGET_NAMES_16, device="cpu")

    # rh_FFJ3 is directly driven (in both actuator_order and joint_order) -> q_cmd == action
    ffj3_act_col = ACTUATOR_ORDER_13.index("rh_FFJ3")
    ffj3_joint_col = TARGET_NAMES_16.index("rh_FFJ3")
    expected = raw["action"][:, ffj3_act_col]
    assert torch.allclose(ds.q_cmd[:, ffj3_joint_col], torch.tensor(expected, dtype=torch.float32), atol=1e-5)


def test_aligned_coupled_joints_use_measured_position_as_target(tmp_path):
    path = tmp_path / "ep1.aligned.npz"
    raw = _write_aligned_npz(path, 20)
    ds = AlignedTrajectoryDataset(str(path), TARGET_NAMES_16, device="cpu")

    # rh_FFJ1/rh_FFJ2 are the coupled pair (not directly in actuator_order) -> q_cmd == gt_pos
    for name in ["rh_FFJ1", "rh_FFJ2"]:
        j = TARGET_NAMES_16.index(name)
        expected = raw["gt_pos"][:, j]
        assert torch.allclose(ds.q_cmd[:, j], torch.tensor(expected, dtype=torch.float32), atol=1e-5)


def test_aligned_q_meas_and_q_torque_are_gt_pos_and_gt_effort(tmp_path):
    path = tmp_path / "ep1.aligned.npz"
    raw = _write_aligned_npz(path, 20)
    ds = AlignedTrajectoryDataset(str(path), TARGET_NAMES_16, device="cpu")
    assert torch.allclose(ds.q_meas, torch.tensor(raw["gt_pos"], dtype=torch.float32), atol=1e-5)
    assert torch.allclose(ds.q_torque, torch.tensor(raw["gt_effort"], dtype=torch.float32), atol=1e-5)


def test_aligned_directory_glob_finds_all_files(tmp_path):
    for i in range(3):
        _write_aligned_npz(tmp_path / f"ep{i}.aligned.npz", 15, seed=i)
    ds = AlignedTrajectoryDataset(str(tmp_path), TARGET_NAMES_16, device="cpu", min_horizon=1)
    assert len(ds.paths) == 3
    assert ds.num_steps == 45
    assert ds.traj_starts.tolist() == [0, 15, 30]
    assert ds.traj_ends.tolist() == [14, 29, 44]


def test_aligned_missing_joint_in_joint_order_raises(tmp_path):
    path = tmp_path / "ep1.aligned.npz"
    _write_aligned_npz(path, 10)
    with pytest.raises(KeyError):
        AlignedTrajectoryDataset(str(path), TARGET_NAMES_16 + ["rh_NOPE"], device="cpu")


def test_aligned_dataset_rate_becomes_rl_dt(tmp_path):
    path = tmp_path / "ep1.aligned.npz"
    _write_aligned_npz(path, 10, rate=60.0)
    ds = AlignedTrajectoryDataset(str(path), TARGET_NAMES_16, device="cpu")
    assert ds.rl_dt == pytest.approx(1.0 / 60.0)


def test_aligned_segments_on_seg_id_change_and_valid_gaps(tmp_path):
    n = 30
    seg_id = np.zeros(n, dtype=np.int64)
    seg_id[15:] = 1
    valid = np.ones(n, dtype=bool)
    valid[10:12] = False  # a gap inside segment 0
    path = tmp_path / "ep1.aligned.npz"
    _write_aligned_npz(path, n, seg_id=seg_id, valid=valid)
    ds = AlignedTrajectoryDataset(str(path), TARGET_NAMES_16, device="cpu", min_horizon=1)
    # segment 0 gets trimmed to [0,9] (stopping before the invalid gap at 10-11),
    # segment 1 [12,14] is the remainder before seg_id changes at 15,
    # segment 2 [15,29] is the seg_id==1 run.
    assert ds.traj_starts.tolist() == [0, 12, 15]
    assert ds.traj_ends.tolist() == [9, 14, 29]


def test_aligned_mismatched_dataset_rate_raises(tmp_path):
    p1, p2 = tmp_path / "a.aligned.npz", tmp_path / "b.aligned.npz"
    _write_aligned_npz(p1, 10, rate=60.0)
    _write_aligned_npz(p2, 10, rate=30.0)
    with pytest.raises(ValueError):
        AlignedTrajectoryDataset([str(p1), str(p2)], TARGET_NAMES_16, device="cpu")


def test_aligned_glob_pattern_path(tmp_path):
    for i in range(2):
        _write_aligned_npz(tmp_path / f"ep{i}.aligned.npz", 10, seed=i)
    pattern = str(tmp_path / "*.aligned.npz")
    ds = AlignedTrajectoryDataset(pattern, TARGET_NAMES_16, device="cpu", min_horizon=1)
    assert ds.num_steps == 20


# ---------------------------------------------------------------------------
# Legacy loader (kept for backward compatibility) -- basic smoke coverage
# ---------------------------------------------------------------------------

TARGET_NAMES_4 = [f"j{i}" for i in range(4)]


def _write_legacy_npz(path, cmd, meas, names, ends, dt=1.0 / 60.0, keys=None):
    keys = keys or DatasetKeys()
    np.savez(
        path,
        **{
            keys.cmd: cmd.astype(np.float32),
            keys.meas: meas.astype(np.float32),
            keys.names: np.array(names),
            keys.ends: np.array(ends, dtype=np.int32),
            keys.dt: np.float32(dt),
        },
    )


def test_legacy_name_permutation_reorders_shuffled_columns(tmp_path):
    recorded_order = ["j2", "j0", "j3", "j1"]
    t = np.arange(10, dtype=np.float32)
    cmd = np.stack([100 * c + t for c in range(4)], axis=1)
    path = tmp_path / "toy.npz"
    _write_legacy_npz(path, cmd, cmd.copy(), recorded_order, ends=[9])
    ds = TrajectoryDataset(str(path), TARGET_NAMES_4, device="cpu")
    expected_j0 = 100 * 1 + t
    assert torch.allclose(ds.q_cmd[:, 0], torch.tensor(expected_j0))


def test_legacy_episode_ends_inclusive(tmp_path):
    n = 12
    cmd = np.tile(np.arange(n, dtype=np.float32).reshape(-1, 1), (1, 4))
    path = tmp_path / "toy.npz"
    _write_legacy_npz(path, cmd, cmd, TARGET_NAMES_4, ends=[4, 11])
    ds = TrajectoryDataset(str(path), TARGET_NAMES_4, device="cpu", min_horizon=1)
    assert ds.traj_starts.tolist() == [0, 5]
    assert ds.traj_ends.tolist() == [4, 11]


def test_legacy_q_torque_is_zero_filled(tmp_path):
    n = 10
    cmd = np.zeros((n, 4), dtype=np.float32)
    path = tmp_path / "toy.npz"
    _write_legacy_npz(path, cmd, cmd, TARGET_NAMES_4, ends=[n - 1])
    ds = TrajectoryDataset(str(path), TARGET_NAMES_4, device="cpu")
    assert torch.all(ds.q_torque == 0.0)

"""Isaac-free ShadowLite joint order/limits, loaded from the single source of
truth: `roto/shadow_pd_id/config/joints.yaml`. That file's own module comment
explains why -- roto's scripts already had three independent hardcoded copies
of this data before it existed; this would otherwise be a fourth.
"""

from __future__ import annotations

import os

import yaml

_DEFAULT_JOINTS_YAML = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "shadow_pd_id", "config", "joints.yaml")
)


def load_joint_config(path: str | None = None) -> tuple[list[str], dict[str, float]]:
    """Return (16 joint names in hardware order, {name: upper_limit_rad}).

    The joint name order matches `hardware_joint_order` in `joints.yaml`,
    which is the order `AlignedTrajectoryDataset`'s aligned `.npz` recordings
    use for `gt_pos`/`gt_vel`/`gt_effort` (`joint_order`).
    """
    path = path or _DEFAULT_JOINTS_YAML
    with open(path) as f:
        cfg = yaml.safe_load(f)
    joint_names = [str(n) for n in cfg["hardware_joint_order"]]
    joint_upper_limits = {name: float(limits["upper"]) for name, limits in cfg["joint_limits_rad"].items()}
    return joint_names, joint_upper_limits

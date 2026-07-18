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

    The 3 coupled driver joints' (rh_FFJ2/rh_MFJ2/rh_RFJ2) limit is
    overridden from `coupled_groups[*].driver_joint_sim_upper_rad`, NOT
    `joint_limits_rad` -- the latter intentionally stores shadow_pd_id's
    hardware-excitation "proxy range" (1.5708 rad), which is narrower than
    the real physical/sim joint limit (1.745 rad, verified against
    SHADOW_TOUCHLAB.urdf). `AlignedTrajectoryDataset`'s coupled-command split
    (`_split_coupled_command`) needs the REAL limit to match what
    `task.py`'s live `UANShadowLiteEnv` uses (`self.robot_joint_pos_upper_limits`)
    -- using the narrower proxy range here silently saturates J2 ~10deg too
    early and hands the remainder to J1 too early, a mismatch between what
    GenAN is trained on and what it sees at deployment.
    """
    path = path or _DEFAULT_JOINTS_YAML
    with open(path) as f:
        cfg = yaml.safe_load(f)
    joint_names = [str(n) for n in cfg["hardware_joint_order"]]
    joint_upper_limits = {name: float(limits["upper"]) for name, limits in cfg["joint_limits_rad"].items()}
    for group in cfg.get("coupled_groups", []):
        if "driver_joint_sim_upper_rad" in group:
            joint_upper_limits[group["driver_joint"]] = float(group["driver_joint_sim_upper_rad"])
    return joint_names, joint_upper_limits

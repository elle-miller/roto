"""Isaac-free per-joint Kp/Kd, loaded from shadow_pd_id's identified gains
(`roto/shadow_pd_id/results/params/<joint>_gains.yaml`) -- the same source
`roto/roto/assets/shadow_hand_lite.py`'s `stiffness`/`damping` dicts were
copy-pasted from (see that file's comment directly above its `stiffness`
dict). Only `kp`/`kd` are read here -- the separately-identified Coulomb/
viscous friction (`fc`/`fv`) in those same yaml files is intentionally left
out, per user decision: it stays part of what a residual-torque label has to
capture, not something subtracted out ahead of time.
"""

from __future__ import annotations

import os

import yaml

_DEFAULT_GAINS_DIR = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "shadow_pd_id", "results", "params")
)

# J1 mimic joints (rh_FFJ1/rh_MFJ1/rh_RFJ1) have no independent command, so
# shadow_pd_id never identified them directly -- shadow_hand_lite.py's own
# convention (see its stiffness dict comment): same Kp/Kd as their driver J2
# (same finger, same physical actuator/tendon driving both).
_J1_ALIAS = {"rh_FFJ1": "rh_FFJ2", "rh_MFJ1": "rh_MFJ2", "rh_RFJ1": "rh_RFJ2"}


def load_pd_gains(joint_name: str, gains_dir: str | None = None) -> tuple[float, float]:
    """Return (kp, kd) for `joint_name`, matching shadow_hand_lite.py's
    ImplicitActuatorCfg stiffness/damping values exactly (same source yaml,
    just not truncated to 4 decimal places).
    """
    gains_dir = gains_dir or _DEFAULT_GAINS_DIR
    source_name = _J1_ALIAS.get(joint_name, joint_name)
    path = os.path.join(gains_dir, f"{source_name}_gains.yaml")
    with open(path) as f:
        gains = yaml.safe_load(f)
    return float(gains["kp"]), float(gains["kd"])

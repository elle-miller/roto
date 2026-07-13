"""Config-driven MLP input construction for the UAN residual-torque policy.

Default vocabulary: `joint_pos` (16, per kinematic DOF), `joint_vel` (13, per
physical motor), `joint_pos_error` (16, per kinematic DOF), `action` (13, per
physical motor) = 58-dim total per frame.

Why 16 for position but 13 for velocity/action: ShadowLite has 16 kinematic
DOFs but only 13 independent motors -- 3 finger pairs (FF/MF/RF) each share
one motor driving both their J1 and J2 via a tendon coupling. *Position* is
still meaningfully independent per DOF (J1 and J2 have distinct instantaneous
angles even though one motor drives both), which is why `joint_pos`/
`joint_pos_error` stay 16-dim. *Velocity*, however, is a per-motor quantity
here -- real hardware doesn't report J1/J2 velocity independently (confirmed
empirically: the aligned recordings' gt_vel duplicates the same value across
a coupled pair's two joints), and more fundamentally there's only one
independent velocity per physical motor to report, so `joint_vel` is taken at
the 13-motor (`control_dof_indices`) level, matching hardware reality and
avoiding the duplicate-value artifact entirely (task.py slices roto's own
`normalised_joint_vel` to `control_dof_indices` rather than
`actuated_dof_indices` -- still always live simulated state, not privileged
replay data, so this is a slicing choice, not a new information source).

`action` is also 13-dim, for a different but related reason: it represents
the physical motor command -- during UAN's own training that's "the 13-motor
command implied by the current PD target" (back-solved via roto's own
`unscale()`, the exact inverse of the `scale()` call that turns a policy's
raw action into a joint target); during a future downstream policy embedding
(the trained, frozen UAN sitting on top of a Bounce/Baoding-style policy),
it's literally that policy's own `self.actions` (also 13-dim, roto's
standard control-action width). Deriving it the same way in both cases keeps
the network's input contract identical regardless of context -- see task.py.

Task.py fills `FeatureContext` directly from roto's own inherited buffers
(`self.normalised_joint_pos`, `self.normalised_joint_vel`, `self.joint_pos_error`)
rather than recomputing normalization here -- this *is* "normalize as
normalized in roto": the same `unscale()` call roto's own
`_compute_intermediate_values` already runs, reused verbatim.

Temporal history (multi-frame stacking) is intentionally not built here --
`observations.obs_stack` in yaml (consumed by roto's own `FrameStack`
wrapper) provides that, so `FeatureBuilder` only ever describes one frame.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

# Per-feature output width, keyed to whichever count applies:
#   "num_joints"  -> 16 kinematic DOFs (actuated_dof_indices)
#   "num_control" -> 13 physical motors (control_dof_indices)
_FEATURE_DIMS = {
    "joint_pos": "num_joints",  # normalised_joint_pos (roto's unscale(), [-1,1]), 16-dim
    "joint_vel": "num_control",  # normalised_joint_vel at the 13-motor level, [-1,1]
    "joint_pos_error": "num_joints",  # raw joint_pos_cmd - joint_pos (radians, unnormalized), 16-dim
    "action": "num_control",  # 13-motor command implied by joint_pos_cmd (see module docstring)
    "last_residual": "num_joints",  # previous step's applied residual torque (N*m, raw), 16-dim
    "real_vel": "num_joints",  # dataset finite-diff velocity -- privileged/replay-only, see DESIGN.md
    "sin_time": 1,
    "cos_time": 1,
}

# The default, spec-matching feature set: 16 + 13 + 16 + 13 = 58-dim.
DEFAULT_FEATURES = ["joint_pos", "joint_vel", "joint_pos_error", "action"]


@dataclass
class FeatureContext:
    """Per-step state the env fills in before calling `FeatureBuilder.build`.

    `joint_pos` and `joint_pos_error` are (num_envs, 16). `joint_vel` and
    `action` are (num_envs, 13). `traj_phase` is (num_envs,), in [0, 1).

    `joint_pos`/`joint_vel` are roto's own normalized buffers (sliced to the
    appropriate index set), NOT recomputed here. `joint_pos_error` is roto's
    own raw `joint_pos_cmd - joint_pos`. `action` is the 13-motor command
    implied by the current PD target (see features.py module docstring).
    """

    joint_pos: torch.Tensor
    joint_vel: torch.Tensor
    joint_pos_error: torch.Tensor
    action: torch.Tensor
    last_residual: torch.Tensor
    real_vel_t: torch.Tensor
    traj_phase: torch.Tensor


class FeatureBuilder:
    """Builds the flat `prop` observation vector from a `FeatureContext`.

    Args:
        feature_list: Ordered list of feature names (see `_FEATURE_DIMS`).
            Defaults to `DEFAULT_FEATURES` (58-dim input).
        num_joints: Number of actuated (kinematic) joints -- 16 for ShadowLite.
        num_control: Number of independently-controlled motors -- 13 for ShadowLite.
    """

    def __init__(self, feature_list: list[str] | None, num_joints: int, num_control: int) -> None:
        # Falsy (None or []) -> DEFAULT_FEATURES, which is always non-empty, so there is
        # no separate "empty list" error case to guard here.
        feature_list = list(feature_list) if feature_list else list(DEFAULT_FEATURES)
        unknown = [f for f in feature_list if f not in _FEATURE_DIMS]
        if unknown:
            raise ValueError(f"Unknown feature(s) {unknown}; valid features are {sorted(_FEATURE_DIMS)}")

        self.feature_list = feature_list
        self.num_joints = num_joints
        self.num_control = num_control

        _dim_lookup = {"num_joints": num_joints, "num_control": num_control}
        width = 0
        for f in self.feature_list:
            dim = _FEATURE_DIMS[f]
            width += _dim_lookup[dim] if dim in _dim_lookup else dim
        self.output_dim = width

    def build(self, ctx: FeatureContext) -> torch.Tensor:
        """Return the (num_envs, output_dim) feature vector for one frame."""
        parts: list[torch.Tensor] = []
        for f in self.feature_list:
            if f == "joint_pos":
                parts.append(ctx.joint_pos)
            elif f == "joint_vel":
                parts.append(ctx.joint_vel)
            elif f == "joint_pos_error":
                parts.append(ctx.joint_pos_error)
            elif f == "action":
                parts.append(ctx.action)
            elif f == "last_residual":
                parts.append(ctx.last_residual)
            elif f == "real_vel":
                parts.append(ctx.real_vel_t)
            elif f == "sin_time":
                parts.append(torch.sin(2.0 * torch.pi * ctx.traj_phase).unsqueeze(-1))
            elif f == "cos_time":
                parts.append(torch.cos(2.0 * torch.pi * ctx.traj_phase).unsqueeze(-1))
            else:  # pragma: no cover - guarded in __init__
                raise ValueError(f"Unhandled feature '{f}'")
        return torch.cat(parts, dim=-1)

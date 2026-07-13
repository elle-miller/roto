"""Real-world trajectory loading for the UAN residual-torque task.

Isaac-free (pure numpy/torch) so it can be unit-tested on CPU without booting
Isaac Sim. Two loaders are provided, sharing the same downstream interface
(`q_cmd`, `q_meas`, `q_meas_vel`, `q_torque`, `traj_starts/ends/lengths`,
`sample_start_indices`, `clamp`, `is_at_boundary`, `traj_progress`):

  * `AlignedTrajectoryDataset` -- reads the current recording format, e.g.
    roto/data/data/aligned/<episode>/*.aligned.npz (one file per recorded
    episode; a directory path auto-globs every `*.aligned.npz` inside it).
  * `TrajectoryDataset` -- the older single/few-file format
    (roto/mimic_recording.npz-style: joint_pos_cmd/joint_pos/actuated_names/
    episode_ends/rl_dt), kept for backward compatibility.

`task.py` picks a loader via `dataset.format` in yaml ("aligned" or
"legacy"); both produce identically-shaped tensors so nothing else in the
env cares which one is active.
"""

from __future__ import annotations

import glob
import os
from dataclasses import dataclass

import numpy as np
import torch

# ---------------------------------------------------------------------------
# Aligned format (current recordings, roto/data/data/aligned/...)
# ---------------------------------------------------------------------------

# The 6 joints that are NOT independently driven on hardware -- each pair's
# combined motor ("J0" in the recording's actuator_order, e.g. "rh_FFJ0")
# drives both DOFs via a tendon. There is no independently-recorded setpoint
# for either DOF individually, only the shared motor's single combined
# command -- see AlignedTrajectoryDataset's docstring for how that combined
# value is split back into individual J1/J2 targets.
COUPLED_JOINT_PAIRS = {
    "rh_FF": ("rh_FFJ1", "rh_FFJ2"),
    "rh_MF": ("rh_MFJ1", "rh_MFJ2"),
    "rh_RF": ("rh_RFJ1", "rh_RFJ2"),
}


class AlignedTrajectoryDataset:
    """Loads `*.aligned.npz` recordings (one file per episode).

    Each file provides, per row `t`:
        act_pos, act_err, act_vel, action, command   float64 (T, 13)  actuator-level
        gt_pos, gt_vel, gt_effort, gt_tactile         float64 (T, 16/64) joint-level
        valid                                         bool    (T,)
        seg_id                                        int64   (T,)
        actuator_order  (13,) <U*   names for the 13 actuator-level columns
        joint_order     (16,) <U*   names for the 16 joint-level columns
        dataset_rate    scalar Hz

    `actuator_order` has 13 entries: 10 directly map 1:1 onto a `joint_order`
    name (independently-driven joints -- their PD target is `action`, the
    real commanded setpoint). The remaining 3 are the combined "J0" channels
    for the FF/MF/RF coupled pairs (see `COUPLED_JOINT_PAIRS`) -- there is no
    per-DOF setpoint for those pairs on real hardware, only the shared
    motor's single combined command; it is split back into individual J1/J2
    targets (see DESIGN NOTE below), so PD keeps a real, physically-in-range
    setpoint to chase for every one of the 16 joints, direct or coupled.

    q_torque = gt_effort (uncalibrated motor effort, 16-dim). Never used as a
    network *input* by task.py -- only ever as an optional, calibration-free
    reward term (correlation/sign-based, not magnitude-based) -- see task.py.

    DESIGN NOTE on splitting the coupled-pair command: `action`'s combined
    channel (e.g. "rh_FFJ0") is on the *combined* J1+J2 angle scale, not
    roto's own internal `RotoEnv._handle_coupled_joints` "compact proxy"
    scale (which compresses both joints' travel into a single joint's own
    limit range via a tuned `coupling_theta` split point) -- confirmed
    empirically against the real recordings: e.g. one FF episode's
    "rh_FFJ0" action ranges [0.06, 1.74] rad, matching rh_FFJ2's own upper
    limit (1.745 rad); one MF episode's "rh_MFJ0" ranges up to 2.92 rad and
    one RF episode's "rh_RFJ0" up to 3.03 rad, both well past a single
    joint's own ~1.4-1.75 rad limit and consistent with the *combined-sum*
    scale (J2_upper + J1_upper = 1.745 + 1.396 = 3.141 rad = pi). Reusing
    `_handle_coupled_joints`'s tuned `coupling_theta` constant unmodified on
    this data would therefore silently apply the wrong transform (it assumes
    a compact [0, J2_upper] input domain, not a combined-sum one).

    The split law's *shape* is still reused, though -- it is the same
    "J2 saturates first, then J1 takes over" two-segment law
    `RotoEnv._asymmetric_backlash` already uses for its stateless "curl"
    case, just reparameterized on the combined-sum scale the real data is
    actually recorded on, with the split point at J2's own physical upper
    limit (the physically meaningful place for it to saturate) rather than
    an arbitrary tuned theta:

        combined = action[:, "<pair>J0"]                          # (T,)
        j2_cmd = clip(combined,            0, j2_upper)            # ramps first
        j1_cmd = clip(combined - j2_upper, 0, j1_upper)            # then this

    `joint_upper_limits` (a plain name->float dict, no Isaac dependency)
    supplies J2_upper/J1_upper for this split; the caller (task.py) reads
    them from the loaded articulation's own joint limits, so this uses
    exactly the same physical bounds the sim itself enforces.
    """

    def __init__(
        self,
        paths: str | list[str],
        joint_names: list[str],
        device: torch.device | str,
        joint_upper_limits: dict[str, float],
        min_horizon: int = 1,
        glob_pattern: str = "*.aligned.npz",
    ) -> None:
        self.device = torch.device(device)
        self.joint_names = list(joint_names)
        self.min_horizon = int(min_horizon)

        files = _expand_paths(paths, glob_pattern)
        if len(files) == 0:
            raise ValueError(f"AlignedTrajectoryDataset found no '{glob_pattern}' files under {paths!r}.")
        self.paths = files

        q_cmd_chunks: list[np.ndarray] = []
        q_meas_chunks: list[np.ndarray] = []
        q_torque_chunks: list[np.ndarray] = []
        seg_starts: list[int] = []
        seg_ends: list[int] = []
        rate = None
        offset = 0

        for path in files:
            raw = np.load(path, allow_pickle=True)
            required = ["action", "gt_pos", "gt_vel", "gt_effort", "valid", "seg_id", "actuator_order", "joint_order", "dataset_rate"]
            missing = [k for k in required if k not in raw.files]
            if missing:
                raise KeyError(f"{path}: missing expected key(s) {missing} (available: {list(raw.files)}).")

            actuator_order = [str(n) for n in raw["actuator_order"]]
            joint_order = [str(n) for n in raw["joint_order"]]

            gt_pos = np.asarray(raw["gt_pos"], dtype=np.float64)
            gt_vel = np.asarray(raw["gt_vel"], dtype=np.float64)
            gt_effort = np.asarray(raw["gt_effort"], dtype=np.float64)
            action = np.asarray(raw["action"], dtype=np.float64)
            valid = np.asarray(raw["valid"], dtype=bool)
            seg_id = np.asarray(raw["seg_id"])

            file_rate = float(raw["dataset_rate"])
            if rate is None:
                rate = file_rate
            elif not np.isclose(rate, file_rate):
                raise ValueError(f"{path}: dataset_rate={file_rate} != first file's {rate} -- mixed control rates unsupported.")

            joint_perm = _name_permutation(joint_order, self.joint_names, path, "joint_order")
            meas = gt_pos[:, joint_perm]
            meas_vel = gt_vel[:, joint_perm]
            torque = gt_effort[:, joint_perm]

            cmd = _build_cmd_from_action(action, actuator_order, self.joint_names, joint_upper_limits, path)

            n = cmd.shape[0]
            # Segment this file on (a) seg_id changes and (b) valid==False gaps;
            # in every currently-recorded file this yields exactly one segment
            # spanning the whole file, but this handles files with internal
            # gaps/discontinuities without special-casing them. A break occurs
            # at row i whenever seg_id changes OR row i-1/i is invalid -- this
            # isolates every invalid row into its own single-row "run" (breaks
            # on both sides of it), which is then dropped, so an invalid gap
            # ANYWHERE (including mid-run, not just at a seg_id edge) correctly
            # splits its surrounding run into two separate valid segments.
            breaks = np.zeros(n, dtype=bool)
            breaks[0] = True
            breaks[1:] |= seg_id[1:] != seg_id[:-1]
            breaks[1:] |= ~valid[1:] | ~valid[:-1]
            run_id = np.cumsum(breaks) - 1
            for rid in np.unique(run_id):
                idx = np.nonzero(run_id == rid)[0]
                if valid[idx[0]]:  # invalid rows are always isolated into their own (excluded) run
                    seg_starts.append(offset + int(idx[0]))
                    seg_ends.append(offset + int(idx[-1]))

            q_cmd_chunks.append(cmd)
            q_meas_chunks.append(meas)
            q_torque_chunks.append(torque)
            offset += n

        self.rl_dt: float = 1.0 / float(rate)
        self.q_cmd = torch.as_tensor(np.concatenate(q_cmd_chunks, axis=0), dtype=torch.float32, device=self.device)
        self.q_meas = torch.as_tensor(np.concatenate(q_meas_chunks, axis=0), dtype=torch.float32, device=self.device)
        self.q_torque = torch.as_tensor(np.concatenate(q_torque_chunks, axis=0), dtype=torch.float32, device=self.device)
        self.num_steps = self.q_cmd.shape[0]
        self.num_joints = self.q_cmd.shape[1]

        if len(seg_starts) == 0:
            raise ValueError(f"No valid segments found across {len(files)} file(s).")

        self.traj_starts = torch.as_tensor(seg_starts, dtype=torch.long, device=self.device)
        self.traj_ends = torch.as_tensor(seg_ends, dtype=torch.long, device=self.device)
        self.traj_lengths = self.traj_ends - self.traj_starts + 1

        self.q_meas_vel = self._finite_diff_velocity(self.q_meas, seg_starts, seg_ends)
        # gt_vel is also available directly (measured, not finite-differenced);
        # kept as a second option since the coupled-pair velocities in gt_vel
        # are duplicated across J1/J2 (hardware doesn't measure them
        # independently) -- finite-diff of gt_pos is used as the primary
        # `q_meas_vel` for consistency with the legacy loader.

        self._segment_id = torch.zeros(self.num_steps, dtype=torch.long, device=self.device)
        valid_start_mask = torch.zeros(self.num_steps, dtype=torch.bool, device=self.device)
        for seg_idx in range(len(seg_starts)):
            s, e = int(self.traj_starts[seg_idx]), int(self.traj_ends[seg_idx])
            self._segment_id[s : e + 1] = seg_idx
            last_valid = max(s, e - self.min_horizon + 1)
            if last_valid >= s:
                valid_start_mask[s : last_valid + 1] = True
        self._valid_start_indices = torch.nonzero(valid_start_mask, as_tuple=False).squeeze(-1)
        if self._valid_start_indices.numel() == 0:
            raise ValueError(
                f"No valid start indices with min_horizon={self.min_horizon} across "
                f"{len(seg_starts)} segment(s) of lengths {self.traj_lengths.tolist()}."
            )

    def _finite_diff_velocity(self, q: torch.Tensor, seg_starts: list[int], seg_ends: list[int]) -> torch.Tensor:
        vel = torch.zeros_like(q)
        for s, e in zip(seg_starts, seg_ends):
            if e > s:
                vel[s:e] = (q[s + 1 : e + 1] - q[s:e]) / self.rl_dt
                vel[e] = vel[e - 1]
        return vel

    def sample_start_indices(self, n: int, generator: torch.Generator | None = None) -> torch.Tensor:
        idx = torch.randint(0, self._valid_start_indices.numel(), (n,), device=self.device, generator=generator)
        return self._valid_start_indices[idx]

    def segment_start(self, t: torch.Tensor) -> torch.Tensor:
        return self.traj_starts[self._segment_id[t]]

    def is_at_boundary(self, t: torch.Tensor) -> torch.Tensor:
        return t >= self.traj_ends[self._segment_id[t.clamp(max=self.num_steps - 1)]]

    def clamp(self, t: torch.Tensor) -> torch.Tensor:
        return t.clamp(min=0, max=self.num_steps - 1)

    def traj_progress(self, t: torch.Tensor) -> torch.Tensor:
        t = self.clamp(t)
        seg = self._segment_id[t]
        start = self.traj_starts[seg]
        length = self.traj_lengths[seg]
        return (t - start).to(torch.float32) / length.to(torch.float32)


def _expand_paths(paths: str | list[str], glob_pattern: str) -> list[str]:
    """Expand a mix of directory paths, glob patterns, and explicit file paths."""
    if isinstance(paths, str):
        paths = [paths]
    files: list[str] = []
    for p in paths:
        if os.path.isdir(p):
            files.extend(sorted(glob.glob(os.path.join(p, glob_pattern))))
        elif any(ch in p for ch in "*?[]"):
            files.extend(sorted(glob.glob(p)))
        else:
            files.append(p)
    return files


def _name_permutation(source_names: list[str], target_names: list[str], path: str, source_label: str) -> list[int]:
    name_to_col = {n: i for i, n in enumerate(source_names)}
    missing = [n for n in target_names if n not in name_to_col]
    if missing:
        raise KeyError(f"{path}: {source_label} is missing required joint name(s) {missing}. Has: {source_names}")
    return [name_to_col[n] for n in target_names]


def _split_coupled_command(combined: np.ndarray, j2_upper: float, j1_upper: float) -> tuple[np.ndarray, np.ndarray]:
    """Split one pair's combined motor command into individual J2/J1 targets.

    Two-segment law (see AlignedTrajectoryDataset's DESIGN NOTE): J2 ramps
    from 0 to its own upper limit first; only once J2 has saturated does the
    remainder drive J1. Both outputs are always within their own joint's
    physical range by construction (clip), so PD always has an in-bounds
    target to chase, whatever the recorded combined value does.
    """
    j2_cmd = np.clip(combined, 0.0, j2_upper)
    j1_cmd = np.clip(combined - j2_upper, 0.0, j1_upper)
    return j2_cmd, j1_cmd


def _build_cmd_from_action(
    action: np.ndarray,
    actuator_order: list[str],
    joint_names: list[str],
    joint_upper_limits: dict[str, float],
    path: str,
) -> np.ndarray:
    """Build the 16-dim PD target: real setpoint for directly-driven joints,
    split real setpoint for the 6 coupled-pair DOFs (see class docstring).
    """
    actuator_col = {n: i for i, n in enumerate(actuator_order)}
    name_to_col = {n: j for j, n in enumerate(joint_names)}

    n_rows = action.shape[0]
    cmd = np.zeros((n_rows, len(joint_names)), dtype=np.float64)
    handled: set[str] = set()

    for pair_key, (j1_name, j2_name) in COUPLED_JOINT_PAIRS.items():
        motor_name = f"{pair_key}J0"
        if motor_name not in actuator_col:
            raise KeyError(f"{path}: coupled motor '{motor_name}' not found in actuator_order {actuator_order}.")
        for name in (j1_name, j2_name):
            if name not in joint_upper_limits:
                raise KeyError(f"{path}: joint_upper_limits is missing required coupled joint '{name}'.")
        combined = action[:, actuator_col[motor_name]]
        j2_cmd, j1_cmd = _split_coupled_command(combined, joint_upper_limits[j2_name], joint_upper_limits[j1_name])
        if j1_name in name_to_col:
            cmd[:, name_to_col[j1_name]] = j1_cmd
            handled.add(j1_name)
        if j2_name in name_to_col:
            cmd[:, name_to_col[j2_name]] = j2_cmd
            handled.add(j2_name)

    for j, name in enumerate(joint_names):
        if name in handled:
            continue
        if name in actuator_col:
            cmd[:, j] = action[:, actuator_col[name]]
        else:
            raise KeyError(
                f"{path}: joint '{name}' is neither a coupled DOF nor present in actuator_order {actuator_order}."
            )
    return cmd


# ---------------------------------------------------------------------------
# Legacy format (single/few-file .npz: joint_pos_cmd/joint_pos/actuated_names)
# ---------------------------------------------------------------------------


@dataclass
class DatasetKeys:
    """Names of the arrays to read out of each legacy-format .npz file."""

    cmd: str = "joint_pos_cmd"
    meas: str = "joint_pos"
    names: str = "actuated_names"
    ends: str = "episode_ends"
    dt: str = "rl_dt"


class TrajectoryDataset:
    """Legacy loader for `.npz` recordings with `joint_pos_cmd`/`joint_pos`
    (both 16-dim) and inclusive `episode_ends`. See `AlignedTrajectoryDataset`
    for the current recording format. Kept for backward compatibility with
    roto/mimic_recording.npz-style files.
    """

    def __init__(
        self,
        paths: str | list[str],
        joint_names: list[str],
        device: torch.device | str,
        keys: DatasetKeys | None = None,
        min_horizon: int = 1,
    ) -> None:
        self.device = torch.device(device)
        self.keys = keys if keys is not None else DatasetKeys()
        self.joint_names = list(joint_names)
        self.min_horizon = int(min_horizon)

        if isinstance(paths, str):
            paths = [paths]
        if len(paths) == 0:
            raise ValueError("TrajectoryDataset requires at least one recording path.")
        self.paths = list(paths)

        q_cmd_chunks: list[np.ndarray] = []
        q_meas_chunks: list[np.ndarray] = []
        seg_starts: list[int] = []
        seg_ends: list[int] = []
        rl_dt = None
        offset = 0

        for path in self.paths:
            raw = np.load(path, allow_pickle=True)
            self._check_keys_present(raw, path)

            names = [str(n) for n in raw[self.keys.names]]
            perm = self._name_permutation(names, path)

            cmd = np.asarray(raw[self.keys.cmd], dtype=np.float32)[:, perm]
            meas = np.asarray(raw[self.keys.meas], dtype=np.float32)[:, perm]
            if cmd.shape != meas.shape:
                raise ValueError(
                    f"{path}: '{self.keys.cmd}' shape {cmd.shape} != '{self.keys.meas}' shape {meas.shape}"
                )

            ends = np.asarray(raw[self.keys.ends], dtype=np.int64)
            file_dt = float(np.asarray(raw[self.keys.dt]))
            if rl_dt is None:
                rl_dt = file_dt
            elif not np.isclose(rl_dt, file_dt):
                raise ValueError(
                    f"{path}: rl_dt={file_dt} does not match the first recording's rl_dt={rl_dt}. "
                    "Mixing recordings at different control rates is not supported."
                )

            seg_start = 0
            for e in ends:
                seg_starts.append(offset + seg_start)
                seg_ends.append(offset + int(e))
                seg_start = int(e) + 1

            q_cmd_chunks.append(cmd)
            q_meas_chunks.append(meas)
            offset += cmd.shape[0]

        q_cmd = np.concatenate(q_cmd_chunks, axis=0)
        q_meas = np.concatenate(q_meas_chunks, axis=0)

        self.rl_dt: float = float(rl_dt)
        self.q_cmd = torch.as_tensor(q_cmd, dtype=torch.float32, device=self.device)
        self.q_meas = torch.as_tensor(q_meas, dtype=torch.float32, device=self.device)
        # Legacy format has no torque field; zero-filled so task.py's optional
        # torque reward term is simply inert (never contributes) for this loader.
        self.q_torque = torch.zeros_like(self.q_meas)
        self.num_steps = self.q_cmd.shape[0]
        self.num_joints = self.q_cmd.shape[1]

        self.q_meas_vel = self._finite_diff_velocity(self.q_meas, seg_starts, seg_ends)

        self.traj_starts = torch.as_tensor(seg_starts, dtype=torch.long, device=self.device)
        self.traj_ends = torch.as_tensor(seg_ends, dtype=torch.long, device=self.device)
        self.traj_lengths = self.traj_ends - self.traj_starts + 1
        if (self.traj_lengths <= 0).any():
            raise ValueError(f"Non-positive trajectory segment length(s) found: {self.traj_lengths.tolist()}")

        self._segment_id = torch.zeros(self.num_steps, dtype=torch.long, device=self.device)
        valid_start_mask = torch.zeros(self.num_steps, dtype=torch.bool, device=self.device)
        for seg_idx in range(len(seg_starts)):
            s, e = int(self.traj_starts[seg_idx]), int(self.traj_ends[seg_idx])
            self._segment_id[s : e + 1] = seg_idx
            last_valid = max(s, e - self.min_horizon + 1)
            if last_valid >= s:
                valid_start_mask[s : last_valid + 1] = True
        self._valid_start_indices = torch.nonzero(valid_start_mask, as_tuple=False).squeeze(-1)
        if self._valid_start_indices.numel() == 0:
            raise ValueError(
                f"No valid start indices with min_horizon={self.min_horizon} across "
                f"{len(seg_starts)} segment(s) of lengths {self.traj_lengths.tolist()}."
            )

    def _check_keys_present(self, raw: np.lib.npyio.NpzFile, path: str) -> None:
        required = [self.keys.cmd, self.keys.meas, self.keys.names, self.keys.ends, self.keys.dt]
        missing = [k for k in required if k not in raw.files]
        if missing:
            raise KeyError(
                f"{path}: missing expected key(s) {missing} (available: {list(raw.files)}). "
                "Set `dataset.keys.<field>` in the yaml to match this recording's field names."
            )

    def _name_permutation(self, dataset_names: list[str], path: str) -> list[int]:
        name_to_col = {n: i for i, n in enumerate(dataset_names)}
        missing = [n for n in self.joint_names if n not in name_to_col]
        if missing:
            raise KeyError(
                f"{path}: recording is missing required joint name(s) {missing}. "
                f"Recording has: {dataset_names}"
            )
        return [name_to_col[n] for n in self.joint_names]

    def _finite_diff_velocity(self, q: torch.Tensor, seg_starts: list[int], seg_ends: list[int]) -> torch.Tensor:
        vel = torch.zeros_like(q)
        for s, e in zip(seg_starts, seg_ends):
            if e > s:
                vel[s:e] = (q[s + 1 : e + 1] - q[s:e]) / self.rl_dt
                vel[e] = vel[e - 1]
        return vel

    def sample_start_indices(self, n: int, generator: torch.Generator | None = None) -> torch.Tensor:
        idx = torch.randint(0, self._valid_start_indices.numel(), (n,), device=self.device, generator=generator)
        return self._valid_start_indices[idx]

    def segment_start(self, t: torch.Tensor) -> torch.Tensor:
        return self.traj_starts[self._segment_id[t]]

    def is_at_boundary(self, t: torch.Tensor) -> torch.Tensor:
        return t >= self.traj_ends[self._segment_id[t.clamp(max=self.num_steps - 1)]]

    def clamp(self, t: torch.Tensor) -> torch.Tensor:
        return t.clamp(min=0, max=self.num_steps - 1)

    def traj_progress(self, t: torch.Tensor) -> torch.Tensor:
        t = self.clamp(t)
        seg = self._segment_id[t]
        start = self.traj_starts[seg]
        length = self.traj_lengths[seg]
        return (t - start).to(torch.float32) / length.to(torch.float32)

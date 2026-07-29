"""Build the encoder's (prop, tactile) input frames from sim and hardware trajectory npz files.

Reuses the verified joint-limit tables / normalisation / 13-joint remap from
../scripts/finetune_bc.py rather than redefining them (single source of truth for the
documented URDF limits and the coupled-finger [0, 1.745] convention).

Per-frame layout (matches roto/tasks/roto_env.py::_get_proprioception exactly):
    prop    = [norm_joint_pos(13), norm_joint_vel(13), joint_pos_error(13), action(13)]  # 52
    tactile = tac(24)  # already the fixed pad/BioTac scatter vector, binary 0/1

Frame stacking (oldest -> newest, cold-start replicates frame 0) matches
multimodal_rl.wrappers.frame_stack.FrameStack's reset-fill behaviour.
"""

import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
import finetune_bc as bc  # noqa: E402  (CONTROL_JOINT_NAMES, LOWER/UPPER/VEL_LIMITS, normalise)

OBS_FRAME_DIM = bc.OBS_FRAME_DIM  # 52
NUM_TACTILE_CHANNELS = 24


def stack_frames(frames: np.ndarray, obs_stack: int) -> np.ndarray:
    """(T, D) -> (T, D*obs_stack), oldest->newest, cold-start replicates frame 0.

    Matches FrameStack's reset behaviour: "the observation returned by reset consists of
    obs_stack many identical frames" (multimodal_rl/wrappers/frame_stack.py).
    """
    T, D = frames.shape
    out = np.empty((T, D * obs_stack), dtype=frames.dtype)
    for t in range(T):
        idx = [max(0, t - obs_stack + 1 + i) for i in range(obs_stack)]
        out[t] = frames[idx].reshape(-1)
    return out


def build_sim_frames(npz):
    """Reconstruct (prop(T,52), tactile(T,24), action(T,13)) from a sim_policy_log_*.npz.

    Two schemas are supported:

    NEW (preferred; has 'q13'/'qd13'/'cmd13'/'pos_err13'): these are already in
    CONTROL_JOINT_NAMES order and self-consistent (pos_err13 == cmd13 - q13 exactly,
    verified). qd13 is TRUE PhysX joint velocity (not a finite difference) -- this is
    what roto_env._compute_intermediate_values actually normalises, so use it directly.
    The coupled slots (FFJ2/MFJ2/RFJ2) are FFJ2-ALONE here; sim trains with J1 locked
    near 0, so FFJ2-alone already equals the full curl (no summing needed on the sim
    side -- only the hardware side needs the FFJ1+FFJ2 combined reading).

    LEGACY (only 'q'/'cmd'/'joints', 16-wide, no velocity): falls back to central-
    difference velocity (noisier / does not match the frozen encoder's true training
    input on the velocity dims -- prefer regenerating with the new schema).

    In both cases the 4th prop term uses `act` RAW (unclipped, unsquashed) policy
    output -- this is exactly `self.actions` as stored by roto_env._pre_physics_step /
    used by _get_proprioception, so it must NOT be passed through normalise() again.
    """
    if "q13" in npz.files:
        q13 = npz["q13"].astype(np.float64)
        vel = npz["qd13"].astype(np.float64)          # TRUE PhysX velocity
        pos_err = npz["pos_err13"].astype(np.float64)  # == cmd13 - q13 (verified exact)
    else:
        joints16 = list(npz["joints"])
        cols = [joints16.index(name) for name in bc.CONTROL_JOINT_NAMES]
        q13 = npz["q"][:, cols].astype(np.float64)
        cmd13 = npz["cmd"][:, cols].astype(np.float64)
        vel = np.gradient(q13, axis=0) * 60.0  # legacy fallback: no recorded velocity
        pos_err = cmd13 - q13

    # Safety clip on the 3 coupled slots (see finetune_bc.py's UPPER_LIMITS comment):
    # normally already in-range (sim trains with J1 locked, FFJ2-alone stays <=1.745).
    q13 = q13.copy()
    for i in bc.COUPLED_SLOTS:
        q13[:, i] = np.clip(q13[:, i], bc.LOWER_LIMITS[i], bc.UPPER_LIMITS[i])

    action13 = npz["act"].astype(np.float32)  # RAW raw policy output, CONTROL_JOINT_NAMES order

    pos_norm = bc.normalise(q13, bc.LOWER_LIMITS, bc.UPPER_LIMITS)
    vel_norm = bc.normalise(vel, -bc.VEL_LIMITS, bc.VEL_LIMITS)

    prop = np.concatenate(
        [pos_norm, vel_norm, pos_err, action13], axis=-1
    ).astype(np.float32)
    assert prop.shape[1] == OBS_FRAME_DIM

    tactile = npz["tac"].astype(np.float32)  # (T,24), already binary/scattered
    assert tactile.shape[1] == NUM_TACTILE_CHANNELS

    return prop, tactile, action13


def compute_hw_last_action(pos_norm, vel_norm, pos_err, tactile, obs_stack,
                            frozen_encoder, frozen_policy_head, device="cpu", seg_id=None,
                            use_tactile=True):
    """Reconstruct the raw sim-space `last_action` term for one hardware trajectory by
    sequentially replaying the FROZEN sim encoder + policy over the observed hw state.

    Matches sim's actual convention: at step t, the 4th prop term is `self.actions`,
    the raw policy output computed from THIS step's own observation
    (roto_env._pre_physics_step / _get_proprioception) -- not the command that was
    actually sent to hardware. Hardware never logged the raw pre-scale policy output,
    so this replay recovers it in sim's exact space (unbounded, matches training),
    unlike normalising the logged command (which is policy-native [-1,1], a different
    and weaker approximation -- see build_hw_frames).

    Must use FROZEN weights only: feeding the (being-trained) online encoder back into
    its own input would make training non-stationary.

    Args:
        pos_norm, vel_norm, pos_err: (T,13) unstacked single-frame arrays.
        tactile: (T,24) unstacked (ignored entirely if use_tactile=False).
        obs_stack: frame-stack width (must match the frozen encoder's input dim).
        frozen_encoder, frozen_policy_head: eval()'d, requires_grad_(False) modules.
        seg_id: optional (T,) int array; last_action resets to 0 at segment boundaries
            (mirrors FrameStack's cold-start-replicates-frame-0 behaviour per segment).
        use_tactile: must match whatever `frozen_encoder` was actually built with (a
            prop-only encoder has no "tactile" key in its observation_space at all).

    Returns:
        action (T,13) float32 raw unbounded values; action[0] (and the first frame of
        each segment) is 0, matching RotoEnv's zero-initialised `self.actions`.
    """
    T = pos_norm.shape[0]
    action = np.zeros((T, 13), dtype=np.float32)
    prop_no_action = np.concatenate([pos_norm, vel_norm, pos_err], axis=-1).astype(np.float32)  # (T,39)
    if use_tactile:
        tactile = tactile.astype(np.float32)

    def frame(t):
        return np.concatenate([prop_no_action[t], action[t]])

    prop_hist = [frame(0)] * obs_stack
    if use_tactile:
        tac_hist = [tactile[0]] * obs_stack

    with torch.no_grad():
        for t in range(T):
            new_segment = seg_id is not None and t > 0 and seg_id[t] != seg_id[t - 1]
            if new_segment:
                prop_hist = [frame(t)] * obs_stack
                if use_tactile:
                    tac_hist = [tactile[t]] * obs_stack
            else:
                prop_hist = prop_hist[1:] + [frame(t)]
                if use_tactile:
                    tac_hist = tac_hist[1:] + [tactile[t]]

            obs = {"prop": torch.tensor(np.concatenate(prop_hist), dtype=torch.float32, device=device).unsqueeze(0)}
            if use_tactile:
                obs["tactile"] = torch.tensor(np.concatenate(tac_hist), dtype=torch.float32, device=device).unsqueeze(0)
            z = frozen_encoder(obs)
            a = frozen_policy_head(z)[0].detach().cpu().numpy()

            if t + 1 < T:
                if seg_id is not None and seg_id[t + 1] != seg_id[t]:
                    action[t + 1] = 0.0  # next frame starts a new segment: cold start
                else:
                    action[t + 1] = a
    return action


def build_hw_frames(npz, frozen_encoder=None, frozen_policy_head=None, obs_stack=None, device="cpu",
                     use_tactile=True):
    """Reconstruct (prop(T,52), tactile(T,24), action(T,13)) from an .aligned.npz hardware log.

    The 4th prop term (last_action):
      - If `frozen_encoder`/`frozen_policy_head` are given: reconstructed via
        `compute_hw_last_action` -- a sequential closed-loop replay that recovers the
        raw, sim-space action (matches what the frozen encoder was trained on).
        `obs_stack` is then required (sets the rolling history width).
      - Otherwise (default, unchanged from before): falls back to
        `normalise(commanded position)` in [-1,1] (policy-native) -- a DIFFERENT,
        weaker approximation, since hardware has no logged equivalent of the raw
        pre-scale policy output without the replay above.

    `tactile` (T,24) is always computed and returned regardless of `use_tactile` (cheap,
    and other callers may still want it for e.g. --hw_tactile_source substitution); only
    the last_action replay's own encoder calls respect `use_tactile` (must match whatever
    `frozen_encoder` was actually built with).
    """
    actuator_order = list(npz["actuator_order"])
    cols = bc.build_policy_column_map(actuator_order)

    act_pos = npz["act_pos"][:, cols].astype(np.float64)
    act_vel = npz["act_vel"][:, cols].astype(np.float64)
    action = npz["action"][:, cols].astype(np.float64)

    # Safety clip on the 3 coupled slots (see finetune_bc.py's UPPER_LIMITS comment):
    # act_pos/action[FFJ0] is the combined-tendon reading, expected in [0, 1.745].
    act_pos = act_pos.copy()
    action = action.copy()
    for i in bc.COUPLED_SLOTS:
        act_pos[:, i] = np.clip(act_pos[:, i], bc.LOWER_LIMITS[i], bc.UPPER_LIMITS[i])
        action[:, i] = np.clip(action[:, i], bc.LOWER_LIMITS[i], bc.UPPER_LIMITS[i])

    pos_norm = bc.normalise(act_pos, bc.LOWER_LIMITS, bc.UPPER_LIMITS)
    vel_norm = bc.normalise(act_vel, -bc.VEL_LIMITS, bc.VEL_LIMITS)
    pos_err = action - act_pos  # rad, command - measured
    tactile = npz["tac"].astype(np.float32)

    if frozen_encoder is not None and frozen_policy_head is not None:
        assert obs_stack is not None, "obs_stack is required when reconstructing last_action via replay"
        seg_id = npz["seg_id"] if "seg_id" in npz.files else None
        last_action = compute_hw_last_action(
            pos_norm, vel_norm, pos_err, tactile, obs_stack,
            frozen_encoder, frozen_policy_head, device=device, seg_id=seg_id, use_tactile=use_tactile,
        )
    else:
        last_action = bc.normalise(action, bc.LOWER_LIMITS, bc.UPPER_LIMITS).astype(np.float32)

    prop = np.concatenate(
        [pos_norm, vel_norm, pos_err, last_action], axis=-1
    ).astype(np.float32)
    assert prop.shape[1] == OBS_FRAME_DIM
    assert tactile.shape[1] == NUM_TACTILE_CHANNELS

    return prop, tactile, last_action

"""Fine-tune a Baoding policy checkpoint via behavior cloning on recorded `.aligned.npz` data.

Given a checkpoint (dict with 'policy'/'value'/'encoder'/... keys, produced by training with
multimodal_rl.rl.ppo.PPO) and a recorded trajectory (as produced by the hardware/sim alignment
pipeline -- see roto/replay_motion_test/*.aligned.npz), this script:

  1. Rebuilds the exact observation the checkpoint's encoder expects, offline (no Isaac Sim),
     mirroring roto/tasks/roto_env.py::_get_proprioception / roto/my_policy_node.py.
  2. Runs obs -> encoder (frozen) -> policy (trainable) -> predicted action.
  3. Regresses the predicted action onto the recorded action via MSE, in the policy's native
     [-1, 1] output space.
  4. Saves a new checkpoint with the same dict format (only 'policy' replaced), reloadable by
     play.py / record_policy.py / my_policy_node.py.

Only the policy head is fine-tuned; the encoder is frozen (matches how the checkpoint's learned
representation was produced under PPO, and avoids overfitting the encoder to a single short
trajectory).

Usage:
    python finetune_bc.py \
        --checkpoint /path/to/best_agent_fixed_j1.pt \
        --data /path/to/position_baoding_noballs_nalin_speed1p0.aligned.npz \
        --agent_cfg_dir ../roto/tasks/baoding/agents/shadowlite \
        --agent_cfg rl_only_pt \
        --output /path/to/best_agent_fixed_j1__ft_position_baoding_noballs_nalin_speed1p0.pt
"""

import argparse
import copy
import os

import numpy as np
import torch
import torch.nn.functional as F
import yaml

from multimodal_rl.models.encoder import Encoder
from multimodal_rl.rl.policy_value import GaussianPolicy

# =============================================================================
# Documented robot constants (shadowlite, 13 policy-controlled joints).
#
# CONTROL_JOINT_NAMES / coupling from
#   roto/roto/tasks/robots/shadowlite/shadowlite.py:152-163 (control_joint_names,
#   coupled_joint_map). LOWER/UPPER/VEL limits (non-coupled joints) transcribed from
#   the <limit .../> tags in roto/assets/shadow_lite/sr_hand_mimic_touchlab.urdf.
#
# The 3 coupled slots (FFJ2/MFJ2/RFJ2, indices 8/9/10) are driven by the combined
# hardware joint (npz actuator names rh_FFJ0/rh_MFJ0/rh_RFJ0, i.e. FFJ1+FFJ2 summed).
# The frozen sim encoder was trained with J1 hard-locked at 0, so FFJ2 alone carried
# the full curl and was normalised over its own individual soft limit [0, 1.745] rad
# (100 deg) -- NOT the combined joint's mechanical range [0, pi] (180 deg). The npz
# logs act_pos/act_vel[FFJ0] in this un-doubled [0, 1.745] regime already (measured
# max ~1.15 rad across all collected seeds), so normalise with [0, 1.745] here too.
# The ROS `ffj0_command = 2 x proxy` doubling (roto/run_shadow.py) is the actuation/
# publish path only and does not belong in this observation normalisation.
# =============================================================================

CONTROL_JOINT_NAMES = [
    "rh_FFJ4", "rh_MFJ4", "rh_RFJ4", "rh_THJ5",   # 0,1,2,3
    "rh_FFJ3", "rh_MFJ3", "rh_RFJ3", "rh_THJ4",   # 4,5,6,7
    "rh_FFJ2", "rh_MFJ2", "rh_RFJ2",              # 8,9,10  (coupled drivers)
    "rh_THJ2", "rh_THJ1",                          # 11,12
]

# Joint whose npz actuator-space column stands in for each CONTROL_JOINT_NAMES entry.
# Non-coupled joints map 1:1; the 3 coupled drivers read the combined FFJ0/MFJ0/RFJ0
# actuator column instead of the (unobservable on hardware) individual FFJ2 angle.
NPZ_ACTUATOR_NAME = {
    "rh_FFJ2": "rh_FFJ0",
    "rh_MFJ2": "rh_MFJ0",
    "rh_RFJ2": "rh_RFJ0",
}

COUPLED_SLOTS = {8, 9, 10}  # indices into CONTROL_JOINT_NAMES / LOWER/UPPER/VEL below

LOWER_LIMITS = np.array([
    -0.3490658503988659,   # FFJ4
    -0.3490658503988659,   # MFJ4
    -0.3490658503988659,   # RFJ4
    -1.0471975511965976,   # THJ5
    -0.2617993877991494,   # FFJ3
    -0.2617993877991494,   # MFJ3
    -0.2617993877991494,   # RFJ3
     0.0,                  # THJ4
     0.0,                  # FFJ2 (coupled: combined FFJ0, [0, 1.745])
     0.0,                  # MFJ2 (coupled)
     0.0,                  # RFJ2 (coupled)
    -0.6981317007977318,   # THJ2
    -0.2617993877991494,   # THJ1
], dtype=np.float64)

UPPER_LIMITS = np.array([
    0.3490658503988659,    # FFJ4
    0.3490658503988659,    # MFJ4
    0.3490658503988659,    # RFJ4
    1.0471975511965976,    # THJ5
    1.5707963267948966,    # FFJ3
    1.5707963267948966,    # MFJ3
    1.5707963267948966,    # RFJ3
    1.2217304763960306,    # THJ4
    1.7450,                 # FFJ2 (coupled: sim FFJ2 individual soft limit, [0, 100 deg])
    1.7450,                 # MFJ2 (coupled)
    1.7450,                 # RFJ2 (coupled)
    0.6981317007977318,    # THJ2
    1.5707963267948966,    # THJ1
], dtype=np.float64)

# Velocity limits (rad/s) for normalising act_vel. Coupled slots reuse the individual-
# joint URDF velocity limit (2.0) since no combined-joint figure is documented; edit
# here if a different value is known.
VEL_LIMITS = np.array([
    2.0, 2.0, 2.0, 4.0,
    2.0, 2.0, 2.0, 4.0,
    2.0, 2.0, 2.0,
    2.0, 4.0,
], dtype=np.float64)

OBS_FRAME_DIM = 52  # pos(13) + vel(13) + pos_error(13) + prev_action(13)


def normalise(x, lower, upper):
    """Map x from [lower, upper] -> [-1, 1] (matches roto_env.unscale)."""
    return (2.0 * x - upper - lower) / (upper - lower)


# =============================================================================
# Config loading (plain YAML, no Isaac Lab / hydra dependency -- offline).
# =============================================================================

def _deep_merge(base: dict, overlay: dict) -> dict:
    out = dict(base)
    for k, v in overlay.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def load_agent_cfg(agent_cfg_dir: str, agent_cfg_name: str | None) -> dict:
    """Load default.yaml, optionally merged with an overlay yaml (e.g. rl_only_pt)."""
    with open(os.path.join(agent_cfg_dir, "default.yaml")) as f:
        cfg = yaml.safe_load(f)
    if agent_cfg_name and agent_cfg_name != "default":
        overlay_path = os.path.join(agent_cfg_dir, f"{agent_cfg_name}.yaml")
        if os.path.exists(overlay_path):
            with open(overlay_path) as f:
                overlay = yaml.safe_load(f)
            cfg = _deep_merge(cfg, overlay)
    return cfg


# =============================================================================
# npz loading + policy-order remap + observation/target reconstruction.
# =============================================================================

def build_policy_column_map(actuator_order: list[str]) -> np.ndarray:
    """For each CONTROL_JOINT_NAMES entry, the column index into the npz's actuator_order."""
    cols = []
    for name in CONTROL_JOINT_NAMES:
        npz_name = NPZ_ACTUATOR_NAME.get(name, name)
        if npz_name not in actuator_order:
            raise ValueError(f"Joint '{npz_name}' (for policy joint '{name}') not found in "
                              f"npz actuator_order={actuator_order}")
        cols.append(actuator_order.index(npz_name))
    return np.array(cols, dtype=np.int64)


def load_npz_dataset(path: str, obs_list: list[str], obs_stack: int):
    """Reconstruct stacked observations + BC targets from one .aligned.npz file.

    Returns:
        obs: dict of {"prop": (N, 52*obs_stack) [, "tactile": (N, NUM_TACTILE*obs_stack)]}
        targets: (N, 13) float32, normalised to [-1, 1]
    """
    if "gt" in obs_list:
        raise ValueError("obs_list contains 'gt' (ball state) which cannot be reconstructed "
                          "from hand-only .aligned.npz files; unsupported by this script.")

    d = np.load(path, allow_pickle=True)
    actuator_order = list(d["actuator_order"])
    cols = build_policy_column_map(actuator_order)

    act_pos = d["act_pos"][:, cols].astype(np.float64)   # (T, 13) policy order
    act_vel = d["act_vel"][:, cols].astype(np.float64)   # (T, 13)
    action = d["action"][:, cols].astype(np.float64)     # (T, 13)
    valid = d["valid"].astype(bool)
    seg_id = d["seg_id"]

    T = act_pos.shape[0]

    # Safety clip on the 3 coupled slots: act_pos/action for FFJ2/MFJ2/RFJ2 is the
    # combined FFJ0/MFJ0/RFJ0 actuator reading, expected in [0, 1.745] (see the
    # coupled-slot note above); never observed to exceed it, but clip defensively
    # so a transient sensor spike can't produce an out-of-training-range norm value.
    act_pos = act_pos.copy()
    action = action.copy()
    for i in COUPLED_SLOTS:
        act_pos[:, i] = np.clip(act_pos[:, i], LOWER_LIMITS[i], UPPER_LIMITS[i])
        action[:, i] = np.clip(action[:, i], LOWER_LIMITS[i], UPPER_LIMITS[i])

    pos_norm = normalise(act_pos, LOWER_LIMITS, UPPER_LIMITS)
    vel_norm = normalise(act_vel, -VEL_LIMITS, VEL_LIMITS)
    act_norm = normalise(action, LOWER_LIMITS, UPPER_LIMITS)
    pos_err = action - act_pos  # rad, command - measured (sim convention)

    frame = np.concatenate([pos_norm, vel_norm, pos_err, act_norm], axis=-1).astype(np.float32)
    assert frame.shape[1] == OBS_FRAME_DIM

    use_tactile = "tactile" in obs_list
    if use_tactile:
        tactile_cfg = None  # threshold applied below with a fixed default; see NOTE
        tac = d["gt_tactile"].astype(np.float32)
        tac_bin = (tac > 0.01).astype(np.float32)  # matches default binary_threshold=0.01

    prop_frames = []
    tac_frames = [] if use_tactile else None
    targets = []
    sample_mask = []

    for t in range(T - 1):
        # Need obs at t (valid) and target action at t+1 (valid), same segment.
        ok = valid[t] and valid[t + 1] and (seg_id[t] == seg_id[t + 1])
        sample_mask.append(ok)
        if not ok:
            prop_frames.append(np.zeros((obs_stack, OBS_FRAME_DIM), dtype=np.float32))
            if use_tactile:
                tac_frames.append(np.zeros((obs_stack, tac_bin.shape[1]), dtype=np.float32))
            targets.append(np.zeros(13, dtype=np.float32))
            continue

        # Stack the last `obs_stack` frames within this segment, cold-starting by
        # replicating the first frame of the segment (matches deque-based stacking
        # at episode start in my_policy_node.py / FrameStack).
        seg = seg_id[t]
        lo = t
        while lo - 1 >= 0 and valid[lo - 1] and seg_id[lo - 1] == seg:
            lo -= 1
        stack_idx = [max(lo, t - obs_stack + 1 + i) for i in range(obs_stack)]
        prop_frames.append(frame[stack_idx])
        if use_tactile:
            tac_frames.append(tac_bin[stack_idx])
        targets.append(act_norm[t + 1])

    prop = np.stack(prop_frames, axis=0).reshape(len(prop_frames), -1)  # (N, 52*obs_stack)
    targets = np.stack(targets, axis=0)  # (N, 13)
    sample_mask = np.array(sample_mask, dtype=bool)

    obs = {"prop": prop[sample_mask]}
    if use_tactile:
        tac_stacked = np.stack(tac_frames, axis=0).reshape(len(tac_frames), -1)
        obs["tactile"] = tac_stacked[sample_mask]

    return obs, targets[sample_mask].astype(np.float32)


# =============================================================================
# Model construction (mirrors roto/my_policy_node.py:684-724, offline).
# =============================================================================

def build_models(agent_cfg: dict, device: str):
    obs_cfg = agent_cfg["observations"]
    obs_list = obs_cfg["obs_list"]
    obs_stack = obs_cfg.get("obs_stack", 1)

    observation_space = {"prop": np.zeros(OBS_FRAME_DIM * obs_stack, dtype=np.float32)}
    if "tactile" in obs_list:
        # NUM_TACTILE inferred from the dataset at load time; caller must patch this in
        # before construction if tactile is used (see main()).
        observation_space["tactile"] = np.zeros(0, dtype=np.float32)  # placeholder, patched later

    action_space = np.zeros(len(CONTROL_JOINT_NAMES), dtype=np.float32)

    encoder_cfg = {"encoder": agent_cfg["encoder"]}
    policy_cfg = dict(agent_cfg["policy"])

    encoder = Encoder(observation_space, action_space, {}, encoder_cfg, device=device)
    policy = GaussianPolicy(
        z_dim=encoder.num_outputs,
        observation_space=observation_space,
        action_space=action_space,
        device=device,
        **policy_cfg,
    )
    return encoder, policy, obs_list, obs_stack


# =============================================================================
# Training.
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Behavior-clone fine-tune a policy checkpoint.")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--data", type=str, required=True, help="Path to a .aligned.npz file.")
    parser.add_argument("--agent_cfg_dir", type=str,
                         default=os.path.join(os.path.dirname(__file__),
                                               "..", "roto", "tasks", "baoding",
                                               "agents", "shadowlite"))
    parser.add_argument("--agent_cfg", type=str, default="rl_only_pt")
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--val_frac", type=float, default=0.1,
                         help="Fraction of timesteps (time-split, tail) held out for validation.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    agent_cfg = load_agent_cfg(os.path.abspath(args.agent_cfg_dir), args.agent_cfg)
    obs_cfg = agent_cfg["observations"]
    obs_list = obs_cfg["obs_list"]
    obs_stack = obs_cfg.get("obs_stack", 1)

    print(f"[INFO] obs_list={obs_list} obs_stack={obs_stack}")

    obs_np, targets_np = load_npz_dataset(args.data, obs_list, obs_stack)
    n_total = targets_np.shape[0]
    print(f"[INFO] Loaded {n_total} samples from {args.data}")
    print(f"[INFO] prop dim = {obs_np['prop'].shape[1]} (expected {OBS_FRAME_DIM * obs_stack})")
    assert obs_np["prop"].shape[1] == OBS_FRAME_DIM * obs_stack

    encoder, policy, _, _ = build_models(agent_cfg, args.device)
    if "tactile" in obs_np:
        # Rebuild encoder now that we know the true tactile width from the data.
        num_tactile = obs_np["tactile"].shape[1]
        observation_space = {
            "prop": np.zeros(OBS_FRAME_DIM * obs_stack, dtype=np.float32),
            "tactile": np.zeros(num_tactile, dtype=np.float32),
        }
        action_space = np.zeros(len(CONTROL_JOINT_NAMES), dtype=np.float32)
        encoder_cfg = {"encoder": agent_cfg["encoder"]}
        encoder = Encoder(observation_space, action_space, {}, encoder_cfg, device=args.device)

    checkpoint = torch.load(args.checkpoint, map_location=args.device)
    encoder.load_state_dict(checkpoint["encoder"])
    policy.load_state_dict(checkpoint["policy"])
    print(f"[INFO] Loaded checkpoint: {args.checkpoint}")

    encoder = encoder.to(args.device)
    policy = policy.to(args.device)

    # Freeze the encoder; fine-tune only the policy head.
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad_(False)
    policy.train()

    obs_t = {k: torch.tensor(v, dtype=torch.float32, device=args.device) for k, v in obs_np.items()}
    targets_t = torch.tensor(targets_np, dtype=torch.float32, device=args.device)

    n_val = int(round(n_total * args.val_frac))
    n_train = n_total - n_val
    train_idx = torch.arange(0, n_train)
    val_idx = torch.arange(n_train, n_total)
    print(f"[INFO] train={n_train} val={n_val}")

    def compute_loss(idx):
        batch_obs = {k: v[idx] for k, v in obs_t.items()}
        with torch.no_grad():
            z = encoder(batch_obs)
        pred, _, _ = policy.act(z, deterministic=True)
        return F.mse_loss(pred, targets_t[idx])

    with torch.no_grad():
        baseline_train = compute_loss(train_idx).item()
        baseline_val = compute_loss(val_idx).item() if n_val > 0 else float("nan")
    print(f"[INFO] baseline MSE: train={baseline_train:.6f} val={baseline_val:.6f}")

    opt = torch.optim.Adam(policy.parameters(), lr=args.lr)

    for epoch in range(args.epochs):
        perm = train_idx[torch.randperm(n_train)]
        epoch_loss = 0.0
        n_batches = 0
        for start in range(0, n_train, args.batch_size):
            batch_idx = perm[start:start + args.batch_size]
            batch_obs = {k: v[batch_idx] for k, v in obs_t.items()}
            with torch.no_grad():
                z = encoder(batch_obs)
            pred, _, _ = policy.act(z, deterministic=True)
            loss = F.mse_loss(pred, targets_t[batch_idx])

            opt.zero_grad()
            loss.backward()
            opt.step()

            epoch_loss += loss.item()
            n_batches += 1

        if epoch % max(1, args.epochs // 20) == 0 or epoch == args.epochs - 1:
            policy.eval()
            with torch.no_grad():
                val_loss = compute_loss(val_idx).item() if n_val > 0 else float("nan")
            policy.train()
            print(f"[INFO] epoch {epoch:4d}  train_mse={epoch_loss / n_batches:.6f}  val_mse={val_loss:.6f}")

    policy.eval()
    with torch.no_grad():
        final_train = compute_loss(train_idx).item()
        final_val = compute_loss(val_idx).item() if n_val > 0 else float("nan")
    print(f"[INFO] final MSE: train={final_train:.6f} (baseline {baseline_train:.6f})  "
          f"val={final_val:.6f} (baseline {baseline_val:.6f})")

    output = args.output
    if output is None:
        ckpt_dir = os.path.dirname(os.path.abspath(args.checkpoint))
        ckpt_stem = os.path.splitext(os.path.basename(args.checkpoint))[0]
        data_stem = os.path.splitext(os.path.basename(args.data))[0].replace(".aligned", "")
        output = os.path.join(ckpt_dir, f"{ckpt_stem}__ft_{data_stem}.pt")

    out_ckpt = copy.deepcopy(checkpoint)
    out_ckpt["policy"] = {k: v.detach().cpu() for k, v in policy.state_dict().items()}
    torch.save(out_ckpt, output)
    print(f"[INFO] Saved fine-tuned checkpoint -> {output}")


if __name__ == "__main__":
    main()

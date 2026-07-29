"""Train an encoder via frozen-target multi-step latent prediction (sim-gap closing).

Pipeline: obs -> online_encoder -> z -> [forward_model chained H steps] -> loss against a
FROZEN sim encoder's z of the true future observation.

    z_0        = online_enc(obs[t])
    z_hat_h    = forward_model(z_hat_{h-1}, action[t+h-1])         for h = 1..H  (chained)
    target_h   = frozen_enc(obs[t+h])                              (no grad)
    loss       = sum_h  gamma**h * MSE(z_hat_h, target_h)          (z_hat_0 := z_0)

NOTE on why there's no projector head (unlike the repo's existing ForwardDynamics SSL task):
that class compares against a self-referential EMA copy of the SAME network, where an
asymmetric predictor head is needed to avoid representational collapse (the classic BYOL
concern). Here the target is a FIXED, independently-trained encoder -- not a moving copy of
the online one -- so there is nothing to "cheat" by collapsing, and no need for that
asymmetry. Critically, at inference only raw `online_enc(obs)` is ever used (no projector), so
the loss must act directly on raw `z` -- inserting a projector here would let a learned
head absorb all the pressure while raw z drifts unconstrained, making the loss numerically nice
without ever closing the actual sim gap in what gets deployed. (Confirmed empirically: with a
projector, the 1-step gap `MSE(online_enc(obs), frozen_enc(obs))` INCREASED over training even
as the projected training loss fell.)

At inference only `online_enc(obs)` is ever used -- forward_model/horizon exist purely to shape
a better encoder during training (see fine-tune plan for the full rationale).

Phase 1 (this script's default data mode): SIM-ONLY pairs -- input and target observations both
come from the same sim trajectory. This proves the multi-step training machinery end-to-end
before hardware data (which needs the not-yet-built alignment step) is wired in. Once
`align.py` exists, `--input_domain hw` switches z_0's input to aligned hardware observations
while the target/rollout-action supervision stays on the sim side.

The encoder architecture (hiddens, layernorm, input dim) is INFERRED from the frozen
checkpoint's own state_dict shapes, not hardcoded -- so this script is invariant to encoder
size (works for any early-fusion prop[+tactile] MLP encoder of this family).
"""

import argparse
import copy
import glob
import os
import re

import numpy as np
import torch
import torch.nn.functional as F

import align as al
import obs_build as ob
from multimodal_rl.models.dynamics import DynamicsMLP
from multimodal_rl.models.encoder import Encoder
from multimodal_rl.models.mlp import MLP
from multimodal_rl.models.running_standard_scaler import RunningStandardScaler

NUM_ACTIONS = 13


def infer_encoder_arch(encoder_state_dict):
    """Read (input_dim, hiddens, layernorm) off the encoder's own Linear/LayerNorm shapes."""
    linear = []
    for k, v in encoder_state_dict.items():
        m = re.match(r"net\.(\d+)\.weight$", k)
        if m and v.dim() == 2:
            linear.append((int(m.group(1)), v))
    linear.sort(key=lambda x: x[0])
    if not linear:
        raise ValueError("No Linear layers found in encoder state_dict under 'net.*.weight'")

    hiddens = [v.shape[0] for _, v in linear]
    input_dim = linear[0][1].shape[1]

    has_layernorm = any(
        f"net.{i + 1}.weight" in encoder_state_dict
        and encoder_state_dict[f"net.{i + 1}.weight"].dim() == 1
        for i, _ in linear
    )
    return input_dim, hiddens, has_layernorm


def build_encoder(input_dim, hiddens, layernorm, obs_dims, device):
    observation_space = {k: np.zeros(d, dtype=np.float32) for k, d in obs_dims.items()}
    action_space = np.zeros(NUM_ACTIONS, dtype=np.float32)
    encoder_cfg = {
        "encoder": {
            "method": "early",
            "hiddens": hiddens,
            "activations": ["elu"] * len(hiddens),
            "layernorm": layernorm,
            "state_preprocessor": None,
        }
    }
    enc = Encoder(observation_space, action_space, {}, encoder_cfg, device=device)
    assert enc.num_inputs == input_dim, (
        f"Rebuilt encoder input dim {enc.num_inputs} != checkpoint's {input_dim} "
        f"(obs_dims={obs_dims})"
    )
    return enc


def infer_policy_arch(policy_state_dict):
    """Read (input_dim, hiddens_incl_output, layernorm) off the policy's own
    `policy_net.*` Linear/LayerNorm shapes (same technique as infer_encoder_arch).

    GaussianPolicy.act(z, deterministic=True) returns `mean_actions =
    self.policy_net(z)` directly -- no sampling, no clipping (policy_value.py:131-
    135) -- so only `policy_net`'s architecture is needed to reproduce it; the
    log_std_parameter and other stochastic-path state are irrelevant here.
    """
    linear = []
    for k, v in policy_state_dict.items():
        m = re.match(r"policy_net\.(\d+)\.weight$", k)
        if m and v.dim() == 2:
            linear.append((int(m.group(1)), v))
    linear.sort(key=lambda x: x[0])
    if not linear:
        raise ValueError("No Linear layers found in policy state_dict under 'policy_net.*.weight'")

    hiddens = [v.shape[0] for _, v in linear]  # last entry = num_actions
    input_dim = linear[0][1].shape[1]           # should equal the encoder's z_dim

    has_layernorm = any(
        f"policy_net.{i + 1}.weight" in policy_state_dict
        and policy_state_dict[f"policy_net.{i + 1}.weight"].dim() == 1
        for i, _ in linear
    )
    return input_dim, hiddens, has_layernorm


def build_policy_head(input_dim, hiddens, layernorm, device):
    """Rebuild just the policy's deterministic action head (the `policy_net` MLP).

    `hiddens` already includes the final num_actions layer (GaussianPolicy appends
    it before building `policy_net` -- policy_value.py:77-79); the final activation
    is always "identity" there, matching the trained checkpoint exactly.
    """
    activations = ["elu"] * (len(hiddens) - 1) + ["identity"]
    return MLP(input_dim, hiddens, activations, layernorm=layernorm).to(device)


def resolve_obs_stack_and_tactile(input_dim, prop_dim, tactile_dim):
    """Auto-detect (obs_stack, use_tactile) purely from the checkpoint's own input_dim --
    no CLI flag or hardcoded assumption needed. A prop+tactile encoder has
    input_dim == (prop_dim+tactile_dim)*obs_stack (e.g. 304 == (52+24)*4); a prop-only
    encoder (no "tactile" key in its observation_space at all) has
    input_dim == prop_dim*obs_stack (e.g. 208 == 52*4). Prop+tactile is checked first
    since it's the more specific (larger frame) match.
    """
    frame_with_tactile = prop_dim + tactile_dim
    if input_dim % frame_with_tactile == 0:
        return input_dim // frame_with_tactile, True
    if input_dim % prop_dim == 0:
        return input_dim // prop_dim, False
    raise ValueError(
        f"input_dim={input_dim} isn't a multiple of prop_dim={prop_dim} or "
        f"prop_dim+tactile_dim={frame_with_tactile} -- can't infer obs_stack/use_tactile."
    )


def load_sim_seeds(sim_dir, seeds=None):
    """Returns dict[seed] -> (prop_stacked (T,D), tactile_stacked (T,D), action (T,13))."""
    paths = sorted(glob.glob(os.path.join(sim_dir, "sim_policy_log_*_seed*.npz")))
    out = {}
    for p in paths:
        m = re.search(r"seed(\d+)", os.path.basename(p))
        if not m:
            continue
        seed = int(m.group(1))
        if seeds is not None and seed not in seeds:
            continue
        d = np.load(p, allow_pickle=True)
        prop, tactile, action = ob.build_sim_frames(d)
        out[seed] = (prop, tactile, action)
    return out


def load_hw_seeds_aligned(hw_dir, sim_dir, frozen_enc, frozen_policy, obs_stack, device, seeds=None,
                           use_tactile=True):
    """Returns dict[seed] -> (prop (T_hw,D), tactile (T_hw,D), action (T_hw,13),
                              hw_to_sim (T_hw,) int64 index into the matching sim seed's
                              600 frames).

    Each hw seed needs a matching `sim_policy_log_*_seed{seed}.npz` in sim_dir -- used
    both to DTW-align (align.align_seed, on achieved position; see align.py for why not
    commanded) and as the frozen-target/forward-model-action source during training
    (sim's own dynamics are what's being distilled). Hw seeds without a matching sim
    file are skipped with a warning. The last_action block of `prop` is reconstructed
    via the frozen-encoder/policy replay (ob.build_hw_frames), not approximated from the
    logged command, so it lands in the same raw sim-space last_action uses.
    """
    hw_paths = sorted(glob.glob(os.path.join(hw_dir, "*.aligned*.npz")))
    sim_paths = sorted(glob.glob(os.path.join(sim_dir, "sim_policy_log_*_seed*.npz")))
    sim_path_by_seed = {}
    for p in sim_paths:
        m = re.search(r"seed(\d+)", os.path.basename(p))
        if m:
            sim_path_by_seed[int(m.group(1))] = p

    out = {}
    for p in hw_paths:
        m = re.search(r"seed(\d+)", os.path.basename(p))
        if not m:
            continue
        seed = int(m.group(1))
        if seeds is not None and seed not in seeds:
            continue
        if seed not in sim_path_by_seed:
            print(f"[WARN] no matching sim seed{seed} in {sim_dir}; skipping hw file {os.path.basename(p)}")
            continue
        hw_npz = np.load(p, allow_pickle=True)
        sim_npz = np.load(sim_path_by_seed[seed], allow_pickle=True)

        hw_to_sim = al.align_seed(sim_npz, hw_npz)
        prop, tactile, action = ob.build_hw_frames(
            hw_npz, frozen_encoder=frozen_enc, frozen_policy_head=frozen_policy,
            obs_stack=obs_stack, device=device, use_tactile=use_tactile,
        )
        out[seed] = (prop, tactile, action, hw_to_sim)
    return out


def pad_stack_hw_seeds(seed_list, hw_data, obs_stack):
    """Stack per-seed hw (prop, tactile, hw_to_sim) into padded (num_seeds, T_max, ...)
    arrays plus a (num_seeds,) valid-length array.

    Hw sequences differ in length per seed (1725-1760 in the collected data, unlike
    sim's uniform 600), so they can't be np.stack'd directly. Padding repeats each
    seed's last frame / last alignment index; sample_batch_hw only ever draws indices
    below valid_len, so the padded tail is never actually trained on -- it exists only
    so downstream indexing can use a single rectangular array.
    """
    props_raw = [ob.stack_frames(hw_data[s][0], obs_stack) for s in seed_list]
    tacs_raw = [ob.stack_frames(hw_data[s][1], obs_stack) for s in seed_list]
    hw_to_sim_raw = [hw_data[s][3] for s in seed_list]
    valid_len = np.array([p.shape[0] for p in props_raw], dtype=np.int64)
    T_max = int(valid_len.max())

    n = len(seed_list)
    props = np.zeros((n, T_max, props_raw[0].shape[1]), dtype=np.float32)
    tacs = np.zeros((n, T_max, tacs_raw[0].shape[1]), dtype=np.float32)
    hw_to_sim = np.zeros((n, T_max), dtype=np.int64)
    for i, (p, t, h) in enumerate(zip(props_raw, tacs_raw, hw_to_sim_raw)):
        L = p.shape[0]
        props[i, :L], props[i, L:] = p, p[-1]
        tacs[i, :L], tacs[i, L:] = t, t[-1]
        hw_to_sim[i, :L], hw_to_sim[i, L:] = h, h[-1]
    return props, tacs, hw_to_sim, valid_len


def sample_batch_hw(valid_len, hw_to_sim, horizon, sim_T, batch_size):
    """Sample (seed_idx, t_idx_hw) uniformly from each seed's valid hw range, restricted
    to frames whose ALIGNED sim index leaves room for the full horizon rollout in sim's
    own (uniform, 600-frame) timeline: hw_to_sim[seed, t] + horizon < sim_T.
    """
    n_seeds = len(valid_len)
    seed_idx = np.random.randint(0, n_seeds, size=batch_size)
    t_idx = np.empty(batch_size, dtype=np.int64)
    max_sim = sim_T - horizon - 1
    for k, s in enumerate(seed_idx):
        # hw_to_sim is monotonic non-decreasing (see align.dtw_align) -> searchsorted
        # finds the largest valid hw prefix in one step.
        cutoff = np.searchsorted(hw_to_sim[s, :valid_len[s]], max_sim, side="right")
        cutoff = max(int(cutoff), 1)  # t=0 always satisfies (hw_to_sim[...,0] == 0)
        t_idx[k] = np.random.randint(0, cutoff)
    return seed_idx, t_idx


def zero_prop_block(prop_array, block_start, obs_stack, frame_dim=ob.OBS_FRAME_DIM, block_width=13):
    """Zero a 13-wide sub-block (e.g. the pos_err block) in every one of the `obs_stack`
    stacked 52-wide frames within a (num_seeds, T, frame_dim*obs_stack) prop array.
    Mutates and returns `prop_array`.
    """
    for f in range(obs_stack):
        s = f * frame_dim + block_start
        prop_array[:, :, s:s + block_width] = 0.0
    return prop_array


def rollout_loss(online_enc, frozen_enc, forward_model, z_scaler,
                  prop_all, tactile_all, action_all, seed_idx, t_idx, horizon, gamma, device,
                  use_tactile=True):
    """seed_idx, t_idx: 1-D int arrays of length B, indexing into the (num_seeds, T, ...) arrays.

    z_scaler: a FIXED (already-fitted, never updated here) RunningStandardScaler over the
    frozen encoder's own latent distribution. Measured: raw z has a 6.4x max/min per-dim
    std ratio across the 256 dims, so unweighted MSE lets a handful of high-variance
    dimensions dominate the gradient. Whitening by fixed per-dim stats is a linear,
    invertible reweighting -- the global minimum (z == target elementwise) is IDENTICAL
    either way, so this only changes gradient priority across dimensions, not what's
    ultimately being matched (the deployed z's raw magnitude is still what h=0 pursues).

    use_tactile: must match the encoder's own architecture -- a prop-only checkpoint's
    observation_space has no "tactile" key at all, so it must never be passed one.
    """
    def obs_at(h):
        obs = {"prop": torch.tensor(prop_all[seed_idx, t_idx + h], dtype=torch.float32, device=device)}
        if use_tactile:
            obs["tactile"] = torch.tensor(tactile_all[seed_idx, t_idx + h], dtype=torch.float32, device=device)
        return obs

    obs0 = obs_at(0)
    z = online_enc(obs0)
    with torch.no_grad():
        target0 = frozen_enc(obs0)
    # h=0: direct term, WHITENED z vs whitened frozen target -- no projector. Whitening is a
    # fixed diagonal reweighting (see z_scaler note above), so this is still literally the
    # deployed representation being matched, just with fairer per-dimension gradient weight.
    z_w = z_scaler(z, train=False, no_grad=False)
    target0_w = z_scaler(target0, train=False)
    loss = F.mse_loss(z_w, target0_w)

    # h>=1: DynamicsMLP is a residual predictor (`state + delta`). Chained H times from an
    # UNTRAINED forward_model, the norm compounds roughly geometrically (empirically ~17.8 ->
    # ~1.4e10 over 30 unnormalised steps -- verified by direct measurement), which both
    # destabilises optimisation and risks fp32 overflow at longer horizons. Re-normalising
    # z_hat to unit norm after every step breaks that feedback loop (the model can still learn
    # meaningful direction changes, just not blow up in magnitude); gradients still flow back
    # through the normalise op into raw z, so these terms still shape the deployed encoder too.
    # Whiten BEFORE unit-normalising so "direction" is measured in fairly-scaled coordinates.
    z_hat = F.normalize(z_w, dim=-1)
    for h in range(1, horizon + 1):
        a_prev = torch.tensor(action_all[seed_idx, t_idx + h - 1], dtype=torch.float32, device=device)
        z_hat = forward_model(z_hat, a_prev)
        z_hat = F.normalize(z_hat, dim=-1)
        with torch.no_grad():
            target_h = F.normalize(z_scaler(frozen_enc(obs_at(h)), train=False), dim=-1)
        loss = loss + (gamma ** h) * F.mse_loss(z_hat, target_h)

    return loss


def direct_gap(online_enc, frozen_enc, prop_all, tactile_all, seed_idx, t_idx, device, use_tactile=True):
    """1-step latent gap: MSE(online_enc(obs), frozen_enc(obs)) -- no rollout, no projector."""
    obs = {"prop": torch.tensor(prop_all[seed_idx, t_idx], dtype=torch.float32, device=device)}
    if use_tactile:
        obs["tactile"] = torch.tensor(tactile_all[seed_idx, t_idx], dtype=torch.float32, device=device)
    with torch.no_grad():
        return F.mse_loss(online_enc(obs), frozen_enc(obs)).item()


def rollout_loss_hw(online_enc, frozen_enc, forward_model, z_scaler,
                     hw_prop_all, hw_tac_all, hw_to_sim_all,
                     sim_prop_all, sim_tac_all, sim_action_all,
                     seed_idx, t_idx_hw, horizon, gamma, device, use_tactile=True):
    """Like rollout_loss, but h=0's ONLINE input comes from a hardware observation while
    every frozen target (h=0..horizon) AND the forward-model's action supervision come
    from the DTW-ALIGNED sim trajectory -- sim's own dynamics/actions are the ground
    truth being distilled, exactly as in the sim-only path; only h=0's input domain
    differs. `seed_idx` must index consistently into both the hw_* and sim_* arrays
    (same seed order -- caller must build both from the identical seed_list).

    z_scaler: see rollout_loss's docstring -- a fixed per-dim whitening over the frozen
    encoder's own latent distribution, applied here identically.
    use_tactile: see rollout_loss's docstring.
    """
    t_idx_sim = hw_to_sim_all[seed_idx, t_idx_hw]  # (B,) aligned anchor into sim's 0..599

    def sim_obs(h):
        obs = {"prop": torch.tensor(sim_prop_all[seed_idx, t_idx_sim + h], dtype=torch.float32, device=device)}
        if use_tactile:
            obs["tactile"] = torch.tensor(sim_tac_all[seed_idx, t_idx_sim + h], dtype=torch.float32, device=device)
        return obs

    hw_obs0 = {"prop": torch.tensor(hw_prop_all[seed_idx, t_idx_hw], dtype=torch.float32, device=device)}
    if use_tactile:
        hw_obs0["tactile"] = torch.tensor(hw_tac_all[seed_idx, t_idx_hw], dtype=torch.float32, device=device)
    z = online_enc(hw_obs0)
    with torch.no_grad():
        target0 = frozen_enc(sim_obs(0))
    z_w = z_scaler(z, train=False, no_grad=False)
    target0_w = z_scaler(target0, train=False)
    loss = F.mse_loss(z_w, target0_w)

    z_hat = F.normalize(z_w, dim=-1)
    for h in range(1, horizon + 1):
        a_prev = torch.tensor(sim_action_all[seed_idx, t_idx_sim + h - 1], dtype=torch.float32, device=device)
        z_hat = forward_model(z_hat, a_prev)
        z_hat = F.normalize(z_hat, dim=-1)
        with torch.no_grad():
            target_h = F.normalize(z_scaler(frozen_enc(sim_obs(h)), train=False), dim=-1)
        loss = loss + (gamma ** h) * F.mse_loss(z_hat, target_h)

    return loss


def direct_gap_hw(online_enc, frozen_enc, hw_prop_all, hw_tac_all, hw_to_sim_all,
                   sim_prop_all, sim_tac_all, seed_idx, t_idx_hw, device, use_tactile=True):
    """1-step gap: MSE(online_enc(hw_obs), frozen_enc(sim_obs @ aligned target)) -- no rollout."""
    t_idx_sim = hw_to_sim_all[seed_idx, t_idx_hw]
    hw_obs = {"prop": torch.tensor(hw_prop_all[seed_idx, t_idx_hw], dtype=torch.float32, device=device)}
    sim_obs = {"prop": torch.tensor(sim_prop_all[seed_idx, t_idx_sim], dtype=torch.float32, device=device)}
    if use_tactile:
        hw_obs["tactile"] = torch.tensor(hw_tac_all[seed_idx, t_idx_hw], dtype=torch.float32, device=device)
        sim_obs["tactile"] = torch.tensor(sim_tac_all[seed_idx, t_idx_sim], dtype=torch.float32, device=device)
    with torch.no_grad():
        return F.mse_loss(online_enc(hw_obs), frozen_enc(sim_obs)).item()


def main():
    parser = argparse.ArgumentParser(description="Frozen-target multi-step latent encoder training.")
    parser.add_argument("--frozen_encoder", type=str, required=True,
                         help="Checkpoint (dict with 'encoder' key) whose encoder is the frozen sim ground truth.")
    parser.add_argument("--sim_dir", type=str,
                         default=os.path.join(os.path.dirname(__file__), "data", "sim"))
    parser.add_argument("--input_domain", type=str, default="sim", choices=["sim", "hw"],
                         help="'sim': input+target both sim (Phase 1, proves the training "
                              "machinery). 'hw': input is a DTW-aligned hardware observation, "
                              "target/forward-model actions stay on the sim side.")
    parser.add_argument("--hw_dir", type=str,
                         default=os.path.join(os.path.dirname(__file__), "data", "hw"),
                         help="Only used when --input_domain hw.")
    parser.add_argument("--hw_tactile_source", type=str, default="hw", choices=["hw", "sim"],
                         help="'hw' (default): use the real hardware tactile array. 'sim': replace "
                              "it with the DTW-ALIGNED sim tactile array, making that one prop-block "
                              "byte-identical between domains -- an experiment to isolate how much of "
                              "the gap is closeable from pos/vel/err/action alone, decoupled from the "
                              "tactile sim<->hw sensing gap. NOTE: an encoder fine-tuned this way expects "
                              "sim-perfect tactile at deployment, not real noisy hw readings -- for "
                              "diagnosing the ceiling, not (yet) a deployment-ready checkpoint.")
    parser.add_argument("--zero_err_block", action="store_true", default=False,
                         help="Zero the pos_err block (dims 26:39 of each 52-wide prop frame) in "
                              "BOTH sim and hw prop arrays. pos_err is ~21x larger in raw scale on "
                              "sim (unbounded actions) than hw (tight position control) -- measured "
                              "to be the 2nd-largest identified contributor to the untrained gap after "
                              "tactile (~9% of it). Unlike tactile, there's no clean 'matched' value to "
                              "substitute (it's not a shared sensor reading), so this zeroes it out in "
                              "both domains rather than aligning it, to isolate what's left in pos/vel/act.")
    parser.add_argument("--val_seeds", type=str, default=None,
                         help="Comma-separated seeds held out for validation (default: highest available seed).")
    parser.add_argument("--horizon", type=int, default=30)
    parser.add_argument("--gamma", type=float, default=0.8)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--steps_per_epoch", type=int, default=100)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--lr_factor", type=float, default=1 / 1.5,
                         help="ReduceLROnPlateau: multiply lr by this when val_1step_gap plateaus. "
                              "Default matches this codebase's own KLAdaptiveLR convention "
                              "(multimodal_rl/rl/kl_adaptive_scheduler.py uses a gentle 1.5x step, "
                              "not a 2x/halving cut) -- val_1step_gap is noisy epoch-to-epoch, so an "
                              "aggressive factor (e.g. 0.5) mistakes normal fluctuation for a genuine "
                              "plateau and crushes the lr toward min_lr too early.")
    parser.add_argument("--lr_patience", type=int, default=8,
                         help="ReduceLROnPlateau: epochs with no gap improvement before cutting lr. "
                              "Kept a bit longer than the early-stop patience default's neighbourhood "
                              "since the gap metric is noisy (see --lr_factor note).")
    parser.add_argument("--min_lr", type=float, default=1e-6)
    parser.add_argument("--early_stop_patience", type=int, default=15,
                         help="Stop if val_1step_gap hasn't improved for this many epochs. "
                              "Set <=0 to disable early stopping (still tracks/saves the best checkpoint).")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    checkpoint = torch.load(args.frozen_encoder, map_location=args.device, weights_only=False)
    input_dim, hiddens, layernorm = infer_encoder_arch(checkpoint["encoder"])
    obs_stack, use_tactile = resolve_obs_stack_and_tactile(input_dim, ob.OBS_FRAME_DIM, ob.NUM_TACTILE_CHANNELS)
    obs_dims = {"prop": ob.OBS_FRAME_DIM * obs_stack}
    if use_tactile:
        obs_dims["tactile"] = ob.NUM_TACTILE_CHANNELS * obs_stack
    print(f"[INFO] inferred encoder arch: input_dim={input_dim} hiddens={hiddens} "
          f"layernorm={layernorm} obs_stack={obs_stack} use_tactile={use_tactile}")
    if not use_tactile and args.hw_tactile_source == "sim":
        print("[WARN] --hw_tactile_source sim has no effect: this checkpoint's encoder "
              "is prop-only (no tactile input at all)")

    frozen_enc = build_encoder(input_dim, hiddens, layernorm, obs_dims, args.device)
    frozen_enc.load_state_dict(checkpoint["encoder"])
    frozen_enc.eval()
    for p in frozen_enc.parameters():
        p.requires_grad_(False)

    # Frozen policy head -- used only to reconstruct hardware's raw last_action via
    # replay (ob.compute_hw_last_action); never trained, never used at inference.
    policy_input_dim, policy_hiddens, policy_layernorm = infer_policy_arch(checkpoint["policy"])
    assert policy_input_dim == hiddens[-1], (
        f"Policy input dim {policy_input_dim} != encoder z_dim {hiddens[-1]}"
    )
    frozen_policy = build_policy_head(policy_input_dim, policy_hiddens, policy_layernorm, args.device)
    policy_net_sd = {
        k[len("policy_net."):]: v
        for k, v in checkpoint["policy"].items() if k.startswith("policy_net.")
    }
    frozen_policy.load_state_dict(policy_net_sd)
    frozen_policy.eval()
    for p in frozen_policy.parameters():
        p.requires_grad_(False)
    print(f"[INFO] inferred policy arch: input_dim={policy_input_dim} hiddens={policy_hiddens} "
          f"layernorm={policy_layernorm}")

    online_enc = build_encoder(input_dim, hiddens, layernorm, obs_dims, args.device)
    online_enc.load_state_dict(checkpoint["encoder"])  # init from frozen weights
    online_enc.train()

    z_dim = hiddens[-1]
    forward_model = DynamicsMLP(state_dim=z_dim, action_dim=NUM_ACTIONS).to(args.device)

    all_seeds_data = load_sim_seeds(args.sim_dir)
    all_seeds = sorted(all_seeds_data.keys())
    print(f"[INFO] loaded sim seeds: {all_seeds}")

    hw_data = None
    if args.input_domain == "hw":
        hw_data = load_hw_seeds_aligned(args.hw_dir, args.sim_dir, frozen_enc, frozen_policy,
                                         obs_stack, args.device, use_tactile=use_tactile)
        hw_seeds = sorted(hw_data.keys())
        print(f"[INFO] loaded + DTW-aligned hw seeds: {hw_seeds}")
        all_seeds = sorted(set(all_seeds) & set(hw_seeds))
        print(f"[INFO] seeds usable for hw training (sim ∩ hw): {all_seeds}")

    if args.val_seeds is not None:
        val_seeds = [int(s) for s in args.val_seeds.split(",")]
    else:
        val_seeds = [all_seeds[-1]]
    train_seeds = [s for s in all_seeds if s not in val_seeds]
    print(f"[INFO] train_seeds={train_seeds} val_seeds={val_seeds}")

    def stack_seeds(seed_list):
        props = np.stack([ob.stack_frames(all_seeds_data[s][0], obs_stack) for s in seed_list])
        tacs = np.stack([ob.stack_frames(all_seeds_data[s][1], obs_stack) for s in seed_list])
        acts = np.stack([all_seeds_data[s][2] for s in seed_list])
        return props.astype(np.float32), tacs.astype(np.float32), acts.astype(np.float32)

    # Sim arrays are always needed: as the training domain in "sim" mode, and as the
    # frozen-target / forward-model-action source in "hw" mode.
    train_sim_prop, train_sim_tac, train_sim_act = stack_seeds(train_seeds)
    val_sim_prop, val_sim_tac, val_sim_act = stack_seeds(val_seeds)
    sim_T = train_sim_prop.shape[1]
    assert sim_T - args.horizon - 1 > 0, f"horizon {args.horizon} too large for sim sequence length {sim_T}"

    POS_ERR_BLOCK_START = 26  # prop frame layout: pos(0:13) vel(13:26) err(26:39) act(39:52)
    if args.zero_err_block:
        zero_prop_block(train_sim_prop, POS_ERR_BLOCK_START, obs_stack)
        zero_prop_block(val_sim_prop, POS_ERR_BLOCK_START, obs_stack)
        print("[INFO] --zero_err_block: pos_err zeroed in sim prop arrays "
              "(and hw prop arrays too, once loaded, if --input_domain hw)")

    # Fixed (fit once, never updated again) per-dimension whitening over the frozen
    # encoder's own latent distribution. Measured: raw z has a 6.4x max/min per-dim std
    # ratio across the 256 dims (top 10 dims alone carry 16.7% of total variance) -- plain
    # MSE lets those high-variance dims dominate the gradient. Whitening is a fixed linear
    # reweighting (global minimum z==target is unchanged), so it only rebalances gradient
    # priority across dimensions, not what the loss ultimately drives z toward.
    z_scaler = RunningStandardScaler(size=z_dim, device=args.device)
    with torch.no_grad():
        fit_obs = {
            "prop": torch.tensor(train_sim_prop.reshape(-1, train_sim_prop.shape[-1]),
                                  dtype=torch.float32, device=args.device),
        }
        if use_tactile:
            fit_obs["tactile"] = torch.tensor(train_sim_tac.reshape(-1, train_sim_tac.shape[-1]),
                                               dtype=torch.float32, device=args.device)
        z_scaler(frozen_enc(fit_obs), train=True)
    z_std = z_scaler.running_variance.sqrt()
    print(f"[INFO] fit z-whitening on {fit_obs['prop'].shape[0]} frozen sim latents "
          f"(per-dim std range [{z_std.min().item():.4f}, {z_std.max().item():.4f}])")

    opt = torch.optim.Adam(
        list(online_enc.parameters()) + list(forward_model.parameters()),
        lr=args.lr,
    )
    # Cuts lr when the held-out 1-step gap plateaus -- the metric that's actually
    # noisy/plateauing across epochs in practice, not the (smoothly-decreasing)
    # training loss, which would trigger far too late or not at all.
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="min", factor=args.lr_factor, patience=args.lr_patience, min_lr=args.min_lr,
    )

    # train_loss_fn/val_loss_fn/gap_fn close over whichever arrays this run's
    # --input_domain needs; the epoch loop below is identical either way.
    if args.input_domain == "sim":
        max_t = sim_T - args.horizon - 1

        def sample_batch(n_seeds, max_t_local, batch_size):
            seed_idx = np.random.randint(0, n_seeds, size=batch_size)
            t_idx = np.random.randint(0, max_t_local, size=batch_size)
            return seed_idx, t_idx

        def train_loss_fn():
            seed_idx, t_idx = sample_batch(len(train_seeds), max_t, args.batch_size)
            return rollout_loss(online_enc, frozen_enc, forward_model, z_scaler,
                                 train_sim_prop, train_sim_tac, train_sim_act, seed_idx, t_idx,
                                 args.horizon, args.gamma, args.device, use_tactile=use_tactile)

        def val_loss_fn():
            seed_idx, t_idx = sample_batch(len(val_seeds), val_sim_prop.shape[1] - args.horizon - 1, 256)
            return rollout_loss(online_enc, frozen_enc, forward_model, z_scaler,
                                 val_sim_prop, val_sim_tac, val_sim_act, seed_idx, t_idx,
                                 args.horizon, args.gamma, args.device, use_tactile=use_tactile).item()

        def gap_fn():
            seed_idx, t_idx = sample_batch(len(val_seeds), val_sim_prop.shape[1] - 1, 512)
            return direct_gap(online_enc, frozen_enc, val_sim_prop, val_sim_tac, seed_idx, t_idx, args.device,
                               use_tactile=use_tactile)

    else:  # hw
        train_hw_prop, train_hw_tac, train_hw_to_sim, train_valid_len = pad_stack_hw_seeds(
            train_seeds, hw_data, obs_stack)
        val_hw_prop, val_hw_tac, val_hw_to_sim, val_valid_len = pad_stack_hw_seeds(
            val_seeds, hw_data, obs_stack)
        print(f"[INFO] hw seed lengths: train={dict(zip(train_seeds, train_valid_len.tolist()))} "
              f"val={dict(zip(val_seeds, val_valid_len.tolist()))}")

        if args.zero_err_block:
            zero_prop_block(train_hw_prop, POS_ERR_BLOCK_START, obs_stack)
            zero_prop_block(val_hw_prop, POS_ERR_BLOCK_START, obs_stack)
            print("[INFO] --zero_err_block: pos_err also zeroed in hw prop arrays")

        if args.hw_tactile_source == "sim":
            # Replace the hw tactile array with the DTW-aligned sim tactile, per seed, so
            # that block is byte-identical between domains -- isolates the rest of the gap.
            for i in range(len(train_seeds)):
                train_hw_tac[i] = train_sim_tac[i, train_hw_to_sim[i]]
            for i in range(len(val_seeds)):
                val_hw_tac[i] = val_sim_tac[i, val_hw_to_sim[i]]
            print("[INFO] --hw_tactile_source sim: hw tactile REPLACED with DTW-aligned sim "
                  "tactile for both train and val (tactile block gap == 0 by construction)")

        def train_loss_fn():
            seed_idx, t_idx = sample_batch_hw(train_valid_len, train_hw_to_sim, args.horizon,
                                               sim_T, args.batch_size)
            return rollout_loss_hw(online_enc, frozen_enc, forward_model, z_scaler,
                                    train_hw_prop, train_hw_tac, train_hw_to_sim,
                                    train_sim_prop, train_sim_tac, train_sim_act,
                                    seed_idx, t_idx, args.horizon, args.gamma, args.device,
                                    use_tactile=use_tactile)

        def val_loss_fn():
            seed_idx, t_idx = sample_batch_hw(val_valid_len, val_hw_to_sim, args.horizon, sim_T, 256)
            return rollout_loss_hw(online_enc, frozen_enc, forward_model, z_scaler,
                                    val_hw_prop, val_hw_tac, val_hw_to_sim,
                                    val_sim_prop, val_sim_tac, val_sim_act,
                                    seed_idx, t_idx, args.horizon, args.gamma, args.device,
                                    use_tactile=use_tactile).item()

        def gap_fn():
            seed_idx, t_idx = sample_batch_hw(val_valid_len, val_hw_to_sim, 0, sim_T, 512)
            return direct_gap_hw(online_enc, frozen_enc,
                                  val_hw_prop, val_hw_tac, val_hw_to_sim,
                                  val_sim_prop, val_sim_tac, seed_idx, t_idx, args.device,
                                  use_tactile=use_tactile)

    baseline_gap = gap_fn()
    print(f"[INFO] baseline (init) 1-step val gap: {baseline_gap:.6f}")

    best_gap = baseline_gap
    best_epoch = -1
    best_encoder_state = copy.deepcopy(online_enc.state_dict())
    best_forward_model_state = copy.deepcopy(forward_model.state_dict())
    epochs_since_improve = 0

    for epoch in range(args.epochs):
        online_enc.train()
        epoch_loss = 0.0
        for step in range(args.steps_per_epoch):
            loss = train_loss_fn()
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(online_enc.parameters()) + list(forward_model.parameters()),
                max_norm=1.0,
            )
            opt.step()
            epoch_loss += loss.item()

        online_enc.eval()
        with torch.no_grad():
            val_loss = val_loss_fn()
            gap = gap_fn()

        lr_before = opt.param_groups[0]["lr"]
        scheduler.step(gap)
        lr_after = opt.param_groups[0]["lr"]

        improved = gap < best_gap
        if improved:
            best_gap, best_epoch = gap, epoch
            best_encoder_state = copy.deepcopy(online_enc.state_dict())
            best_forward_model_state = copy.deepcopy(forward_model.state_dict())
            epochs_since_improve = 0
        else:
            epochs_since_improve += 1

        if epoch % max(1, args.epochs // 20) == 0 or epoch == args.epochs - 1 or improved:
            marker = " *" if improved else ""
            print(f"[INFO] epoch {epoch:4d}  train_loss={epoch_loss / args.steps_per_epoch:.6f}  "
                  f"val_loss={val_loss:.6f}  val_1step_gap={gap:.6f}{marker}")
        if lr_after < lr_before:
            print(f"[INFO] epoch {epoch:4d}  lr reduced {lr_before:.2e} -> {lr_after:.2e} "
                  f"(no gap improvement for {args.lr_patience} epochs)")

        if args.early_stop_patience > 0 and epochs_since_improve >= args.early_stop_patience:
            print(f"[INFO] early stopping at epoch {epoch} "
                  f"({epochs_since_improve} epochs since last improvement)")
            break

    print(f"[INFO] final val_1step_gap={gap:.6f}  best val_1step_gap={best_gap:.6f} "
          f"at epoch {best_epoch}  (baseline {baseline_gap:.6f})")

    output = args.output
    if output is None:
        ckpt_dir = os.path.join(os.path.dirname(__file__), "checkpoints")
        os.makedirs(ckpt_dir, exist_ok=True)
        stem = os.path.splitext(os.path.basename(args.frozen_encoder))[0]
        output = os.path.join(ckpt_dir, f"{stem}__fd_h{args.horizon}.pt")

    # Save the BEST checkpoint (by held-out val_1step_gap), not just whatever the
    # final epoch happened to land on -- the observed training behaviour plateaus/
    # oscillates after an early best, so "final" and "best" are often different epochs.
    out_ckpt = copy.deepcopy(checkpoint)
    out_ckpt["encoder"] = {k: v.detach().cpu() for k, v in best_encoder_state.items()}
    out_ckpt["forward_model"] = {k: v.detach().cpu() for k, v in best_forward_model_state.items()}
    torch.save(out_ckpt, output)
    print(f"[INFO] Saved BEST checkpoint (epoch {best_epoch}, gap {best_gap:.6f}) -> {output}")


if __name__ == "__main__":
    main()

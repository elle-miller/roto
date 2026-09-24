#!/usr/bin/env python3
"""Fit rh_FFJ1's own Kp/Kd (currently just aliased from rh_FFJ2, see
genan/pd_gains.py's _J1_ALIAS) by sim-rollout position-matching against real
free-space data -- the SAME method shadow_pd_id/src/{collect_rollouts,
optimize,loss}.py already use for the 13 policy joints (drive the sim with
the real recorded command, sample candidate gains, pick whichever makes the
SIMULATED joint move like the REAL joint did), just applied to a joint that
can't be excited in isolation (J1 only moves as the tendon consequence of J2
being driven -- see shadow_pd_id/config/joints.yaml's coupled_groups comment).

Only rh_FFJ1's stiffness/damping are swept per candidate. Every other one of
the 16 joints -- including rh_FFJ2 itself, and all 13 policy joints -- keeps
whatever gains are already active in the loaded SHADOW_HAND_LITE_CFG (the 13
policy joints' real identified {Kp,Kd}, confirmed already baked in per
shadow_pd_id/DECISIONS.md 2026-07-10; rh_MFJ1/rh_RFJ1 stay at their current
J2-alias placeholder). So although the whole hand moves during the rollout
(free-space data, not single-joint excitation), only ONE joint's dynamics are
actually unknown -- the loss is scored on rh_FFJ1's own position column only,
exactly like shadow_pd_id's per-joint loss, just computed on multi-joint data
instead of isolated-joint data (rh_FFJ1 physically cannot be excited alone).

KNOWN SIMPLIFICATION (not fixed here): real inter-finger self-contact during
free-space motion (see roto/scripts/diag_self_contact.py) is not detected or
down-weighted -- any sim/real contact-timing mismatch could bias the fit.
Left as a follow-up if the resulting gains don't validate well.

Usage:
    python fit_j1_gains.py --headless --n_candidates 24 --fit_traj_idx 0 1 2 3 4
"""

import argparse
import os
import sys

sys.stdout.reconfigure(line_buffering=True)

from isaaclab.app import AppLauncher

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_ROTO_ROOT = os.path.dirname(_THIS_DIR)
_GENAN_DIR = os.path.join(_ROTO_ROOT, "genan")
_SHADOW_PD_ID_SRC = os.path.join(_ROTO_ROOT, "shadow_pd_id", "src")

parser = argparse.ArgumentParser(description="Fit rh_FFJ1's Kp/Kd from free-space data.")
parser.add_argument("--joint", type=str, default="rh_FFJ1", help="Which J1 mimic joint to fit.")
parser.add_argument(
    "--config", type=str, default=os.path.join(_GENAN_DIR, "agents", "shadowlite", "default.yaml"),
    help="Agent yaml providing dataset.paths (same free-space data GenAN trains on).",
)
parser.add_argument("--dataset", type=str, action="append", default=None, help="Override dataset.paths (repeatable).")
parser.add_argument("--n_candidates", type=int, default=24, help="Latin-Hypercube sample count.")
parser.add_argument("--sampling_seed", type=int, default=0)
parser.add_argument("--kp_bounds", type=float, nargs=2, default=[0.1, 30.0])
parser.add_argument("--kd_bounds", type=float, nargs=2, default=[0.0001, 2.0])
parser.add_argument("--fit_traj_idx", type=int, nargs="+", default=[0, 1, 2, 3, 4], help="Segments to fit on.")
parser.add_argument("--test_traj_idx", type=int, nargs="+", default=[5, 6], help="Held-out segments to report on.")
parser.add_argument("--velocity_weight", type=float, default=0.1, help="Matches shadow_pd_id/config/optim.yaml.")
parser.add_argument("--warmup_samples", type=int, default=30)
parser.add_argument("--warmup_weight", type=float, default=0.1)
parser.add_argument("--unstable_penalty", type=float, default=1.0e6)
parser.add_argument("--max_steps_per_segment", type=int, default=600, help="Cap per-segment rollout length.")
parser.add_argument("--out", type=str, default=None, help="Output gains yaml path (default: shadow_pd_id/results/params/<joint>_gains.yaml).")
# --device is registered by AppLauncher.add_app_launcher_args below.

AppLauncher.add_app_launcher_args(parser)
args_cli, _unused = parser.parse_known_args()
args_cli.num_envs = 1
args_cli.video = False
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import numpy as np  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402
from scipy.stats import qmc  # noqa: E402

sys.path.insert(0, _THIS_DIR)
sys.path.insert(0, _ROTO_ROOT)
sys.path.insert(0, _SHADOW_PD_ID_SRC)
from common_utils import LOG_PATH, make_env, update_env_cfg  # noqa: E402
from multimodal_rl.tools.writer import Writer  # noqa: E402

from loss import compute_loss  # noqa: E402

from roto.tasks import uan_shadowlite  # noqa: E402,F401
from roto.tasks.uan_shadowlite.task import UANShadowLiteEnvCfg  # noqa: E402


def build_env(agent_cfg: dict) -> object:
    env_cfg = UANShadowLiteEnvCfg()
    env_cfg.dataset = dict(agent_cfg["dataset"])
    env_cfg.uan = dict(agent_cfg.get("uan", {}))
    env_cfg.uan["action_scale"] = 1.0
    env_cfg.uan["residual_clip"] = 1000.0
    env_cfg.uan["reset_to_random"] = False
    env_cfg.num_eval_envs = 0

    writer = Writer(agent_cfg, play=True)
    env_cfg = update_env_cfg(args_cli, env_cfg, agent_cfg)

    args_cli.task = "UAN_Shadowlite"
    args_cli.gym_env_id = "UAN_Shadowlite"
    return make_env(agent_cfg, env_cfg, writer, args_cli)


def reset_to_segment(env, traj_idx: int) -> None:
    """Identical pattern to fit_torque_scale.py's reset_to_segment (see that
    file's docstring for why joint state must be rewritten, not just traj_t).
    """
    env.reset(hard=True)
    unwrapped = env.unwrapped
    t0 = unwrapped.dataset.traj_starts[traj_idx].expand(env.num_envs).clone()
    unwrapped.traj_t[:] = t0
    t = unwrapped.dataset.clamp(t0)
    q0 = unwrapped.dataset.q_meas[t]
    qd0 = unwrapped.dataset.q_meas_vel[t]
    full_pos = unwrapped.robot.data.joint_pos.clone()
    full_pos[:, unwrapped.actuated_dof_indices] = q0
    full_vel = torch.zeros_like(full_pos)
    full_vel[:, unwrapped.actuated_dof_indices] = qd0
    unwrapped.robot.write_joint_state_to_sim(full_pos, full_vel)
    unwrapped.robot.set_joint_position_target(full_pos)


def set_candidate_gains(robot, joint_id: int, kp: float, kd: float, device) -> None:
    stiffness = torch.full((robot.num_instances, 1), float(kp), device=device)
    damping = torch.full((robot.num_instances, 1), float(kd), device=device)
    robot.write_joint_stiffness_to_sim(stiffness, joint_ids=[joint_id])
    robot.write_joint_damping_to_sim(damping, joint_ids=[joint_id])


def rollout_segment(env, traj_idx: int, joint_sim_idx: int, max_steps: int) -> tuple[np.ndarray, np.ndarray]:
    """PURE PD -- actions are always zero (no residual/torque injection). Only
    the candidate joint's stiffness/damping (set by the caller beforehand)
    differs between calls; every other joint uses whatever's already active
    in the loaded SHADOW_HAND_LITE_CFG. Returns (sim_q, real_q) 1-D arrays
    for `joint_sim_idx` only.
    """
    unwrapped = env.unwrapped
    n_steps = min(int(unwrapped.dataset.traj_lengths[traj_idx].item()) - 1, max_steps)
    sim_q, real_q = [], []
    with torch.inference_mode():
        reset_to_segment(env, traj_idx)
        actions = torch.zeros(env.num_envs, unwrapped.cfg.num_actions, device=env.device)
        for _ in range(n_steps):
            t = unwrapped.dataset.clamp(unwrapped.traj_t)
            real_q.append(unwrapped.dataset.q_meas[t][0, joint_sim_idx].item())
            env.step(actions)
            sim_q.append(unwrapped.joint_pos[0, unwrapped.actuated_dof_indices][joint_sim_idx].item())
    return np.array(sim_q), np.array(real_q)


def score_candidate(env, kp: float, kd: float, robot_joint_id: int, joint_sim_idx: int,
                     traj_indices: list[int], max_steps: int, control_rate_hz: float,
                     loss_kwargs: dict, device) -> tuple[float, dict]:
    set_candidate_gains(env.unwrapped.robot, robot_joint_id, kp, kd, device)
    per_seg_losses = []
    for ti in traj_indices:
        sim_q, real_q = rollout_segment(env, ti, joint_sim_idx, max_steps)
        loss, subs = compute_loss(sim_q, real_q, control_rate_hz, **loss_kwargs)
        per_seg_losses.append(loss)
        if subs["unstable"]:
            break
    mean_loss = float(np.mean(per_seg_losses))
    return mean_loss, {"per_segment": per_seg_losses}


def main() -> None:
    with open(args_cli.config) as f:
        agent_cfg = yaml.safe_load(f)
    if args_cli.dataset is not None:
        agent_cfg["dataset"]["paths"] = args_cli.dataset
    agent_cfg["log_path"] = LOG_PATH
    agent_cfg["experiment"]["video_dir"] = None

    env = build_env(agent_cfg)
    unwrapped = env.unwrapped
    device = unwrapped.device

    joint_names_env = [unwrapped.robot.joint_names[i] for i in unwrapped.actuated_dof_indices]
    joint_sim_idx = joint_names_env.index(args_cli.joint)
    robot_joint_id = unwrapped.robot.joint_names.index(args_cli.joint)
    control_rate_hz = 1.0 / unwrapped.dataset.rl_dt
    n_segments = int(unwrapped.dataset.traj_starts.shape[0])
    print(f"[INFO] {n_segments} free-space segments loaded, control_rate_hz={control_rate_hz:.2f}")
    print(f"[INFO] Fitting {args_cli.joint}: sim actuated_dof_idx={joint_sim_idx}, robot joint id={robot_joint_id}")

    orig_kp = unwrapped.robot.data.joint_stiffness[0, robot_joint_id].item()
    orig_kd = unwrapped.robot.data.joint_damping[0, robot_joint_id].item()
    print(f"[INFO] Current (pre-fit) gains for {args_cli.joint}: kp={orig_kp:.4f} kd={orig_kd:.4f} "
          f"(likely the J2-alias placeholder)")

    loss_kwargs = dict(
        velocity_weight=args_cli.velocity_weight, warmup_samples=args_cli.warmup_samples,
        warmup_weight=args_cli.warmup_weight, unstable_penalty=args_cli.unstable_penalty,
    )

    sampler = qmc.LatinHypercube(d=2, seed=args_cli.sampling_seed)
    unit_samples = sampler.random(n=args_cli.n_candidates)
    lows = np.array([args_cli.kp_bounds[0], args_cli.kd_bounds[0]])
    highs = np.array([args_cli.kp_bounds[1], args_cli.kd_bounds[1]])
    candidates = qmc.scale(unit_samples, lows, highs)

    print(f"[INFO] Evaluating {args_cli.n_candidates} candidates on fit segments {args_cli.fit_traj_idx}")
    results = []
    for i, (kp, kd) in enumerate(candidates):
        loss, subs = score_candidate(
            env, kp, kd, robot_joint_id, joint_sim_idx, args_cli.fit_traj_idx,
            args_cli.max_steps_per_segment, control_rate_hz, loss_kwargs, device,
        )
        results.append({"kp": float(kp), "kd": float(kd), "loss": loss})
        print(f"  [{i+1:3d}/{args_cli.n_candidates}] kp={kp:8.4f} kd={kd:7.4f} -> loss={loss:.6f}")

    stable = [r for r in results if r["loss"] < args_cli.unstable_penalty]
    if not stable:
        raise RuntimeError("Every candidate was unstable -- widen bounds or check the env setup.")
    best = min(stable, key=lambda r: r["loss"])
    print(f"\n[RESULT] Best: kp={best['kp']:.4f} kd={best['kd']:.4f} loss={best['loss']:.6f} "
          f"({len(stable)}/{args_cli.n_candidates} stable)")

    print(f"\n[RESULT] === Held-out check, segments {args_cli.test_traj_idx} ===")
    test_loss, test_subs = score_candidate(
        env, best["kp"], best["kd"], robot_joint_id, joint_sim_idx, args_cli.test_traj_idx,
        args_cli.max_steps_per_segment, control_rate_hz, loss_kwargs, device,
    )
    print(f"[RESULT] Held-out loss={test_loss:.6f}  per_segment={test_subs['per_segment']}")

    print("\n[RESULT] === Reference: current J2-alias gains on the SAME fit segments ===")
    ref_loss, _ = score_candidate(
        env, orig_kp, orig_kd, robot_joint_id, joint_sim_idx, args_cli.fit_traj_idx,
        args_cli.max_steps_per_segment, control_rate_hz, loss_kwargs, device,
    )
    print(f"[RESULT] J2-alias (kp={orig_kp:.4f} kd={orig_kd:.4f}) loss={ref_loss:.6f}  "
          f"vs fitted loss={best['loss']:.6f}  (lower is better)")

    out_path = args_cli.out or os.path.join(
        _ROTO_ROOT, "shadow_pd_id", "results", "params", f"{args_cli.joint}_gains.yaml"
    )
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        yaml.safe_dump({
            "joint_name": args_cli.joint,
            "kp": best["kp"],
            "kd": best["kd"],
            "fc": 0.0,
            "fv": 0.0,
            "loss": best["loss"],
            "loss_by_type": {"free_space": best["loss"]},
            "held_out_loss": test_loss,
            "reference_j2_alias_loss": ref_loss,
            "n_candidates_evaluated": args_cli.n_candidates,
            "n_stable": len(stable),
            "fit_traj_idx": args_cli.fit_traj_idx,
            "test_traj_idx": args_cli.test_traj_idx,
            "note": ("Selected by fit_j1_gains.py: sim-rollout position-matching on free-space "
                     "multi-joint data (J1 cannot be excited in isolation), scored on this joint's "
                     "own position column only. Fc fixed at 0.0 (not fit -- see module docstring)."),
        }, f, default_flow_style=False, sort_keys=False)
    print(f"\n[INFO] Saved to {out_path}")

    env.close()


if __name__ == "__main__":
    try:
        main()
    except Exception as err:
        print("ERROR DURING FIT:", err)
        raise
    finally:
        simulation_app.close()

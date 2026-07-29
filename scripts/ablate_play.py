"""Ablation harness for the Baoding proprioceptive policy: does it sense, or just cycle?

The checkpoint's encoder input is purely proprioceptive -- 4 stacked 52-dim frames of
[normalised joint pos (13), normalised joint vel (13), joint pos error (13), previous
action (13)] -- with no ball/goal channels at all (see roto/tasks/roto_env.py
_get_proprioception). This script answers two questions about a trained checkpoint:

  1. Which of those 4 blocks does the policy actually rely on? (zero/freeze/noise-mask
     any combination of blocks, closed-loop, and watch what happens to task performance)
  2. Does ball contact change the motion at all, or does the hand just run an open-loop
     cycle? (--no_ball parks both balls out of contact and disables drop-termination so
     the hand keeps moving; compare its trajectory to a normal run)

Every run auto-records a video tagged with its condition (ablate/mode/no_ball/seed), so
each test leaves behind a clip you can watch -- pass --no_video to skip that for a fast,
many-env metrics-only sweep.

Videos are saved under ./ablation/, tagged with their condition (ablate/mode/no_ball/seed)
so successive tests never overwrite each other. On a headless server (no display), pass
--headless too -- that's what makes off-screen video capture possible.

Usage:
    # Visual check of one condition (small num_envs, saves ablation/ablate-..._<ts>.mp4)
    python ablate_play.py --checkpoint <ckpt> --agent_cfg rl_only_pt --ablate pos_error --headless

    # No-ball trajectory comparison
    python ablate_play.py --checkpoint <ckpt> --agent_cfg rl_only_pt --no_ball \
        --log_traj noball.npz --headless

    # Metrics sweep (many envs, no video)
    python ablate_play.py --checkpoint <ckpt> --agent_cfg rl_only_pt \
        --ablate vel,pos_error,prev_action --ablate_mode freeze --no_video --num_envs 256 --headless
"""

import argparse
import glob
import math
import os
import shutil
import sys
import time

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(
    description="Ablate proprioception blocks / remove balls and play a Baoding policy checkpoint."
)
parser.add_argument("--task", type=str, default="Baoding")
parser.add_argument("--robot", type=str, default="shadowlite")
parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint.")
parser.add_argument("--agent_cfg", type=str, default="rl_only_pt", help="Name of the agent configuration.")
parser.add_argument("--num_envs", type=int, default=None, help="Default: 4 if recording video, else 256.")
parser.add_argument("--episodes", type=int, default=5, help="Number of episode-length windows to run.")
parser.add_argument("--seed", type=int, default=None)

parser.add_argument(
    "--ablate", type=str, default="",
    help="Comma list of proprioception blocks to remove: pos,vel,pos_error,prev_action",
)
parser.add_argument("--ablate_mode", type=str, default="zero", choices=["zero", "freeze", "noise"])
parser.add_argument(
    "--no_ball", action="store_true", default=False,
    help="Park both balls out of contact and suppress drop-termination; hand runs to time-out.",
)
parser.add_argument(
    "--log_traj", type=str, default=None,
    help="If set, dump env-0 actions/joint_pos/joint_pos_cmd/pos_error/num_rotations to this .npz path.",
)
parser.add_argument(
    "--print_pos_error", action="store_true", default=False,
    help="Print env-0 per-control-joint pos_error (the exact obs block, rad) live during rollout.",
)
parser.add_argument(
    "--print_every", type=int, default=30,
    help="Steps between --print_pos_error readouts (RL runs ~60 Hz, so 30 ~ 2 Hz).",
)

parser.add_argument("--video", dest="video", action="store_true", default=True, help="Record a video (default: on).")
parser.add_argument("--no_video", dest="video", action="store_false", help="Disable video recording.")
parser.add_argument(
    "--video_length", type=int, default=None,
    help="Video length in steps (default: one full episode, computed from episode_length_s).",
)

parser.add_argument("--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O.")
parser.add_argument(
    "--renderer", type=str, default="RayTracedLighting", choices=["RayTracedLighting", "PathTracing"],
)
parser.add_argument("--samples_per_pixel_per_frame", type=int, default=1)

AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
if args_cli.video:
    args_cli.enable_cameras = True
    if not args_cli.headless:
        print(
            "[WARN] --video is on without --headless. Off-screen video capture needs "
            "--headless (with --enable_cameras, set automatically here) unless a real "
            "display is attached. On a remote/server box, re-run with --headless."
        )
sys.argv = [sys.argv[0]] + hydra_args
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import numpy as np
import torch

import isaaclab_tasks  # noqa: F401
from common_utils import (
    LOG_PATH,
    load_hand_task_agent_cfg,
    make_env,
    make_models,
    register_hand_task_to_hydra,
    resolve_gym_env_id,
    set_seed,
    update_env_cfg,
)
from isaaclab.utils import update_dict

from multimodal_rl.rl.ppo import PPO, PPO_DEFAULT_CONFIG
from multimodal_rl.tools.writer import Writer

OBS_FRAME_DIM = 52  # pos(13) + vel(13) + pos_error(13) + prev_action(13)
BLOCKS = {"pos": (0, 13), "vel": (13, 26), "pos_error": (26, 39), "prev_action": (39, 52)}

# RecordVideo (via common_utils.make_env) always writes to ./videos/; every run's clip is
# moved+renamed out of there into this directory so ablation runs stay in one place.
ABLATION_VIDEO_DIR = "./ablation"

# Where balls are teleported to for --no_ball, relative to each env's origin: to the
# side and above the drop-termination height so they never re-enter the hand's reach.
BALL_PARK_OFFSET = (0.5, 0.5, 0.5)


def build_ablate_cols(ablate_arg: str, obs_stack: int) -> list[int]:
    names = [n.strip() for n in ablate_arg.split(",") if n.strip()]
    for n in names:
        if n not in BLOCKS:
            raise ValueError(f"Unknown ablate block {n!r}. Choose from {sorted(BLOCKS)}.")
    cols = []
    for frame_idx in range(obs_stack):
        base = frame_idx * OBS_FRAME_DIM
        for n in names:
            lo, hi = BLOCKS[n]
            cols.extend(range(base + lo, base + hi))
    return cols


def build_video_tag(args_cli) -> str:
    blocks = args_cli.ablate.replace(",", "+") or "none"
    parts = [f"ablate-{blocks}", args_cli.ablate_mode]
    if args_cli.no_ball:
        parts.append("noball")
    seed = args_cli.seed if args_cli.seed is not None else "default"
    parts.append(f"seed{seed}")
    parts.append(time.strftime("%Y%m%d-%H%M%S"))
    return "_".join(parts)


def main():
    args_cli.gym_env_id = resolve_gym_env_id(args_cli.task, args_cli.robot)
    if args_cli.task in ("Bounce", "Baoding"):
        env_cfg, agent_cfg = register_hand_task_to_hydra(args_cli.task, args_cli.robot, "default_cfg")
        specialised_cfg = load_hand_task_agent_cfg(args_cli.task, args_cli.robot, args_cli.agent_cfg)
    else:
        raise ValueError("ablate_play.py only supports --task Bounce/Baoding.")
    agent_cfg = update_dict(agent_cfg, specialised_cfg)
    dtype = torch.float32

    agent_cfg["seed"] = args_cli.seed if args_cli.seed is not None else agent_cfg["seed"]
    set_seed(agent_cfg["seed"])
    agent_cfg["log_path"] = LOG_PATH
    agent_cfg["experiment"]["video_dir"] = None

    # Video defaults to on; many-env visual recordings are cluttered/slow, so keep
    # num_envs small unless the caller opts out of video for a metrics-only sweep.
    if args_cli.num_envs is None:
        args_cli.num_envs = 4 if args_cli.video else 256

    env_cfg = update_env_cfg(args_cli, env_cfg, agent_cfg)
    env_cfg.num_eval_envs = 0

    # Video length must be known before make_env wraps RecordVideo; compute one
    # full episode's worth of steps from the cfg (mirrors roto_env.py's own print).
    if args_cli.video_length is None:
        steps_per_s = 1.0 / (env_cfg.sim.dt * env_cfg.decimation)
        args_cli.video_length = int(math.ceil(env_cfg.episode_length_s * steps_per_s)) + 1

    video_dir = "./videos/"
    existing_videos = set()
    if args_cli.video:
        os.makedirs(video_dir, exist_ok=True)
        existing_videos = set(glob.glob(os.path.join(video_dir, "*.mp4")))

    writer = Writer(agent_cfg, play=True)
    env = make_env(agent_cfg, env_cfg, writer, args_cli)

    obs_stack = agent_cfg["observations"]["obs_stack"]
    cols = build_ablate_cols(args_cli.ablate, obs_stack)
    if cols:
        print(f"[INFO] Ablating blocks={args_cli.ablate} mode={args_cli.ablate_mode} "
              f"({len(cols)} of {OBS_FRAME_DIM * obs_stack} obs dims)")
    if args_cli.no_ball:
        print("[INFO] --no_ball: balls parked out of contact, drop-termination suppressed")

    policy, value, encoder, value_preprocessor = make_models(env, env_cfg, agent_cfg, dtype)

    ppo_cfg = PPO_DEFAULT_CONFIG.copy()
    ppo_cfg.update(agent_cfg["agent"])
    agent = PPO(
        encoder, policy, value, value_preprocessor,
        memory=None, cfg=ppo_cfg,
        observation_space=env.observation_space,
        action_space=env.action_space,
        device=env.device,
        writer=writer, ssl_task=None, dtype=dtype,
        debug=agent_cfg["experiment"]["debug"],
    )

    resume_path = os.path.abspath(args_cli.checkpoint)
    agent.load(resume_path)
    print(f"[INFO] Loaded checkpoint: {resume_path}")

    raw = env.env.unwrapped
    device = env.device
    num_envs = env.num_envs

    # --- no-ball hooks: suppress physics termination, teleport balls aside -------
    orig_get_dones = raw._get_dones

    def _no_ball_dones():
        termination, time_out = orig_get_dones()
        return torch.zeros_like(termination), time_out

    if args_cli.no_ball:
        raw._get_dones = _no_ball_dones

    park_offset = torch.tensor(BALL_PARK_OFFSET, device=device)

    def park_balls():
        env_ids = raw.robot._ALL_INDICES
        for ball in (raw.ball_1, raw.ball_2):
            state = ball.data.default_root_state.clone()
            state[:, 0:3] = park_offset + raw.scene.env_origins
            state[:, 3:7] = torch.tensor([1.0, 0.0, 0.0, 0.0], device=device)
            state[:, 7:] = 0.0
            ball.write_root_pose_to_sim(state[:, :7], env_ids)
            ball.write_root_velocity_to_sim(state[:, 7:], env_ids)

    # --- observation ablation ----------------------------------------------------
    state_holder = {"frozen_prop": None}

    def apply_ablation(states):
        if not cols:
            return states
        prop = states["policy"]["prop"][:].clone()
        if args_cli.ablate_mode == "zero":
            prop[:, cols] = 0.0
        elif args_cli.ablate_mode == "freeze":
            prop[:, cols] = state_holder["frozen_prop"][:, cols]
        elif args_cli.ablate_mode == "noise":
            prop[:, cols] = torch.randn_like(prop[:, cols])
        states["policy"]["prop"] = prop
        return states

    def hard_reset():
        s, i = env.reset(hard=True)
        state_holder["frozen_prop"] = s["policy"]["prop"][:].clone()
        if args_cli.no_ball:
            park_balls()
        return s, i

    ep_length = raw.max_episode_length - 1
    print(f"[INFO] Episode length: {ep_length} steps, running {args_cli.episodes} episode(s)")

    actuated_idx = sorted(raw.actuated_dof_indices)
    actuated_names = [raw.robot.joint_names[i] for i in actuated_idx]

    # Exact pos_error the policy sees: joint_pos_cmd - joint_pos, sliced to the 13 control
    # joints in policy order (roto_env.py _get_proprioception uses prop_dof_indices ==
    # control_dof_indices). Raw radians -- this block is NOT normalised in the obs.
    control_idx = list(raw.control_dof_indices)
    control_names = [raw.robot.joint_names[i] for i in control_idx]
    if args_cli.print_pos_error:
        print("[INFO] pos_error control joints:", control_names)

    traj = None
    if args_cli.log_traj:
        traj = {"actions": [], "joint_pos_cmd": [], "joint_pos": [],
                "pos_error": [], "num_rotations": [], "terminated": []}

    with torch.inference_mode():
        states, infos = hard_reset()

    all_returns, all_rotations, all_drop_rate, all_survival = [], [], [], []

    for ep in range(args_cli.episodes):
        returns = torch.zeros((num_envs, 1), device=device)
        mask = torch.ones((num_envs, 1), device=device)
        term_step = torch.full((num_envs,), ep_length, dtype=torch.long, device=device)

        for t in range(ep_length):
            if not simulation_app.is_running():
                break
            with torch.inference_mode():
                states = apply_ablation(states)

                # What the policy actually receives for pos_error, AFTER masking, from the
                # newest stacked frame ([26:39] within frame obs_stack-1). This is 0 when
                # pos_error is ablated with mode=zero -- proving the mask hit the input,
                # even though the real sim tracking error below stays nonzero.
                nb = (obs_stack - 1) * OBS_FRAME_DIM
                obs_pos_err_0 = states["policy"]["prop"][:][0, nb + 26: nb + 39].cpu().float().numpy().copy()

                z = encoder(states)
                actions, _, _ = agent.policy.act(z, deterministic=True)
                states, rewards, terminated, truncated, infos = env.step(actions)
                if args_cli.no_ball:
                    park_balls()

                # env-0 per-control-joint pos_error (rad) -- the REAL sim tracking error
                # (joint_pos_cmd - joint_pos); unaffected by masking the policy's input.
                pos_err_0 = raw.joint_pos_error[0, control_idx].cpu().float().numpy().copy()

                if traj is not None:
                    traj["actions"].append(raw.actions[0].cpu().float().numpy().copy())
                    traj["joint_pos_cmd"].append(raw.joint_pos_cmd[0, actuated_idx].cpu().float().numpy().copy())
                    traj["joint_pos"].append(raw.joint_pos[0, actuated_idx].cpu().float().numpy().copy())
                    traj["pos_error"].append(pos_err_0)
                    traj["num_rotations"].append(int(raw.num_rotations[0].item()))
                    traj["terminated"].append(bool(terminated[0].item()))

                if args_cli.print_pos_error and (t % args_cli.print_every == 0):
                    cells = "  ".join(f"{n.replace('rh_',''):>5}:{v:+.3f}"
                                       for n, v in zip(control_names, pos_err_0))
                    print(f"[ep{ep + 1} t{t:4d}] pos_err(rad)  {cells}  "
                          f"|sim_L2|={np.linalg.norm(pos_err_0):.3f}  "
                          f"|obs_L2|={np.linalg.norm(obs_pos_err_0):.3f}")

                this_done = torch.logical_or(terminated, truncated).squeeze(-1)
                alive_before = mask.squeeze(-1) > 0.5
                term_step[this_done & alive_before] = t

                returns += rewards * mask
                mask *= (1.0 - this_done.float()).unsqueeze(-1)

        rotations_snapshot = raw.num_rotations.clone().float()
        all_returns.append(returns.mean().item())
        all_rotations.append(rotations_snapshot.mean().item())
        all_drop_rate.append((term_step < ep_length).float().mean().item())
        all_survival.append(term_step.float().mean().item())

        print(
            f"[Episode {ep + 1}/{args_cli.episodes}] "
            f"mean_return={all_returns[-1]:.3f}  "
            f"mean_num_rotations={all_rotations[-1]:.3f}  "
            f"drop_rate={all_drop_rate[-1]:.1%}  "
            f"mean_survival_steps={all_survival[-1]:.1f}/{ep_length}"
        )

        if ep + 1 < args_cli.episodes:
            with torch.inference_mode():
                states, infos = hard_reset()

    print("\n===== SUMMARY =====")
    print(f"checkpoint:  {resume_path}")
    print(f"condition:   ablate={args_cli.ablate or 'none'} mode={args_cli.ablate_mode} no_ball={args_cli.no_ball}")
    print(f"num_envs:    {num_envs}  episodes: {args_cli.episodes}")
    print(f"mean_return:         {np.mean(all_returns):.3f}")
    print(f"mean_num_rotations:  {np.mean(all_rotations):.3f}")
    print(f"drop_rate:           {np.mean(all_drop_rate):.1%}")
    print(f"mean_survival_steps: {np.mean(all_survival):.1f} / {ep_length}")

    env.close()

    if traj is not None:
        out_path = os.path.abspath(args_cli.log_traj)
        np.savez_compressed(
            out_path,
            actions=np.array(traj["actions"], dtype=np.float32),
            joint_pos_cmd=np.array(traj["joint_pos_cmd"], dtype=np.float32),
            joint_pos=np.array(traj["joint_pos"], dtype=np.float32),
            pos_error=np.array(traj["pos_error"], dtype=np.float32),
            num_rotations=np.array(traj["num_rotations"], dtype=np.int32),
            terminated=np.array(traj["terminated"], dtype=bool),
            actuated_names=np.array(actuated_names),
            control_names=np.array(control_names),
            ablate=args_cli.ablate,
            ablate_mode=args_cli.ablate_mode,
            no_ball=args_cli.no_ball,
        )
        print(f"[INFO] Saved trajectory ({len(traj['actions'])} steps) -> {out_path}")

    if args_cli.video:
        new_videos = sorted(
            set(glob.glob(os.path.join(video_dir, "*.mp4"))) - existing_videos, key=os.path.getmtime
        )
        os.makedirs(ABLATION_VIDEO_DIR, exist_ok=True)
        tag = build_video_tag(args_cli)
        for i, f in enumerate(new_videos):
            suffix = f"_{i}" if len(new_videos) > 1 else ""
            dst = os.path.join(ABLATION_VIDEO_DIR, f"{tag}{suffix}.mp4")
            shutil.move(f, dst)
            print(f"[INFO] Saved video: {dst}")
        if not new_videos:
            print(f"[WARN] --video was set but no new .mp4 was found in {video_dir}")


if __name__ == "__main__":
    main()
    simulation_app.close()

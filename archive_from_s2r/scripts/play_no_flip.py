"""Play a trained Baoding checkpoint with its own DR (FSR corrupt_max) intact,
but with the per-step FSR flip/dither noise disabled (tactile_flip_prob_* = 0).

This is a thin variant of ablate_play.py (no ablation applied -- --ablate is
always empty here): same checkpoint, same robot/agent_cfg, same ball-mass DR,
same tactile_fsr_corrupt_max (some taxels can still be stuck this episode --
FSR is NOT "perfect"), just without the additional per-step 0/1 flip dither
on top of the stuck channels.

Usage:
    python play_no_flip.py --robot shadowlite_padtac_bt --agent_cfg rl_only_pt_padtac_bt \
        --checkpoint <ckpt> --seed 123 --headless
"""

import argparse
import glob
import math
import os
import shutil
import sys
import time

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Play a Baoding checkpoint with FSR flip-dither disabled.")
parser.add_argument("--task", type=str, default="Baoding")
parser.add_argument("--robot", type=str, default="shadowlite")
parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint.")
parser.add_argument("--agent_cfg", type=str, default="rl_only_pt", help="Name of the agent configuration.")
parser.add_argument("--num_envs", type=int, default=None, help="Default: 4 if recording video, else 256.")
parser.add_argument("--episodes", type=int, default=2, help="Number of episode-length windows to run.")
parser.add_argument("--seed", type=int, default=None)
parser.add_argument("--tag", type=str, default="run", help="Extra tag for the saved video filename.")

parser.add_argument("--video", dest="video", action="store_true", default=True, help="Record a video (default: on).")
parser.add_argument("--no_video", dest="video", action="store_false", help="Disable video recording.")
parser.add_argument("--video_length", type=int, default=None)

parser.add_argument("--disable_fabric", action="store_true", default=False)
parser.add_argument("--renderer", type=str, default="RayTracedLighting", choices=["RayTracedLighting", "PathTracing"])
parser.add_argument("--samples_per_pixel_per_frame", type=int, default=1)

AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
if args_cli.video:
    args_cli.enable_cameras = True
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

OUT_DIR = "./ablation"


def build_video_tag(args_cli) -> str:
    seed = args_cli.seed if args_cli.seed is not None else "default"
    return f"noflip_{args_cli.robot}_{args_cli.tag}_seed{seed}_" + time.strftime("%Y%m%d-%H%M%S")


def main():
    args_cli.gym_env_id = resolve_gym_env_id(args_cli.task, args_cli.robot)
    env_cfg, agent_cfg = register_hand_task_to_hydra(args_cli.task, args_cli.robot, "default_cfg")
    specialised_cfg = load_hand_task_agent_cfg(args_cli.task, args_cli.robot, args_cli.agent_cfg)
    agent_cfg = update_dict(agent_cfg, specialised_cfg)
    dtype = torch.float32

    agent_cfg["seed"] = args_cli.seed if args_cli.seed is not None else agent_cfg["seed"]
    set_seed(agent_cfg["seed"])
    agent_cfg["log_path"] = LOG_PATH
    agent_cfg["experiment"]["video_dir"] = None

    if args_cli.num_envs is None:
        args_cli.num_envs = 4 if args_cli.video else 256

    env_cfg = update_env_cfg(args_cli, env_cfg, agent_cfg)
    env_cfg.num_eval_envs = 0

    # --- disable the per-step FSR flip/dither noise, keep everything else (incl.
    # tactile_fsr_corrupt_max stuck-taxel DR, ball mass DR, coupling, etc.) as-is ---
    before = (
        getattr(env_cfg, "tactile_flip_prob_off_to_on", None),
        getattr(env_cfg, "tactile_flip_prob_on_to_off", None),
        getattr(env_cfg, "tactile_flip_prob_unsel_off_to_on", None),
        getattr(env_cfg, "tactile_flip_prob_unsel_on_to_off", None),
    )
    env_cfg.tactile_flip_prob_off_to_on = 0.0
    env_cfg.tactile_flip_prob_on_to_off = 0.0
    # tactile_flip_scope="both" dithers the (12-k) unselected pads at a separate
    # rate pair; zero it too or the "no flip" run still carries that noise.
    env_cfg.tactile_flip_prob_unsel_off_to_on = 0.0
    env_cfg.tactile_flip_prob_unsel_on_to_off = 0.0
    print(f"[INFO] tactile_flip_prob (sel_off_on, sel_on_off, unsel_off_on, unsel_on_off): {before} -> (0.0, 0.0, 0.0, 0.0)")
    print(f"[INFO] tactile_fsr_corrupt_max left as-is: {getattr(env_cfg, 'tactile_fsr_corrupt_max', None)}")

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

    def hard_reset():
        return env.reset(hard=True)

    ep_length = raw.max_episode_length - 1
    print(f"[INFO] Episode length: {ep_length} steps, running {args_cli.episodes} episode(s)")

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
                z = encoder(states)
                actions, _, _ = agent.policy.act(z, deterministic=True)
                states, rewards, terminated, truncated, infos = env.step(actions)

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
    print(f"robot:       {args_cli.robot}   agent_cfg: {args_cli.agent_cfg}")
    print(f"condition:   flip_dither=OFF  fsr_corrupt_max={getattr(env_cfg, 'tactile_fsr_corrupt_max', None)} (unchanged)")
    print(f"num_envs:    {num_envs}  episodes: {args_cli.episodes}")
    print(f"mean_return:         {np.mean(all_returns):.3f}")
    print(f"mean_num_rotations:  {np.mean(all_rotations):.3f}")
    print(f"drop_rate:           {np.mean(all_drop_rate):.1%}")
    print(f"mean_survival_steps: {np.mean(all_survival):.1f} / {ep_length}")

    env.close()

    if args_cli.video:
        new_videos = sorted(
            set(glob.glob(os.path.join(video_dir, "*.mp4"))) - existing_videos, key=os.path.getmtime
        )
        os.makedirs(OUT_DIR, exist_ok=True)
        tag = build_video_tag(args_cli)
        for i, f in enumerate(new_videos):
            suffix = f"_{i}" if len(new_videos) > 1 else ""
            dst = os.path.join(OUT_DIR, f"{tag}{suffix}.mp4")
            shutil.move(f, dst)
            print(f"[INFO] Saved video: {dst}")
        if not new_videos:
            print(f"[WARN] --video was set but no new .mp4 was found in {video_dir}")


if __name__ == "__main__":
    main()
    simulation_app.close()

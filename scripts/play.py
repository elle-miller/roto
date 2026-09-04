# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play a trained RL agent with multimodal_rl.

Author: Elle Miller 
"""


import argparse
import os
import sys

import numpy as np

from isaaclab.app import AppLauncher

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Play a checkpoint of an RL agent from skrl.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during playback.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--robot",
    type=str,
    default=None,
    help="Robot: Bounce/Baoding → shadow|shadowlite|orca|allegro; Find → franka. Defaults: shadow or franka.",
)
parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint.")
parser.add_argument("--video_dir", type=str, default=None, help="Directory to save recorded videos.")
parser.add_argument("--agent_cfg", type=str, default=None, help="Name of the agent configuration.")

parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment.")
parser.add_argument(
    "--record_steps",
    type=int,
    default=300,
    help="Number of control steps to log into sim_policy_log_seed*.npz (60 Hz → 300=5s, 600=10s, 900=15s).",
)
# Rendering options (useful for RTX5090 and similar GPUs)
parser.add_argument(
    "--renderer", type=str, default="PathTracing", choices=["RayTracedLighting", "PathTracing"], help="Renderer to use."
)
parser.add_argument("--samples_per_pixel_per_frame", type=int, default=1, help="Number of samples per pixel per frame.")

# --- Presentation overrides (camera / background / resolution) ---------------
# All default to None = leave the env cfg exactly as it ships, so training and
# every existing caller are unaffected. Used to render the ICRA comparison
# clips from a fixed front viewpoint; pick the numbers with
# scripts/random_forces/calibrate_camera.py and paste them in.
parser.add_argument("--cam_eye", type=float, nargs=3, default=None, metavar=("X", "Y", "Z"),
                    help="Camera position, world metres. Requires --cam_lookat.")
parser.add_argument("--cam_lookat", type=float, nargs=3, default=None, metavar=("X", "Y", "Z"),
                    help="Camera target, world metres. Requires --cam_eye.")
parser.add_argument("--cam_res", type=int, nargs=2, default=None, metavar=("W", "H"),
                    help="Render resolution for the recorded video, e.g. 3840 2160. Read ONCE "
                         "when the render product is built, so it is applied before make_env.")
parser.add_argument("--hdr", type=str, default=None,
                    help="HDRI background stem from roto/assets/rooms/ (no .hdr), e.g. "
                         "qwantani_dusk_2_puresky_4k. Default: whatever baoding.py sets.")

# --- Tactile ablation --------------------------------------------------------
parser.add_argument("--zero_tactile", action="store_true", default=False,
                    help="Force the env's tactile output to all-zero at the source, whatever the "
                         "agent_cfg says. Lets one tactile-trained checkpoint be run both with and "
                         "without touch, changing exactly one variable. Also the native condition "
                         "for checkpoints trained under a zero_tactile agent_cfg.")

# --- Episode length ----------------------------------------------------------
# --- Domain randomisation pinning -------------------------------------------
# BaodingShadowLitePadTacBTCfg ships with the full robustness stack ON:
# tactile_fsr_corrupt_max=8, tactile_flip_prob=0.1, ball kicks (0.1-0.3 m/s every
# 0.5-1.5 s), ball mass DR (45-100 g) and cmd slew DR (0.3-1.0). That is correct
# for training and for robustness evals, and wrong for a presentation clip or for
# any A/B where one variable is supposed to differ -- corrupted taxels in
# particular cripple a tactile policy while a prop-only one does not notice.
# These default to None = leave the cfg alone, so nothing else changes.
parser.add_argument("--ball_mass_g", type=float, default=None,
                    help="Pin BOTH balls in EVERY env to this mass (grams), overriding mass DR.")
parser.add_argument("--ball_disturb_off", action="store_true", default=False,
                    help="Force the random ball push/force disturbance DR OFF.")
parser.add_argument("--fsr_corrupt_max", type=int, default=None,
                    help="Per-episode FSR taxel corruption DR. 0 disables it (clean tactile).")
parser.add_argument("--tactile_flip_prob", type=float, default=None,
                    help="Per-step taxel dither probability, both directions. 0 = off. Needed "
                         "alongside --fsr_corrupt_max 0 for genuinely clean tactile.")
parser.add_argument("--cmd_speed_frac_range", type=float, nargs=2, default=None, metavar=("LO", "HI"),
                    help="Per-episode command-slew DR, resampled on reset. This is the real "
                         "training plant for slew-DR checkpoints -- (0.3, 1.0) for the aug4 "
                         "family. Prefer this over a pinned --cmd_speed_frac when reproducing "
                         "training conditions: a fixed value at the TOP of the range is not "
                         "representative of it, since most draws land lower and the policy "
                         "behaves very differently there.")
parser.add_argument("--cmd_speed_frac", type=float, default=None,
                    help="Pin the command-rate limiter to a fixed value (e.g. 0.6 to match the "
                         "hardware deploy SPEED_FRAC), disabling the slew DR range. Pass a "
                         "negative value to disable slew entirely, which is what pre-2026-08-01 "
                         "checkpoints need -- they were trained without it.")
# The temporal taxel debounce. The agent YAML that enabled it
# (rl_only_pt_padtac_bt_smooth.yaml, k_on=3 k_off=1) was DELETED in commit
# 7f9500f on 2026-08-16, but the env still implements it
# (shadowlite.py _init_tactile_smoothing, reads tactile_cfg.smoothing). Any
# checkpoint trained before that date under the smooth config therefore cannot
# be reproduced from the YAMLs alone -- this flag restores the setting.
parser.add_argument("--tactile_smoothing", type=int, nargs=2, default=None, metavar=("K_ON", "K_OFF"),
                    help="Enable the temporal taxel hold: a taxel must read ON for K_ON "
                         "consecutive steps before the policy sees 1, and OFF for K_OFF before it "
                         "returns to 0. The deleted smooth config used 3 1.")
parser.add_argument("--log_out", type=str, default=None,
                    help="Path for the joint+tactile npz log. Default is "
                         "sim_policy_log_seed<SEED>.npz in the cwd, which collides when several "
                         "conditions are run at the same seed -- set this per run.")
parser.add_argument("--episode_seconds", type=float, default=None,
                    help="Override episode_length_s. A clip longer than the stock 10 s otherwise "
                         "contains a reset teleport mid-video. Drop-termination stays ACTIVE: a "
                         "real failure should still be visible.")

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
if args_cli.video:
    args_cli.enable_cameras = True
sys.argv = [sys.argv[0]] + hydra_args
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app
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
from isaaclab_tasks.utils.hydra import register_task_to_hydra
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry

from multimodal_rl.rl.ppo import PPO, PPO_DEFAULT_CONFIG
from multimodal_rl.tools.writer import Writer


def main():
    """Play a trained RL agent from a checkpoint.

    Loads a checkpoint and runs the agent in the environment, optionally recording videos.
    """
    # Parse configuration
    args_cli.gym_env_id = resolve_gym_env_id(args_cli.task, args_cli.robot)
    if args_cli.task in ("Bounce", "Baoding"):
        env_cfg, agent_cfg = register_hand_task_to_hydra(args_cli.task, args_cli.robot, "default_cfg")
        specialised_cfg = load_hand_task_agent_cfg(args_cli.task, args_cli.robot, args_cli.agent_cfg)
    else:
        env_cfg, agent_cfg = register_task_to_hydra(args_cli.gym_env_id, "default_cfg")
        specialised_cfg = load_cfg_from_registry(args_cli.gym_env_id, args_cli.agent_cfg)
    agent_cfg = update_dict(agent_cfg, specialised_cfg)
    dtype = torch.float32

    # Set seed (important for seed-deterministic runs)
    agent_cfg["seed"] = args_cli.seed if args_cli.seed is not None else agent_cfg["seed"]
    set_seed(agent_cfg["seed"])
    agent_cfg["log_path"] = LOG_PATH
    agent_cfg["experiment"]["video_dir"] = args_cli.video_dir

    # Update the environment config
    env_cfg = update_env_cfg(args_cli, env_cfg, agent_cfg)

    # ----- presentation overrides (all before make_env) -----
    # Baoding._setup_scene reads the module-level _BAODING_HDR when it spawns
    # /World/bglight, so rebinding the attribute here is enough and leaves
    # baoding.py untouched for training runs.
    if args_cli.hdr is not None:
        import roto.tasks.baoding.baoding as _baoding_mod
        _hdr_path = _baoding_mod._BAODING_HDR.parent / f"{args_cli.hdr}.hdr"
        if not _hdr_path.is_file():
            raise SystemExit(f"Unknown HDRI {args_cli.hdr!r}: no such file {_hdr_path}")
        _baoding_mod._BAODING_HDR = _hdr_path
        print(f"[INFO] HDRI background -> {_hdr_path}")

    # viewer.resolution is read exactly once, lazily, when DirectRLEnv builds the
    # render product on the first render(). Setting it after that point fails
    # SILENTLY at the old resolution -- hence before make_env, and verify the
    # output with ffprobe rather than trusting this line.
    if args_cli.cam_res is not None:
        env_cfg.viewer.resolution = tuple(args_cli.cam_res)
        print(f"[INFO] render resolution -> {env_cfg.viewer.resolution}")

    if (args_cli.cam_eye is None) != (args_cli.cam_lookat is None):
        raise SystemExit("--cam_eye and --cam_lookat must be given together.")
    if args_cli.cam_eye is not None:
        env_cfg.viewer.eye = tuple(args_cli.cam_eye)
        env_cfg.viewer.lookat = tuple(args_cli.cam_lookat)
        # eye/lookat are offsets from the frame chosen by origin_type. The default
        # "world" combined with num_envs > 1 puts env 0 away from the origin and
        # silently shifts the framing; "env" pins it to env 0 regardless.
        env_cfg.viewer.origin_type = "env"
        env_cfg.viewer.env_index = 0
        print(f"[INFO] camera eye={env_cfg.viewer.eye} lookat={env_cfg.viewer.lookat} (origin_type=env)")

    if args_cli.fsr_corrupt_max is not None:
        env_cfg.tactile_fsr_corrupt_max = (
            None if args_cli.fsr_corrupt_max <= 0 else int(args_cli.fsr_corrupt_max))
        print(f"[INFO] tactile_fsr_corrupt_max -> {env_cfg.tactile_fsr_corrupt_max}")
    if args_cli.tactile_flip_prob is not None:
        env_cfg.tactile_flip_prob_off_to_on = float(args_cli.tactile_flip_prob)
        env_cfg.tactile_flip_prob_on_to_off = float(args_cli.tactile_flip_prob)
        print(f"[INFO] tactile_flip_prob -> {args_cli.tactile_flip_prob:g}")
    if args_cli.ball_disturb_off:
        env_cfg.ball_push_vel_range = None
        env_cfg.ball_push_angvel_range = None
        env_cfg.ball_force_range = None
        env_cfg.ball_torque_range = None
        print("[INFO] ball disturbance DR forced OFF")
    if args_cli.cmd_speed_frac_range is not None:
        lo, hi = args_cli.cmd_speed_frac_range
        env_cfg.cmd_speed_frac = None
        env_cfg.cmd_speed_frac_range = (float(lo), float(hi))
        print(f"[INFO] cmd slew DR -> ({lo:g}, {hi:g}) per episode")
    elif args_cli.cmd_speed_frac is not None:
        # Negative = slew OFF entirely. Checkpoints trained before the command-rate
        # slew DR landed (2026-08-01) never saw a rate-limited plant, so pinning
        # them to any fixed fraction evaluates them off-distribution.
        if args_cli.cmd_speed_frac < 0:
            env_cfg.cmd_speed_frac = None
            env_cfg.cmd_speed_frac_range = None
            print("[INFO] cmd slew forced OFF (pre-slew checkpoint plant)")
        else:
            env_cfg.cmd_speed_frac = float(args_cli.cmd_speed_frac)
            env_cfg.cmd_speed_frac_range = None
            print(f"[INFO] cmd slew pinned to {env_cfg.cmd_speed_frac:g}")

    if args_cli.episode_seconds is not None:
        print(f"[INFO] episode_length_s {env_cfg.episode_length_s:g} -> {args_cli.episode_seconds:g} s")
        env_cfg.episode_length_s = float(args_cli.episode_seconds)

    # Setup logging
    writer = Writer(agent_cfg, play=True)

    # Make environment (order: gymnasium Env -> FrameStack -> IsaacLab)
    env_cfg.num_eval_envs = 0 # don't need the visualization of eval envs
    env = make_env(agent_cfg, env_cfg, writer, args_cli)

    if args_cli.ball_mass_g is not None:
        _m = args_cli.ball_mass_g / 1000.0
        env.env.unwrapped.cfg.ball_mass_range = (_m, _m)
        print(f"[INFO] ball mass pinned to {args_cli.ball_mass_g:g} g")
    if args_cli.zero_tactile:
        _raw = env.env.unwrapped
        if _raw.tactile_cfg is None:
            _raw.tactile_cfg = {"binary_tactile": True, "binary_threshold": 0.0}
        _raw.tactile_cfg["zero_tactile"] = True
        print("[INFO] Tactile forced to ZERO at the source (--zero_tactile)")

    print("\n===== BODY NAMES =====")
    try:
        robot = env.env.unwrapped.robot
        for i, name in enumerate(robot.body_names):
            print(i, name)
    except Exception as e:
        print("Error:", e)
    print("======================\n")
    print("Joint names and limits:")
    r = env.env.unwrapped
    print([r.robot.joint_names[i] for i in r.actuated_dof_indices])
    print(r.robot_joint_pos_lower_limits, r.robot_joint_pos_upper_limits, r.robot_joint_vel_limits)

    # Setup models
    policy, value, encoder, value_preprocessor = make_models(env, env_cfg, agent_cfg, dtype)

    # Configure and instantiate PPO agent
    ppo_agent_cfg = PPO_DEFAULT_CONFIG.copy()
    ppo_agent_cfg.update(agent_cfg["agent"])
    agent = PPO(
        encoder,
        policy,
        value,
        value_preprocessor,
        memory=None,
        cfg=ppo_agent_cfg,
        observation_space=env.observation_space,
        action_space=env.action_space,
        device=env.device,
        writer=writer,
        ssl_task=None,
        dtype=dtype,
        debug=agent_cfg["experiment"]["debug"],
    )

    # Load checkpoint
    resume_path = os.path.abspath(args_cli.checkpoint)
    agent.load(resume_path)
    print(f"[INFO] Loading model checkpoint from: {resume_path}")
    modules = torch.load(resume_path, map_location=env.device)
    if isinstance(modules, dict):
        for name in modules.keys():
            print(f"  - {name}")

    # Reset environment
    timestep = 0
    ep_length = env.env.unwrapped.max_episode_length - 1

    returns = torch.zeros(size=(env.num_envs, 1), device=env.device)
    mask = torch.Tensor([[1] for _ in range(env.num_envs)]).to(env.device)

    states, infos = env.reset(hard=True)

    idx = r.actuated_dof_indices          # 16: full actuated (incl. J1 mimics)
    prop_idx = r.prop_dof_indices         # 13: policy proprio / control joints (ShadowLite)
    N_RECORD = int(args_cli.record_steps)
    print(f"[INFO] Recording {N_RECORD} steps (~{N_RECORD / 60.0:.1f}s at 60 Hz) into sim_policy_log_seed*.npz")
    # After env.step, roto_env._compute_intermediate_values() has filled
    # r.joint_vel and r.joint_pos_error (= joint_pos_cmd - joint_pos).
    rec = {
        "act": [],          # (13,) policy output (unitless)
        "q": [],            # (16,) achieved pos rad
        "cmd": [],          # (16,) joint_pos_cmd rad
        "qd": [],           # (16,) joint vel rad/s
        "pos_err": [],      # (16,) cmd - q  rad
        "q13": [],          # (13,) policy-order achieved pos
        "qd13": [],         # (13,) policy-order vel
        "cmd13": [],        # (13,) policy-order cmd
        "pos_err13": [],    # (13,) policy-order cmd - q  (matches obs cmd_error)
        "tac": [],          # tactile (binary)
        "num_rotations": [],  # cumulative rotation count
    }

    # Simulate environment
    while simulation_app.is_running():
        with torch.inference_mode():
            # Agent stepping
            z = encoder(states)
            actions, _, _ = agent.policy.act(z, deterministic=True)

            # Environment stepping
            states, rewards, terminated, truncated, infos = env.step(actions)
            # Compute evaluation rewards
            mask_update = 1 - torch.logical_or(terminated, truncated).float()

            # Update evaluation metrics
            returns += rewards * mask
            mask *= mask_update

            # Manually reset eval episodes every ep_length
            if timestep % ep_length == 0:
                mean_eval_return = returns.mean().item()
                print("Reset - Max eval return", returns.max().item())
                print("Reset - Mean eval return", mean_eval_return)
                states, infos = env.reset(hard=True)

                returns = torch.zeros(size=(env.num_envs, 1), device=env.device)
                mask = torch.Tensor([[1] for _ in range(env.num_envs)]).to(env.device)

            if len(rec["act"]) < N_RECORD:
                rec["act"].append(actions[0].detach().cpu().numpy().copy())
                rec["q"].append(r.robot.data.joint_pos[0, idx].detach().cpu().numpy().copy())
                rec["cmd"].append(r.joint_pos_cmd[0, idx].detach().cpu().numpy().copy())
                rec["qd"].append(r.joint_vel[0, idx].detach().cpu().numpy().copy())
                rec["pos_err"].append(r.joint_pos_error[0, idx].detach().cpu().numpy().copy())
                rec["q13"].append(r.joint_pos[0, prop_idx].detach().cpu().numpy().copy())
                rec["qd13"].append(r.joint_vel[0, prop_idx].detach().cpu().numpy().copy())
                rec["cmd13"].append(r.joint_pos_cmd[0, prop_idx].detach().cpu().numpy().copy())
                rec["pos_err13"].append(r.joint_pos_error[0, prop_idx].detach().cpu().numpy().copy())
                rec["tac"].append(r.tactile[0].detach().cpu().numpy().copy())
                rec["num_rotations"].append(float(r.num_rotations[0].item()))
                if len(rec["act"]) >= N_RECORD:
                    fname = args_cli.log_out or f"sim_policy_log_seed{agent_cfg['seed']}.npz"
                    np.savez(
                        fname,
                        **{k: np.array(v) for k, v in rec.items()},
                        joints=[r.robot.joint_names[i] for i in idx],
                        joints13=[r.robot.joint_names[i] for i in prop_idx],
                    )
                    print("saved", fname, len(rec["act"]), "steps")
                    _tac = np.array(rec["tac"], dtype=np.float32)
                    print("[VERIFY] tactile: mean=%.6f  frac_on=%.3f%%  zero_tactile=%s"
                          % (_tac.mean(), 100.0 * (_tac > 0.5).mean(), args_cli.zero_tactile))
                    print("[VERIFY] rotations=%.1f" % float(r.num_rotations[0].item()))
                    break

        if args_cli.video:
            # Exit the play loop after recording one video
            if timestep == args_cli.video_length:
                break

        timestep += 1

    # Close the simulator
    env.close()


if __name__ == "__main__":
    main()


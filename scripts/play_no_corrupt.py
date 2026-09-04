"""Record tactile over one episode with FSR corruption DR fully disabled.

Same checkpoint/robot as a normal play, but tactile_fsr_corrupt_max is set to
None before make_env, so no taxel is forced to a constant. Any channel that
still reads constant is genuinely never contacted by the behaviour, which is
what separates "never touched" from "DR-forced to 0".

Writes tac_no_corrupt_seed<seed>.npz alongside play.py's own log.
"""

import argparse
import math
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Record tactile with FSR corruption DR off.")
parser.add_argument("--task", type=str, default="Baoding")
parser.add_argument("--robot", type=str, default="shadowlite")
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--agent_cfg", type=str, default="rl_only_pt")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--record_steps", type=int, default=599)
parser.add_argument("--seed", type=int, default=None)
parser.add_argument("--out", type=str, default=None)
parser.add_argument("--disable_fabric", action="store_true", default=False)
parser.add_argument("--renderer", type=str, default="RayTracedLighting")
parser.add_argument("--samples_per_pixel_per_frame", type=int, default=1)
parser.add_argument("--video", action="store_true", default=False)

AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
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

    env_cfg = update_env_cfg(args_cli, env_cfg, agent_cfg)
    env_cfg.num_eval_envs = 0

    # Must be set BEFORE make_env: _init_tactile_fsr_corrupt() reads it at
    # construction, so a later write would not rebuild the corrupt buffers.
    print(f"[INFO] tactile_fsr_corrupt_max: {getattr(env_cfg, 'tactile_fsr_corrupt_max', None)} -> None")
    env_cfg.tactile_fsr_corrupt_max = None

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
    agent.load(args_cli.checkpoint)
    print(f"[INFO] Loaded checkpoint: {args_cli.checkpoint}")

    raw = env.env.unwrapped
    tac = []

    with torch.inference_mode():
        states, _ = env.reset(hard=True)
        for _ in range(int(args_cli.record_steps)):
            if not simulation_app.is_running():
                break
            z = encoder(states)
            actions, _, _ = agent.policy.act(z, deterministic=True)
            states, _, _, _, _ = env.step(actions)
            tac.append(raw.tactile[0].detach().cpu().numpy().copy())

    out = args_cli.out or f"tac_no_corrupt_seed{agent_cfg['seed']}.npz"
    np.savez(out, tac=np.array(tac))
    print(f"saved {out} {len(tac)} steps")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()

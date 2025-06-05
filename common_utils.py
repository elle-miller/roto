"""Rest everything follows."""

import gymnasium as gym

# stop log files from being generated!!
import logging
import numpy as np
import os
import random
import torch
import traceback
from copy import deepcopy
from datetime import datetime

logging.getLogger("hydra").setLevel(logging.CRITICAL)
logging.getLogger("hydra._internal").setLevel(logging.CRITICAL)

from isaaclab_rl.algorithms.memories import Memory
from isaaclab_rl.algorithms.policy_value import DeterministicValue, GaussianPolicy
from isaaclab_rl.algorithms.ppo import PPO, PPO_DEFAULT_CONFIG
from isaaclab_rl.algorithms.trainer import Trainer
from isaaclab_rl.models.encoder import Encoder
from isaaclab_rl.models.running_standard_scaler import RunningStandardScaler
from isaaclab_rl.tools.writer import Writer
from isaaclab_rl.wrappers.frame_stack import FrameStack
from isaaclab_rl.wrappers.isaaclab_wrapper import IsaacLabWrapper

# ADD YOUR ENVS HERE
from tasks import franka


def make_env(env_cfg, args_cli, obs_stack=1):
    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # wrap for video recording
    if args_cli.video:
        log_dir = os.cwd()
        log_dir = "/home/elle/code/external/IsaacLab/isaaclab_rl"
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap around environment for skrl
    env = IsaacLabWrapper(env)
    return env


def make_models(env, env_cfg, agent_cfg):
    observation_space = env.observation_space["policy"]
    action_space = env.action_space

    encoder = Encoder(observation_space, action_space, env_cfg, agent_cfg["models"], device=env.device)
    z_dim = encoder.num_outputs

    policy = GaussianPolicy(
        z_dim=z_dim,
        observation_space=observation_space,
        action_space=env.action_space,
        device=env.device,
        **agent_cfg["models"]["policy"],
    )

    value = DeterministicValue(
        z_dim=z_dim,
        observation_space=observation_space,
        action_space=env.action_space,
        device=env.device,
        **agent_cfg["models"]["value"],
    )

    print("*****Encoder*****")
    print(encoder)
    print("*****RL models*****")
    print(policy)
    print(value)

    return policy, value, encoder


def make_memory(env, env_cfg, size, num_envs):
    memory = Memory(
        memory_size=size,
        num_envs=num_envs,
        device=env.device,
        env_cfg=env_cfg,
    )
    return memory


def make_agent_cfg(env, agent_cfg):

    # configure and instantiate PPO agent
    default_agent_cfg = PPO_DEFAULT_CONFIG.copy()
    agent_cfg["agent"]["rewards_shaper"] = None  # avoid 'dictionary changed size during iteration'
    default_agent_cfg.update(agent_cfg["agent"])
    default_agent_cfg["state_preprocessor"] = None
    default_agent_cfg["state_preprocessor_kwargs"].update(
        {"size": env.observation_space["policy"], "device": env.device}
    )
    default_agent_cfg["rewards_shaper"] = (
        lambda rewards, *args, **kwargs: rewards * agent_cfg["agent"]["rewards_shaper_scale"]
    )
    print(default_agent_cfg)
    return default_agent_cfg


def make_trainer(env, agent, agent_cfg, auxiliary_task=None, writer=None):

    num_timesteps_M = agent_cfg["trainer"]["max_global_timesteps_M"]
    num_eval_envs = agent_cfg["trainer"]["num_eval_envs"]
    trainer = Trainer(
        env=env,
        agents=agent,
        num_timesteps_M=num_timesteps_M,
        num_eval_envs=num_eval_envs,
        auxiliary_task=auxiliary_task,
        writer=writer,
    )
    return trainer


def update_env_cfg(args_cli, env_cfg, agent_cfg):

    env_cfg.seed = agent_cfg["seed"]

    # override configurations with either config file or args
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    env_cfg.obs_list = agent_cfg["models"]["obs_list"]
    env_cfg.num_eval_envs = agent_cfg["trainer"]["num_eval_envs"]
    env_cfg.obs_stack = agent_cfg["models"]["obs_stack"]

    # variables that impact how env obs are processed
    env_cfg.normalise_prop = agent_cfg["models"]["preprocess"]["normalise_prop"]
    env_cfg.binary_tactile = agent_cfg["models"]["preprocess"]["binary_tactile"]
    
    return env_cfg


def set_seed(seed: int = 42) -> None:

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

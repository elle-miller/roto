"""Rest everything follows."""

import gymnasium as gym
import numpy as np
import os
import random
import torch

from isaaclab_rl.algorithms.memories import Memory
from isaaclab_rl.algorithms.policy_value import DeterministicValue, GaussianPolicy
from isaaclab_rl.algorithms.trainer import Trainer
from isaaclab_rl.models.encoder import Encoder
from isaaclab_rl.models.running_standard_scaler import RunningStandardScaler
from isaaclab_rl.wrappers.frame_stack import FrameStack
from isaaclab_rl.wrappers.isaaclab_wrapper import IsaacLabWrapper

# ADD YOUR ENVS HERE

# change this to something else if you want
LOG_PATH = os.getcwd()


def make_env(env_cfg, args_cli, obs_stack=1):

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # FrameStack expects a gym env
    if obs_stack > 1:
        env = FrameStack(env, num_stack=obs_stack)

    # wrap for video recording
    if args_cli.video:
        log_dir = os.cwd()
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # Isaac Lab wrapper
    env = IsaacLabWrapper(env)
    return env


def make_models(env, env_cfg, agent_cfg, dtype):
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

    value_preprocessor = RunningStandardScaler(size=1, device=env.device, dtype=dtype)

    print("*****Encoder*****")
    print(encoder)
    print("*****RL models*****")
    print(policy)
    print(value)
    print(value_preprocessor)

    return policy, value, encoder, value_preprocessor


def make_memory(env, env_cfg, size, num_envs):
    memory = Memory(
        memory_size=size,
        num_envs=num_envs,
        device=env.device,
        env_cfg=env_cfg,
    )
    return memory


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

Common helper utilities for RL scripts.

This module provides factory functions used by the training and play scripts.
Edits focus on documentation, duplicate import removal, and formatting.

"""

import os
import random
from typing import Any, Dict, Optional

import gymnasium as gym
import numpy as np
import torch

from isaaclab.utils import update_dict
from isaaclab_rl.rl.memories import Memory
from isaaclab_rl.rl.policy_value import DeterministicValue, GaussianPolicy
from isaaclab_rl.rl.trainer import Trainer
from isaaclab_rl.models.encoder import Encoder
from isaaclab_rl.models.running_standard_scaler import RunningStandardScaler
from isaaclab_rl.wrappers.frame_stack import FrameStack
from isaaclab_rl.wrappers.isaaclab_wrapper import IsaacLabWrapper
from isaaclab_rl.ssl.reconstruction import Reconstruction
from isaaclab_rl.ssl.dynamics import ForwardDynamics
from isaaclab_rl.rl.ppo import PPO, PPO_DEFAULT_CONFIG
from isaaclab_rl.tools.writer import Writer
from isaaclab_tasks.utils.hydra import hydra_task_config, register_task_to_hydra
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry


# ADD YOUR ENVS HERE
from roto.tasks import franka, shadow  # noqa: F401

# change this to something else if you want
LOG_PATH = os.getcwd()

def make_aux(env: Any,
             rl_memory: Memory,
             encoder: Encoder,
             value: DeterministicValue,
             value_preprocessor: Any,
             env_cfg: Any,
             agent_cfg: Dict[str, Any],
             writer: Writer) -> Optional[Any]:
    """
    Create the auxiliary SSL task if configured.

    Args:
        env: The environment instance.
        rl_memory: Replay/memory object used for SSL.
        encoder: Encoder model instance.
        value: Value model instance.
        value_preprocessor: Preprocessor for the value network.
        env_cfg: Environment configuration.
        agent_cfg: Agent configuration dictionary.
        writer: Writer instance for logging and videos.

    Returns:
        An SSL task instance or None when not configured.
    """
    rl_rollout = agent_cfg["agent"]["rollouts"]
    ssl_task = None

    if "ssl_task" in agent_cfg:
        task_type = agent_cfg["ssl_task"].get("type")
        if task_type == "reconstruction":
            ssl_task = Reconstruction(agent_cfg["ssl_task"], rl_rollout, rl_memory,
                                      encoder, value, value_preprocessor, env, env_cfg, writer)
        elif task_type == "forward_dynamics":
            ssl_task = ForwardDynamics(agent_cfg["ssl_task"], rl_rollout, rl_memory,
                                       encoder, value, value_preprocessor, env, env_cfg, writer)
        else:
            # Unknown auxiliary task type: intentionally permissive at top level.
            print("No auxiliary task configured or unknown type:", task_type)

    return ssl_task

def make_env(env_cfg: Any, writer: Writer, args_cli: Any, obs_stack: int = 1) -> gym.Env:
    """
    Create and configure the gymnasium environment used for training/evaluation.

    Args:
        env_cfg: Environment configuration object.
        writer: Writer instance holding video_dir and logging configuration.
        args_cli: Parsed CLI arguments.
        obs_stack: Number of observations to stack for each observation entry.

    Returns:
        A wrapped gymnasium environment ready to use.
    """
    env = gym.make(args_cli.task, cfg=env_cfg,
                   render_mode="rgb_array" if args_cli.video else None)

    obs, reward = env.reset()

    gym_dict: Dict[str, gym.Space] = {}
    for k, v in obs["policy"].items():
        obs_shape = v.shape[1] * obs_stack
        gym_dict[k] = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(obs_shape,))

    single_obs_space = gym.spaces.Dict()
    single_obs_space["policy"] = gym.spaces.Dict(gym_dict)

    obs_space = gym.vector.utils.batch_space(single_obs_space, env_cfg.scene.num_envs)
    single_action_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(env_cfg.num_actions,))
    action_space = gym.vector.utils.batch_space(single_action_space, env_cfg.scene.num_envs)
    env.unwrapped.set_spaces(single_obs_space, obs_space, single_action_space, action_space)
    env.obs_stack = obs_stack

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": writer.video_dir,
            # record only the first episode for quick debugging
            "step_trigger": lambda step: step == 0,
        }
        env = FrameStack(env, obs_stack)
        # If the project has a video wrapper, apply it here.
        # env = VideoWrapper(env, **video_kwargs)

    return env
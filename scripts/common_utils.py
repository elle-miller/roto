"""Utility helpers shared between the RoTO training / inference scripts."""

import gymnasium as gym
import numpy as np
import os
import random
import torch

from multimodal_rl.models.encoder import Encoder
from multimodal_rl.models.running_standard_scaler import RunningStandardScaler
from multimodal_rl.rl.memories import Memory
from multimodal_rl.rl.policy_value import DeterministicValue, GaussianPolicy
from multimodal_rl.rl.ppo import PPO, PPO_DEFAULT_CONFIG
from multimodal_rl.rl.trainer import Trainer
from multimodal_rl.ssl.dynamics import ForwardDynamics
from multimodal_rl.ssl.reconstruction import Reconstruction
from multimodal_rl.wrappers.frame_stack import FrameStack
from multimodal_rl.wrappers.isaaclab_wrapper import IsaacLabWrapper

# Import task modules to register environments
from roto.tasks import franka, shadow, shadow_lite  # noqa: F401

# Logging directory (change this to a custom path if desired)
LOG_PATH = os.getcwd()


def make_aux(env, rl_memory, encoder, value, value_preprocessor, env_cfg, agent_cfg, writer):
    """Instantiate the optional self-supervised auxiliary task.

    Args:
        env: The gymnasium environment.
        rl_memory: Rollout memory buffer for RL.
        encoder: Encoder network.
        value: Value network.
        value_preprocessor: Value preprocessor.
        env_cfg: Environment configuration.
        agent_cfg: Agent configuration dictionary.
        writer: Writer for logging.

    Returns:
        SSL task instance or None if no SSL task is configured.
    """
    ssl_cfg = agent_cfg.get("ssl_task")
    if not ssl_cfg:
        return None

    rl_rollout = agent_cfg["agent"]["rollouts"]
    task_type = ssl_cfg.get("type")
    task_map = {
        "reconstruction": Reconstruction,
        "forward_dynamics": ForwardDynamics,
    }

    task_cls = task_map.get(task_type)
    if task_cls is None:
        return None

    return task_cls(
        ssl_cfg,
        rl_rollout,
        rl_memory,
        encoder,
        value,
        value_preprocessor,
        env,
        env_cfg,
        writer,
    )


def make_env(agent_cfg, env_cfg, writer, args_cli):
    """Create and wrap the Isaac Lab environment with gym + writer utilities.

    Args:
        agent_cfg: Agent configuration dictionary.
        env_cfg: Environment configuration.
        writer: Writer for logging.
        args_cli: Command-line arguments.

    Returns:
        Wrapped gymnasium environment.
    """
    # Update env_cfg with observation settings from agent_cfg
    # Note: configclass instances don't have .update() method, so we assign attributes directly
    if "observations" in agent_cfg:
        obs_cfg = agent_cfg["observations"]
        env_cfg.obs_list = obs_cfg.get("obs_list", getattr(env_cfg, "obs_list", []))
        env_cfg.obs_stack = obs_cfg.get("obs_stack", getattr(env_cfg, "obs_stack", 1))
        if "pixel_cfg" in obs_cfg:
            env_cfg.pixel_cfg = obs_cfg["pixel_cfg"]
        if "tactile_cfg" in obs_cfg:
            env_cfg.tactile_cfg = obs_cfg["tactile_cfg"]

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    obs, _ = env.reset()

    # Build observation space dictionary accounting for frame stacking
    gym_dict = {}
    for k, v in obs["policy"].items():
        obs_shape = list(v.shape)
        # Multiply the last dimension (channels) by the stack size
        obs_shape[-1] = obs_shape[-1] * env.unwrapped.obs_stack
        if k == "rgb":
            gym_dict[k] = gym.spaces.Box(
                low=0,
                high=255,
                shape=obs_shape[1:],
                dtype=np.uint8,
            )
        elif k == "depth":
            gym_dict[k] = gym.spaces.Box(
                low=0.0,
                high=1.0,
                shape=obs_shape[1:],
                dtype=np.float32,
            )
        else:
            gym_dict[k] = gym.spaces.Box(low=-np.inf, high=np.inf, shape=obs_shape[1:], dtype=np.float32)

    single_obs_space = gym.spaces.Dict()
    single_obs_space["policy"] = gym.spaces.Dict(gym_dict)
    obs_space = gym.vector.utils.batch_space(single_obs_space, env_cfg.scene.num_envs)
    single_action_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(env_cfg.num_actions,))
    action_space = gym.vector.utils.batch_space(single_action_space, env_cfg.scene.num_envs)
    env.unwrapped.set_spaces(single_obs_space, obs_space, single_action_space, action_space)

    # Wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": "./videos/", 
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training to", writer.video_dir)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # Apply frame stacking if needed
    if env.unwrapped.obs_stack > 1:
        env = FrameStack(env, obs_stack=env.unwrapped.obs_stack)

    # Apply Isaac Lab wrapper
    env = IsaacLabWrapper(env, env_cfg.num_eval_envs, obs_stack=env.unwrapped.obs_stack, debug=env_cfg.debug)
    return env


def make_models(env, env_cfg, agent_cfg, dtype):
    """Build encoder, policy, and value networks.

    Args:
        env: The gymnasium environment.
        env_cfg: Environment configuration.
        agent_cfg: Agent configuration dictionary.
        dtype: Data type for tensors.

    Returns:
        Tuple of (policy, value, encoder, value_preprocessor) networks.
    """
    observation_space = env.observation_space["policy"]
    action_space = env.action_space

    encoder = Encoder(observation_space, action_space, env_cfg, agent_cfg, device=env.device)
    z_dim = encoder.num_outputs

    policy = GaussianPolicy(
        z_dim=z_dim,
        observation_space=observation_space,
        action_space=env.action_space,
        device=env.device,
        **agent_cfg["policy"],
    )

    value = DeterministicValue(
        z_dim=z_dim,
        observation_space=observation_space,
        action_space=env.action_space,
        device=env.device,
        **agent_cfg["value"],
    )

    value_preprocessor = RunningStandardScaler(size=1, device=env.device, dtype=dtype, debug=env_cfg.debug)

    print("*****Encoder*****")
    print(encoder)
    print("*****RL models*****")
    print(policy)
    print(value)
    print(value_preprocessor)

    return policy, value, encoder, value_preprocessor


def make_memory(env, env_cfg, size, num_envs):
    """Allocate rollout storage for PPO.

    Args:
        env: The gymnasium environment.
        env_cfg: Environment configuration.
        size: Size of the memory buffer (number of rollout steps).
        num_envs: Number of parallel environments.

    Returns:
        Memory buffer instance.
    """
    memory = Memory(
        memory_size=size,
        num_envs=num_envs,
        device=env.device,
        env_cfg=env_cfg,
    )
    return memory


def make_trainer(env, agent, agent_cfg, ssl_task=None, writer=None):
    """Create the high-level Trainer wrapper.

    Args:
        env: The gymnasium environment.
        agent: The RL agent (PPO).
        agent_cfg: Agent configuration dictionary.
        ssl_task: Optional self-supervised learning task.
        writer: Optional writer for logging.

    Returns:
        Trainer instance.
    """
    num_timesteps_M = agent_cfg["trainer"]["max_global_timesteps_M"]
    num_eval_envs = agent_cfg["trainer"]["num_eval_envs"]
    trainer = Trainer(
        env=env,
        agents=agent,
        agent_cfg=agent_cfg,
        num_timesteps_M=num_timesteps_M,
        num_eval_envs=num_eval_envs,
        ssl_task=ssl_task,
        writer=writer,
    )
    return trainer


def update_env_cfg(args_cli, env_cfg, agent_cfg):
    """Sync Isaac Lab config with CLI + agent overrides.

    Args:
        args_cli: Command-line arguments.
        env_cfg: Environment configuration to update.
        agent_cfg: Agent configuration dictionary.

    Returns:
        Updated environment configuration.
    """
    env_cfg.seed = agent_cfg["seed"]
    env_cfg.debug = agent_cfg["experiment"]["debug"]

    # Override configurations with either config file or CLI args
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    env_cfg.obs_list = agent_cfg["observations"]["obs_list"]
    env_cfg.num_eval_envs = agent_cfg["trainer"]["num_eval_envs"]
    env_cfg.obs_stack = agent_cfg["observations"]["obs_stack"]

    return env_cfg


def set_seed(seed: int = 42) -> None:
    """Apply the same seed across numpy/torch/random."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_one_seed(args_cli, env, agent_cfg=None, env_cfg=None, writer=None, seed=None):
    """Train the PPO agent for a single seed configuration.

    Args:
        args_cli: Command-line arguments.
        env: The gymnasium environment.
        agent_cfg: Agent configuration dictionary.
        env_cfg: Environment configuration.
        writer: Writer for logging.
        seed: Random seed for training.
    """
    dtype = torch.float32

    agent_cfg["seed"] = seed
    set_seed(agent_cfg["seed"])

    # Setup models
    policy, value, encoder, value_preprocessor = make_models(env, env_cfg, agent_cfg, dtype)

    # Create tensors in memory for RL (only for the training envs, not eval envs)
    env.num_train_envs = env_cfg.scene.num_envs - agent_cfg["trainer"]["num_eval_envs"]
    rl_memory = make_memory(env, env_cfg, size=agent_cfg["agent"]["rollouts"], num_envs=env.num_train_envs)
    ssl_task = make_aux(env, rl_memory, encoder, value, value_preprocessor, env_cfg, agent_cfg, writer)

    # Configure and instantiate PPO agent
    ppo_agent_cfg = PPO_DEFAULT_CONFIG.copy()
    ppo_agent_cfg.update(agent_cfg["agent"])
    agent = PPO(
        encoder,
        policy,
        value,
        value_preprocessor,
        memory=rl_memory,
        cfg=ppo_agent_cfg,
        observation_space=env.observation_space,
        action_space=env.action_space,
        device=env.device,
        writer=writer,
        ssl_task=ssl_task,
        dtype=dtype,
        debug=agent_cfg["experiment"]["debug"],
    )

    # Start training
    trainer = make_trainer(env, agent, agent_cfg, ssl_task, writer)
    trainer.train()
    print("Training complete!")

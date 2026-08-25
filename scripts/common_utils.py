"""Utility helpers shared between the RoTO training / inference scripts."""

import os
import random

import gymnasium as gym
import numpy as np
import torch
import yaml

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
from roto.tasks import baoding, bounce, find, peace  # noqa: F401
from roto.tasks.robots import allegro, franka, orca, shadow, shadowlite  # noqa: F401


def resolve_gym_env_id(task: str | None, robot: str | None) -> str:
    if task is None:
        raise ValueError("task is required")
    if task == "Find":
        if robot is None or robot.strip().lower() in ("franka",):
            return "Find"
        raise ValueError("Task Find only supports robot franka.")
    if task in ("Bounce", "Baoding"):
        r = normalize_hand_robot(robot)
        if r in ("shadowlite_padtac", "shadowlite_padtac_bt"):
            if task != "Baoding":
                raise ValueError(f"Robot {r!r} is only supported for Baoding, got task {task!r}.")
            suffix = "PadTacBT" if r == "shadowlite_padtac_bt" else "PadTac"
            return f"Baoding_Shadowlite_{suffix}"
        if r == "shadowlite":
            return f"{task}_Shadowlite"
        if r == "orca":
            return f"{task}_Orca"
        if r == "allegro":
            return f"{task}_Allegro"
        return task  # shadow → "Bounce" / "Baoding"
    if task == "PeaceSign":
        r = normalize_hand_robot(robot)
        if r != "shadowlite":
            raise ValueError(f"Task PeaceSign only supports robot shadowlite, got {r!r}.")
        return "PeaceSign_Shadowlite"
    return task


def normalize_hand_robot(robot: str | None) -> str:
    """Return ``shadow``, ``shadowlite``, ``orca``, ``allegro``, ``shadowlite_padtac``, or
    ``shadowlite_padtac_bt`` (default ``shadow``)."""
    r = (robot or "shadow").strip().lower()
    if r not in ("shadow", "shadowlite", "orca", "allegro", "shadowlite_padtac", "shadowlite_padtac_bt"):
        raise ValueError(
            f"Unknown robot {robot!r}. Use one of: shadow, shadowlite, orca, allegro, "
            f"shadowlite_padtac, shadowlite_padtac_bt."
        )
    return r


# Agent YAML filenames per robot (paths: ``tasks/<task>/agents/<robot>/<file>``).
_BOUNCE_SHADOW_AGENT_FILES = {
    "default_cfg": "default.yaml",
    "rl_only_pt": "rl_only_pt.yaml",
    "rl_only_ptd": "rl_only_ptd.yaml",
    "rl_only_ptg": "rl_only_ptg.yaml",
    "tac_recon": "tac_recon.yaml",
    "full_recon": "full_recon.yaml",
    "forward_dynamics": "forward_dynamics.yaml",
    "forward_dynamics_memory": "forward_dynamics_memory.yaml",
    "tac_dynamics": "tac_dynamics.yaml",
}
_BAODING_SHADOW_AGENT_FILES = {
    "default_cfg": "default.yaml",
    "rl_only_pt": "rl_only_pt.yaml",
    "rl_only_pt_padtac": "rl_only_pt_padtac.yaml",
    "rl_only_pt_padtac_bt": "rl_only_pt_padtac_bt.yaml",
    "rl_only_pt_padtac_bt_sweep": "rl_only_pt_padtac_bt_sweep.yaml",
    "rl_only_ptd": "rl_only_ptd.yaml",
    "rl_only_ptg": "rl_only_ptg.yaml",
    "tac_recon": "tac_recon.yaml",
    "full_recon": "full_recon.yaml",
    "forward_dynamics": "forward_dynamics.yaml",
    "forward_dynamics_padtac_bt": "forward_dynamics_padtac_bt.yaml",
    "forward_dynamics_memory": "forward_dynamics_memory.yaml",
    "tac_dynamics": "tac_dynamics.yaml",
}
_ORCA_ALLEGRO_AGENT_FILES = {
    "default_cfg": "default.yaml",
    "rl_only_pt": "rl_only_pt.yaml",
    "rl_only_ptg": "rl_only_ptg.yaml",
    "forward_dynamics": "forward_dynamics.yaml",
}

_PEACE_SHADOWLITE_VARIANT_FILES = {
    "default_cfg": "default.yaml",
    "rl_only_pt": "rl_only_pt.yaml",
    "rl_only_ptd": "rl_only_ptd.yaml",
    "rl_only_ptg": "rl_only_ptg.yaml",
    "tac_recon": "tac_recon.yaml",
    "full_recon": "full_recon.yaml",

    "forward_dynamics": "forward_dynamics.yaml",
    "forward_dynamics_memory": "forward_dynamics_memory.yaml",
    "tac_dynamics": "tac_dynamics.yaml",
    
}


def _hand_agent_files(task_name: str, robot: str) -> dict[str, str]:
    if robot in ("orca", "allegro"):          # removed "shadowlite" from here
        return _ORCA_ALLEGRO_AGENT_FILES
    if task_name == "Bounce":
        return _BOUNCE_SHADOW_AGENT_FILES
    if task_name == "Baoding":
        if robot == "shadowlite":
            return _BAODING_SHADOW_AGENT_FILES  # or a dedicated _BAODING_SHADOWLITE_AGENT_FILES
        return _BAODING_SHADOW_AGENT_FILES
    if task_name == "PeaceSign":
        return _PEACE_SHADOWLITE_VARIANT_FILES
    raise ValueError(f"Unsupported task/robot combo: {task_name!r} / {robot!r}")


def _hand_agent_yaml_path(task_name: str, robot: str, entry_point_key: str) -> str:
    from roto.tasks.baoding import agents as baoding_agents
    from roto.tasks.bounce import agents as bounce_agents

    files = _hand_agent_files(task_name, robot)
    if entry_point_key not in files:
        raise ValueError(
            f"Agent config key {entry_point_key!r} is not available for task {task_name!r} "
            f"with robot {robot!r}. Available keys: {sorted(files)}."
        )
    base = os.path.dirname(bounce_agents.__file__ if task_name == "Bounce" else baoding_agents.__file__)
    agent_robot = "shadowlite" if robot in ("shadowlite_padtac", "shadowlite_padtac_bt") else robot
    return os.path.join(base, agent_robot, files[entry_point_key])


def register_hand_task_to_hydra(
    task_name: str, robot: str | None, agent_cfg_entry_point: str
):
    """Like ``register_task_to_hydra`` but resolves env cfg + agent yaml from ``--robot`` for Bounce/Baoding."""
    from hydra.core.config_store import ConfigStore

    from isaaclab.envs.utils.spaces import replace_env_cfg_spaces_with_strings
    from isaaclab.utils import replace_slices_with_strings

    r = normalize_hand_robot(robot)
    if task_name == "Bounce":
        from roto.tasks.bounce.bounce import (
            BounceAllegroCfg,
            BounceCfg,
            BounceOrcaCfg,
            BounceShadowLiteCfg,
        )

        env_cfg_cls = {
            "shadow": BounceCfg,
            "shadowlite": BounceShadowLiteCfg,
            "orca": BounceOrcaCfg,
            "allegro": BounceAllegroCfg,
        }[r]
    elif task_name == "Baoding":
        from roto.tasks.baoding.baoding import (
            BaodingAllegroCfg,
            BaodingCfg,
            BaodingOrcaCfg,
            BaodingShadowLiteCfg,
            BaodingShadowLitePadTacCfg,
            BaodingShadowLitePadTacBTCfg,
        )

        env_cfg_cls = {
            "shadow": BaodingCfg,
            "shadowlite": BaodingShadowLiteCfg,
            "orca": BaodingOrcaCfg,
            "allegro": BaodingAllegroCfg,
            "shadowlite_padtac": BaodingShadowLitePadTacCfg,
            "shadowlite_padtac_bt": BaodingShadowLitePadTacBTCfg,
        }[r]

    elif task_name == "Peace":
        from roto.tasks.peace.peace import PeaceSignCfg

        env_cfg_cls = {
            "shadowlite": PeaceSignCfg,
        }[r]
    else:
        raise ValueError(f"register_hand_task_to_hydra only supports Bounce/Baoding/Peace, got {task_name!r}")

    env_cfg = env_cfg_cls()
    if task_name == "Baoding":
        from roto.tasks.baoding.baoding import apply_baoding_object_cfgs_from_scalars

        apply_baoding_object_cfgs_from_scalars(env_cfg)

    
    agent_yaml = _hand_agent_yaml_path(task_name, r, agent_cfg_entry_point)
    with open(agent_yaml, encoding="utf-8") as f:
        agent_cfg = yaml.full_load(f)

    env_cfg = replace_env_cfg_spaces_with_strings(env_cfg)
    env_cfg_dict = env_cfg.to_dict()
    agent_cfg_dict = agent_cfg
    cfg_dict = {"env": env_cfg_dict, "agent": agent_cfg_dict}
    cfg_dict = replace_slices_with_strings(cfg_dict)
    ConfigStore.instance().store(name=task_name, node=cfg_dict)
    return env_cfg, agent_cfg


def load_hand_task_agent_cfg(task_name: str, robot: str | None, entry_point_key: str):
    """Load an agent YAML for Bounce/Baoding from the correct ``agents/<robot>/`` directory."""
    r = normalize_hand_robot(robot)
    path = _hand_agent_yaml_path(task_name, r, entry_point_key)
    print(f"[INFO]: Parsing configuration from: {path}")
    with open(path, encoding="utf-8") as f:
        return yaml.full_load(f)

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

    gym_id = getattr(args_cli, "gym_env_id", None) or args_cli.task
    env = gym.make(gym_id, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    obs, _ = env.reset()

    # Debug: print articulation joint names
    try:
        robot = env.unwrapped.scene["robot"]
        print("\n===== ROBOT JOINT NAMES =====")
        for i, name in enumerate(robot.joint_names):
            print(i, name)
        print("=============================\n")
        print(env_cfg.num_actions, ' total actions======================================')
        #print(env.action_space.shape, '======================================')
        # Print all attributes that contain the word 'space' or 'action'
        # If vectorized (e.g., shape is [4096, 13])
        # print(f"Action Space Shape: {env._action_space.shape}")

        # # If you want the dimension of a single hand:
        # if hasattr(env.action_space, 'n'):
        #     print(f"Action Count (Discrete): {env.action_space.shape}")
        # else:
        #     # This is the most common way to check for continuous control
        #     print(f"Action Count (Continuous): {env.action_space.shape[-1]}")

        # First, see what is inside the observation dictionary
        print(f"Observation Keys: {env.observation_space.keys()}")

        # Check the policy observations specifically
        policy_obs = env.observation_space['policy']
        print(f"Policy Keys: {policy_obs.keys()}")

        # Verify the joint position dimension (Expected: 16)
        if 'joint_pos' in policy_obs.spaces:
            joint_dim = policy_obs.spaces['joint_pos'].shape[-1]
            print(f"Joint Observation Dimension: {joint_dim}")

        # Add this around line 245
        # Verify the dimensions before the wrapper takes over

    except Exception as e:
        print(f"[WARN] Could not access robot joint_names: {e}")

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

    obs_dims = {k: v.shape for k, v in gym_dict.items()}
    total_obs_dim = sum(int(np.prod(v.shape)) for v in gym_dict.values())
    print(f"[INFO] Observation shapes: {obs_dims}")
    print(f"[INFO] Total observation input dim: {total_obs_dim}")

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
    rl_memory = make_memory(
        env, env_cfg, size=agent_cfg["agent"]["rollouts"], num_envs=env.num_train_envs
    )
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

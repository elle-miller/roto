"""Rest everything follows."""

import gymnasium as gym
import os
import random
import torch
import traceback
from datetime import datetime
from copy import deepcopy
import numpy as np

# stop log files from being generated!!
import logging
logging.getLogger('hydra').setLevel(logging.CRITICAL)
logging.getLogger('hydra._internal').setLevel(logging.CRITICAL)

from isaaclab_rl.models.encoder import Encoder
from isaaclab_rl.algorithms.memories import Memory
from isaaclab_rl.algorithms.policy_value import DeterministicValue, GaussianPolicy
from isaaclab_rl.algorithms.ppo import PPO, PPO_DEFAULT_CONFIG
from isaaclab_rl.algorithms.sequential import SequentialTrainer
from isaaclab_rl.wrappers.frame_stack import FrameStack
from isaaclab_rl.wrappers.skrl_wrapper import SkrlVecEnvWrapper, process_skrl_cfg
from isaaclab_rl.models.running_standard_scaler import RunningStandardScaler
from isaaclab_rl.tools.writer import Writer

# Import extensions to set up environment tasks
from tasks import franka  # noqa: F401

os.environ["WANDB_DIR"] = "./wandb"
os.environ["WANDB_CACHE_DIR"] = "./wandb"
os.environ["WANDB_CONFIG_DIR"] = "./wandb"
os.environ["WANDB_DATA_DIR"] = "./wandb"



def print_all_modules(model):
    print(f"All modules in {model.__class__.__name__}:")
    for name, module in model.named_modules():
        if name:  # Skip the root module
            print(f"  • {name}: {module.__class__.__name__}")


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
    env = SkrlVecEnvWrapper(env, ml_framework="torch")
    return env

def make_models(env, env_cfg, agent_cfg, skrl_config_dict):
    observation_space = env.observation_space["policy"]
    action_space = env.action_space

    models = {}

    # The shape of the input to the MLP may differ between prop-only and prop-mixed observation spaces, so let's handle
    # that information passing right here.
    encoder = Encoder(observation_space, action_space, env_cfg, skrl_config_dict, device=env.device)
    z_dim = encoder.num_outputs

    # non-shared models
    if agent_cfg["models"]["separate"]:

        # print("policy skrl config", skrl_config_dict["policy"])
        # print("value skrl config", skrl_config_dict["value"])

        models["policy"] = GaussianPolicy(
            z_dim=z_dim,
            observation_space=observation_space,
            action_space=env.action_space,
            device=env.device,
            **skrl_config_dict["policy"],
        )

        # print("policy skrl config", skrl_config_dict["policy"])
        # print("value skrl config", skrl_config_dict["value"])

        models["value"] = DeterministicValue(
            z_dim=z_dim,
            observation_space=observation_space,
            action_space=env.action_space,
            device=env.device,
            **skrl_config_dict["value"],
        )
        # print("policy skrl config", skrl_config_dict["policy"])
        # print("value skrl config", skrl_config_dict["value"])
    else:
        raise ValueError("We do not support shared models.")
    
    print("*****Encoder*****")
    # print_all_modules(encoder)
    print(encoder)
    print("*****RL models*****")
    # print_all_modules(models["policy"])
    # print_all_modules(models["value"])
    print(models["policy"])
    print(models["value"])

    return models, encoder

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
    default_agent_cfg.update(process_skrl_cfg(agent_cfg["agent"], ml_framework="torch"))
    default_agent_cfg["state_preprocessor"] = None
    # default_agent_cfg["value_preprocessor"] = RunningStandardScaler #None # agent_cfg["agent"]["value_preprocessor"]
    default_agent_cfg["state_preprocessor_kwargs"].update({"size": env.observation_space["policy"], "device": env.device})
    # default_agent_cfg["value_preprocessor_kwargs"].update({"size": 1, "device": env.device})
    default_agent_cfg["rewards_shaper"] = lambda rewards, *args, **kwargs: rewards * agent_cfg["agent"]["rewards_shaper_scale"]
    print(default_agent_cfg)
    return default_agent_cfg


def make_trainer(env, agent, agent_cfg, auxiliary_task=None, writer=None):

    train_timesteps = int(agent_cfg["trainer"]["max_global_timesteps_M"] * 1e6 / env.num_envs)
    agent_cfg["trainer"]["timesteps"] = train_timesteps

    # configure and instantiate a custom RL trainer for logging episode events
    trainer_cfg = agent_cfg["trainer"]
    trainer_cfg["close_environment_at_exit"] = False
    trainer_cfg["disable_progressbar"] = True
    trainer_cfg["observation_spaces"] = env.observation_space
    num_eval_envs = trainer_cfg["num_eval_envs"]
    trainer = SequentialTrainer(cfg=trainer_cfg, env=env, agents=agent, num_eval_envs=num_eval_envs, auxiliary_task=auxiliary_task, writer=writer)
    return trainer

def update_env_cfg(args_cli, env_cfg, agent_cfg, skrl_config_dict):

    env_cfg.seed = agent_cfg["seed"]

    # override configurations with either config file or args
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    env_cfg.obs_list = skrl_config_dict["obs_list"]
    env_cfg.num_eval_envs = agent_cfg["trainer"]["num_eval_envs"]

    # variables that impact how env obs are processed
    env_cfg.normalise_prop = (
        skrl_config_dict["preprocess"]["normalise_prop"]
        if "normalise_prop" in skrl_config_dict["preprocess"]
        else False
    )
    env_cfg.binary_tactile = (
        skrl_config_dict["preprocess"]["binary_tactile"]
        if "binary_tactile" in skrl_config_dict["preprocess"]
        else False
    )
    env_cfg.normalise_pixels = (
        skrl_config_dict["preprocess"]["normalise_pixels"]
        if "normalise_pixels" in skrl_config_dict["preprocess"]
        else False
    )
    env_cfg.random_crop = (
        skrl_config_dict["preprocess"]["random_crop"] if "random_crop" in skrl_config_dict["preprocess"] else False
    )
    # variables that impact the network construction
    if "pixels" in skrl_config_dict:
        env_cfg.img_dim = skrl_config_dict["pixels"]["img_dim"]

    return env_cfg



def set_seed(seed: int = 42) -> None:
    
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
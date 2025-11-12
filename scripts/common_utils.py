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
from isaaclab_rl.auxiliary.reconstruction import Reconstruction
from isaaclab_rl.auxiliary.dynamics import ForwardDynamics
from isaaclab_rl.algorithms.ppo import PPO, PPO_DEFAULT_CONFIG
from isaaclab_rl.tools.writer import Writer
from isaaclab_tasks.utils.hydra import hydra_task_config, register_task_to_hydra
from isaaclab.utils import update_dict


import torch

from isaaclab.utils import update_dict
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry
# ADD YOUR ENVS HERE
from tasks import franka,shadow  # noqa: F401

# change this to something else if you want
LOG_PATH = os.getcwd()



def make_aux(env, rl_memory, encoder, value, value_preprocessor, env_cfg, agent_cfg, writer):

    # configure auxiliary task
    rl_rollout = agent_cfg["agent"]["rollouts"]
    if "ssl_task" in agent_cfg.keys():

        match agent_cfg["ssl_task"]["type"]:
            case "reconstruction":
                ssl_task = Reconstruction(agent_cfg["ssl_task"], rl_rollout, rl_memory, encoder, value, value_preprocessor, env, env_cfg, writer)
            case "forward_dynamics":
                ssl_task = ForwardDynamics(agent_cfg["ssl_task"], rl_rollout, rl_memory, encoder, value, value_preprocessor, env, env_cfg, writer)
            case _:  # default case
                print("No auxiliary task")
                ssl_task = None

    else:
        ssl_task = None
    return ssl_task

def make_env(env_cfg, writer, args_cli, obs_stack=1):

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    obs, reward = env.reset()

    gym_dict = {}
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
            # "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training to", writer.video_dir)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)
    
    # FrameStack expects a gym env
    if obs_stack > 1:
        env = FrameStack(env, obs_stack=obs_stack)

    # Isaac Lab wrapper
    env = IsaacLabWrapper(env, env_cfg.num_eval_envs, obs_stack=obs_stack, debug=env_cfg.debug)
    return env


def make_models(env, env_cfg, agent_cfg, dtype):
    observation_space = env.observation_space["policy"]
    action_space = env.action_space

    encoder = Encoder(observation_space, action_space, env_cfg, agent_cfg["encoder"], device=env.device)
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
    memory = Memory(
        memory_size=size,
        num_envs=num_envs,
        device=env.device,
        env_cfg=env_cfg,
    )
    return memory


def make_trainer(env, agent, agent_cfg, ssl_task=None, writer=None):

    num_timesteps_M = agent_cfg["trainer"]["max_global_timesteps_M"]
    num_eval_envs = agent_cfg["trainer"]["num_eval_envs"]
    trainer = Trainer(
        env=env,
        agents=agent,
        num_timesteps_M=num_timesteps_M,
        num_eval_envs=num_eval_envs,
        ssl_task=ssl_task,
        writer=writer,
    )
    return trainer


def update_env_cfg(args_cli, env_cfg, agent_cfg):

    env_cfg.seed = agent_cfg["seed"]
    env_cfg.debug = agent_cfg["experiment"]["debug"]

    # override configurations with either config file or args
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    env_cfg.obs_list = agent_cfg["observations"]["obs_list"]
    env_cfg.num_eval_envs = agent_cfg["trainer"]["num_eval_envs"]
    env_cfg.obs_stack = agent_cfg["observations"]["obs_stack"]

    # variables that impact how env obs are processed
    env_cfg.normalise_prop = agent_cfg["observations"]["preprocess"]["normalise_prop"]
    env_cfg.binary_tactile = agent_cfg["observations"]["preprocess"]["binary_tactile"]

    return env_cfg


def set_seed(seed: int = 42) -> None:

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# @hydra_task_config(args_cli.task, "default_cfg")
def train_one_seed(args_cli, env, agent_cfg=None, env_cfg=None, writer=None, seed=None):
    """Train with skrl agent."""

    dtype = torch.float32

    agent_cfg["seed"] = seed
    set_seed(agent_cfg["seed"])

    # setup models
    policy, value, encoder, value_preprocessor = make_models(env, env_cfg, agent_cfg, dtype)

    # create tensors in memory for RL stuff [only for the training envs]
    num_training_envs = env_cfg.scene.num_envs - agent_cfg["trainer"]["num_eval_envs"]
    rl_memory = make_memory(env, env_cfg, size=agent_cfg["agent"]["rollouts"], num_envs=num_training_envs)
    ssl_task = make_aux(env, rl_memory, encoder, value, value_preprocessor, env_cfg, agent_cfg, writer)

    # configure and instantiate PPO agent
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
        debug=agent_cfg["experiment"]["debug"]
    )

    # Let's go!
    trainer = make_trainer(env, agent, agent_cfg, ssl_task, writer)
    trainer.train()
    print("Training complete!")


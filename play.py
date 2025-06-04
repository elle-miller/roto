# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Script to play a checkpoint of an RL agent from skrl.

Visit the skrl documentation (https://skrl.readthedocs.io) to see the examples structured in
a more user-friendly way.
"""

"""Launch Isaac Sim Simulator first."""


import argparse
import sys
import traceback

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Play a checkpoint of an RL agent from skrl.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
if args_cli.video:
    args_cli.enable_cameras = True
sys.argv = [sys.argv[0]] + hydra_args
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from isaaclab_tasks.utils.hydra import hydra_task_config
from isaaclab_rl.utils.models.running_standard_scaler import RunningStandardScaler
from isaaclab_rl.wrappers.isaaclab_wrapper import IsaacLabWrapper

from isaaclab_rl.utils.skrl.run_utils import *

    
@hydra_task_config(args_cli.task, "skrl_cfg_entry_point")
def main(env_cfg, agent_cfg: dict):
    """Train with skrl agent."""

    # SEED (environment AND agent)
    # note: we lose determinism when using pixels due to GPU renderer
    agent_cfg["seed"] = args_cli.seed if args_cli.seed is not None else agent_cfg["seed"]
    set_seed(agent_cfg["seed"])

    # UPDATE CFGS
    skrl_config_dict = process_skrl_cfg(agent_cfg["models"], ml_framework="torch")
    env_cfg = update_env_cfg(args_cli, env_cfg, agent_cfg, skrl_config_dict)

    # LOGGING SETUP
    wandb_session = setup_wandb(agent_cfg, skrl_config_dict)
    tb_writer, agent_cfg = setup_logging(agent_cfg)

    # Expose environment creation so I can configure obs_stack as tunable hparam
    obs_stack=skrl_config_dict["obs_stack"]
    env_cfg.obs_stack = obs_stack
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # wrap for video recording
    if args_cli.video:
        log_dir = "/home/elle/code/external/IsaacLab/isaaclab_rl"
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # FrameStack expects a gymnasium.Env
    if obs_stack != 1:
        env = FrameStack(env, num_stack=obs_stack)
    
    env = IsaacLabWrapper(env)
    models, encoder = make_models(env, env_cfg, agent_cfg, skrl_config_dict)
    num_training_envs = env.num_envs - agent_cfg["trainer"]["num_eval_envs"]
    default_agent_cfg = make_agent_cfg(env, agent_cfg)
    value_preprocessor = RunningStandardScaler(size=1, device=env.device)

    # specify directory for logging experiments (load checkpoint)
    log_root_path = os.path.join(
        LOG_ROOT_DIR,
        "logs",
        "skrl",
        agent_cfg["agent"]["experiment"]["directory"],
        agent_cfg["agent"]["experiment"]["experiment_name"],
    )
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    # get checkpoint path
    if args_cli.checkpoint:
        resume_path = os.path.abspath(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, other_dirs=["checkpoints"])
    log_dir = os.path.dirname(os.path.dirname(resume_path))

    agent = PPO(
        value_preprocessor=value_preprocessor,
        models=models,
        memory=None,  # memory is optional during evaluation
        cfg=default_agent_cfg,
        observation_space=env.observation_space,
        action_space=env.action_space,
        device=env.device,
        encoder=encoder,
        wandb_session=wandb_session,
    )

    # initialize agent
    agent.init()
    print(f"[INFO] Loading model checkpoint from: {resume_path}")
    agent.load(resume_path)

    modules = torch.load(resume_path, map_location=env.device)
    if type(modules) is dict:
        for name, data in modules.items():
            print(name)

    # Let's go!
    import time
    # get environment (step) dt for real-time evaluation
    try:
        dt = env.step_dt
    except AttributeError:
        dt = env.unwrapped.step_dt
    # reset environment
    states, _ = env.reset()
    timestep = 0
    real_time = True
    # simulate environment
    while simulation_app.is_running():
        start_time = time.time()

        # run everything in inference mode
        with torch.inference_mode():
            # agent stepping
            z = encoder(states)
            actions, _, _ = agent.policy.act(z, deterministic=True)

            # env stepping
            states, _, _, _, _ = env.step(actions)
        
        if args_cli.video:
            timestep += 1
            # exit the play loop after recording one video
            if timestep == args_cli.video_length:
                break

        # time delay for real-time evaluation
        sleep_time = dt - (time.time() - start_time)
        if real_time and sleep_time > 0:
            time.sleep(sleep_time)


    # close the simulator
    env.close()


if __name__ == "__main__":
    try:
        # run the main function
        main()
    except Exception as err:
        carb.log_error(err)
        carb.log_error(traceback.format_exc())
        raise
    finally:
        # close sim app
        simulation_app.close()

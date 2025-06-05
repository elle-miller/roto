# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Script to train RL agent with skrl.

Visit the skrl documentation (https://skrl.readthedocs.io) to see the examples structured in
a more user-friendly way.
"""

"""Launch Isaac Sim Simulator first."""


import argparse
import gc
import sys

import optuna
from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with skrl.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=600, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=500, help="Interval between video recordings (in steps).")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--study", type=str, default="default", help="study name")


# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
if args_cli.video:
    args_cli.enable_cameras = True
sys.argv = [sys.argv[0]] + hydra_args
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import isaaclab_tasks  # noqa: F401
from isaaclab_rl.utils.models.running_standard_scaler import RunningStandardScaler
from isaaclab_rl.utils.skrl.run_utils import *
from isaaclab_rl.wrappers.isaaclab_wrapper import IsaacLabWrapper
from isaaclab_tasks.utils.hydra import hydra_task_config, register_task_to_hydra

# hparam_dir = agent_cfg["agent"]["experiment"]["directory"]
# hparam_exp = agent_cfg["agent"]["experiment"]["experiment_name"]


class OptimisationRunner:
    def __init__(self, study_name, n_startup_trials, n_warmup_steps, interval_steps):

        self.sampler = optuna.samplers.TPESampler(n_startup_trials=n_startup_trials)

        # self.pruner = optuna.pruners.MedianPruner(
        #     n_startup_trials=n_startup_trials,
        #     n_warmup_steps=n_warmup_steps,
        #     interval_steps=interval_steps
        # )
        self.pruner = optuna.pruners.NopPruner()

        self.study = optuna.create_study(
            storage=storage,
            sampler=self.sampler,
            pruner=self.pruner,
            study_name=study_name,
            direction="maximize",
            load_if_exists=True,
        )

    def run(self, n_trials=50):

        self.study.optimize(
            lambda trial: self.objective(
                trial, env=env, env_cfg=env_cfg, agent_cfg=agent_cfg, skrl_config_dict=skrl_config_dict
            ),
            n_trials=n_trials,
            show_progress_bar=True,
            gc_after_trial=True,
        )

        # Antonin's code
        print(f"Number of finished trials: {len(self.study.trials)}")
        print("Best trial:")
        trial = self.study.best_trial
        print("  Value: ", trial.value)
        print("  Params: ")
        for key, value in trial.params.items():
            print(f"    {key}: {value}")
        print("  User attrs:")
        for key, value in trial.user_attrs.items():
            print(f"    {key}: {value}")
        return self.study.best_trial

    def free_memory(self):
        torch.cuda.empty_cache()
        gc.collect()

    def objective(self, trial: optuna.Trial, env, env_cfg, agent_cfg, skrl_config_dict) -> float:
        print(f"Starting trial: {trial.number}")
        # parameters to optimize
        # https://optuna.readthedocs.io/en/stable/reference/generated/optuna.trial.Trial.html
        # mini_batches = trial.suggest_categorical("mini_batches", [4, 8, 16, 32, 64])
        # learning_rate = trial.suggest_float("learning_rate", low=1e-5, high=1e-3, log=True)
        # learning_epochs = trial.suggest_categorical("learning_epochs", [4, 8, 16, 32])
        # rollouts = trial.suggest_categorical("rollouts", [16, 32, 64])
        # entropy_loss_scale = trial.suggest_categorical("entropy_loss_scale", [0.0, 0.05, 0.1])
        # value_clip = trial.suggest_categorical("value_clip", [0.1, 0.15, 0.2])

        ## aux
        if skrl_config_dict["auxiliary_task"]["type"] != None:
            # mini_batches_aux = trial.suggest_categorical("mini_batches_aux", [8, 16, 32, 64])
            learning_rate_aux = trial.suggest_float("learning_rate_aux", low=1e-5, high=1e-3, log=True)
            loss_weight_aux = trial.suggest_float("loss_weight_aux", low=1e-3, high=10, log=True)
            rl_per_aux = trial.suggest_categorical("rl_per_aux", [2, 3, 4])

            skrl_config_dict["auxiliary_task"]["rl_per_aux"] = rl_per_aux
            skrl_config_dict["auxiliary_task"]["learning_rate"] = learning_rate_aux
            skrl_config_dict["auxiliary_task"]["loss_weight"] = loss_weight_aux
            # max obs stack size
            if skrl_config_dict["auxiliary_task"]["type"] == "reconstruction":
                # print("manual last_n_obs")
                last_n_obs = trial.suggest_categorical("last_n_obs", [1, 2, 4, 8, 16])
                # if last_n_obs == 8 or last_n_obs == 16:
                last_n_obs = 16
                skrl_config_dict["auxiliary_task"]["last_n_obs"] = last_n_obs

            elif skrl_config_dict["auxiliary_task"]["type"] == "forward_dynamics":

                # it can take quite long, cap at8
                n_f = trial.suggest_categorical("n_f", [2, 3, 4, 10])
                skrl_config_dict["auxiliary_task"]["n_f"] = n_f

        tb_writer, agent_cfg = setup_logging(agent_cfg)
        wandb_session = setup_wandb(agent_cfg, skrl_config_dict, group_name=study_name, run_name=f"{trial.number}")
        models, encoder = make_models(env, env_cfg, agent_cfg, skrl_config_dict)
        num_training_envs = env.num_envs - agent_cfg["trainer"]["num_eval_envs"]
        rl_memory = make_memory(env, env_cfg, num_envs=num_training_envs, size=agent_cfg["agent"]["rollouts"])
        default_agent_cfg = make_agent_cfg(env, agent_cfg)
        value_preprocessor = RunningStandardScaler(size=1, device=env.device)
        auxiliary_task = make_aux(
            env,
            rl_memory,
            encoder,
            models["value"],
            value_preprocessor,
            env_cfg,
            agent_cfg,
            skrl_config_dict,
            wandb_session,
        )

        # update default_agent_cfg with trial
        # default_agent_cfg["rollouts"] = rollouts
        # default_agent_cfg["mini_batches"] = mini_batches
        # default_agent_cfg["learning_epochs"] = learning_epochs
        # default_agent_cfg["learning_rate"] = learning_rate
        # default_agent_cfg["entropy_loss_scale"] = entropy_loss_scale
        # default_agent_cfg["value_clip"] = value_clip
        # default_agent_cfg["value_loss_scale"] = value_loss_scale
        # default_agent_cfg["rewards_shaper_scale"] = rewards_shaper_scale
        # default_agent_cfg["rewards_shaper"] = lambda rewards, *args, **kwargs: rewards * rewards_shaper_scale
        # PPO
        agent = PPO(
            encoder,
            value_preprocessor,
            models=models,
            memory=rl_memory,
            cfg=default_agent_cfg,
            observation_space=env.observation_space,
            action_space=env.action_space,
            device=env.device,
            wandb_session=wandb_session,
            auxiliary_task=auxiliary_task,
        )

        trainer = make_trainer(env, agent, agent_cfg, auxiliary_task)
        if skrl_config_dict["auxiliary_task"]["type"] == "forward_dynamics":
            agent.checkpoint_modules["forward_model"] = auxiliary_task.forward_model
            agent.checkpoint_modules["projector"] = auxiliary_task.projector
            if skrl_config_dict["auxiliary_task"]["tactile_only"] == True:
                agent.checkpoint_modules["tactile_decoder"] = auxiliary_task.tactile_decoder
        if skrl_config_dict["auxiliary_task"]["type"] == "reconstruction":
            agent.checkpoint_modules["decoder"] = auxiliary_task.decoder

        try:
            best_return, should_prune = trainer.train(wandb_session=wandb_session, tb_writer=tb_writer, trial=trial)
        except AssertionError as e:
            # Sometimes, random hyperparams can generate NaN.
            print(e)

        # prune trial
        if should_prune:
            wandb.finish()
            raise optuna.TrialPruned()

            # completed_trials = [t for t in self.study.trials if t.state == optuna.trial.TrialState.COMPLETE]
            # other_values = []
            # for t in completed_trials:
            #     if step in t.intermediate_values:
            #         other_values.append(t.intermediate_values[step])

            # if other_values:
            #     median_value = np.median(other_values)
            #     current_value = best_return
            #     logger.info(f"⚠️ Trial {trial.number} PRUNED at step {step}:")
            #     logger.info(f"   - Current value: {current_value:.4f}")
            #     logger.info(f"   - Median value: {median_value:.4f}")
            #     logger.info(f"   - Difference: {current_value - median_value:.4f}")
            #     logger.info(f"   - Warmup steps: {self.pruner._n_warmup_steps}")
            #     logger.info(f"   - Startup trials needed: {self.pruner._n_startup_trials}")

        return best_return


if __name__ == "__main__":

    # parse configuration
    cfg = hydra_task_config(args_cli.task, "skrl_cfg_entry_point")
    env_cfg, agent_cfg = register_task_to_hydra(args_cli.task, "skrl_cfg_entry_point")

    # SEED (environment AND agent)
    # note: we lose determinism when using pixels due to GPU renderer
    agent_cfg["seed"] = args_cli.seed if args_cli.seed is not None else agent_cfg["seed"]
    set_seed(agent_cfg["seed"])

    # UPDATE CFGS
    skrl_config_dict = process_skrl_cfg(agent_cfg["models"], ml_framework="torch")
    env_cfg = update_env_cfg(args_cli, env_cfg, agent_cfg, skrl_config_dict)

    # create isaac environment
    # Expose environment creation so I can configure obs_stack as tunable hparam
    obs_stack = skrl_config_dict["obs_stack"]
    env_cfg.obs_stack = obs_stack
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    ep_length = env.unwrapped.max_episode_length
    train_timesteps = agent_cfg["trainer"]["max_global_timesteps_M"] * 1e6
    num_envs = env.unwrapped.num_envs
    num_env_train_steps = int(train_timesteps / num_envs)
    num_evaluations = int(num_env_train_steps / ep_length)
    print("TRAINING STEPS:", train_timesteps)
    print("NUM EVALUATIONS: ", num_evaluations)

    # FrameStack expects a gymnasium.Env
    # env.configure_gym_env_spaces(obs_stack)
    if obs_stack != 1:
        env = FrameStack(env, num_stack=obs_stack)

    env = IsaacLabWrapper(env)

    # https://optuna.readthedocs.io/en/stable/reference/generated/optuna.create_study.html
    storage = "sqlite:///paper_agents5.db"
    study_name = args_cli.study

    # Usage
    total_trials = 50
    n_startup_trials = 5
    n_warmup_steps = int(num_evaluations / 3)
    interval_steps = 10

    runner = OptimisationRunner(study_name, n_startup_trials, n_warmup_steps, interval_steps)

    try:
        best_trial = runner.run(total_trials)

    except Exception as err:
        carb.log_error(err)
        carb.log_error(traceback.format_exc())
        raise
    except KeyboardInterrupt:
        pass
    finally:
        # close sim app
        simulation_app.close()

    # rewards_shaper_scale = trial.suggest_categorical("rewards_shaper_scale", [1, 0.1, 0.01])
    # value_loss_scale = trial.suggest_categorical("value_loss_scale", [1.0, 2.0])
    # obs_stack = trial.suggest_categorical("obs_stack", [1,2,4,8,16])

    # policy_hiddens = trial.suggest_categorical("policy_hiddens", [[32, 32], [64, 32], [128, 64, 32]])
    # encoder_hiddens = trial.suggest_categorical("encoder_hiddens", [[2048, 1024, 512, 256], [1024, 512, 256]])

    # # arches
    # skrl_config_dict["encoder"]["hiddens"] = encoder_hiddens
    # skrl_config_dict["encoder"]["activations"] = ["elu"] * len(encoder_hiddens)

    # skrl_config_dict["policy"]["hiddens"] = policy_hiddens
    # skrl_config_dict["value"]["hiddens"] = policy_hiddens
    # spec_activations= ["elu"] * len(policy_hiddens)
    # skrl_config_dict["policy"]["activations"] = spec_activations + ["identity"]
    # skrl_config_dict["value"]["activations"] = spec_activations + ["identity"]
    # print("ACVITATIONS", spec_activations)

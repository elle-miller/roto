# RoTO: Robot Tactile Olympiad
<img src="readme_assets/images/roto2.png" 
     width="800" 
     border="1"
     style="display: block; margin: 0 auto;"/>


RoTO is a **reinforcement learning benchmark environment** designed to standardise and promote future research in tactile-based manipulation. The environments are designed to cover a wide range of tactile interactions: sparse (Find), intermittent (Bounce), and sustained (Baoding). We will continue to add more environments and strongly welcome contributions 🤗
- `roto 1.0` included the Find (Franka), Bounce & Baoding (Shadow Hand) tasks. It was introduced in [Enhancing Tactile-based RL for Robotic Control](https://elle-miller.github.io/tactile_rl/) (NeurIPS 2025), which shows that blind superhuman dexterity is possible with sparse binary contacts + self-supervision.
- `roto 2.0` is extended to include the Allegro, ORCA, and Shadow Dexterous Hand Lite robots for the Bounce & Baoding tasks. We swept hyperparameters for the full state & blind agents, and benchmarked the results in a 2-page writeup [here](). The checkpoints/logs are [here]().

<!-- <img src="readme_assets/images/roto.png" 
     width="400" 
     border="1"
     style="display: block; margin: 0 auto;"/> -->

## ✨ Overview

<img src="readme_assets/images/setup.png" width="1000" border="1"/>

We split the paper code across two repositories. Imagine the typical RL loop: you can think of `multimodal_rl` as the agent, and `roto` as the environment. We did this for modularity, in case you want to use your own RL repository instead of ours (there will be some integration to achieve this but happy to help).

`multimodal_rl`: The motto of this repo is _"doing good RL with Isaac Lab as painlessly as possible"_. We started from the [skrl](https://github.com/Toni-SM/skrl) library and made significant changes to better handle multimodal dictionary observations, observation stacking and associated memory management, and integrated self-supervision. Many existing libraries did not provide support for doing robust RL research (correct evaluation metrics, distinct train/evaluation envs, integrated hyperparameter optimisation). These are well established norms in the RL research community, but are not yet consistently present in RL+robotics research, which we want to encourage 🚀

`roto`: This repo just contains the robot configurations and task definitions. We take advantage of class inheritance to heavily reduce repeated code. `RotoEnv` is a child of `DirectRLEnv`, and sets up basic functions to perform joint position control of a robot and reset it. `[Robot]Env` is a child of `RotoEnv`, defining robot-specific functions that do not change task-to-task, e.g. the proprioceptive observation key. Finally, `[Task]Env` defines task-specific functions such as setting up the environment, rewards, and episode resets.


## 🤖 Environments

The agents are all joint position controlled. Franka has 9 joints, Shadow has 20 actuated joints.

| Environment | Description | Rewards | Robots|
| :---: | :--- | :--- |  :--- |
| <img src="readme_assets/images/find.png" alt="Find Environment" width="400px"> | The agent must locate a fixed ball on a plate as quickly as possible. | Distance reward from end-effector to ball | Franka |
| <img src="readme_assets/images/bounce.png" alt="Bounce Environment" width="400px"> | The agent must bounce a ball as many times as possible within 10s. | Sparse bounce bonus  | Shadow, ORCA, Allegro, Shadow Lite |
| <img src="readme_assets/images/baoding.png" alt="Baoding Environment" width="400px"> | The agent must rotate two small balls around each other as many times as possible within 10s. |  Small distance reward to ball target + successful rotation bonus  | Shadow, ORCA, Allegro, Shadow Lite |

## Observations

We use dictionary-style observations, and categorising into proprioception, tactile, rgb, depth, and gt (ground-truth). The proprioception & tactile methods should be defined in `{Robot}Env`, but gt information is task-dependent. To specify which observations are used, add the keys to `obs_list` in the agent cfg..
```
observations:
  obs_list:
  - prop
  - tactile
  - rgb
  - depth
  - gt
  obs_stack: 3
  tactile_cfg:
    binary_tactile: true
    binary_threshold: 0.01
  pixel_cfg:
    width: 80
    height: 80
    latent_pixel_dim: 128 
    normalise_rgb: true
    max_depth: 2.0  # meters
```
Here is an example rendering of raw RGB, normalised RGB, and depth of Shadow Baoding agent.
<img src="readme_assets/rgb.gif" 
     width="200" 
     border="1"
     style="display: block; margin: 0 auto;"/>
<img src="readme_assets/rgb_normalise.gif" 
     width="200" 
     border="1"
     style="display: block; margin: 0 auto;"/>
<img src="readme_assets/depth.gif" 
     width="200" 
     border="1"
     style="display: block; margin: 0 auto;"/>

## 🛠️ Installation

We need to install Isaac Sim, Isaac Lab, `multimodal_rl` and `roto` in a conda environment. We recommend using the latest Isaac Sim for maximum performance.

1. Create conda environment and install Isaac Lab and Isaac Sim (easiest to install both as [pip packages](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/isaaclab_pip_installation.html#))

2. Install [multimodal_rl](https://github.com/elle-miller/multimodal_rl) as a local editable package
```
git clone git@github.com:elle-miller/multimodal_rl.git
cd multimodal_rl
pip install -e .
```
3. Install `roto` as a local editable package
```
git clone git@github.com:elle-miller/roto.git
cd roto
pip install -e .
```
4. Test the installation by playing a trained agent.
```
# play in isaac sim viewer
python scripts/play.py --task Baoding --robot Shadow --num_envs 512 --agent_cfg forward_dynamics_memory --checkpoint readme_assets/checkpoints/baoding_memory.pt

# save a video
python scripts/play.py --task Baoding --robot Shadow --num_envs 512 --agent_cfg forward_dynamics_memory --video --video_length 1200 --headless --checkpoint readme_assets/checkpoints/baoding_memory.pt
```
The video should pop up in a `./videos` folder and look like this:

<img src="readme_assets/baoding_memory.gif" 
     width="400" 
     border="1"
     style="display: block; margin: 0 auto;"/>

You can find more trained checkpoints in the [roto_paper_results](https://github.com/elle-miller/roto_paper_results) repository.


## 🏃 Usage
Mostly the same as default Isaac Lab setup. The only breaking change is that a given task is not linked to a cfg file. The cfgs must be defined in the task `__init__.py` and specified as an `agent_cfg` argument.

We provide 3 environments x 7 cfgs, corresponding to the paper
```
gym.register(
    id="Baoding",
    entry_point="roto.tasks.baoding.baoding:BaodingShadowEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": baoding.BaodingCfg,
        "default_cfg": baoding_default_cfg,
        "rl_only_pt": baoding_rl_only_pt,
        "tac_recon": baoding_tactile_recon,
        "full_recon": baoding_full_recon,
        "forward_dynamics": baoding_forward_dynamics,
        "tac_dynamics": baoding_tactile_dynamics,
    }
)
```
### Training
Here is how you would train a Find agent just with RL, a Bounce agent with RL + Tactile Reconstruction, and a Baoding agent with RL + Forward Dynamics.
```
python scripts/train.py --task Find --num_envs 4196 --headless --seed 1234 --agent_cfg rl_only_pt
python scripts/train.py --task Bounce --num_envs 4196 --headless --seed 1234 --agent_cfg tac_recon
python scripts/train.py --task Baoding --num_envs 4196 --headless --seed 1234 --agent_cfg forward_dynamics
```

### Sweeping
We use `opunta` for integrated hyperparameter optimisation. The command is the same as for `train.py`, but with an additional `--study` name argument. You can specify the pruner, number of trials, number of warm up steps etc. I recommend [this blogpost](https://araffin.github.io/post/hyperparam-tuning/)  if you are new to sweeping :)
```
python scripts/sweep.py --task Find --num_envs 4196 --headless --seed 1234 --agent_cfg rl_only_pt --study find_rl_only_pt
python scripts/sweep.py --task Bounce --num_envs 4196 --headless --seed 1234 --agent_cfg tac_recon --study bounce_tac_recon
python scripts/sweep.py --task Baoding --num_envs 4196 --headless --seed 1234 --agent_cfg forward_dynamics --study baoding_forward_dynamics
```

### Playing
See last step in installation.


## 📁 Data

The data in the paper (checkpoints, training logs, plot scripts) is available in the [roto_paper_results](https://github.com/elle-miller/roto_paper_results) repo.


## 📧 Contact

For any questions, issues, or collaborations, please feel free to post an issue/start a discussion/reach out.

- Maintainer: Elle Miller
- Project Website: https://elle-miller.github.io/tactile_rl

This project is licensed under the BSD-3 License.


## 🤗 Contributing
This is our plan for future additions, but we highly welcome community contributions and PRs!

- More environments
- Integrate TacSL for high-resolution touch sensing when it becomes released: https://github.com/isaac-sim/IsaacGymEnvs/issues/244

## 📄 Citation

If you use this benchmark environment in your academic or professional research, please cite the following work:

```
@inproceedings{miller2025tactilerl,
  author    = {Miller, Elle and McInroe, Trevor and Abel, David and Mac Aodha, Oisin and Vijayakumar, Sethu},
  title     = {Enhancing Tactile-based Reinforcement Learning for Robotic Control},
  booktitle = {NeurIPS},
  year      = {2025},
}
```

## Videos

### Baoding: Shadow, Allegro, ORCA, Shadow Lite

Videos are paired left-to-right: **States+Proprio+Binary Tactile** (vision) | **Proprio+Binary Tactile** (blind).

**Shadow Hand**

| Vision | Blind |
| :---: | :---: |
| <video src="readme_assets/roto2_videos/shadow_baoding_vision.mp4" controls width="400"></video> | <video src="readme_assets/roto2_videos/shadow_baoding_blind.mp4" controls width="400"></video> |

**Allegro Hand**

| Vision | Blind |
| :---: | :---: |
| <video src="readme_assets/roto2_videos/allegro_baoding_vision.mp4" controls width="400"></video> | <video src="readme_assets/roto2_videos/allegro_baoding_blind.mp4" controls width="400"></video> |

**ORCA Hand**

| Vision | Blind |
| :---: | :---: |
| <video src="readme_assets/roto2_videos/orca_baoding_vision.mp4" controls width="400"></video> | <video src="readme_assets/roto2_videos/orca_baoding_blind.mp4" controls width="400"></video> |

**Shadow Dexterous Hand Lite**

| Vision | Blind |
| :---: | :---: |
| <video src="readme_assets/roto2_videos/lite_baoding_vision.mp4" controls width="400"></video> | <video src="readme_assets/roto2_videos/lite_baoding_blind.mp4" controls width="400"></video> |

### Bounce: Shadow, Allegro, ORCA, Shadow Lite

Videos are paired left-to-right: **States+Proprio+Binary Tactile** (vision) | **Proprio+Binary Tactile** (blind).

**Shadow Hand**

| Vision | Blind |
| :---: | :---: |
| <video src="readme_assets/roto2_videos/shadow_bounce_vision.mp4" controls width="400"></video> | <video src="readme_assets/roto2_videos/shadow_bounce_blind.mp4" controls width="400"></video> |

**Allegro Hand**

| Vision | Blind |
| :---: | :---: |
| <video src="readme_assets/roto2_videos/allegro_bounce_vision.mp4" controls width="400"></video> | <video src="readme_assets/roto2_videos/allegro_bounce_blind.mp4" controls width="400"></video> |

**ORCA Hand**

| Vision | Blind |
| :---: | :---: |
| <video src="readme_assets/roto2_videos/orca_bounce_vision.mp4" controls width="400"></video> | <video src="readme_assets/roto2_videos/orca_bounce_blind.mp4" controls width="400"></video> |

**Shadow Dexterous Hand Lite**

| Vision | Blind |
| :---: | :---: |
| <video src="readme_assets/roto2_videos/lite_bounce_vision.mp4" controls width="400"></video> | <video src="readme_assets/roto2_videos/lite_bounce_blind.mp4" controls width="400"></video> |

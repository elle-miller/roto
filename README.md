# RoTO: Robot Tactile Olympiad
<img src="images/roto.png" width="1000" border="1"/>

---

## 🧭 Table of Contents
* [✨ Overview](#-overview)
* [🛠️ Installation](#️-installation)
* [🏃 Usage](#-usage)
* [📊 Benchmark Results](#-benchmark-results)
* [📚 Documentation](#-documentation)
* [📄 Citation](#-citation)
* [📧 Contact](#-contact)

---

## ✨ Overview

RoTO is an **open-source Reinforcement Learning benchmark environment** designed to standardise and promote future research in tactile-based manipulation. The environments are designed to cover a wide range of tactile interactions (sparse, intermittent, and sustained).

---

## 🛠️ Installation

1. Install Isaac Lab (recommend [pip installation](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/isaaclab_pip_installation.html#))
2. Install [isaaclab_rl](https://github.com/elle-miller/isaaclab_rl) as a local editable package
```
git clone git@github.com:elle-miller/isaaclab_rl.git
cd isaaclab_rl
pip install -e .
```
3. Clone this repository
```
git clone git@github.com:elle-miller/roto.git
```
---

## 🏃 Usage
Mostly the same as default Isaac Lab arguments, except need to specify an `agent_cfg` yaml file.

### Training
Here is how you would train a Find agent just with RL, a Bounce agent with RL + Tactile Reconstruction, and a Baoding agent with RL + Forward Dynamics.
```
python train.py --task Find --num_envs 4196 --headless --seed 1234 --agent_cfg rl_only_pt
python train.py --task Bounce --num_envs 4196 --headless --seed 1234 --agent_cfg tac_recon
python train.py --task Baoding --num_envs 4196 --headless --seed 1234 --agent_cfg full_dynamics
```

### Sweeping
Here are all the sweeps run in the paper. Same setup as `train.py`, but with an additional `--study` name argument.
```
python sweep.py --task Find --num_envs 4196 --headless --seed 1234 --agent_cfg rl_only_pt --study find_rl_only_pt
python sweep.py --task Find --num_envs 4196 --headless --seed 1234 --agent_cfg full_recon --study find_full_recon
python sweep.py --task Find --num_envs 4196 --headless --seed 1234 --agent_cfg tac_recon --study find_tac_recon
python sweep.py --task Find --num_envs 4196 --headless --seed 1234 --agent_cfg full_dynamics --study find_full_dynamics
python sweep.py --task Find --num_envs 4196 --headless --seed 1234 --agent_cfg tac_dynamics --study find_tac_dynamics

python sweep.py --task Bounce --num_envs 4196 --headless --seed 1234 --agent_cfg rl_only_pt --study bounce_rl_only_pt
python sweep.py --task Bounce --num_envs 4196 --headless --seed 1234 --agent_cfg full_recon --study bounce_full_recon
python sweep.py --task Bounce --num_envs 4196 --headless --seed 1234 --agent_cfg tac_recon --study bounce_tac_recon
python sweep.py --task Bounce --num_envs 4196 --headless --seed 1234 --agent_cfg full_dynamics --study bounce_full_dynamics
python sweep.py --task Bounce --num_envs 4196 --headless --seed 1234 --agent_cfg tac_dynamics --study bounce_tac_dynamics

python sweep.py --task Baoding --num_envs 4196 --headless --seed 1234 --agent_cfg rl_only_pt --study baoding_rl_only_pt
python sweep.py --task Baoding --num_envs 4196 --headless --seed 1234 --agent_cfg full_recon --study baoding_full_recon
python sweep.py --task Baoding --num_envs 4196 --headless --seed 1234 --agent_cfg tac_recon --study baoding_tac_recon
python sweep.py --task Baoding --num_envs 4196 --headless --seed 1234 --agent_cfg full_dynamics --study baoding_full_dynamics
python sweep.py --task Baoding --num_envs 4196 --headless --seed 1234 --agent_cfg tac_dynamics --study baoding_tac_dynamics
```

### Playing
To play with the viewer:
```
python play.py --task Bounce --agent_cfg rl_only_pt --num_envs 1 --headless --checkpoint paper_data/checkpoints/bounce_pt.pt
```
To generate a video:
```
python play.py --task Bounce --agent_cfg rl_only_pt --num_envs 1 --checkpoint paper_data/checkpoints/bounce_pt.pt --video
```
---
## 📊 Benchmark Results

Please see the paper. Note that the environments in this repo have been improved since the paper:
- Removed unnecessary dense rewards (time in air for Bounce, distance rewards for Baoding)
- Explicitly added joint control errors to the proprioceptive observation
- Fixed joint velocity normalisation

To run the paper checkpoints in the original environments, please... **TODO**.

---
## Data
The data in the paper is available.....
---
## 📧 Contact
For any questions, issues, or collaborations, please feel free to reach out:

Maintainer: Elle Miller
Email: elle.miller@ed.ac.uk
Project Website: https://elle-miller.github.io/tactile_rl

This project is licensed under the MIT License - see the LICENSE file for details.
---
## 📄 Citation
If you use this benchmark environment in your academic or professional research, please cite the following work:

```
@inproceedings{miller2025tactilerl,
  author    = {Miller, Elle and McInroe, Trevor and Abel, David and Mac Aodha, Oisin and Vijayakumar, Sethu},
  title     = {Enhancing Tactile-based Reinforcement Learning for Robotic Control},
  journal   = {NeurIPS},
  year      = {2025},
}
```
---
## TODO

`RotoEnv` inherits from `DirectRLEnv`
- _configure_gym_env_spaces
- _pre_physics_step
- _apply_action sets joint pos targets (with moving average)
- _get_observations
- _reset_robot(env_ids, joint_pos_noise) resets robot joint pos
- _compute_intermediate_values(env_ids) computes normalised joint pos/vel

`${Robot}$Env` inherits from `RotoEnv`
- FrankaEnv, ShadowEnv
- _setup_scene
- _get_proprioception
- _get_tactile
- _compute_intermediate_values
- _get_dones with null termination and timeout truncation

`${Task}$Env` inherets from `${Robot}$Env`
- _get_gt

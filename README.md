# RoTO: Robot Tactile Olympiad
RoTO is a **reinforcement learning benchmark environment** designed to standardise and promote future research in tactile-based manipulation. It is introduced in detail in [Enhancing Tactile-based RL for Robotic Control](https://elle-miller.github.io/tactile_rl/) (NeurIPS 2025).  The environments are designed to cover a wide range of tactile interactions: sparse (Find), intermittent (Bounce), and sustained (Baoding). We will continue to add more environments and strongly welcome contributions 🤗

> **📌 New tactile hardware (PadTac + BioTac).** Everything below this point is the
> original RoTO documentation. Every change made for this project on top of it — new
> FSR "pad" + BioTac fingertip tactile sensing for the Shadow Hand Lite's Baoding task,
> the new robots/configs that come with it, and how to run all of it — is addressed in
> one place: the **"🧩 New Tactile Hardware: PadTac + BioTac (Shadow Hand Lite)"**
> section near the bottom of this README. Start there if you're picking this project up.

<img src="readme_assets/images/roto.png" 
     width="400" 
     border="1"
     style="display: block; margin: 0 auto;"/>

## ✨ Overview

<img src="readme_assets/images/setup.png" width="1000" border="1"/>

We split the paper code across two repositories. Imagine the typical RL loop: you can think of `multimodal_rl` as the agent, and `roto` as the environment. We did this for modularity, in case you want to use your own RL repository instead of ours (there will be some integration to achieve this but happy to help).

`multimodal_rl`: The motto of this repo is _"doing good RL with Isaac Lab as painlessly as possible"_. We started from the [skrl](https://github.com/Toni-SM/skrl) library and made significant changes to better handle multimodal dictionary observations, observation stacking and associated memory management, and integrated self-supervision. Many existing libraries did not provide support for doing robust RL research (correct evaluation metrics, distinct train/evaluation envs, integrated hyperparameter optimisation). These are well established norms in the RL research community, but are not yet consistently present in RL+robotics research, which we want to encourage 🚀

`roto`: This repo just contains the robot configurations and task definitions. We take advantage of class inheritance to heavily reduce repeated code. `RotoEnv` is a child of `DirectRLEnv`, and sets up basic functions to perform joint position control of a robot and reset it. `[Robot]Env` is a child of `RotoEnv`, defining robot-specific functions that do not change task-to-task, e.g. the proprioceptive observation key. Finally, `[Task]Env` defines task-specific functions such as setting up the environment, rewards, and episode resets.


## 🤖 Environments

The agents are all joint position controlled. Franka has 9 joints, Shadow has 20 actuated joints.

| Environment | Description | Observations | Rewards | Resets |
| :---: | :--- | :--- | :--- | :--- |
| <img src="readme_assets/images/find.png" alt="Find Environment" width="400px"> | The agent must locate a fixed ball on a plate as quickly as possible. | Proprioception + 2 binary contacts | Distance reward from end-effector to ball | Timestep limit |
| <img src="readme_assets/images/bounce.png" alt="Bounce Environment" width="400px"> | The agent must bounce a ball as many times as possible within 10s. | Proprioception + 17 binary contacts | Small airtime reward + bounce bonus | Timestep limit, ball falls |
| <img src="readme_assets/images/baoding.png" alt="Baoding Environment" width="400px"> | The agent must rotate two small balls around each other without letting them  drop. | Proprioception + 17 binary contacts | Small distance reward to ball target + successful rotation bonus | Timestep limit, ball falls |

## Observations

We use dictionary-style observations, and categorising into proprioception, tactile, rgb, depth, and gt (ground-truth). The proprioception & tactile methods should be defined in `RobotEnv`, but gt information is task-dependent. To specify which observations are used, add the keys to `obs_list` in the agent cfg..
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
4. Test the installation by playing a trained agent in the viewer or saving a video. Note that the viewer playback is much slower than real-time.
```
python scripts/play.py --task Baoding --num_envs 512 --agent_cfg forward_dynamics_memory --checkpoint readme_assets/checkpoints/baoding_memory.pt
python scripts/play.py --task Baoding --num_envs 512 --agent_cfg forward_dynamics_memory --video --video_length 1200 --headless --checkpoint readme_assets/checkpoints/baoding_memory.pt
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


## 📊 Benchmark Results [in-progress]

Please see the paper for now.

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
- Observation augmentations (code exists just need to integrate)
- Integrate TacSL for high-resolution touch sensing when it becomes released: https://github.com/isaac-sim/IsaacGymEnvs/issues/244
- Provide transformer architectures
- Action chunking

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

---

## 🧩 New Tactile Hardware: PadTac + BioTac (Shadow Hand Lite)

Everything in this section is **new on top of the RoTO documentation above** — a new
tactile sensing modality for Shadow Lite **Baoding**, the robots/configs needed to
train it, and how to run every part of it. Nothing above this line changed; this is
the single place documenting what's different and how to use it.

### What changed

The rest of this README describes per-link binary contact tactile sensing. This project
adds a second tactile modality for **Shadow Lite Baoding only** that mirrors discrete
sensors on the real hand instead of one channel per touched link:

- **PadTac** — 12 discrete FSR ("pad") tactile sensor sites placed at specific points on
  the palm and phalanges. Poses are authored in
  [`roto/assets/shadow_lite/PAD_POSES.yaml`](roto/assets/shadow_lite/PAD_POSES.yaml) and
  baked into the `shadow_padtac.usd` robot asset.
- **PadTac + BioTac (`padtac_bt`)** — the same 12 FSR pads, plus 4 BioTac SP fingertip
  sensors on the distal links (`shadow_padtac_biotac.usd`). This is the configuration
  actively deployed to hardware.
- Both scatter contact into the same **24-channel** tactile vector used elsewhere in
  this codebase; only 12–16 of those channels are ever active per variant (the rest
  stay hard 0). New env/config classes: `ShadowLitePadTacEnv(Cfg)` and
  `ShadowLitePadTacBTEnv(Cfg)` in
  [`roto/tasks/robots/shadowlite/shadowlite.py`](roto/tasks/robots/shadowlite/shadowlite.py),
  `BaodingShadowLitePadTacCfg`/`BaodingShadowLitePadTacBTCfg` in
  [`roto/tasks/baoding/baoding.py`](roto/tasks/baoding/baoding.py).
- **Only Baoding on Shadow Lite has PadTac support** — Find, Bounce, and the other
  robots (Shadow, ORCA, Allegro) are untouched.
- **Obs/action contract:** 13 control-joint actions, 304-d observation (per-step
  `prop(52) + tactile(24)`, stacked ×4) — same shape as `rl_only_pt_padtac`.
- New domain randomization shipped with `BaodingShadowLitePadTacBTCfg`: ball mass range
  45–100g (was a fixed 55g), ball friction range, an opt-in command "slew" matching the
  hardware's `SPEED_FRAC` rate limiter (off by default), and FSR "taxel" corruption/flip
  DR (a random subset of the 12 FSR channels forced stuck + intermittently dithered each
  episode; the 4 BioTac channels are never touched by this). See the docstring on
  `BaodingShadowLitePadTacBTCfg` for the exact knobs.
- These configs also override `coupling_theta = 0.785` (vs. the sysid-derived `0.875` on
  plain `shadowlite`) to match the coupling law the trial15/27 hardware checkpoints were
  trained and are evaluated under.
- `default.yaml` (shared by all `shadowlite*` agent configs) now has `tactile` commented
  out of `obs_list` by default — each PadTac(+BT) agent config re-enables `prop + tactile`
  explicitly, so don't assume tactile is on unless the specific `--agent_cfg` says so.
- `scripts/play.py` now also dumps a `sim_policy_log_seed<seed>.npz` trace (actions,
  positions, commands, velocities, pos-error, tactile) on every run — controlled by
  `--record_steps` (default 300 steps = 5s @ 60Hz). This is the sim-side half of the
  sim-vs-hardware comparisons used when validating a deploy.

### New `--robot` options (Baoding only)

| `--robot` value | Tactile sensors | Robot asset |
|---|---|---|
| `shadowlite_padtac` | 12 FSR pads only | `shadow_padtac.usd` |
| `shadowlite_padtac_bt` | 12 FSR pads + 4 BioTac fingertips | `shadow_padtac_biotac.usd` |

Agent configs for these live under the same
`roto/tasks/baoding/agents/shadowlite/` folder used by plain `shadowlite`.

### New agent configs (`roto/tasks/baoding/agents/shadowlite/`)

| `--agent_cfg` | Use with `--robot` | Purpose |
|---|---|---|
| `rl_only_pt_padtac` | `shadowlite_padtac` | Scratch RL, pads only |
| `rl_only_pt_padtac_bt` | `shadowlite_padtac_bt` | Scratch RL, pads + BioTac |
| `forward_dynamics_padtac_bt` | `shadowlite_padtac_bt` | + self-supervised forward-dynamics auxiliary loss |
| `rl_only_pt_padtac_bt_sweep` | `shadowlite_padtac_bt` | Scratch Optuna sweep, ships with `zero_tactile: true` baked in — this is specifically the **prop-only ablation** sweep (same 304-d obs shape, tactile zeroed at the source), not a general tactile sweep config |

### Running it

These flags work with the same `train.py` / `play.py` / `sweep.py` entry points used
elsewhere in this README — `--robot` just wasn't shown above since the rest of this
file predates the multi-robot / multi-tactile work.

**Train from scratch:**
```bash
python scripts/train.py --task Baoding --robot shadowlite_padtac_bt --agent_cfg rl_only_pt_padtac_bt --num_envs 4096 --headless --seed 1234

# + forward-dynamics self-supervision
python scripts/train.py --task Baoding --robot shadowlite_padtac_bt --agent_cfg forward_dynamics_padtac_bt --num_envs 4096 --headless --seed 1234
```

**Sweep (prop-only ablation, scratch, no checkpoint):**
```bash
python scripts/sweep.py --task Baoding --robot shadowlite_padtac_bt --agent_cfg rl_only_pt_padtac_bt_sweep --num_envs 4096 --headless --seed 1234 --study my_ablation_sweep
```

**Play / evaluate a checkpoint** (also dumps `sim_policy_log_seed<seed>.npz`):
```bash
python scripts/play.py --task Baoding --robot shadowlite_padtac_bt --agent_cfg rl_only_pt_padtac_bt --checkpoint <path/to/best_agent.pt> --num_envs 512 --record_steps 300
```

There is no separate warm-start/fine-tune CLI script for PadTac(+BT) checkpoints in
this repo yet — [`fine-tune/FINETUNING.md`](fine-tune/FINETUNING.md) documents the
(encoder) fine-tuning approach actually used to close the sim-to-real observation gap
for these checkpoints; read it before starting any sim-to-real fine-tuning work.
`scripts/ablate_play.py` and `replay_motion_test/` provide the broader ablation /
open-loop-replay tooling this hand's checkpoints have been evaluated with, but they
aren't PadTac-specific, so they aren't re-documented here.

### Deploying to real hardware

[`scripts/deploy_policy_simtactile_curlamp.py`](scripts/deploy_policy_simtactile_curlamp.py)
is the current real-hardware deploy script for these checkpoints (ROS + the physical
Shadow Hand Lite, reading the FSR mux over serial and, optionally, BioTac). Unlike
`train.py`/`play.py` it is **not** argparse-driven — edit the mode/protocol constants
documented at the top of the file (SIM / ZERO / REAL tactile source, plus an
experimental curl-joint `pos_err` amplification toggle) and run it directly:
```bash
python scripts/deploy_policy_simtactile_curlamp.py
```
Read its module docstring first — it explains exactly what each mode does and why, and
flags which parts are diagnostic-only vs. policy-affecting. It imports
`from fsr_pad_map import FSR_CHANNELS` and falls back to an inlined copy of the same
12-value channel list if that module isn't present (`fsr_pad_map.py` itself isn't part
of this repo), so hardware runs are already wired to the pad channel layout above
without extra setup.

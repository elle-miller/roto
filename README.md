# isaaclab_rl_project

## Installation

1. Install Isaac Lab via pip with [these instructions](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/isaaclab_pip_installation.html)

2. Install `isaaclab_rl` as a local editable package.

```
git clone git@github.com:elle-miller/isaaclab_rl.git
cd isaaclab_rl
pip install -e .
```
You should now see it with `pip show isaaclab_rl`.

3. Create your own project repo 

```
git clone git@github.com:elle-miller/isaaclab_rl_project.git
mv isaaclab_rl_project my_cool_project_name
cd my_cool_project_name
```

4. Test everything is working OK with the Franka Lift environment
```
python train.py --task Franka_Lift --num_envs 8192 --headless
```
You should hit a return of ~6000 by 40 million timesteps. 

5. Make your own environment


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

3. Create your own project

```
git clone git@github.com:elle-miller/isaaclab_rl_project.git
mv isaaclab_rl_project my_cool_project_name
cd my_cool_project_name
python scripts/train.py --task Franka_Lift --headless
```
See the `isaaclab_rl_project` README for instructions on how to create your own environments.
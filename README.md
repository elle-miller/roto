# isaaclab_rl_project

![isaaclab_rl](https://github.com/user-attachments/assets/72036a2f-41ab-4317-ad30-8a165afa83a5)

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

# play checkpoint with viewer
python play.py --task Franka_Lift --num_envs 256 --checkpoint logs/franka/lift/.../checkpoints/best_agent.pt

# save video
```
You should hit a return of ~8000 by 40 million timesteps (check "Eval episode returns / returns" on wandb)

5. Make your own environment!

TODO


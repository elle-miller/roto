# isaaclab_rl_project

![isaaclab_rl](https://github.com/user-attachments/assets/72036a2f-41ab-4317-ad30-8a165afa83a5)

1. Follow instructions here to install Isaac Lab and twin library [isaaclab_rl](https://github.com/elle-miller/isaaclab_rl)

2. Create your own project repo 

```
git clone git@github.com:elle-miller/isaaclab_rl_project.git
mv isaaclab_rl_project my_cool_project_name
cd my_cool_project_name
```

3. Test everything is working OK with the Franka Lift environment
```
python train.py --task Franka_Lift --num_envs 8192 --headless

# play checkpoint with viewer
python play.py --task Franka_Lift --num_envs 256 --checkpoint logs/franka/lift/.../checkpoints/best_agent.pt
```
You should hit a return of ~8000 by 40 million timesteps (check "Eval episode returns / returns" on wandb)

4. Make your own environment

TODO
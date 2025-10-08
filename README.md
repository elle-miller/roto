# roto



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

### training

```

python train.py --task Bounce --num_envs 4196 --headless --seed 1234 --agent_cfg full_dynamics


```

### sweeping

```
python sweep.py --task Bounce --num_envs 4196 --headless --seed 1234 --agent_cfg full_dynamics --study bounce_full_dynamics
```
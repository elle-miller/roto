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


### playing

```
python play.py --task Bounce --num_envs 1 --headless --seed 1234 --video --checkpoint results/cam_ready/checkpoints/bounce_pt.pt

```
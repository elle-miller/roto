#!/bin/bash
#
#SBATCH --job-name=bo_pt
#
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=100:00:00
#SBATCH --mem-per-cpu=4G
# 
#SBATCH --gres=gpu:1

eval "$(conda shell.bash hook)"
conda activate env_isaaclab


# python sweep.py --task Bounce --num_envs 4196 --headless --seed 1234 --agent_cfg rl_only_pt --study bounce_rl_only_pt

python sweep.py --task Bounce --num_envs 4196 --headless --seed 1234 --agent_cfg full_dynamics --study bounce_full_dynamics_friyay
# python sweep.py --task Bounce --num_envs 4196 --headless --seed 1234 --agent_cfg tac_recon --study bounce_tac_recon
# python sweep.py --task Bounce --num_envs 4196 --headless --seed 1234 --agent_cfg tac_recon --study bounce_tac_recon

# TEST_SEEDS=(5 6 7 8 9)

# for SEED in "${TEST_SEEDS[@]}"; do
#     echo "Running with seed $SEED"
#     python train.py --task Bounce --num_envs 4196 --headless --seed "$SEED" --agent_cfg full_dynamics
# done
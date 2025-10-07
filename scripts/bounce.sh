#!/bin/bash
#
#SBATCH --job-name=bo_tr
#
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=100:00:00
#SBATCH --mem-per-cpu=4G
# 
#SBATCH --gres=gpu:1

eval "$(conda shell.bash hook)"
conda activate env_isaaclab


# python sweep.py --task Shadow_Bounce_TactileRecon --num_envs 4196 --headless --seed 1234 --study TactileRecon_ireland --env bounce --ssl tac_recon
# python sweep.py --task Shadow_Bounce_Recon --num_envs 4196 --headless --seed 1234 --study FullRecon --env bounce --ssl full_recon

python sweep.py --task Shadow_Bounce --num_envs 4196 --headless --seed 1234 --study testing --env bounce --ssl full_recon


# TEST_SEEDS=(6 7 8 9)

# for SEED in "${TEST_SEEDS[@]}"; do
#     echo "Running with seed $SEED"
#     python train.py --task Shadow_Bounce_TactileRecon --num_envs 4196 --headless --seed "$SEED"
# done
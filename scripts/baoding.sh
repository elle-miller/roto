#!/bin/bash
#
#SBATCH --job-name=baoding_fr
#
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=48:00:00
#SBATCH --mem-per-cpu=4G
# 
#SBATCH --gres=gpu:1


eval "$(conda shell.bash hook)"
conda activate env_isaaclab

python sweep.py --task Shadow_Baoding_Recon --num_envs 4196 --headless --seed 1234 --study full_recon_ireland --env baoding --ssl full_recon


# TEST_SEEDS=(5 6 7 8 9)

# for SEED in "${TEST_SEEDS[@]}"; do
#     echo "Running with seed $SEED"
#     python train.py --task Shadow_Baoding_TactileRecon --num_envs 4196 --headless --seed "$SEED"
# done
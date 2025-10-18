#!/bin/bash
#
#SBATCH --job-name=find_fd
#
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=100:00:00
#SBATCH --mem-per-cpu=4G
# 
#SBATCH --gres=gpu:1

eval "$(conda shell.bash hook)"
conda activate env_isaaclab

# python sweep.py --task FrankaFind_TactileRecon --num_envs 4196 --headless --seed 1234 --env find --ssl tac_recon --study TactileRecon_ireland
# python sweep.py --task FrankaFind_FullRecon --num_envs 4196 --headless --seed 1234 --env find --ssl full_recon --study full_recon
python sweep.py --task Find --num_envs 4196 --headless --seed 1234 --agent_cfg full_dynamics --study find_full_dynamics


# TEST_SEEDS=(5 6 7 8 9)

# # for SEED in "${TEST_SEEDS[@]}"; do
# #     echo "Running with seed $SEED"
# #     python train.py --task FrankaFind --num_envs 4196 --headless --seed "$SEED"
# # done

# for SEED in "${TEST_SEEDS[@]}"; do
#     echo "Running with seed $SEED"
#     python train.py --task FrankaFind_TactileRecon --num_envs 4196 --headless --seed "$SEED"
# done
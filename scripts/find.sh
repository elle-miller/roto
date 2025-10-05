#!/bin/bash
eval "$(conda shell.bash hook)"
conda activate env_isaaclab

python sweep.py --task FrankaFind_TactileRecon --num_envs 4196 --headless --seed 1234 --env find --ssl tac_recon --study FrankaFind_TactileRecon_ireland


# TEST_SEEDS=(5 6 7 8 9)

# # for SEED in "${TEST_SEEDS[@]}"; do
# #     echo "Running with seed $SEED"
# #     python train.py --task FrankaFind --num_envs 4196 --headless --seed "$SEED"
# # done

# for SEED in "${TEST_SEEDS[@]}"; do
#     echo "Running with seed $SEED"
#     python train.py --task FrankaFind_TactileRecon --num_envs 4196 --headless --seed "$SEED"
# done
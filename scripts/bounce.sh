#!/bin/bash
eval "$(conda shell.bash hook)"
conda activate env_isaaclab


python sweep.py --task Shadow_Bounce_TactileRecon --num_envs 4196 --headless --seed 1234 --study tacShadow_Bounce_TactileRecon_ireland --env bounce --ssl tac_recon



# TEST_SEEDS=(5 6 7 8 9)

# # for SEED in "${TEST_SEEDS[@]}"; do
# #     echo "Running with seed $SEED"
# #     python train.py --task FrankaFind --num_envs 4196 --headless --seed "$SEED"
# # done

# for SEED in "${TEST_SEEDS[@]}"; do
#     echo "Running with seed $SEED"
#     python train.py --task Shadow_Bounce --num_envs 4196 --headless --seed "$SEED"
# done
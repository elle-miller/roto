#!/bin/bash
eval "$(conda shell.bash hook)"
conda activate env_isaaclab

TEST_SEEDS=(5 6 7 8 9)

for SEED in "${TEST_SEEDS[@]}"; do
    echo "Running with seed $SEED"
    python train.py --task Shadow_Baoding --num_envs 4196 --headless --seed "$SEED"
done
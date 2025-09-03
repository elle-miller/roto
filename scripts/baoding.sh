#!/bin/bash
eval "$(conda shell.bash hook)"
conda activate env_isaaclab
SEEDS=(1 12 123 1234 12345)

for SEED in "${SEEDS[@]}"; do
    echo "Running with seed $SEED"
    python train.py --task Shadow_Baoding --num_envs 4096 --headless --seed "$SEED"
done
#!/bin/bash
#
#SBATCH --job-name=job_name
#
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=100:00:00
#SBATCH --mem-per-cpu=4G
# 
#SBATCH --gres=gpu:1


eval "$(conda shell.bash hook)"
conda activate env_isaaclab

python sweep.py --task Baoding --num_envs 4196 --headless --seed 1234 --agent_cfg forward_dynamics --study baoding_sweep

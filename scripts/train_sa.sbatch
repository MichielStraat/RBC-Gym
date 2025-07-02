#!/bin/bash

#SBATCH --job-name=rbc3d_sa
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=0-02:00:00
#SBATCH --output=/homes/mstraat/Projects/RBC-Gym/slurmout/%x_%j.out
#SBATCH --error=/homes/mstraat/Projects/RBC-Gym/slurmout/%x_%j.err

source ~/miniconda3/etc/profile.d/conda.sh
conda activate rbcgym
cd ~/Projects/RBC-Gym

# srun xvfb-run --auto-servernum -s "-screen 0 1024x768x24" python example/run_3D.py
srun xvfb-run --auto-servernum -s "-screen 0 1024x768x24" python experiments/run_sarl.py

#!/bin/bash

#SBATCH --job-name=rbc3d_sa
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=0-06:00:00
#SBATCH --output=/homes/mstraat/Projects/RBC-Gym/slurmout/%x_%j.out
#SBATCH --error=/homes/mstraat/Projects/RBC-Gym/slurmout/%x_%j.err

# ----------------------------------------
# Activate conda
# ----------------------------------------
source ~/miniconda3/etc/profile.d/conda.sh
conda activate rbcgym

# ----------------------------------------
# Change to project directory
# ----------------------------------------
cd ~/Projects/RBC-Gym

# ----------------------------------------
# Experiment config (frozen at submission)
# ----------------------------------------
export RL_N_STEPS=150
export RL_N_ENVS=1
export RL_BATCH_SIZE=30 # should be a factor of N_STEPS * N_ENVS
export RL_N_EPOCHS=10
export RL_ENT_COEF=0.01
export RL_STAT_WINDOW_SIZE=1    # Number of episodes to average over for rollout training statistics
export RL_NR_ITERATIONS=150  # Number of iterations (rollout collections), i.e. model update episodes on a dataset collected from a rollout

export RBC_HEATER_DURATION=0.375
export RBC_HEATER_LIMIT=0.9
export RBC_RAYLEIGH_NUMBER=2500
export RBC_EPISODE_LENGTH=225   # RL_N_STEPS * RBC_HEATER_DURATION * physicalheight^2

# below are the defaults as used in the script
# export RL_N_STEPS=4
# export RL_N_ENVS=1
# export RL_BATCH_SIZE=4
# export RL_N_EPOCHS=10
# export RL_ENT_COEF=0.01
# export RL_STAT_WINDOW_SIZE=50

# export RBC_HEATER_DURATION=0.375
# export RBC_HEATER_LIMIT=0.9
# export RBC_RAYLEIGH_NUMBER=2500
# export RBC_EPISODE_LENGTH=10

# ----------------------------------------
# Create unique run directory
# ----------------------------------------
RUN_DIR="/homes/mstraat/Projects/RBC-Gym/results/run_${SLURM_JOB_ID}"
mkdir -p "$RUN_DIR"
mkdir -p "$RUN_DIR/slurmout"

# Copy this sbatch script for reproducibility
cp "$0" "$RUN_DIR/"

# Save config snapshot
# Save config snapshot for reproducibility
cat <<EOF > "$RUN_DIR/config.yaml"
rl_n_steps: $RL_N_STEPS
rl_n_envs: $RL_N_ENVS
rl_batch_size: $RL_BATCH_SIZE
rl_n_epochs: $RL_N_EPOCHS
rl_ent_coef: $RL_ENT_COEF
rl_stat_window_size: $RL_STAT_WINDOW_SIZE
rl_nr_iterations: $RL_NR_ITERATIONS
rbc_heater_duration: $RBC_HEATER_DURATION
rbc_heater_limit: $RBC_HEATER_LIMIT
rbc_rayleigh_number: $RBC_RAYLEIGH_NUMBER
rbc_episode_length: $RBC_EPISODE_LENGTH
EOF

# ----------------------------------------
# Run experiment
# ----------------------------------------
srun xvfb-run --auto-servernum -s "-screen 0 1024x768x24" \
    python experiments/run_sarl.py --config "$RUN_DIR/config.yaml" --output_dir "$RUN_DIR"

# ----------------------------------------
# Move Slurm logs into run folder
# ----------------------------------------
LOG_NAME="${SLURM_JOB_NAME}_${SLURM_JOB_ID}"

# Original logs are in: slurmout/LOG_NAME.out + .err
cp "/homes/mstraat/Projects/RBC-Gym/slurmout/${LOG_NAME}.out" "$RUN_DIR/slurmout/${LOG_NAME}.out"
cp "/homes/mstraat/Projects/RBC-Gym/slurmout/${LOG_NAME}.err" "$RUN_DIR/slurmout/${LOG_NAME}.err"

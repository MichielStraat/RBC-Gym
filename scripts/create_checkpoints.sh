#!/bin/sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Set the path to your Julia environment
export JULIA_PROJECT="${SCRIPT_DIR}/../.venv/julia_env"

# params
ra=10000
nr_episodes=20
episode_length=400

# Now run your Julia script
julia ${SCRIPT_DIR}/../src/rbc_gym/sim/rbc_sim2D.jl --dir ${SCRIPT_DIR}/../data/ --random_inits ${nr_episodes} --Ra ${ra} --N 96 64 --min_b 1 --random_kick 0.02 --delta_t 0.03 --delta_t_snap 0.3 --duration ${episode_length} --use_cpu
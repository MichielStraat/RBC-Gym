#!/bin/sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Set the path to your Julia environment
export JULIA_PROJECT="${SCRIPT_DIR}/../.venv/julia_env"

# params
if [ $# -lt 1 ]; then
    echo "Usage: $0 <ra>"
    exit 1
fi
ra=$1
dir=${SCRIPT_DIR}/../data/checkpoints/
episode_length=600

# Now run your Julia script
julia ${SCRIPT_DIR}/../src/rbc_gym/sim/rbc_sim2D.jl --dir ${dir}/train --seed 42 --random_inits 20 --Ra ${ra} --N 96 64 --min_b 1 --random_kick 0.02 --delta_t 0.03 --delta_t_snap 0.3 --duration ${episode_length} --use_cpu
julia ${SCRIPT_DIR}/../src/rbc_gym/sim/rbc_sim2D.jl --dir ${dir}/test --seed 62 --random_inits 10 --Ra ${ra} --N 96 64 --min_b 1 --random_kick 0.02 --delta_t 0.03 --delta_t_snap 0.3 --duration ${episode_length} --use_cpu
julia ${SCRIPT_DIR}/../src/rbc_gym/sim/rbc_sim2D.jl --dir ${dir}/val --seed 72 --random_inits 10 --Ra ${ra} --N 96 64 --min_b 1 --random_kick 0.02 --delta_t 0.03 --delta_t_snap 0.3 --duration ${episode_length} --use_cpu
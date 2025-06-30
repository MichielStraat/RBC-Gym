#!/bin/bash

# Script to run flow statistics experiment for Rayleigh-Bénard convection
# Runs the experiment for a predefined list of Rayleigh numbers
# Usage: ./run_flowstats.sh

# Predefined list of Rayleigh numbers
# RAYLEIGH_NUMBERS=(500 750 1000 1500 2000 4000 8000 16000 32000 64000 128000 256000 512000 1000000)
RAYLEIGH_NUMBERS=(64000 128000 256000 512000 1000000)



SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo "Running flow statistics experiments for multiple Rayleigh numbers"
echo "Project root: $PROJECT_ROOT"
echo "Rayleigh numbers to process: ${RAYLEIGH_NUMBERS[@]}"
echo "Total experiments: ${#RAYLEIGH_NUMBERS[@]}"
echo ""

# Change to project root directory
cd "$PROJECT_ROOT"

# Check if virtual environment exists and activate it
if [ -d ".venv" ]; then
    echo "Activating virtual environment (.venv)..."
    source .venv/bin/activate
    echo "Virtual environment activated"
    echo "Python executable: $(which python)"
    echo ""
else
    echo "Warning: No .venv directory found. Using system Python."
    echo "Python executable: $(which python)"
    echo ""
fi

# Counter for tracking progress
counter=1
total=${#RAYLEIGH_NUMBERS[@]}

# Run the Python script for each Rayleigh number separately
for ra in "${RAYLEIGH_NUMBERS[@]}"; do
    echo "[$counter/$total] Starting experiment for Ra=$ra"
    
    # Run the Python script with a single Rayleigh number
    python experiments/flowstats/flowstats_ra.py --rayleigh_numbers "$ra"
    
    # Check if the command was successful
    if [ $? -eq 0 ]; then
        echo "[$counter/$total] Experiment completed successfully for Ra=$ra"
    else
        echo "[$counter/$total] ERROR: Experiment failed for Ra=$ra"
        echo "Continuing with next Rayleigh number..."
    fi
    
    echo ""
    ((counter++))
done

echo "All experiments completed!"

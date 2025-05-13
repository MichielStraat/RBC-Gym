# Rayleigh-Bènard Convection Simulation

## Setup
We use [uv](https://docs.astral.sh/uv/) to manage python dependencies. Install virtual environment via:
```bash
uv sync
source .venv/bin/activate
```

## Get started


## TODOs
- [] vectorized environments
- [] GPU compatibility
- [] Checkpoints
- [] 3D simulation
- [] Timing analysis
- [] Multi-Agent Environment via PettingZoo
- [] Seed; does not change with no given seed -> desired behavior?

## Timing
```text
Julia based simulation

Julia init time: 7.68 seconds
Average time to reset env: 0.02 seconds
Average time to step one timestep (dt=1): 0.12 seconds
Average time to step with rendering: 0.12 seconds
```
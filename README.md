# Rayleigh-Bènard Convection Simulation

## Setup
We use [uv](https://docs.astral.sh/uv/) to manage python dependencies. Install virtual environment via:
```bash
uv sync
source .venv/bin/activate
```

The simulation is based on Julia. Please install according to their [docs](https://julialang.org/install/) or via their standalone installer:
```bash
curl -fsSL https://install.julialang.org | sh
```

## Get started
### Run plain environment
Small example to run a 2D Rayleigh-Benard Convection simulation wrapped in a gym environment:
```bash
python example/run.py
```
### Run in parallel utilizing vectorized envs
```bash
python example/run_vectorized.py
```
### Run with wrappers
```bash
python example/run_wrappers.py
```


## Roadmap
- [x] Timing analysis
- [x] vectorized environments
- [x] Wrappers
- [x] Checkpoints
- [x] Make installable

- [ ] Normalize Reward Wrapper
- [ ] GPU compatibility
- [ ] Create Dataset from python gym
- [ ] PyTests
- [ ] 3D simulation
- [ ] Multi-Agent Environment via PettingZoo

## Time efficiency
Time comparison vs a shenfun based simulation. Averaged over 1000 iterations on an apple sillicone based system.
```text
Julia based simulation

Julia init time: 7.68 seconds
Average time to reset env: 0.02 seconds
Average time to step one timestep (dt=1): 0.12 seconds (sim dt=0.03)
Average time to step with rendering: 0.12 seconds
```

```text
Shenfun based simulation

Shenfun init time: 0.00 seconds
Average time to reset env: 0.19 seconds
Average time to step one timestep (dt=1): 0.56 seconds (sim dt=0.03)
```
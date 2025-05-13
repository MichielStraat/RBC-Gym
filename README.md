# Rayleigh-Bènard Convection Simulation

## Setup
We use [uv](https://docs.astral.sh/uv/) to manage python dependencies. Install virtual environment via:
```bash
uv sync
source .venv/bin/activate
```

## Get started


## TODOs
- [ ] vectorized environments
- [ ] GPU compatibility
- [ ] Checkpoints
- [ ] 3D simulation
- [ ] Timing analysis
- [ ] Multi-Agent Environment via PettingZoo
- [ ] Seed; does not change with no given seed -> desired behavior?

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
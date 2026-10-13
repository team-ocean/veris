# Veris

Veris is a standalone differentiable sea-ice model in JAX. Define an experiment,
initialize it, run successive timesteps and save output through one public
lifecycle. Dynamics, thermodynamic growth and coupled experiments share that
workflow on CPU, GPU and explicitly configured JAX meshes.

Install the current checkout in an activated Python environment:

```bash
python -m pip install -e .
```

The central [command line interface](docs/cli.rst) lists, copies and runs setups:

```bash
veris show-setups
veris copy-setup growth --to ./my-growth
veris run ./my-growth/setup.py --steps 2 --output growth.nc
```

This coupled example supplies prescribed ocean geometry, masks and atmospheric
fields without an external ocean model:

```python
from veris import Execution, Experiment, Grid, Simulation

experiment = Experiment.builtin(
    "island", grid=Grid(6, 8), steps=3,
    settings={"dtype": "float32", "nEVPsteps": 2},
    execution=Execution(backend="cpu"),
)
with Simulation.initialize(experiment) as simulation:
    result = simulation.run()
    simulation.snapshot("final.nc", variables=("hIceMean", "Area"))
    simulation.save_restart("checkpoint.nc")
```

Use new output destinations for subsequent executions. Grid dimensions count
global physical cells in `(x, y)` order, excluding halos. The two EVP subcycles
above demonstrate the API rather than establish solver convergence. Float64
requires enabling JAX x64 before allocation; importing Veris changes no backend
or precision settings.

The [primary quickstart](docs/quickstart/user-guide.rst) includes scheduled output,
complete save-and-resume and optional observations. The
[custom ocean example](docs/quickstart/custom-ocean.rst) defines geometry, masks,
initial conditions and time-varying forcing using public imports. The
[integration guide](docs/reference/integration.rst) describes pure rollouts and
differentiation.

The source hierarchy groups experiment definitions and state in `veris.model`,
geometry and halos in `veris.grid`, scientific laws in `veris.physics`, numerical
solvers in `veris.numerics`, and initialization and execution in `veris.execution`.
Host storage and restart live in `veris.io`; examples and presets live in
`veris.setups`. The [package hierarchy](docs/reference/package-hierarchy.rst)
lists the complete tree and owning modules for specialized imports.

`Experiment` keeps numerical settings, physical constants, initial fields,
forcing, execution resources and output controls separate. Registry defaults and
validation remain authoritative. Definitions use Python or versioned JSON;
outputs save resolved definitions. `Simulation.step()` advances one timestep;
`run()` advances the remaining configured target. Absolute forcing and model time
remain correct across successive runs.

State remains an array-only JAX PyTree. Public `advance` performs one pure physics
timestep and `step` retains its existing supplied-transition rollout contract.
Use these functions for validated JIT, grad, JVP and VJP paths through initial
conditions and JAX forcing. Static configuration/constants, clocks and file I/O
stay outside differentiated computation. The wheel includes `py.typed`.

Snapshots and selected-field loading use physical cells without halos. Complete
restart preserves all numerical arrays including halos, the definition, constants,
forcing position and exact clock. It supports serial CPU or one GPU device on the
saved backend; distributed restart is unsupported. Restored scheduled output uses
a new file, the original cadence/calendar and only fully covered resumed mean
windows. See the [lifecycle API](docs/api/simulation.rst) for precise contracts.

Maintained dynamics, growth and parallel command-line examples use the same
lifecycle. Run `python -m veris.setups.run_dyn --help`,
`python -m veris.setups.run_growth --help` or
`python -m veris.setups.run_parallel --help`. Distributed initialization and mesh
ownership remain explicit; configure the runtime before enumerating devices.

Sphinx documentation and the scientific test harness live in `docs/` and `tests/`:

```bash
python -m pip install -r docs/requirements.txt
sphinx-build -E -W -b html docs docs/_build/html
JAX_PLATFORMS=cpu pytest tests/ -q --fast
JAX_PLATFORMS=cpu pytest tests/ -n 3 --dist=loadscope --partition-cpus --max-worker-restart=0
```

Run one pytest instance at a time and the full correctness suite before commits.
`--fast` selects a deterministic sample before fixture setup. See
[tests/README.md](tests/README.md) for reference provenance, gradients and hardware
checks, and [DESIGN.md](DESIGN.md) for architecture and verification design.

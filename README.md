# Veris

Veris is a standalone differentiable sea-ice model in JAX. Define an experiment,
initialize it, run successive timesteps and save output through one public
lifecycle. Dynamics, thermodynamic growth and coupled experiments share that
workflow on CPU, GPU and explicitly configured JAX meshes.

Start with [Getting started](docs/introduction/get-started.rst) for installation
from `jax-only`, a first command-line simulation and reading its output.
The [command line reference](docs/cli.rst) covers setup discovery, copying,
execution and overrides.

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

Use new output destinations when repeating this example. Its two EVP subcycles
demonstrate the API; they do not establish solver convergence.

- [Python lifecycle tutorial](docs/quickstart/user-guide.rst): successive runs,
  scheduled output, observations and exact restart.
- [Custom ocean](docs/quickstart/custom-ocean.rst) and
  [setup blueprint](docs/quickstart/setup-blueprint.rst): prescribed geometry,
  fields and reusable experiment factories.
- [Pure integration](docs/reference/integration.rst) and
  [automatic differentiation](docs/reference/automatic-differentiation.rst):
  rollouts, forcing sensitivities and derivative conventions.
- [Model setups](docs/reference/setup-gallery.rst): physical assumptions and
  serial, GPU and distributed examples.
- [Python API](docs/api/index.rst) and
  [package hierarchy](docs/reference/package-hierarchy.rst): contracts and ownership.
- [Development](docs/development.rst): checks, documentation builds and contribution
  workflow. [DESIGN.md](DESIGN.md) and [CHANGELOG.md](CHANGELOG.md) record
  architecture and verified progress.

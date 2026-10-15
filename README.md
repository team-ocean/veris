# Veris

Veris models how sea ice moves, grows and melts without an external ocean model.
It uses JAX, a Python library for calculations with arrays (tables of numbers).
The model is differentiable: it can calculate how changes in inputs affect its
results. Define an experiment (its inputs and settings), prepare its initial
values, advance it in timesteps (intervals of simulated time) and save the
results. The same workflow covers dynamics (ice motion), thermodynamic growth
(growth and melting through heat exchange) and coupled experiments (both
together). It runs on computer processors (CPU), graphics processors (GPU) and
explicitly configured JAX meshes (groups of devices).

Install the [latest release](https://github.com/team-ocean/veris/releases/latest)
with Python 3.12 or newer:

```sh
python -m pip install --upgrade veris
```

Start with [Getting started](https://veris.readthedocs.io/en/stable/introduction/get-started.html)
for installation, a first command-line simulation and reading its output.
The [stable documentation](https://veris.readthedocs.io/en/stable/) describes the
latest published release.
The [command line reference](docs/cli.rst) explains how to list and copy setups,
run them and override their settings.

This coupled example supplies the ocean's shape and size, masks (arrays marking
land and ocean) and atmospheric fields (weather values across the grid, the
cells dividing the modeled area). These prescribed inputs come from the setup,
without an external ocean model:

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
(repeated calculations in the elastic-viscous-plastic ice-motion solver) demonstrate the programming
interface; they do not show that the solver has converged to a solution.

- [Python lifecycle tutorial](docs/quickstart/user-guide.rst): successive runs,
  output saved on a schedule, observations and exact restart.
- [Custom ocean](docs/quickstart/custom-ocean.rst) and
  [setup blueprint](docs/quickstart/setup-blueprint.rst): prescribed geometry,
  fields and reusable functions that create experiments.
- [Pure integration](docs/reference/integration.rst) and
  [automatic differentiation](docs/reference/automatic-differentiation.rst):
  runs over several timesteps, sensitivity to supplied inputs and rules for
  calculating derivatives.
- [Model setups](docs/reference/setup-gallery.rst): physical assumptions and
  serial (single-device), GPU and distributed (multiple-process) examples.
- [Python API](docs/api/index.rst) and
  [package hierarchy](docs/reference/package-hierarchy.rst): interface rules and
  each part of the package's responsibilities.
- [Development](docs/development.rst): checks, documentation builds and contribution
  workflow. [DESIGN.md](DESIGN.md) and [CHANGELOG.md](CHANGELOG.md) record
  the model's design and verified progress.

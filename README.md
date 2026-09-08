# Veris

This branch contains the redesigned version of Veris, which uses JAX's sharded arrays to enable parallel execution.


### How to use

Veris can be installed from this repository via ```pip install -e .```. To run Veris as a standalone model, see the scripts provided [here](https://github.com/jpgaertner/veris_minimum_working_example/tree/jax_halo_exchange) [![DOI](https://zenodo.org/badge/953970891.svg)](https://doi.org/10.5281/zenodo.20642360).

### Local standalone example

The package has no Veros dependency. The former geographic ocean-coupled setup
and its downloaded forcing assets have been removed. A small artificial-island
example runs dynamics, transport, and thermodynamic growth with prescribed ocean
and atmospheric fields:

```python
import jax
from veris.setup.artificial import initialize, step

jax.config.update("jax_enable_x64", True)
state, settings = initialize()
for _ in range(3):
    state = step(state, settings, cooling=100.0)
jax.block_until_ready(state)
```

Use a fresh process for this serial example. See `tests/README.md` for the
validated environment and numerical test commands.

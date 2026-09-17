Dynamics only
=============

A 512-km basin has land on its last x and y lines, rotating ocean currents and
the reference's fixed wind snapshot 15 (day 5.875). Wind does not evolve with
time. Defaults are 1024 by 1024 physical cells, 100 steps of 600 seconds and 120
adaptive EVP iterations per step (alpha/beta 500, absolute wind forcing).
There is no thermodynamic growth. A small local CPU run is::

   python -m veris.setups.run_dyn --nx 32 --ny 48 --steps 2 --evp-steps 4 \
       --backend cpu --output output/dynamics.nc

Use ``--backend gpu`` for a GPU directly on the current node. If the environment
already sets ``JAX_PLATFORMS=cpu``, unset it or set ``JAX_PLATFORMS=cuda`` first.
The output contains all final State fields with halos removed, in ``(x, y)``
order. The runner reports compilation and integration times separately.
Compilation lowers and compiles the required scan shapes without executing a
warmup trajectory; integration executes exactly the requested model steps.

The Python API returns independent model objects::

   import jax
   from functools import partial
   from veris import step
   from veris.setups import run_dyn

   jax.config.update("jax_enable_x64", True)
   state, settings, constants = run_dyn.initialize(32, 48)
   advance = partial(run_dyn.step, conf=settings, phys=constants)
   state = step(state, advance, 3)

The setup-level ``step`` returns just State; ``compiled_step`` compiles the same
calculation. The public ``veris.step`` runs a scan with checkpointing; see
:doc:`../integration` for forcing, diagnostic histories and sharded rollouts.
Ocean stresses are separate Diagnostics fields. No coupling output is added to
State. ``settings_overrides``, ``physical_overrides`` and ``scenario_overrides``
allow controlled experiments without changing source files.

Reference adaptations
---------------------

The runnable cases correct the old serial mask's five-cell neighbor offset,
use the physical one-cell face stencil, and initialize periodic halos
consistently. They replace generated initializer imports, fixed home paths and
all-rank output writes. Fields unused by each isolated experiment retain the
shared VARIABLES defaults rather than the old blanket zero initialization;
all prescribed fields consumed by the experiment retain their reference values.
Experimental constants live in scenario registries, while universal physical
constants stay in PhysicalConstants. No changes to the physics kernels are needed.

Derivative validation
---------------------

The exact dynamics initial state has zero strain and closed walls. Guarded
norm operations now provide finite selected linearizations there; tests check
the actual initial State, its full pullback, and evolving wind sensitivities.
Smooth reference oracles and the growth column's longwave sensitivity remain
finite-difference checked. See :doc:`../automatic-differentiation` for zero-norm
conventions, physical branch thresholds and the genuine free-drift degeneracy.

Scenario controls
-----------------

.. exec::

   from veris.setups.run_dyn import DYNAMICS_SETTINGS
   print(".. list-table:: Dynamics only scenario controls")
   print("   :header-rows: 1\n")
   print("   * - Setting\n     - Default\n     - Units\n     - Description")
   for name, metadata in DYNAMICS_SETTINGS.items():
       print(f"   * - ``{name}``\n     - ``{metadata.default!r}``")
       print(f"     - {metadata.units}\n     - {metadata.description}")

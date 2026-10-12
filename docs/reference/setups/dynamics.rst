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

The command-line runner uses ``Experiment.builtin("dynamics", ...)`` and
``Simulation`` for the same lifecycle as coupled and growth experiments. Global
physical dimensions and settings belong to the definition; forcing, time and
output share :doc:`/api/simulation`. A public Python experiment is::

   from veris import Execution, Experiment, Grid, Simulation

   experiment = Experiment.builtin(
       "dynamics", grid=Grid(32, 48), steps=3,
       settings={"dtype": "float32", "nEVPsteps": 4},
       execution=Execution(backend="cpu"),
   )
   with Simulation.initialize(experiment) as simulation:
       result = simulation.run()

The preset factory returns initial fields and static model objects::

   import jax
   from functools import partial
   from veris import advance, step
   from veris.setups import run_dyn

   jax.config.update("jax_enable_x64", True)
   state, settings, constants = run_dyn.initialize(32, 48)
   transition = partial(
       advance, configuration=settings, constants=constants, physics="dynamics",
   )
   state = step(state, transition, 3, has_aux=True)

``advance`` returns State and Diagnostics. The public ``veris.step`` runs a
scan with checkpointing; see
:doc:`../integration` for forcing, diagnostic histories and sharded rollouts.
Ocean stresses are separate Diagnostics fields. No coupling output is added to
State. ``settings_overrides``, ``physical_overrides`` and ``scenario_overrides``
allow controlled experiments without changing source files.

Reference adaptations
---------------------

The runnable cases correct the old serial mask's five-cell neighbor offset,
use the physical one-cell face stencil, and initialize periodic halos
consistently. They replace generated initializer imports, fixed home paths and
all-rank output writes. Prescribed fields retain their reference values.
Unprescribed fields, including ocean and air temperature, start at zero as in
``initialize_dyn.py``. Dynamics-only execution does not consume those temperatures.
Independent original-source regression data verifies every startup field,
shared numerical setting and physical constant, then physical fields after
100 timesteps with 120 adaptive EVP iterations on a reduced 16 by 16 grid.
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

Dynamics only
=============

A 512-km basin has land on its last x and y grid lines, rotating ocean currents
and the reference's fixed wind snapshot 15 (day 5.875). Wind stays constant
throughout the run. Defaults are 1024 by 1024 physical cells, 100 steps of 600
seconds and 120 adaptive EVP iterations per step. EVP, or elastic-viscous-plastic,
is the method used to solve for ice velocity and stress (alpha/beta 500,
absolute wind forcing, which uses wind velocity without subtracting ice
velocity). This case has no thermodynamic growth, meaning freezing or melting.
A small local CPU run is::

   python -m veris.setups.run_dyn --nx 32 --ny 48 --steps 2 --evp-steps 4 \
       --backend cpu --output output/dynamics.nc

Use ``--backend gpu`` for a GPU directly on the current node. If the environment
already sets ``JAX_PLATFORMS=cpu``, unset it or set ``JAX_PLATFORMS=cuda`` first.
The output contains all final State fields with halos removed, in ``(x, y)``
order. State is the collection of model arrays. Halos are extra cells that hold
boundary or neighboring data. The runner reports compilation and integration
times separately. Compilation prepares executable code for the required loop
shapes without a warmup model run. Integration executes exactly the requested
model steps.

The command-line runner uses ``Experiment.builtin("dynamics", ...)`` and
``Simulation`` for the same run process as coupled (motion plus growth) and
growth experiments.
The definition holds global physical dimensions and settings. Forcing
(prescribed inputs), time and output use the shared :doc:`/api/simulation`
interface. A public Python experiment is::

   from veris import Execution, Experiment, Grid, Simulation

   experiment = Experiment.builtin(
       "dynamics", grid=Grid(32, 48), steps=3,
       settings={"dtype": "float32", "nEVPsteps": 4},
       execution=Execution(backend="cpu"),
   )
   with Simulation.initialize(experiment) as simulation:
       result = simulation.run()

The shared initializer returns initial fields and model objects that stay
fixed during compilation::

   import jax
   from functools import partial
   from veris import Experiment, Grid, advance, initialize_experiment, step

   jax.config.update("jax_enable_x64", True)
   state, settings, constants = initialize_experiment(
       Experiment.builtin("dynamics", grid=Grid(32, 48))
   )
   transition = partial(
       advance, configuration=settings, constants=constants, physics="dynamics",
   )
   state = step(state, transition, 3, has_aux=True)

``advance`` returns State and Diagnostics. The public ``veris.step`` repeats
timesteps with checkpointing, which recalculates intermediate values to save
memory during reverse differentiation. See :doc:`../integration` for forcing,
diagnostic histories and runs split across devices. Ocean stresses are
separate Diagnostics fields. Outputs for exchange with an ocean model do not
add fields to State. Experiment ``settings``, ``constants`` and ``scenario``
groups let you control experiments without changing source files.

Reference adaptations
---------------------

The runnable cases correct the old serial mask's five-cell neighbor offset.
Masks identify ocean and land. The cases use the physical one-cell face
stencil, a calculation based on neighboring cells, and initialize periodic
halos consistently. They replace generated initializer imports, fixed home
paths and file writes by every process. Prescribed fields keep their reference
values. Fields not prescribed, including ocean and air temperature, start at
zero as in ``initialize_dyn.py``. Dynamics-only execution does not use those
temperatures. Independent data from the original source checks every startup
field, shared numerical setting and physical constant. It also checks physical
fields after 100 timesteps with 120 adaptive EVP iterations on a reduced 16 by
16 grid. Experiment-specific constants live in scenario registries; universal
physical constants stay in PhysicalConstants. This requires no changes to the
physics calculations.

Derivative validation
---------------------

The exact dynamics initial state has zero strain (no ice deformation) and
closed walls. Safeguarded magnitude calculations give a finite chosen local
linear response there. Tests check the actual initial State, its full reverse
derivative and changing wind sensitivities. Finite differences, which compare
results after small input changes, check independent smooth reference
calculations and the growth column's sensitivity to longwave radiation.
See :doc:`../automatic-differentiation` for choices at zero magnitude, physical
branch thresholds and the special free-drift case with an unbounded sensitivity.

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

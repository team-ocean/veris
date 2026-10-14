Define, run and resume a sea-ice experiment
===========================================

Complete :doc:`/introduction/get-started` first to install Veris. This tutorial
uses Python to run a coupled experiment, which combines ice motion and growth.
It saves output on a schedule and resumes a checkpoint, a complete saved model
that lets a run continue exactly. Run its blocks in order from a fresh working
directory.

The immutable ``Experiment`` describes the inputs and settings; you cannot
change it in place. The host ``Simulation`` uses ordinary Python code to manage
execution and files. This CPU (computer processor) example combines momentum
(equations for ice motion), ice/snow transport (movement between grid cells)
and thermodynamic growth (growth and melting through heat exchange). Its
prescribed ocean and atmospheric fields are supplied values, such as ocean
temperature and wind::

   from datetime import timedelta
   import jax
   import numpy as np

   from veris import Execution, Experiment, Grid, Simulation
   from veris.io import OutputSettings, Stream, read_record

   output = OutputSettings(streams=(
       Stream("instant", ("hIceMean", "Area"), timedelta(seconds=600)),
       Stream("mean", ("hIceMean",), timedelta(seconds=600),
              timedelta(seconds=1200)),
   ))
   experiment = Experiment.builtin(
       "island", grid=Grid(6, 8), steps=4,
       settings={"dtype": "float32", "nEVPsteps": 2},
       execution=Execution(backend="cpu"), output=output,
   )
   with Simulation.initialize(experiment, output_path="history.nc") as simulation:
       simulation.run(2)
       simulation.save_restart("checkpoint.nc")
       result = simulation.run()
       simulation.snapshot("final.nc", variables=("hIceMean", "Area"))

   with Simulation.from_restart(
       "checkpoint.nc", output_path="resumed-history.nc",
   ) as simulation:
       resumed = simulation.run()

   assert resumed.completed_steps == result.completed_steps == 4
   assert resumed.elapsed == result.elapsed == timedelta(seconds=2400)
   for left, right in zip(jax.tree.leaves(result.state),
                          jax.tree.leaves(resumed.state), strict=True):
       np.testing.assert_array_equal(left, right)
   final = read_record("final.nc")
   assert final.fields["hIceMean"].shape == (6, 8)
   print(final.time, float(final.fields["hIceMean"].mean()))

Use new file destinations when running the example again. ``Grid(6, 8)`` counts
the whole model's physical cells in ``(x, y)`` order, with the x direction first.
It excludes halos, extra cells at grid edges used for boundary calculations.
The example's two EVP subcycles (repeated calculations in the ice-motion solver)
demonstrate the workflow; they do not show that the solver has converged to a
solution. Initialization prepares the island's shape and the ocean cells. The
builtin setup reapplies the supplied heat on every step; see
:doc:`/reference/setups/island` for its physics.

The first run stops after two timesteps, or intervals of simulated time; the
second finishes the four-step target. Restoring the intermediate checkpoint
reproduces the same final State, the arrays holding the model's values. For
the exact rules for run counts, the simulation clock and failures, see
:doc:`/api/simulation`.

Observing the trajectory
------------------------

Request an observation when you need a time series in memory. An observation
calculates selected values from each timestep. Keep it a pure JAX calculation:
use JAX, a Python library for array calculations, without changing anything
outside the function::

   import jax.numpy as jnp

   with Simulation.initialize(
       Experiment.builtin("growth", settings={"dtype": "float32"}, steps=3),
   ) as simulation:
       result = simulation.run(observe=lambda state, diagnostics: {
           "ice": jnp.mean(state.hIceMean[2:-2, 2:-2]),
           "freshwater": jnp.mean(diagnostics.EmPmR[2:-2, 2:-2]),
       })
   assert result.observations["ice"].shape == (3,)

The history contains samples taken after each step. Default runs keep only their
final State and Diagnostics, the extra calculated output values. Observations
need more memory as you select more data or run more steps. For physical input
names, units, JSON definitions (experiment definitions stored in text files)
and rules for overriding settings, see :doc:`/api/experiment`. To supply your
own masks (arrays marking land and ocean) and fields (values across the grid),
continue with :doc:`custom-ocean`.

Loading selected fields
-----------------------

A snapshot, which saves selected fields at one time, can set their starting
values without resuming the model clock. Reload the final thickness and
concentration (the fraction of each cell covered by ice) into a new serial
experiment, one without a device mesh::

   from veris import Experiment, Grid, Simulation
   from veris.io import read_record

   with Simulation.initialize(Experiment.builtin(
       "island", grid=Grid(6, 8), steps=0, settings={"dtype": "float32"},
   )) as simulation:
       simulation.load_fields("final.nc", variables=("hIceMean", "Area"))
       assert simulation.completed_steps == 0
       loaded = simulation.state
   assert read_record("final.nc").fields["hIceMean"].shape == (6, 8)

``load_fields`` keeps the new simulation's clock and forcing definition, which
specifies supplied inputs such as wind and ocean temperature. ``from_restart``
restores the saved run so it can continue exactly. Their input checks and
execution limits are documented in :doc:`/api/simulation`.

Next steps
-----------

* :doc:`/reference/io` explains output schedules and the intervals used for averages.
* :doc:`/reference/integration` demonstrates runs through several timesteps and
  how to calculate the effect of input changes on results.
* :doc:`/api/experiment` describes explicit meshes (groups of devices) and the
  input sizes for the whole grid.
* :doc:`/reference/setups/parallel-dynamics` covers runs on one computer and
  across several Python processes.

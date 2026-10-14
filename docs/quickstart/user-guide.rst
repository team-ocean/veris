Define, run and resume a sea-ice experiment
===========================================

Complete :doc:`/introduction/get-started` first to install Veris. This tutorial
uses the Python lifecycle to run a coupled experiment, schedule output and
resume a checkpoint. Run its blocks in order from a fresh working directory.

The immutable ``Experiment`` describes the model; the host ``Simulation`` owns
execution and files. This CPU example couples momentum, ice/snow transport and
thermodynamic growth with prescribed ocean and atmospheric fields::

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
global physical cells in ``(x, y)`` order, excluding halos. The example's two EVP
subcycles demonstrate the lifecycle; they do not establish solver convergence.
Initialization prepares the island and wet-cell geometry. The builtin reapplies
prescribed heat on every step; see :doc:`/reference/setups/island` for its physics.

The first run stops after two timesteps; the second finishes the four-step
target. Restoring the intermediate checkpoint reproduces the same final State.
For precise run-count, clock and failure behavior, see :doc:`/api/simulation`.

Observing the trajectory
------------------------

Request a small pure JAX observation when you need a time series in memory::

   import jax.numpy as jnp

   with Simulation.initialize(
       Experiment.builtin("growth", settings={"dtype": "float32"}, steps=3),
   ) as simulation:
       result = simulation.run(observe=lambda state, diagnostics: {
           "ice": jnp.mean(state.hIceMean[2:-2, 2:-2]),
           "freshwater": jnp.mean(diagnostics.EmPmR[2:-2, 2:-2]),
       })
   assert result.observations["ice"].shape == (3,)

The history contains post-step samples. Default runs retain only their final
State and Diagnostics; observation memory grows with the selected data and run
length. For physical input names, units, JSON definitions and override rules,
see :doc:`/api/experiment`. To supply your own masks and fields, continue with
:doc:`custom-ocean`.

Loading selected fields
-----------------------

A snapshot can initialize selected fields without resuming a model clock.
Reload the final thickness and concentration into a new serial experiment::

   from veris import Experiment, Grid, Simulation
   from veris.io import read_record

   with Simulation.initialize(Experiment.builtin(
       "island", grid=Grid(6, 8), steps=0, settings={"dtype": "float32"},
   )) as simulation:
       simulation.load_fields("final.nc", variables=("hIceMean", "Area"))
       assert simulation.completed_steps == 0
       loaded = simulation.state
   assert read_record("final.nc").fields["hIceMean"].shape == (6, 8)

``load_fields`` retains the new simulation's clock and forcing definition;
``from_restart`` restores an exact continuation. Their validation and backend
limits are documented in :doc:`/api/simulation`.

Next steps
-----------

* :doc:`/reference/io` explains stream calendars and averaging windows.
* :doc:`/reference/integration` demonstrates pure rollouts and differentiation.
* :doc:`/api/experiment` describes explicit meshes and global input dimensions.
* :doc:`/reference/setups/parallel-dynamics` covers local and multi-process launches.

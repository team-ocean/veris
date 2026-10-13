Define, run and resume a sea-ice experiment
===========================================

Install Veris into an activated Python environment::

   python -m pip install -e .

For discovery, editable setup templates and execution from a shell, see
:doc:`/cli`.

Veris is standalone. Its public lifecycle uses an immutable ``Experiment`` and
a host ``Simulation``; it requires no external ocean model. This complete CPU
example couples momentum, ice/snow transport and thermodynamic growth with
prescribed ocean and atmospheric fields::

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
Heat fluxes are in W/m², positive upward. The island preset reapplies its
prescribed heat before growth replaces those fields with ocean-coupling fluxes.
Its central island and wet-cell geometry are prepared during initialization.

``run()`` advances the remaining configured steps; ``run(n)`` advances exactly n
steps, and ``step()`` advances one. Forcing and time retain their absolute
positions across calls. ``RunResult`` returns final ``state``, final
``diagnostics``, per-call ``steps``, absolute ``completed_steps``/``elapsed`` and
execution timings. Histories are optional. Request only values you need::

   import jax.numpy as jnp

   with Simulation.initialize(
       Experiment.builtin("growth", settings={"dtype": "float32"}, steps=3),
   ) as simulation:
       result = simulation.run(observe=lambda state, diagnostics: {
           "ice": jnp.mean(state.hIceMean[2:-2, 2:-2]),
           "freshwater": jnp.mean(diagnostics.EmPmR[2:-2, 2:-2]),
       })
   assert result.observations["ice"].shape == (3,)

The growth preset preserves its recursive heat-flux policy unless heat is
explicitly prescribed. Builtins are ``island``, ``dynamics`` and ``growth``;
custom geometry and forcing use the same lifecycle. See :doc:`custom-ocean`
for an experiment defined entirely through public interfaces.

Definition, precision and files
-------------------------------

Keep numerical ``settings``, physical ``constants``, ``initial_conditions``,
``forcing``, ``execution`` and host ``output`` separate inside the experiment.
Their defaults come from the existing metadata registries. An example JSON
definition is:

.. code-block:: json

   {
     "schema_version": 1,
     "preset": "island",
     "grid": {"nx": 6, "ny": 8},
     "settings": {"dtype": "float32", "nEVPsteps": 2},
     "forcing": {"series": {"Qnet": [80, 100, 120]}},
     "steps": 3
   }

Load it with ``Experiment.from_file("experiment.json", overrides=...)``.
Registry/preset defaults precede file values, which precede Python overrides.
Mappings merge by key; arrays replace whole values. ``experiment.to_dict()``
returns a resolved JSON-compatible definition. Output files save this definition,
numerical configuration and constants. Units, supported names, dimensions and
duration encodings are described in :doc:`/api/experiment`.

The examples explicitly request float32. For float64, enable
``jax.config.update("jax_enable_x64", True)`` before allocating arrays, and
request ``settings={"dtype": "float64"}``. Precision belongs to numerical
configuration and is inherited by physical constants. Importing Veris does not
choose a backend or change JAX settings.

Snapshots contain selected physical fields at an absolute model time.
``load_fields`` validates and applies supported selected initial/forcing fields
to an existing serial simulation; it retains its time and forcing position.
For example, reload the selected final fields into a new serial experiment::

   from veris import Experiment, Grid, Simulation
   from veris.io import read_record

   with Simulation.initialize(Experiment.builtin(
       "island", grid=Grid(6, 8), steps=0, settings={"dtype": "float32"},
   )) as simulation:
       simulation.load_fields("final.nc", variables=("hIceMean", "Area"))
       assert simulation.completed_steps == 0
       loaded = simulation.state
   assert read_record("final.nc").fields["hIceMean"].shape == (6, 8)

For an exact continuation, use ``save_restart``/``from_restart``. Complete
restart supports serial CPU or one GPU device and restores every numerical field,
including halos, plus the definition, forcing cursor and exact model clock.
Distributed restart is unsupported. Restored Diagnostics become available after
the next physics timestep. Resumed output uses a new file, keeps the original
calendar/cadence, skips the checkpoint instant and discards crossing unfinished
mean windows. See :doc:`/api/simulation` for output and restart policies.

Pure execution and parallel resources
-------------------------------------

Use ``advance(state, configuration, constants, forcing, physics=...)`` for one
pure timestep and ``veris.step`` for a pure fixed-length rollout. State and JAX
forcing arrays retain validated JIT, gradient, JVP and VJP support. Experiment
construction, Simulation, clocks, files, meshes and static settings/constants
remain outside differentiated computation. See :doc:`/reference/integration`
and :doc:`/reference/automatic-differentiation` for examples and branch limits.

CPU, GPU and supported sharded execution use the same workflow. Prepare
distributed JAX explicitly before enumerating devices. Supply a mesh whose
``x``/``y`` shape matches ``Execution.partitions``; global Grid extents must be
divisible into local extents of at least two cells. Regional allocation creates
addressable partitions with packed halos. Mesh and communicator ownership stay
with the caller. See :doc:`/reference/setups/parallel-dynamics` and
:doc:`/api/experiment` for explicit resource setup.

Development checks
------------------

Run one pytest instance at a time from the repository root::

   JAX_PLATFORMS=cpu pytest tests/ -q --fast
   JAX_PLATFORMS=cpu pytest tests/ -q

``--fast`` selects a deterministic approximately 10% sample before fixture
setup; ``VERIS_TEST_SEED`` changes that sample. Run the full correctness suite
before commits. Hardware availability limits GPU and multi-host verification.

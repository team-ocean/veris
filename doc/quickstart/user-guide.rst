Basic usage
===========

Install the current checkout into an activated Python environment::

   python -m pip install -e .

Initialize the model and run the artificial island example::

   import jax
   jax.config.update("jax_enable_x64", True)

   from veris.setup.artificial import initialize, step

   state, settings, constants = initialize(
       nx=8, ny=12, wind=5.0,
       settings_overrides={"artificialCooling": 100.0},
   )
   for _ in range(3):
       state = step(state, settings, constants)
   jax.block_until_ready(state)
   print(float(state.hIceMean[2:-2, 2:-2].mean()))

The example uses a periodic Cartesian grid with a central island. Each time step
updates ice velocities and stresses, transports ice and snow, then computes
thermodynamic growth and ocean heat/salt exchange. Cooling is prescribed in
W/m², positive upward. Ocean fields remain prescribed. Omitted ``cooling`` uses
``settings.artificialCooling``; an explicit scalar or JAX scalar array overrides
it and remains differentiable. Each step restores prescribed atmospheric heat
forcing before growth replaces ``Qnet`` and ``Qsw`` with ocean-coupling fluxes.

Grid extents, spacing, wind, temperatures, initial ice/snow conditions and other
experiment controls are recorded in the initialized settings. Explicit
``nx``, ``ny``, ``wind`` and ``air_temperature`` arguments override their registry
defaults. Use ``settings_overrides`` for other experiment controls and
``physical_overrides`` for material parameters. Explicit ``deltatDyn``,
``deltatTherm`` and ``nEVPsteps`` overrides take precedence over the artificial
scenario defaults.

State, settings and physical constants are frozen dataclasses. Use immutable
updates; timestep reciprocals and density ratios are recomputed automatically::

   from dataclasses import replace
   settings = replace(settings, deltatTherm=300, deltatDyn=300)
   constants = replace(constants, rhoIce=920.0)

Output-only coupling diagnostics are available without adding State leaves::

   from veris.setup.artificial import step_with_diagnostics
   state, diagnostics = step_with_diagnostics(state, settings, constants)

For default registry allocation without experiment-specific forcing, use
``veris.initialization.initialize``. It allocates all fields, including masks
and metrics, and accepts separate settings, physical and field overrides.
Float64 metadata requires ``jax_enable_x64`` as enabled above. Array overrides
do not recompute other arrays; experiment initialization must establish
consistent geometry and intermediate fields before running kernels.

For serial registry allocation, pass
``settings_overrides={"use_sharding": False}``. The artificial initializer already
selects this mode and rejects sharded initialization. Grid extents are stored as
``settings.nx`` and ``settings.ny``; replacing those values later does not resize
existing arrays.

Distributed allocation
----------------------

The general initializer accepts an explicit JAX mesh and places every State
array with ``NamedSharding(mesh, PartitionSpec("x", "y"))``::

   from veris.initialization import initialize as allocate
   from veris.fill_overlap import fill_overlap

   mesh = jax.make_mesh((jax.device_count(), 1), ("x", "y"))
   distributed_state, distributed_settings, constants = allocate(
       nx=8, ny=12, mesh=mesh,
       settings_overrides={"use_sharding": True},
   )
   with jax.set_mesh(mesh):
       refreshed_ice = fill_overlap(
           distributed_state.hIceMean, distributed_settings
       )

Here ``nx`` and ``ny`` count interior cells **per partition**. Each partition
stores its own two-cell halos, so global array storage has shape
``(mesh.shape["x"] * (nx + 4), mesh.shape["y"] * (ny + 4))``. State overrides
must already use that packed layout. Mesh resources remain outside State.

After supplying consistent geometry, masks and forcing in that layout, call
``step`` or ``step_with_diagnostics`` inside the same ``jax.set_mesh(mesh)``
context. The driver maps its stencil calculations over local partitions and
exchanges their halos; EVP residual reductions use the mesh axes. The regular
``step`` returns State, while ``step_with_diagnostics`` returns State and the
separate coupling diagnostics. ``compiled_step`` provides an optional compiled
version of the same driver. Sharded execution currently accepts spatially
uniform cooling as a scalar or replicated scalar JAX array.

The default five EVP subcycles demonstrate the integration sequence; they do not
establish a converged dynamics solution. See :doc:`/reference/setups/artificial`
for the example API and :doc:`/reference/settings` for kernel defaults.

Development checks
------------------

From the repository root, run the full correctness suite or a deterministic
collection-time sample::

   pytest tests/ -q
   pytest tests/ -q --fast

``VERIS_TEST_SEED`` changes the stable approximately 10% sample. CPU execution is
the default test target; selecting a GPU requires a compatible JAX installation.

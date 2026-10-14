Halos and parallel execution
============================

Halo exchange
-------------

After replacing State fields manually, use
:func:`~veris.grid.halos.fill_state_overlap` to refresh field-specific
boundaries. The x direction is periodic; y boundaries follow
``Configuration.enable_cyclic_y``. For sharded arrays, retain the active
``jax.set_mesh(mesh)`` context. See :doc:`/reference/settings` for solid-wall
behavior and :doc:`experiment` for packed storage and meshes.

.. automodule:: veris.grid.halos

.. autofunction:: veris.grid.halos.fill_circular_overlap

.. autofunction:: veris.grid.halos.fill_overlap_shard

.. autofunction:: veris.grid.halos.fill_overlap

.. autofunction:: veris.grid.halos.fill_overlap_uv

.. autofunction:: veris.grid.halos.fill_state_overlap

Reductions
----------

.. automodule:: veris.grid.reductions

.. autofunction:: veris.grid.reductions.global_sum

Parallel dynamics helpers
-------------------------

The public runner prepares distributed JAX before accessing devices or allocating
arrays. The parallel driver selects the dynamics preset and parallel profile,
then delegates to :func:`veris.run_experiment`.
The shared :func:`veris.advance` transition owns its physics sequence.
Launch examples appear in :doc:`/reference/setups/parallel-dynamics`.

``veris.grid.halos.remove_halos`` retains JAX arrays and supports transformations.
The host :func:`~veris.io.distributed.distributed_collector` trims packed halos
and collects selected physical fields for the :doc:`io` API. All processes
participate in collective calls in the same order; only the writing process
creates the output file.

.. autofunction:: veris.execution.runtime.distributed_options

.. autofunction:: veris.execution.runtime.create_mesh

.. autofunction:: veris.grid.halos.remove_halos

Execution resources
-------------------

The public Python runner prepares resources before initializing State. It
validates explicit ``DistributedExecution`` controls or adapts JAX/Slurm launch
variables once. Explicit controls take precedence. Services started by the
framework close on success, bootstrap failure or execution failure; caller-owned
services remain active. Supplied meshes remain borrowed.

``Execution()`` requests serial execution. ``all_devices=True`` requests an
all-device mesh discovered at launch; the parallel dynamics profile uses that
policy. Explicit partitions must use all available devices of the chosen
backend. Multiple equally sized device slices use a hybrid mesh rather than an
unsupported ``jax.make_mesh`` topology. The low-level Simulation facade still
expects callers to prepare distributed services and meshes themselves.

An explicitly sized, framework-owned mesh must provide at least one device per
process. Owned CPU launches use exactly one device per rank, so a known rank
count must match the mesh size before bootstrap. Borrowed services may expose
multiple CPU devices per rank and are checked against the actual mesh. Impossible
known rank/mesh requests fail before launch policy or distributed initialization.

.. autoclass:: veris.model.experiment.DistributedExecution

.. autoclass:: veris.execution.runtime.RuntimeResources

.. autofunction:: veris.execution.runtime.execution_resources

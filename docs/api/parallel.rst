Halos and parallel execution
============================

Halo exchange
-------------

Halos are extra cells around each grid piece. They hold values from neighboring
pieces or from boundary rules. After replacing State fields manually, use
:func:`~veris.grid.halos.fill_state_overlap` to refresh each field's boundaries.
The x direction is periodic: its ends join. The y boundaries follow
``Configuration.enable_cyclic_y``. For sharded arrays, which are split across
devices, keep the ``jax.set_mesh(mesh)`` context active. A mesh arranges the
devices that share the grid. See :doc:`/reference/settings` for solid-wall
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

The public runner prepares JAX for a run across processes before accessing
devices or creating arrays. The parallel driver selects the dynamics preset and
parallel profile, then calls :func:`veris.run_experiment`. The shared
:func:`veris.advance` step function controls the order of its physics
calculations. Launch examples appear in
:doc:`/reference/setups/parallel-dynamics`.

``veris.grid.halos.remove_halos`` keeps its results as JAX arrays and supports
JAX transformations. The host
:func:`~veris.io.distributed.distributed_collector` removes halos from packed
storage and gathers selected interior fields for the :doc:`io` API. Every
process must join calls that exchange data between processes in the same order.
Only the writing process creates the output file.

.. autofunction:: veris.execution.runtime.distributed_options

.. autofunction:: veris.execution.runtime.create_mesh

.. autofunction:: veris.grid.halos.remove_halos

Execution resources
-------------------

The public Python runner prepares resources before initializing State. It checks
explicit ``DistributedExecution`` controls or reads JAX/Slurm launch variables
once. Explicit controls take precedence. Veris closes services it starts after
success, a startup failure or an execution failure. Services started by the
caller remain active. The caller also keeps ownership of supplied meshes.

``Execution()`` requests serial execution. ``all_devices=True`` requests a mesh
using every device discovered at launch. The parallel dynamics profile uses
that policy. Explicit partitions must use all available devices of the chosen
backend, the CPU or GPU system that runs JAX. Multiple equally sized device
groups, called slices, use a hybrid mesh that joins them. These layouts are not
supported by ``jax.make_mesh``. The lower-level Simulation interface still
expects callers to prepare distributed services and meshes themselves.

An explicitly sized mesh owned by Veris must provide at least one device per
process. CPU launches owned by Veris use exactly one device per rank, meaning
each process. If the rank count is known, it must match the mesh size before
startup.
Caller-owned services may expose multiple CPU devices per rank; Veris checks
them against the actual mesh. If known rank and mesh counts cannot work together,
the request fails before applying launch policy or starting distributed JAX.

.. autoclass:: veris.model.experiment.DistributedExecution

.. autoclass:: veris.execution.runtime.RuntimeResources

.. autofunction:: veris.execution.runtime.execution_resources

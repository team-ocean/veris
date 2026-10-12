Halos and parallel execution
============================

Halo exchange
-------------

After replacing State fields manually, use
:func:`~veris.grid.halos.fill_state_overlap` to refresh field-specific
boundaries. The x direction is periodic; y boundaries follow
``Configuration.enable_cyclic_y``. For sharded arrays, retain the active
``jax.set_mesh(mesh)`` context. See :doc:`/reference/settings` for solid-wall
behavior and :doc:`/quickstart/user-guide` for packed storage and meshes.

.. automodule:: veris.grid.halos

.. autofunction:: veris.grid.halos.fill_circular_overlap

.. autofunction:: veris.grid.halos.fill_overlap_shard

.. autofunction:: veris.grid.halos.make_sharded_fill_overlap

.. autofunction:: veris.grid.halos.fill_overlap

.. autofunction:: veris.grid.halos.fill_overlap_uv

.. autofunction:: veris.grid.halos.fill_state_overlap

Reductions
----------

.. automodule:: veris.grid.reductions

.. autofunction:: veris.grid.reductions.global_sum

Parallel dynamics helpers
-------------------------

Initialize distributed JAX before accessing devices or allocating arrays.
The parallel runner uses the dynamics preset through Experiment and Simulation.
The shared :func:`veris.advance` transition owns its physics sequence.
Launch examples appear in :doc:`/reference/setups/parallel-dynamics`.

``veris.grid.halos.remove_halos`` retains JAX arrays and supports transformations.
The host :func:`~veris.io.distributed.distributed_collector` trims packed halos
and collects selected physical fields for the :doc:`io` API. All processes
participate in collective calls in the same order; only the writing process
creates the output file.

.. autofunction:: veris.setups.run_parallel.distributed_options

.. autofunction:: veris.setups.run_parallel.create_mesh

.. autofunction:: veris.grid.halos.remove_halos

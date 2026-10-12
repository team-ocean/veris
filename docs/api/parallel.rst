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
The parallel runner uses :func:`veris.setups.run_dyn.initialize` and
:func:`veris.setups.run_dyn.step`; it has no separate physics transition.
Launch examples appear in :doc:`/reference/setups/parallel-dynamics`.

``veris.grid.halos.remove_halos`` retains JAX arrays and supports transformations.
It is also available from ``veris.setups.run_parallel``. The host
``gather_output`` collects the nine reference fields on every process;
``save_output`` writes them on process zero. All processes must participate
in collective calls in the same order. For arbitrary State selections, use
:func:`~veris.io.distributed.distributed_collector` with the :doc:`io` API.

.. autofunction:: veris.setups.run_parallel.distributed_options

.. autofunction:: veris.setups.run_parallel.create_mesh

.. autofunction:: veris.grid.halos.remove_halos

.. autofunction:: veris.setups.run_parallel.gather_output

.. autofunction:: veris.setups.run_parallel.save_output

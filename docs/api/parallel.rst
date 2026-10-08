Halos and parallel execution
============================

Halo exchange
-------------

After replacing State fields manually, use
:func:`~veris.fill_overlap.fill_state_overlap` to refresh field-specific
boundaries. The x direction is periodic; y boundaries follow
``Configuration.enable_cyclic_y``. For sharded arrays, retain the active
``jax.set_mesh(mesh)`` context. See :doc:`/reference/settings` for solid-wall
behavior and :doc:`/quickstart/user-guide` for packed storage and meshes.

.. automodule:: veris.fill_overlap

.. autofunction:: veris.fill_overlap.fill_circular_overlap

.. autofunction:: veris.fill_overlap.fill_overlap_shard

.. autofunction:: veris.fill_overlap.make_sharded_fill_overlap

.. autofunction:: veris.fill_overlap.fill_overlap

.. autofunction:: veris.fill_overlap.fill_overlap_uv

.. autofunction:: veris.fill_overlap.fill_state_overlap

Reductions
----------

.. automodule:: veris.global_sum

.. autofunction:: veris.global_sum.global_sum

Parallel dynamics helpers
-------------------------

Initialize distributed JAX before accessing devices or allocating arrays.
The parallel runner uses :func:`veris.setups.run_dyn.initialize` and
:func:`veris.setups.run_dyn.step`; it has no separate physics transition.
Launch examples appear in :doc:`/reference/setups/parallel-dynamics`.

``remove_halos`` retains JAX arrays and supports transformations. The host
``gather_output`` collects the nine reference fields on every process;
``save_output`` writes them on process zero. All processes must participate
in collective calls in the same order. For arbitrary State selections, use
:func:`~veris.io.distributed.distributed_collector` with the :doc:`io` API.

.. autofunction:: veris.setups.run_parallel.distributed_options

.. autofunction:: veris.setups.run_parallel.create_mesh

.. autofunction:: veris.setups.run_parallel.remove_halos

.. autofunction:: veris.setups.run_parallel.gather_output

.. autofunction:: veris.setups.run_parallel.save_output

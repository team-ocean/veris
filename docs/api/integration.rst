Integration
===========

Differentiable rollouts
-----------------------

The package-level driver repeats a pure single-step transition. For forcing
histories, observation PyTrees, diagnostics, checkpointing and sharded
examples, see :doc:`/reference/integration`.

.. autofunction:: veris.step

Dynamics and transport stage
----------------------------

Custom setup steps can reuse the shared dynamics stage. It returns
``(state, stress_u, stress_v)``; the stresses are evaluated before transport.
The caller supplies atmospheric forcing, optional thermodynamics and the
final State and Diagnostics halo refresh. Under sharding, call this local
stage inside the setup's ``jax.shard_map``.

.. automodule:: veris.dynamics

.. autofunction:: veris.dynamics.dynamics_transport

Host timing and scheduled output
--------------------------------

These helpers compile and time integration and connect it to an
:class:`~veris.io.OutputManager`. Keep them outside JAX transformations; use
:func:`veris.step` directly for differentiation. The scheduling and timing
contracts are described in :doc:`/reference/integration`.

.. autofunction:: veris.integration_output.run_timed

.. autofunction:: veris.integration_output.output_callbacks

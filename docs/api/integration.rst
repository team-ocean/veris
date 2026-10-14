Integration
===========

Differentiable rollouts
-----------------------

The package-level driver repeats a function that updates the model by one step.
The function is pure: it returns results without changing Python objects or
writing files. JAX can differentiate the repeated calculation to measure how
results change with inputs. See :doc:`/reference/integration` for external inputs
that change over time, observations stored as nested collections called PyTrees,
step diagnostics, checkpointing to limit memory use, and examples that split
arrays across devices.

.. autofunction:: veris.step

Dynamics and transport stage
----------------------------

Custom step functions can reuse the shared dynamics stage, which computes ice
motion and transport. It returns ``(state, stress_u, stress_v)``. The stresses,
or forces per unit area, are computed before transport moves ice across cells.
The caller supplies atmospheric inputs, optional freezing and melting, and the
final refresh of State and Diagnostics halos. Halos are extra cells that hold
boundary or neighboring values. When splitting arrays across devices, call this
local stage inside the caller's ``jax.shard_map``.

.. automodule:: veris.physics.dynamics.driver

.. autofunction:: veris.physics.dynamics.driver.dynamics_transport

Host timing and scheduled output
--------------------------------

These helpers compile and time model runs and connect them to an
:class:`~veris.io.OutputManager`. Keep them outside JAX transformations such as
compilation and automatic differentiation. Use :func:`veris.step` directly to
compute derivatives. See :doc:`/reference/integration` for timing and
:ref:`scheduled-output-integration` for scheduling rules.

.. autofunction:: veris.execution.rollout_output.run_timed

.. autofunction:: veris.execution.rollout_output.output_callbacks

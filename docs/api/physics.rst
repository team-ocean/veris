Physics kernels
===============

These functions expose the equations and stencil operations used by setup
steps. Most accept ``(vs, conf, phys)``: a halo-inclusive State, static
Configuration and static PhysicalConstants. They return arrays or tuples;
the caller assigns results with ``dataclasses.replace`` and supplies the
required halo exchange and stage ordering.

Single physics timestep
-----------------------

``veris.advance`` performs one complete timestep and returns State and
Diagnostics. Select coupled, dynamics or thermodynamics; bind configuration,
constants and physics statically for JAX transformations. Supplied forcing is
dynamic and refreshed before the kernels consume it. This differs from
``veris.step``, which performs a multi-step rollout with a supplied transition.
See :doc:`/reference/integration` for composition and differentiation examples.

.. autofunction:: veris.execution.advance.advance

The :doc:`setup steps <initialization>` retain their existing behavior. The
focused :func:`~veris.physics.dynamics.driver.dynamics_transport` stage remains available to
callers composing specialized transitions.

Dynamics
--------

.. automodule:: veris.physics.dynamics.area_mass

.. autofunction:: veris.physics.dynamics.area_mass.AreaWS

.. autofunction:: veris.physics.dynamics.area_mass.SeaIceMass

.. automodule:: veris.numerics.momentum

.. autofunction:: veris.numerics.momentum.tauXY

.. autofunction:: veris.numerics.momentum.WindForcingXY

.. autofunction:: veris.numerics.momentum.IceVelocities

.. automodule:: veris.numerics.evp

.. autofunction:: veris.numerics.evp.evp_solver

.. automodule:: veris.numerics.freedrift

.. autofunction:: veris.numerics.freedrift.freedrift_solver

.. automodule:: veris.physics.dynamics.rheology

.. autofunction:: veris.physics.dynamics.rheology.SeaIceStrength

.. autofunction:: veris.physics.dynamics.rheology.ocean_drag_coeffs

.. autofunction:: veris.physics.dynamics.rheology.basal_drag_coeffs

.. autofunction:: veris.physics.dynamics.rheology.side_drag

.. autofunction:: veris.physics.dynamics.rheology.strainrates

.. autofunction:: veris.physics.dynamics.rheology.viscosities

.. autofunction:: veris.physics.dynamics.rheology.stress

.. autofunction:: veris.physics.dynamics.rheology.stressdiv

.. automodule:: veris.physics.dynamics.ocean_stress

.. autofunction:: veris.physics.dynamics.ocean_stress.OceanStressUV

Transport
---------

.. automodule:: veris.physics.transport.advection

.. autofunction:: veris.physics.transport.advection.Advection

.. autofunction:: veris.physics.transport.advection.calc_Advection

.. autofunction:: veris.physics.transport.advection.calc_ZonalFlux

.. autofunction:: veris.physics.transport.advection.calc_MeridionalFlux

.. autofunction:: veris.physics.transport.advection.limiter

.. automodule:: veris.grid.averaging

.. autofunction:: veris.grid.averaging.c_point_to_z_point

.. automodule:: veris.physics.transport.cleanup

.. autofunction:: veris.physics.transport.cleanup.clean_up_advection

.. autofunction:: veris.physics.transport.cleanup.ridging

Thermodynamics
--------------

.. automodule:: veris.physics.thermodynamics.growth

.. autofunction:: veris.physics.thermodynamics.growth.Growth

.. automodule:: veris.numerics.surface_temperature

.. autofunction:: veris.numerics.surface_temperature.solve4temp

Atmospheric fluxes
------------------

.. automodule:: veris.physics.thermodynamics.heat_flux_cesm

.. autofunction:: veris.physics.thermodynamics.heat_flux_cesm.qsat

.. autofunction:: veris.physics.thermodynamics.heat_flux_cesm.qsat_august_eqn

.. autofunction:: veris.physics.thermodynamics.heat_flux_cesm.get_press_levs

.. autofunction:: veris.physics.thermodynamics.heat_flux_cesm.compute_z_level

.. autofunction:: veris.physics.thermodynamics.heat_flux_cesm.dqnetdt

.. autofunction:: veris.physics.thermodynamics.heat_flux_cesm.net_lw_ocn

.. autofunction:: veris.physics.thermodynamics.heat_flux_cesm.cdn

.. autofunction:: veris.physics.thermodynamics.heat_flux_cesm.psimhu

.. autofunction:: veris.physics.thermodynamics.heat_flux_cesm.psixhu

.. autofunction:: veris.physics.thermodynamics.heat_flux_cesm.flux_atmOcn

.. autofunction:: veris.physics.thermodynamics.heat_flux_cesm.flux_atmOcn_simple

.. automodule:: veris.physics.thermodynamics.heat_flux_mitgcm

.. autofunction:: veris.physics.thermodynamics.heat_flux_mitgcm.bulkf_formula_lanl

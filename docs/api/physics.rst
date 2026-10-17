Physics kernels
===============

These functions implement the equations and nearby-cell calculations used by
``advance``. Most accept ``(vs, conf, phys)``: a State with halos, a fixed
static Configuration and a dynamic PhysicalConstants PyTree. State holds the model arrays. Halos
are extra cells that hold boundary or neighboring values. The functions return
arrays or tuples. The caller assigns results with ``dataclasses.replace``,
exchanges halo values when required and calls the stages in the required order.

Single physics timestep
-----------------------

``veris.advance`` moves the model forward by one complete timestep and returns
State and Diagnostics, which holds step results. Select coupled for ice motion,
transport, freezing and melting; dynamics for motion and transport; or
thermodynamics for freezing and melting. Keep configuration and the physics
choice static when applying JAX transformations; physical constants are dynamic. Supplied forcing, or
external inputs, can change and is refreshed before the numerical functions
use it. ``veris.step`` repeats a supplied step function over multiple timesteps.
See :doc:`/reference/integration` for examples of combining these functions and
computing derivatives, which describe how results change with inputs.

.. autofunction:: veris.execution.advance.advance

The :func:`~veris.physics.dynamics.driver.dynamics_transport` stage is available
to callers building specialized step functions.

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

See :doc:`/reference/ocean-budgets` for mass, salt, enthalpy and status conventions.

.. automodule:: veris.physics.thermodynamics.budgets

.. autoclass:: veris.physics.thermodynamics.budgets.OceanExchangeFluxes

.. autofunction:: veris.physics.thermodynamics.budgets.ocean_exchange_fluxes

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

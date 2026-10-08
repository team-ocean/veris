Physics kernels
===============

These functions expose the equations and stencil operations used by setup
steps. Most accept ``(vs, conf, phys)``: a halo-inclusive State, static
Configuration and static PhysicalConstants. They return arrays or tuples;
the caller assigns results with ``dataclasses.replace`` and supplies the
required halo exchange and stage ordering. For a complete transition, use
the :doc:`setup steps <initialization>` or
:func:`~veris.dynamics.dynamics_transport`.

Dynamics
--------

.. automodule:: veris.area_mass

.. autofunction:: veris.area_mass.AreaWS

.. autofunction:: veris.area_mass.SeaIceMass

.. automodule:: veris.dynsolver

.. autofunction:: veris.dynsolver.tauXY

.. autofunction:: veris.dynsolver.WindForcingXY

.. autofunction:: veris.dynsolver.IceVelocities

.. automodule:: veris.evp_solver

.. autofunction:: veris.evp_solver.evp_solver

.. automodule:: veris.freedrift_solver

.. autofunction:: veris.freedrift_solver.freedrift_solver

.. automodule:: veris.dynamics_routines

.. autofunction:: veris.dynamics_routines.SeaIceStrength

.. autofunction:: veris.dynamics_routines.ocean_drag_coeffs

.. autofunction:: veris.dynamics_routines.basal_drag_coeffs

.. autofunction:: veris.dynamics_routines.side_drag

.. autofunction:: veris.dynamics_routines.strainrates

.. autofunction:: veris.dynamics_routines.viscosities

.. autofunction:: veris.dynamics_routines.stress

.. autofunction:: veris.dynamics_routines.stressdiv

.. automodule:: veris.ocean_stress

.. autofunction:: veris.ocean_stress.OceanStressUV

Transport
---------

.. automodule:: veris.advection

.. autofunction:: veris.advection.Advection

.. autofunction:: veris.advection.calc_Advection

.. autofunction:: veris.advection.calc_ZonalFlux

.. autofunction:: veris.advection.calc_MeridionalFlux

.. autofunction:: veris.advection.limiter

.. automodule:: veris.averaging

.. autofunction:: veris.averaging.c_point_to_z_point

.. automodule:: veris.clean_up

.. autofunction:: veris.clean_up.clean_up_advection

.. autofunction:: veris.clean_up.ridging

Thermodynamics
--------------

.. automodule:: veris.growth

.. autofunction:: veris.growth.Growth

.. automodule:: veris.solve4temp

.. autofunction:: veris.solve4temp.solve4temp

Atmospheric fluxes
------------------

.. automodule:: veris.heat_flux_CESM

.. autofunction:: veris.heat_flux_CESM.qsat

.. autofunction:: veris.heat_flux_CESM.qsat_august_eqn

.. autofunction:: veris.heat_flux_CESM.get_press_levs

.. autofunction:: veris.heat_flux_CESM.compute_z_level

.. autofunction:: veris.heat_flux_CESM.dqnetdt

.. autofunction:: veris.heat_flux_CESM.net_lw_ocn

.. autofunction:: veris.heat_flux_CESM.cdn

.. autofunction:: veris.heat_flux_CESM.psimhu

.. autofunction:: veris.heat_flux_CESM.psixhu

.. autofunction:: veris.heat_flux_CESM.flux_atmOcn

.. autofunction:: veris.heat_flux_CESM.flux_atmOcn_simple

.. automodule:: veris.heat_flux_MITgcm

.. autofunction:: veris.heat_flux_MITgcm.bulkf_formula_lanl

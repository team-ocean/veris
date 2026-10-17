Ocean boundary budgets
======================

``advance`` returns instantaneous grid-cell mean ocean mass and salt inputs in
``Diagnostics.ocean_mass_flux`` and ``ocean_salt_mass_flux``. Both are downward
positive, in kg/m2/s and kg salt/m2/s. Thicknesses are solid volumes per cell
area in metres. Snow is fresh; ice has one fixed, nonnegative bulk salinity
``saltIce_ref`` in g/kg for a trajectory. For thermodynamic changes ``di, ds``,
signed raw-minus-prepared cleanup ``ci, cs`` and timestep ``dt``,

.. math::

   M = B_m - [\rho_i(di-ci) + \rho_s(ds-cs)]/dt,\qquad
   J = -\rho_i(di-ci) S_i/(1000\,dt).

``B_m`` is diagnosed independently from rain, snowfall, runoff, open-water
liquid evaporation and actual frozen-surface sublimation/deposition. Native
``evap`` is downward water volume rate (negative for evaporation); the surface
solver returns upward sublimation mass rate. Sublimation consumes existing
snow first and then ice, capped by available pre-phase frozen stock. Deposition
adds fresh frost. Surface fluxes and geometry are evaluated on the incoming
column, then the diagnosed vapor event precedes the phase update.

``EmPmR=-M`` is total bulk ocean mass loss. Retained ice salinity does not reduce
that mass. The legacy-named ``forc_salt_surface`` now consistently means

.. math::

   F_S = (-M S_o + 1000 J)/\rho_{fw},

in (g/kg)*m/s. This local-salinity virtual flux is separate from actual salt
mass input. A fixed-volume ocean may instead require its own fixed reference
salinity convention. The coupler must choose that convention explicitly.
The `CICE ocean flux interface <https://docs.cesm.ucar.edu/models/cesm2/config/old/cesmBbrowser/html_code/cice/ice_flux.F90.html>`_
also distinguishes salt mass in kg/m2/s from heat in W/m2 and distinguishes
cell means from ice-area means.

Snow enthalpy and cleanup
-------------------------

Open-water snowfall melts immediately and consumes ``L*P`` before phase change.
At freezing, that heat loss forms ice with mass ``P*dt`` and leaves zero net
ocean mass/heat input. Warm open water receives the snow mass and loses its
fusion heat. Snow retained on ice carries enthalpy ``-L*m`` relative to liquid
at freezing. Liquid rain retained as snow releases its fusion heat. Flooding
converts equal snow and ice mass without an artificial latent heat source.

``external_mass_flux`` and ``external_heat_flux`` are independent downward
atmosphere/land inputs to the modeled column, evaluated from boundary events
and surface thermal fluxes before phase change. External heat includes incoming
snow enthalpy and the frozen-mass reference correction for vapor exchange.
With zero-layer solid enthalpy ``H=-L*(rho_i*h_i+rho_s*h_s)``, the residual ocean
heat satisfies ``Q_down + dH/dt = external_heat_flux``. Native ``State.Qnet`` is
upward, so ``Q_down=-Qnet``. ``Qnet`` already includes transmitted shortwave;
``Qsw`` remains its diagnostic shortwave partition.

Cleanup fields retain signed prepared-minus-raw thickness. Negative overshoot
clipping creates solid stock and withdraws the compensating ocean mass; removing
thin positive stock returns it. ``advance`` clears consumed cleanup fields so
thermodynamic continuation cannot apply the same event twice. Transport itself
is not counted as a thermodynamic ocean exchange. Dry operands are neutralized
before arithmetic; dry surface physics is disabled and finite dry surface
temperatures remain pass-through values.

Validity and independent references
-----------------------------------

``physics_status`` is real and grid-shaped: zero valid, one invalid physical
parameters, two new ice formation with ``ocSalt < saltIce_ref``. Gross formation
is tested even when concurrent melting hides it in net thickness change. Salty
ice may melt into fresh water; fresh-water formation uses ``saltIce_ref=0``.
There is no salinity-dependent replacement of the configured bulk ice salinity.
``Simulation`` retains intermediate failures and rejects the candidate run
without advancing its committed state or clock. Pure callers must check status.

The public ``Growth`` tuple retains its eleven-value prefix and appends actual
mass, actual salt mass, external mass and external heat arrays. Original v1
reference archives remain unchanged. ``ocean_budgets_v2`` records declared
correction deltas and new diagnostics from archived a9d6c0a with explicit phase
hooks and independent NumPy boundary accounting. Its generator verifies every
original field first and records source, generator and archive checksums. It
never imports the corrected production solver. These are modeled conservation
checks; they do not establish observational skill or resolve discontinuous
ice-membership derivatives.

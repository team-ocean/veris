# Coefficient inventory for registry migration

Audited 2026-09-09 against maintained `veris/*.py` and
`veris/setup/artificial.py`, before the approved dataclass migration. This is
an implementation inventory, not the generated end-user registry reference.
Defaults below preserve current formulas, including independent rounded
coefficients. `P` means PhysicalConstants; `S` means Settings. Names for newly
extracted fields are proposals. Existing fields retain their names unless an
explicit canonical replacement is stated.

## Classification decisions applied during implementation

The snow thickness ``hCut = 0.15 m`` belongs to PhysicalConstants: it sets the
physical transition between ice and optically opaque snow albedo in solve4temp.
It is not a solver tolerance. Scenario controls, including grid extents and all
artificial setup defaults, are now initialized fields of Settings; initial
array values remain described by VARIABLES and explicit setup overrides.
The geometry adapter retains its independent historical 273 K surface default
through ``Settings.geometrySurfaceTemperature``. Registry allocation's 273.15 K
temperature values describe initial conditions, not a second Celsius conversion
constant.

## Scattered surface thermodynamics and dynamics

| Source and local expression | Exact current value | Proposed field | Class |
| --- | --- | --- | --- |
| `solve4temp.aa1`, duplicated in artificial saturation forcing | 2663.5 | `iceVaporPressureTemperature` | P |
| `solve4temp.aa2`, duplicated in artificial saturation forcing | 12.537 | `iceVaporPressureLog10Offset` | P |
| `solve4temp.bb1`, artificial humidity, CESM August and LW formulas | 0.622 | `waterVaporDryAirMassRatio` | P |
| `solve4temp.Ppascals`, artificial humidity denominator | 100000 | `iceSurfacePressure` | P |
| `solve4temp` penetrating SW exponential | -1.5 times ice thickness | `iceShortwaveExtinction` = 1.5 | P |
| `solve4temp` Newton loop | 6 | `surfaceTemperatureIterations` | S |
| `growth` minimum actual ice thickness | 0.05 m | `minActualIceThickness` | S |
| `growth.tmpscal0` in McPhee taper | 0.4 | `McPheeTaperArea` | P |
| `growth.tmpscal1` numerator | 7 (`7 / tmpscal0`) | `McPheeTaperSteepness` | P |
| `growth.tmpscal0` for lateral concentration loss | 0.5 (`0.5 * recip_hIceActual`) | `lateralMeltAreaFactor` | P |
| `dynamics_routines.basal_drag_coeff.fac` | 10.0 | `basalDragSmoothing` | S |
| `basal_drag_coeff` minimum concentration | 0.01 | `basalDragMinArea` | S |
| `evp_solver` adaptive minimum cell ice mass | 1e-4 | `aEVPmassMin` | S |
| `evp_solver.aEVPcStar` | 4 | `aEVPcStar` | S |
| `evp_solver.evpRevFac` | 1 | `evpStressRelaxation` | S |
| `evp_solver.recip_evpRevFac` | 0.25 | `evpShearRelaxation` | S |

The last two EVP locals are **not reciprocals** despite their old names.
Do not derive 0.25 as `1 / 1`. They are the current independent coefficients
of the normal/shear stress updates; preserving them is required. The 0.25
may reflect the reference elliptical yield shape, but changing it to derive
from configurable `PlasDefCoeff` would change existing nondefault behavior.
Similarly, the lateral melt concentration factor is a parameterization choice,
not a grid interpolation weight, and merits a named physical coefficient.

Compute vapor-pressure `bb2 = 1 - bb1`, `cc0 = 10**aa2`, `cc1`, `cc2`,
`recip_fac`, and taper `7 / 0.4` locally from initialized fields or expose
read-only derived properties. They are not independent defaults.

## CESM bulk helper coefficients

These helpers currently have no settings argument in several cases; they must
receive initialized constants as part of migration. Avoid hidden global default
instances for calls made inside configurable kernels.

| Function/expression | Exact current value | Proposed field | Class |
| --- | --- | --- | --- |
| `qsat` numerator | 640380.0 | `cesmSaturationHumidityScale` | P |
| `qsat` exponential temperature | 5107.4 | `cesmSaturationHumidityTemperature` | P |
| `qsat_august_eqn` log10 offset | 9.4051 | `augustVaporPressureLog10Offset` | P |
| August exponent and `dqnetdt` derivative | 2353.0 | `augustVaporPressureTemperature` | P |
| August pressure conversion | 133.322 | `mmHgToPa` | P |
| `cdn` reciprocal wind term | 0.0027 | `neutralDragInverseWind` | P |
| `cdn` constant term | 0.000142 | `neutralDragConstant` | P |
| `cdn` linear wind term | 0.0000764 | `neutralDragLinearWind` | P |
| `psimhu` rounded angle offset | 1.571 | `cesmUnstableMomentumOffset` | P |
| `net_lw_ocn` humidity pressure scale | 1000.0 | `longwaveHumidityPressureScale` | P |
| `net_lw_ocn` clear sky offset | 0.39 | `longwaveClearSkyOffset` | P |
| `net_lw_ocn` humidity coefficient | 0.05 | `longwaveHumidityCoefficient` | P |
| `flux_atmOcn` surface humidity salinity factor | 0.98 | `seawaterHumidityFactor` | P |
| `flux_atmOcn` unstable heat transfer square root | 0.0327 | `cesmNeutralHeatUnstable` | P |
| `flux_atmOcn` stable heat transfer square root | 0.018 | `cesmNeutralHeatStable` | P |
| `flux_atmOcn` moisture transfer square root | 0.0346 | `cesmNeutralMoisture` | P |
| CESM and LANL unstable stability coefficient | 16.0 | `bulkUnstableStabilityCoefficient` | P |
| CESM and LANL stable stability coefficient | -5.0 | `bulkStableStabilityCoefficient` = 5.0 | P |
| CESM and LANL maximum absolute height/Obukhov length | 10.0 | `bulkStabilityLimit` | S |
| `flux_atmOcn` potential-to-actual T correction | 0.01 K/m | existing `gamma_blk` = 0.010 | P |

Use the `waterVaporDryAirMassRatio` above for CESM's 0.622; derive its
complement 0.378 as `1 - ratio` (this may differ by the final floating-point
rounding bit from the literal, so verify preserved numerical tolerances).
Do not replace the empirically rounded 1.571 with exact pi/2. LANL intentionally
uses exact `pi * 0.5`; those two parameterizations currently differ.

`flux_atmOcn` contains an initial stability evaluation and one repeated update.
A `cesmBulkIterations = 2` setting can represent those evaluations if the
repeated blocks are converted to a loop with an independently checked numerical
oracle. They currently are not an explicit convergence loop or tolerance.

## CESM cloud interpolation tables

Store these as immutable tuple defaults on PhysicalConstants, converting to JAX
arrays in the consuming kernel. A global device array allocated during import
is not the requested initialization-owned constants object. Require equal
lengths and strictly increasing latitude knots.

- `longwaveCloudLatitudes` (degrees), current `_clat`:
  `(-90.0, -80.0, -70.0, -60.0, -50.0, -40.0, -30.0, -20.0, -10.0, -5.0, 0.0, 5.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0)`.
- `longwaveCloudCoefficients` (dimensionless), current `_cc`:
  `(0.88, 0.84, 0.80, 0.76, 0.72, 0.68, 0.63, 0.59, 0.52, 0.50, 0.50, 0.50, 0.52, 0.59, 0.63, 0.68, 0.72, 0.76, 0.80, 0.84, 0.88)`.

## MITgcm LANL bulk formula

| Local expression | Exact current value | Proposed field | Class |
| --- | --- | --- | --- |
| `ht` | 2.0 m | existing `ztref` (or separate `lanlTemperatureHeight`) | S |
| `zref` | 10.0 m | existing `zref` | S |
| `zice` | 0.0005 m | existing `zzsice` | P |
| `usm` minimum wind | 1.0 m/s | `lanlMinWindSpeed` | S |
| `ssq` humidity scale | 3.797915 | `lanlSaturationHumidityScale` | P |
| `ssq` constant in latent heat exponent | 7.93252e-6 | `lanlSaturationExponentOffset` | P |
| `ssq` temperature factor and `devdt` derivative | 2.166847e-3 | `lanlSaturationExponentTemperature` | P |
| `ssq` reference pressure denominator | 1013.0 | `lanlReferencePressure` | P |
| stability iterations | 5 | `lanlBulkIterations` | S |
| neutral drag polynomial | 2.7e-3, 0.142e-3, 0.0764e-3 | same three neutral drag constants as CESM | P |

The wind floor happens to equal the currently unused ice forcing `umin_i`, but
LANL is an **open ocean** bulk routine. Do not assign it an ice-specific field
merely because its default is equal. The existing CESM `umin_o = 0.5` must
remain different from this LANL minimum 1.0.

The inactive-land replacement inputs (winds 1.0, air T 275.0, humidity 0.003,
surface T 280.0) are AD-safe evaluation sentinels, not physical model inputs:
all associated outputs and sensitivities are identically masked away. Keep
these locally documented numerical sentinels. They must not become allocated
State fields or imply a configurable land model.

## Gravity and radius

`gravity = 9.81 m/s²` already exists in the default dictionary. CESM heights
and turbulent fluxes, plus LANL fluxes, instead request nonexistent `grav`.
Canonicalize these reads to `PhysicalConstants.gravity`; do not add a second
independent `grav` default. `tests/test_heat_flux_CESM.py:32` and
`tests/test_heat_flux_MITgcm.py:20` explicitly supply `grav=sett.gravity`.
The CESM fixture supplies `radius=6371000.0 m`; the typing contract uses the
same radius. Introduce `PhysicalConstants.radius = 6371000.0` and describe it
as the spherical Earth radius used in the geometric altitude conversion.
These defaults follow the repository's current test/reference choices.

## Setup and structural values

Artificial example defaults (nx 8, ny 12, wind 5 m/s, air temperature 260 K,
timesteps 600 s, EVP steps 5, spacing 8000 m, ice thickness 1 m, snow 0.05 m,
concentration 0.8, bottom -100 m, Coriolis 1e-4 s⁻¹, salinity 34.7 g/kg,
cooling 100 W/m²) are experiment configuration or initial array values. Route
experiment controls through settings overrides and initial array values through
VARIABLES/default initialization plus explicit setup overrides. Reuse the ice
vapor law constants instead of reproducing 12.537, 2663.5, 0.622, 100000 and
0.378 in the example. `set_inits.py`'s TSurf 273 K is an initialization default,
not a Celsius conversion constant; explicitly document any change to its
historical default during migration.

Keep algebraic exponents (Stefan-Boltzmann power 4 and derivative factor 4),
0/1 masks and area bounds, category midpoint factors `2*(l+1)-1`, fixed
superbee limiter coefficients, finite-difference weights 0.5/0.25, Taylor
boundary weights 2/3/6, quadratic solution factors 4 and 0.5, axis indices,
array tuple positions, two-cell stencil halo widths, and exact mathematical
pi/log(10) in equations. These define the discretization or mathematics, not
physical/configurable scalar parameters. A generalized halo-width setting
would falsely promise unsupported arbitrary stencils.

## Unused existing defaults

An AST attribute-read inventory excluding declarations/protocols and the
settings dictionary identifies:

`explicitDrag`, `recip_rhoFresh`, `lhEvap`, `saltOcn_ref`, `maxTIce`, `h0`,
`h0_south`, `umin_i`, `bolzc`, `avogad`, `rgas`, `mwdair`, `mwwv`, `rwv`,
`cpwv`, `p0`, `cappa`, `zzsice`, `snow_emissivity`, `ice_emissivity`, `tf0kel`,
`ocean_albedo`, `ice_albedo`.

This is a read-use distinction, not permission to drop scientifically useful
public constants. `h0`/`h0_south` become necessary constructor inputs for
currently used reciprocals. `lhEvap` becomes a dependency of `lhSublim`.
`zzsice` can replace the local LANL value. `saltOcn_ref` can supply the example
salinity default. `explicitDrag` is advertised as a switch but is not checked
by the current EVP implementation. `maxTIce` is not the surface solver cap:
that solver caps at melting temperature. Do not silently activate either
previously unused setting and thereby alter the baseline algorithm.

The forcing constants `rgas`, `rdair`, `rwv`, `zvir`, `cpvir`, and `cappa`
are independently rounded historical defaults; despite comments giving
approximate relationships, do not recompute them from the molecular weights.
For example `rgas=8314.47` differs from `avogad*bolzc`, and `zvir=0.608`
differs from the exact ratio of supplied rounded gas constants.

## Complete existing default classification

The following exact source expressions cover every existing dictionary key.
New scattered fields above supplement this table. Derived fields should be
computed at construction rather than accepted as independently stale overrides.

| Existing field | Exact current default expression | Class |
| --- | --- | --- |
| `deltatTherm` | `86400` | S |
| `recip_deltatTherm` | `1 / 86400` | S (derived) |
| `deltatDyn` | `86400` | S |
| `recip_deltatDyn` | `1 / 86400` | S (derived) |
| `nITC` | `5` | S |
| `recip_nITC` | `1 / 5` | S (derived) |
| `noSlip` | `True` | S |
| `useRelativeWind` | `True` | S |
| `secondOrderBC` | `False` | S |
| `extensiveFld` | `True` | S |
| `useRealFreshWaterFlux` | `False` | S |
| `useFreedrift` | `False` | S |
| `useEVP` | `True` | S |
| `evpAlpha` | `500` | S |
| `evpBeta` | `500` | S |
| `useAdaptiveEVP` | `False` | S |
| `aEVPalphaMin` | `5` | S |
| `aEvpCoeff` | `0.5` | S |
| `explicitDrag` | `True` | S |
| `nEVPsteps` | `400` | S |
| `computeEvpResidual` | `False` | S |
| `use_coastline` | `False` | S |
| `use_sharding` | `True` | S |
| `rhoIce` | `900` | P |
| `rhoFresh` | `1000` | P |
| `rhoSea` | `1026` | P |
| `rhoAir` | `1.3` | P |
| `rhoSnow` | `330` | P |
| `recip_rhoFresh` | `1 / 1000` | P (derived) |
| `recip_rhoSea` | `1 / 1026` | P (derived) |
| `rhoIce2rhoSnow` | `900 / 330` | P (derived) |
| `rhoIce2rhoFresh` | `900 / 1000` | P (derived) |
| `rhoFresh2rhoSnow` | `1000 / 330` | P (derived) |
| `dryIceAlb` | `0.75` | P |
| `dryIceAlb_south` | `0.75` | P |
| `wetIceAlb` | `0.66` | P |
| `wetIceAlb_south` | `0.66` | P |
| `drySnowAlb` | `0.84` | P |
| `drySnowAlb_south` | `0.84` | P |
| `wetSnowAlb` | `0.7` | P |
| `wetSnowAlb_south` | `0.7` | P |
| `wetAlbTemp` | `0` | P |
| `lhFusion` | `3.34e5` | P |
| `lhEvap` | `2.5e6` | P |
| `lhSublim` | `3.34e5 + 2.5e6` | P (derived) |
| `cpAir` | `1005` | P |
| `cpWater` | `3986` | P |
| `stefBoltz` | `5.67e-8` | P |
| `iceEmiss` | `0.95` | P |
| `snowEmiss` | `0.95` | P |
| `iceConduct` | `2.1656` | P |
| `snowConduct` | `0.31` | P |
| `hCut` | `0.15` | P |
| `shortwave` | `0.3` | P |
| `tempFrz` | `-1.96` | P |
| `dtempFrz_dS` | `0` | P |
| `saltIce_ref` | `0` | P |
| `saltOcn_ref` | `34.7` | P |
| `minLWdown` | `60` | S |
| `maxTIce` | `30` | S |
| `minTIce` | `-50` | S |
| `minTAir` | `-50` | S |
| `dalton` | `0.00175` | P |
| `Area_reg` | `0.15**2` | S |
| `hIce_reg` | `0.10**2` | S |
| `celsius2K` | `273.15` | P |
| `stantonNr` | `0.0056` | P |
| `uStarBase` | `0.0125` | P |
| `McPheeTaperFac` | `12.5` | P |
| `h0` | `0.5` | P |
| `recip_h0` | `1 / 0.5` | P (derived) |
| `h0_south` | `0.5` | P |
| `recip_h0_south` | `1 / 0.5` | P (derived) |
| `airTurnAngle` | `0` | P |
| `waterTurnAngle` | `0` | P |
| `sinWat` | `0` | P (derived) |
| `cosWat` | `1` | P (derived) |
| `wSpeedMin` | `1e-10` | S |
| `hIce_min` | `1e-5` | S |
| `Area_min` | `1e-5` | S |
| `airIceDrag` | `0.0012` | P |
| `airIceDrag_south` | `0.0012` | P |
| `waterIceDrag` | `0.0055` | P |
| `waterIceDrag_south` | `0.0055` | P |
| `cDragMin` | `0.25` | S |
| `seaIceLoadFac` | `1` | S |
| `gravity` | `9.81` | P |
| `PlasDefCoeff` | `2` | P |
| `deltaMin` | `2e-9` | S |
| `pressReplFac` | `1` | S |
| `pStar` | `27.5e3` | P |
| `cStar` | `20` | P |
| `basalDragU0` | `5e-5` | P |
| `basalDragK1` | `8` | P |
| `basalDragK2` | `0` | P |
| `cBasalStar` | `20` | P |
| `tensileStrFac` | `0` | P |
| `CrMax` | `1e6` | S |
| `sideDragCoeff` | `0.001` | P |
| `sideDragU0` | `0.01` | P |
| `umin_o` | `0.5` | S |
| `umin_i` | `1.0` | S |
| `zref` | `10.0` | S |
| `ztref` | `2.0` | S |
| `bolzc` | `1.38065e-23` | P |
| `avogad` | `6.02214e26` | P |
| `rgas` | `8314.47` | P |
| `mwdair` | `28.966` | P |
| `mwwv` | `18.016` | P |
| `rdair` | `287.042` | P |
| `rwv` | `461.505` | P |
| `zvir` | `0.608` | P |
| `cpdair` | `1.00464e3` | P |
| `cpwv` | `1.810e3` | P |
| `cpvir` | `0.802` | P |
| `karman` | `0.4` | P |
| `latvap` | `2.501e6` | P |
| `p0` | `1e5` | P |
| `cappa` | `0.286` | P |
| `zzsice` | `0.0005` | P |
| `ch` | `1e-3` | P |
| `ce` | `1.15e-3` | P |
| `eps2` | `1e-20` | S |
| `emissivity` | `1` | P |
| `ocean_emissivity` | `0.985` | P |
| `snow_emissivity` | `0.98` | P |
| `ice_emissivity` | `0.98` | P |
| `tf0kel` | `273.15` | P |
| `gamma_blk` | `0.010` | P |
| `ocean_albedo` | `0.1` | P |
| `ice_albedo` | `0.7` | P |

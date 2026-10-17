Automatic differentiation at model boundaries
=============================================

Automatic differentiation (AD) calculates how model results change when inputs
change. Veris supports JAX's forward and reverse methods through its physics
calculations, frozen State (a collection of arrays whose fields cannot be
reassigned) and compiled timestep drivers. Fixes at difficult mathematical
points preserve the reference equations and their derivatives wherever those
equations vary smoothly. Tests cover zero strain (no ice deformation), calm
winds, absent ice, dry cells (land), freshwater salinity, weak free-drift forcing
(forcing for ice motion without internal stress), both model precisions and
evolving dynamics split across devices.

Use ``advance`` for one physics timestep and ``step`` to repeat a fixed number
of timesteps. These functions are pure: their results depend on their inputs,
and they do not change outside resources. Numerical Configuration and
physics mode are fixed during compilation. Independent PhysicalConstants
values are dynamic PyTree leaves; dtype is static, and reciprocals, ratios and
turning-angle factors are recomputed from base values. The initial State,
physical parameters and supplied JAX forcing arrays support tested JIT (compilation), grad
(gradients), JVP (forward changes along an input direction) and VJP (reverse
changes from an output). Keep Experiment validation, initialization, Simulation,
clocks, files and resource management outside these JAX operations. Arrays in
experiment definitions are copied on the host, the CPU running the Python
program. Build the quantity you want to differentiate from dynamic JAX arrays
instead of differentiating an Experiment constructor. See :doc:`integration`.

Physical parameter continuations
--------------------------------

Use ``PhysicalConstants.with_overrides`` inside differentiated functions. It
accepts independent registry values, rejects derived/numerical/dtype keys and
keeps scalar shapes and the selected precision. Concrete values receive host
domain validation. Host Experiment/JSON construction remains device-free::

   import jax
   import jax.numpy as jnp
   from veris.model.physical_constants import PhysicalConstants

   jax.config.update("jax_enable_x64", True)
   base = PhysicalConstants(h0=0.5)
   reciprocal = lambda h0: base.with_overrides(h0=h0).recip_h0
   sensitivity = jax.grad(reciprocal)(jnp.asarray(0.5))  # -4

For model sensitivities, pass that candidate to ``advance`` and reduce the
returned State to a scalar objective. JIT keeps only ``configuration`` and
``physics`` static, not ``constants``. Same-schema parameter values reuse the
compiled transition. ``base_parameter_names()`` identifies the independent
values; dependent laws are not separate fitting parameters.

A continuation holds the supplied initial State and geometry fixed. Rebuild
parameter-dependent initial conditions explicitly when those should contribute
to a derivative. Hard cleanup/membership switches retain their selected local
rules and one-sided behavior; no membership smoothing or straight-through
estimator is introduced. Nonzero h0, drag, conductivity and illuminated dry-ice
albedo sensitivities are checked with JVP/VJP and central-difference sweeps.

Compiled ``advance`` returns a real, same-dtype ``physics_status`` grid in
Diagnostics: zero means valid, one identifies invalid physical parameters.
Invalid parameters skip the physics transition. Pure callers must check this
status. Simulation keeps failures sticky across its fixed-length rollout,
skips subsequent physics and rejects a failed candidate before updating host
State or model time. Initial status is zero. Geometry validation also rejects
coordinates that collapse to a pole or lose ordering in the selected precision,
before device selection.

Zero norms
----------

Viscosity, the ice's resistance to deformation, depends on a measure of strain,
``sqrt(deltaSq)``. A zero velocity field gives zero strain. Differentiating the
square root without a safeguard can produce an infinite intermediate
derivative and then NaN (not a number), even when the model result is finite.
Adding ``deltaMin`` after the square root keeps the viscosity denominator away
from zero but leaves the square-root derivative infinite.

``veris.numerics.autodiff.norm_sqrt`` safeguards the input before calculating
its square root. A norm measures magnitude, such as speed. The function keeps
the same model result, including exactly zero for a zero input. For positive
squared norms, it uses the usual square-root derivative. At zero, it chooses
zero as the local linear response. The same choice applies to speed norms and
unused helper values for adaptive relaxation or stability calculations.
Negative squared norms still produce NaN so invalid inputs remain visible.

A norm has no unique ordinary derivative at zero. The chosen zero response
agrees with differences taken equally on both sides along any fixed direction.
A difference taken on only one side can give a different response. This choice
preserves the linear response of products such as viscosity times shear strain
(deformation that changes shape), and tests check that response directly. The
model equations have no added positive smoothing constant.

Inactive branches
-----------------

JAX calculates the alternatives in an array condition before selecting one
with ``where``. An unused ``1/0`` or ``sqrt(0)`` can therefore spoil reverse AD.
The physics calculations supply safe inputs to these operations even in unused
alternatives, also called inactive branches. Covered cases include:

* Slope ratios for advection (movement between cells) in a constant field, and
  averaging weights at dry grid corners.
* Heat conductivity when ice is absent, and unused Newton solver updates.
  A Newton solver repeatedly adjusts an estimate to solve an equation.
* Salt-flux expressions in freshwater, and safeguards for zero ice area.
* Calm wind, and atmospheric-stability expressions limited by a cap.

The CESM atmospheric heat-exchange routines replace dry-cell inputs with safe
values before doing arithmetic. Land inputs can therefore use zero or NaN placeholders.
Fluxes (transfers such as heat) and diagnostic outputs are zero on land.
Wet cells and fractional masks keep their existing weights, including the
historical powers of the mask. A mask identifies the ocean fraction of a cell.
Wet cells still require valid temperatures, pressures and heights.

Free drift
----------

Free drift balances external forces without internal ice stress. The solver
expresses the original momentum balance along the grid's perpendicular
Cartesian axes, the x/y directions. It rewrites the quadratic solution to avoid subtracting
nearly equal values under
weak forcing and avoids an undefined direction angle at zero forcing. The
Coriolis frequency describes the effect of Earth's rotation. When mass times
this frequency is nonzero, the response at zero forcing has a finite, nonzero
derivative. Tests verify it with an independent formula and finite differences
(small input changes).

When both net forcing and mass times Coriolis are zero, quadratic drag
(resistance proportional to speed squared) gives a speed proportional to the
square root of forcing. The ordinary sensitivity to forcing is unbounded at
that point. The implementation chooses a finite zero
tangent, or local linear response, exactly there. Tests show that the
one-sided difference quotient (the response change divided by the forcing
change) grows without bound. Sensitivity studies near this special physical
case must account for this behavior. The chosen tangent
does not approximate the unbounded one-sided response.

Thresholds and validation
-------------------------

Clipping values to bounds, ridging (ice piling up), transport limiter switches
and thin-ice removal keep their existing physical thresholds. At a continuous
kink, where the slope changes abruptly, JAX uses the local linear response of
the selected branch. At a jump in the result, a branch derivative does not
describe an input change that crosses the jump. Tests check these choices and
the different responses on each side.

Check a sensitivity with finite differences that stay on the same smooth
branch. Use the initialized precision for State, Configuration and
PhysicalConstants. At an exact kink, check the stated derivative choice and
the one-sided behavior separately. A finite gradient alone does not prove
that the derivative is correct.

Float32 surface-temperature derivatives
---------------------------------------

The Newton solver repeatedly adjusts surface temperature to balance heat
flows. It calculates the humidity slope directly from saturation vapor
pressure, the vapor pressure at which air is saturated. The equivalent formula
using inverse vapor pressure produced large intermediate values. Their
forward-mode derivatives overflowed float32 even when the slope itself was
finite. The rewritten formula keeps the same humidity equation, masks and
temperature limits. Tests compare forward and reverse derivatives with finite
differences and an independent derivative of the heat balance between
conduction and the atmosphere.

Differentiating complete rollouts
---------------------------------

Use the public ``veris.step`` to repeat any setup's physics calculation a fixed
number of times. Checkpointing is enabled by default: it recalculates
intermediate step values during reverse AD to save memory, while keeping the
same physics sequence. State and forcing indexed by time remain
differentiable. See :doc:`integration` for a cooling-gradient example,
diagnostic histories, device-mesh contexts and checkpoint memory limits.

The observer runs inside the differentiated timestep loop. A PyTree is a nested
collection of arrays. Select a small array PyTree when the quantity you want to
differentiate needs intermediate states. Omit the observer when that quantity
uses only the final state. Keep OutputManager and the host timing runner
outside JAX transformations.

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
and they do not change outside resources. Configuration, PhysicalConstants and
physics mode are fixed during compilation and differentiation. The initial
State and supplied JAX forcing arrays support tested JIT (compilation), grad
(gradients), JVP (forward changes along an input direction) and VJP (reverse
changes from an output). Keep Experiment validation, initialization, Simulation,
clocks, files and resource management outside these JAX operations. Arrays in
experiment definitions are copied on the host, the CPU running the Python
program. Build the quantity you want to differentiate from dynamic JAX arrays
instead of differentiating an Experiment constructor. See :doc:`integration`.

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

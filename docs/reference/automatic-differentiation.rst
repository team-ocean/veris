Automatic differentiation at model boundaries
=============================================

Veris supports JAX forward and reverse differentiation through its physics
kernels, frozen State and compiled integration drivers. The AD boundary repairs
preserve reference forward equations and their derivatives on smooth branches.
Tests cover zero strain, calm winds, absent ice, dry cells, freshwater salinity,
weak free-drift forcing, both model precisions and evolving sharded dynamics.

Zero norms
----------

Viscosity depends on the strain invariant ``sqrt(deltaSq)``. A zero velocity
field gives zero strain; differentiating an unguarded square root can introduce
an infinite intermediate derivative and a subsequent NaN, even with a finite
forward solution. Adding ``deltaMin`` after the square root regularizes the
viscosity denominator but leaves that derivative singular.

``veris._ad.norm_sqrt`` guards the square-root input before evaluating it. Its
forward result is unchanged, including exactly zero at the origin. At positive
squared norms it has the ordinary square-root derivative. At zero it selects a
zero linearization. The same convention applies to speed norms and inactive
adaptive-relaxation or stability auxiliaries. Negative squared norms still
produce NaN so invalid inputs remain visible.

A norm has no unique classical derivative at its origin. The zero convention
agrees with symmetric differences of the norm along any fixed direction;
its forward one-sided directional derivative can differ. It preserves the
linear response of products such as viscosity times shear strain, and tests
check that response explicitly. No positive smoothing constant has been added
to the forward equations.

Inactive branches
-----------------

JAX evaluates array branches before selecting with ``where``. An inactive
``1/0`` or ``sqrt(0)`` can contaminate reverse AD. The kernels now supply valid
inputs to singular operations inside inactive branches. Covered cases include:

* Constant-field advection slope ratios and dry-corner averaging weights.
* Absent-ice thermal conductivity and inactive Newton updates.
* Freshwater salt-flux expressions and zero-area regularization.
* Calm wind and capped atmospheric-stability expressions.

CESM atmospheric routines sanitize dry-cell inputs before arithmetic, supporting
zero or NaN placeholders on land. Dry fluxes and diagnostics are zero. Existing
wet-cell and fractional-mask weighting is retained, including its historical
powers of the mask. Valid temperatures, pressures and heights are still
required on wet cells.

Free drift
----------

The solver now expresses the original momentum balance in Cartesian form.
Rationalizing its quadratic solution avoids cancellation for weak forcing,
and eliminates the undefined polar angle at zero forcing. With nonzero
mass times Coriolis frequency, the zero-forcing response has a finite,
nonzero derivative; analytic and finite-difference tests verify it.

When both net forcing and mass times Coriolis are zero, quadratic drag gives a
speed proportional to the square root of forcing. Its classical forcing
sensitivity is unbounded at that point. The implementation selects a finite
zero tangent exactly there, and the tests explicitly demonstrate the divergent
one-sided difference quotient. Sensitivity studies near this particular
physical degeneracy must account for that behavior; the selected tangent
does not approximate its unbounded one-sided response.

Thresholds and validation
-------------------------

Clipping, ridging, limiter switches and thin-ice removal retain their existing
physical thresholds. At continuous kinks, JAX follows selected branch
linearizations; at a discontinuous transition, a local branch derivative
does not describe a perturbation crossing that transition. Existing tests
check these conventions and the distinct one-sided responses.

Validate a sensitivity with finite differences on the same smooth branch,
using the initialized precision for State, Configuration and PhysicalConstants.
At an exact kink, compare against the stated convention and one-sided behavior
separately. A finite gradient alone is insufficient evidence of correctness.

Float32 surface-temperature derivatives
---------------------------------------

The Newton surface-temperature solver evaluates the humidity slope directly
from saturation vapor pressure. The equivalent inverse-vapor-pressure form
produced large intermediates whose forward-mode derivatives overflowed float32,
even when the slope itself was finite. The reformulation retains the same
humidity equation, masks and temperature limits. Solver tests compare forward
and reverse derivatives with finite differences and an independent derivative
of the conductive/atmospheric heat balance.

Differentiating complete rollouts
---------------------------------

Use the public ``veris.step`` for a fixed-length scan through any setup kernel.
Checkpointing is enabled by default; it recomputes step intermediates during
reverse AD without changing the physics sequence. State and time-indexed forcing
remain differentiable. See :doc:`integration` for a cooling-gradient example,
auxiliary diagnostic histories, mesh contexts and checkpoint memory limits.

The observer is part of the differentiated scan body. Select a small array
PyTree when a loss needs intermediate states; omit it for a final-state loss.
Keep OutputManager and the host timing runner outside JAX transformations.

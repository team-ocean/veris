"""Surface-temperature Newton tangents stay finite at both model precisions.

The inverse-vapor-pressure form formerly overflowed intermediate float32
JVPs despite a finite forward humidity derivative. Test the real solver with
snow and nonzero latent heat exchange, independently against centered FD.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from veris.setups.run_growth import initialize
from veris.solve4temp import solve4temp


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_surface_temperature_thickness_jvp_matches_finite_difference(
    dtype: str,
) -> None:
    """Thickness sensitivity propagates through all Newton humidity derivatives."""
    state, conf, phys = initialize(dtype=dtype)
    temperature = jnp.full_like(state.TSurf, 260.0)
    snow = jnp.full_like(state.hSnowMean, 0.1)
    freezing = jnp.full_like(state.theta, phys.celsius2K + phys.tempFrz)

    @jax.jit
    def objective(thickness: jax.Array) -> jax.Array:
        result = solve4temp(
            state,
            conf,
            phys,
            jnp.full_like(state.hIceMean, thickness),
            snow,
            temperature,
            freezing,
        )
        return jnp.mean(result[0])

    point = jnp.asarray(1.4, dtype=dtype)
    value, tangent = jax.jvp(objective, (point,), (jnp.ones_like(point),))
    gradient = jax.grad(objective)(point)
    assert np.isfinite(value), "ERROR nonfinite surface temperature"
    assert np.isfinite(tangent), "ERROR nonfinite surface-temperature thickness JVP"
    assert abs(float(tangent)) > 1e-3, "ERROR missing thickness sensitivity"
    epsilon = 0.01 if dtype == "float32" else 1e-4
    fd = (objective(point + epsilon) - objective(point - epsilon)) / (2 * epsilon)
    # Differentiate the converged conductive/atmospheric balance independently
    # in NumPy float64: k*(Tfreeze-T) = radiation + sensible + latent flux.
    t = float(value)
    pressure = 10 ** (
        phys.iceVaporPressureLog10Offset - phys.iceVaporPressureTemperature / t
    )
    vapor_slope = pressure * phys.iceVaporPressureTemperature * np.log(10) / t**2
    humidity_slope = (
        phys.waterVaporDryAirMassRatio
        * phys.iceSurfacePressure
        * vapor_slope
        / (phys.iceSurfacePressure - (1 - phys.waterVaporDryAirMassRatio) * pressure)
        ** 2
    )
    conductance = 1 / (float(point) / phys.iceConduct + 0.1 / phys.snowConduct)
    wind = max(float(state.wSpeed[0, 0]), phys.wSpeedMin)
    atmospheric_slope = (
        4 * phys.snowEmiss * phys.stefBoltz * t** 3
        + phys.dalton
        * phys.rhoAir
        * wind
        * (phys.cpAir + phys.lhSublim * humidity_slope)
    )
    analytic = (
        -(conductance**2)
        / phys.iceConduct
        * (float(freezing[0, 0]) - t)
        / (conductance + atmospheric_slope)
    )
    np.testing.assert_allclose(tangent, analytic, rtol=3e-5, atol=1e-6)
    np.testing.assert_allclose(tangent, gradient, rtol=3e-5, atol=1e-6)
    np.testing.assert_allclose(
        tangent,
        fd,
        rtol=0.01 if dtype == "float32" else 1e-6,
        atol=1e-4 if dtype == "float32" else 1e-7,
    )

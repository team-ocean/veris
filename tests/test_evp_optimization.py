"""Pre-optimization EVP value and AD oracles on a nonuniform rectangular grid.

The fixture records the original jax-only equations before insertion of the CPU
fusion barrier. Masked values exercise a coastline; derivatives use open ocean
with nonzero velocity and strain to avoid the model's zero-norm singularities.
Run this file directly only on the provenance source revision to regenerate.
"""

from dataclasses import replace
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from numpy.typing import NDArray

from veris.configuration import Settings
from veris.physical_constants import PhysicalConstants
from veris.setup.artificial import initialize
from veris.state import State

REFERENCE = Path(__file__).parent / "reference_data" / "evp_pre_barrier.npz"
CASES = [
    (False, False, 1),
    (False, True, 4),
    (True, False, 4),
    (True, True, 4),
    (False, False, 400),
    (True, False, 400),
]


def oracle_state(masked: bool) -> tuple[State, Settings, PhysicalConstants]:
    """Construct periodic 6x9 interiors with variable ice, stress and forcing."""
    vs, sett, phys = initialize(6, 9)
    x, y = np.meshgrid(np.arange(6), np.arange(9), indexing="ij")

    def field(values: np.ndarray) -> jax.Array:
        return jnp.asarray(np.pad(values, 2, mode="wrap"), dtype=jnp.float64)

    a = field(np.sin(2 * np.pi * x / 6) + 0.3 * np.cos(2 * np.pi * y / 9))
    b = field(np.cos(2 * np.pi * (x / 6 + y / 9)))
    interior = np.ones((6, 9))
    if masked:
        interior[2, 4] = 0
    mask = field(interior)
    west = mask * jnp.roll(mask, 1, 0)
    south = mask * jnp.roll(mask, 1, 1)
    thickness = (1.1 + 0.12 * a) * mask
    area = (0.86 + 0.025 * b) * mask
    vs = replace(
        vs,
        maskInC=mask,
        iceMask=mask,
        maskInU=west,
        iceMaskU=west,
        maskInV=south,
        iceMaskV=south,
        hIceMean=thickness,
        Area=area,
        hSnowMean=0.06 * mask,
        uIce=(0.09 + 0.012 * a) * west,
        vIce=(0.06 + 0.013 * b) * south,
        uOcean=-0.03 + 0.003 * b,
        vOcean=-0.02 + 0.004 * a,
        sigma1=(-700 + 35 * a) * mask,
        sigma2=(80 + 12 * b) * mask,
        sigma12=(25 + 3 * a) * mask,
        WindForcingX=(0.08 + 0.006 * b) * west,
        WindForcingY=(-0.04 + 0.007 * a) * south,
        R_low=-8 - a,
        fCori=1e-4 * jnp.where(b > 0, 1, -1),
    )
    from veris.area_mass import AreaWS, SeaIceMass
    from veris.dynamics_routines import SeaIceStrength

    mass_c, mass_u, mass_v = SeaIceMass(vs, sett, phys)
    area_w, area_s = AreaWS(vs, sett, phys)
    return (
        replace(
            vs,
            SeaIceMassC=mass_c,
            SeaIceMassU=mass_u,
            SeaIceMassV=mass_v,
            AreaW=area_w,
            AreaS=area_s,
            SeaIceStrength=SeaIceStrength(vs, sett, phys),
        ),
        sett,
        phys,
    )


def evaluate(
    case: tuple[bool, bool, int], masked: bool
) -> dict[str, NDArray[np.float64]]:
    """Return scaled field outputs and derivatives with respect to two forcings."""
    vs, sett, phys = oracle_state(masked)
    from veris.evp_solver import evp_solver

    adaptive, no_slip, steps = case
    sett = replace(sett, useAdaptiveEVP=adaptive, noSlip=no_slip, nEVPsteps=steps)

    def solve(parameters: jax.Array) -> jax.Array:
        state = replace(
            vs,
            WindForcingX=parameters[0] * vs.WindForcingX,
            SeaIceStrength=parameters[1] * vs.SeaIceStrength,
        )
        result = evp_solver(state, sett, phys)
        # Scale stresses so a single tolerance resolves velocity and stress alike.
        return jnp.stack(result) / jnp.array([1, 1, 1000, 1000, 1000])[:, None, None]

    parameters = jnp.array([1.07, 0.93], dtype=jnp.float64)
    if masked:
        return {"value": np.asarray(solve(parameters))}
    value, tangent = jax.jvp(solve, (parameters,), (jnp.array([0.7, -0.2]),))
    _, pullback = jax.vjp(solve, parameters)
    weights = jnp.linspace(-0.5, 0.7, value.size).reshape(value.shape)
    epsilon = 1e-4
    direction = jnp.array([0.7, -0.2])
    finite_jvp = (
        solve(parameters + epsilon * direction)
        - solve(parameters - epsilon * direction)
    ) / (2 * epsilon)
    finite_vjp = jnp.stack(
        [
            jnp.sum(
                weights
                * (
                    solve(parameters + epsilon * axis)
                    - solve(parameters - epsilon * axis)
                )
            )
            / (2 * epsilon)
            for axis in jnp.eye(2)
        ]
    )
    np.testing.assert_allclose(
        tangent,
        finite_jvp,
        rtol=3e-6,
        atol=2e-10,
        err_msg="ERROR EVP JVP finite-difference mismatch",
    )
    np.testing.assert_allclose(
        pullback(weights)[0],
        finite_vjp,
        rtol=3e-6,
        atol=2e-10,
        err_msg="ERROR EVP VJP finite-difference mismatch",
    )
    return {
        "value": np.asarray(value),
        "jvp": np.asarray(tangent),
        "vjp": np.asarray(pullback(weights)[0]),
    }


@pytest.mark.parametrize("case", CASES)
@pytest.mark.parametrize("masked", [False, True], ids=["smooth_ad", "coastline"])
def test_evp_matches_pre_optimization_values_and_derivatives(
    case: tuple[bool, bool, int], masked: bool
) -> None:
    """Keep every velocity/stress cell and both AD directions at baseline values."""
    index = CASES.index(case)
    with np.load(REFERENCE) as reference:
        for quantity, actual in evaluate(case, masked).items():
            expected = reference[f"case{index}_masked{int(masked)}_{quantity}"]
            assert np.all(np.isfinite(actual)), f"ERROR nonfinite EVP {quantity}"
            np.testing.assert_allclose(
                actual,
                expected,
                rtol=2e-10,
                atol=2e-12,
                err_msg=f"ERROR EVP pre-optimization {quantity} mismatch",
            )


if __name__ == "__main__":
    raise SystemExit(
        "Use the immutable pre-dataclass generator described in "
        "tests/reference_data/README.md; current tests use the migrated API."
    )

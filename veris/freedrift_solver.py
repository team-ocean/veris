"""Solve the local ice momentum balance without internal stress."""

from __future__ import annotations

from functools import partial

import jax.numpy as jnp
from jax import Array

from veris._ad import norm_sqrt
from veris._typing import State, jit
from veris.configuration import Configuration
from veris.physical_constants import PhysicalConstants


@partial(jit, static_argnames=["conf", "phys"])
def freedrift_solver(
    vs: State, conf: Configuration, phys: PhysicalConstants
) -> tuple[Array, Array]:
    """calculate ice velocities without taking into account internal ice stress"""

    # air-ice stress at c-point
    tauXIceCenter = 0.5 * (vs.WindForcingX + jnp.roll(vs.WindForcingX, -1, 0))
    tauYIceCenter = 0.5 * (vs.WindForcingY + jnp.roll(vs.WindForcingY, -1, 1))

    # mass of ice per unit area times coriolis factor
    mIceCor = phys.rhoIce * vs.hIceMean * vs.fCori

    # ocean surface velocity at c-points
    uOceanCenter = 0.5 * (vs.uOcean + jnp.roll(vs.uOcean, -1, 0))
    vOceanCenter = 0.5 * (vs.vOcean + jnp.roll(vs.vOcean, -1, 1))

    # right hand side of the free drift equation
    rhsX = -tauXIceCenter - mIceCor * vOceanCenter
    rhsY = -tauYIceCenter + mIceCor * uOceanCenter

    # For y = ocean - ice, the Cartesian balance is
    # (drag * |y| I + mIceCor J) y = rhs, with J(x, y) = (-y, x).
    # Rationalizing the positive quadratic root avoids cancellation at weak
    # forcing. Cartesian inversion retains the finite Coriolis response at
    # rhs == 0, where the polar angle representation is singular.
    drag = phys.rhoSea * jnp.where(
        vs.fCori < 0, phys.waterIceDrag_south, phys.waterIceDrag
    )
    rhs_squared = rhsX**2 + rhsY**2
    coriolis_squared = mIceCor**2
    root_denominator = coriolis_squared + norm_sqrt(
        coriolis_squared**2 + 4 * drag**2 * rhs_squared
    )
    safe_root_denominator = jnp.where(root_denominator == 0, 1, root_denominator)
    relative_speed = norm_sqrt(2 * rhs_squared / safe_root_denominator)
    drag_speed = drag * relative_speed
    denominator = drag_speed**2 + coriolis_squared
    safe_denominator = jnp.where(denominator == 0, 1, denominator)

    # Simultaneous zero rhs and zero mass-Coriolis is a genuine square-root
    # response to forcing: no finite classical forcing derivative exists.
    # The guarded zero solution selects a zero forcing tangent at that point;
    # a differentiable physical response there would require drag regularization.
    uIceCenter = uOceanCenter - (drag_speed * rhsX + mIceCor * rhsY) / safe_denominator
    vIceCenter = vOceanCenter - (drag_speed * rhsY - mIceCor * rhsX) / safe_denominator

    # interpolate to velocity points
    uIceFD = 0.5 * (jnp.roll(uIceCenter, 1, 0) + uIceCenter)
    vIceFD = 0.5 * (jnp.roll(vIceCenter, 1, 1) + vIceCenter)

    # apply masks
    uIceFD = uIceFD * vs.iceMaskU
    vIceFD = vIceFD * vs.iceMaskV

    return uIceFD, vIceFD

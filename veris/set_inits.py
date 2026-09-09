"""Initialize immutable sea-ice fields from supplied surface ocean geometry.

This host adapter preserves the original Veris ``set_inits`` metric equations.
Ocean volume masks have shape (x, y, z), with the surface at the last level;
spacing vectors and horizontal arrays include the same halos as State. The
periodic corner area is the mean of four neighboring tracer-cell areas. Ocean
geometry is external initialization input, never an AD leaf in the ice State.
"""

from dataclasses import dataclass, fields, replace

import jax.numpy as npx
import numpy as np

from veris._typing import ArrayInput
from veris.configuration import Settings
from veris.physical_constants import PhysicalConstants
from veris.state import State


@dataclass(frozen=True)
class Geometry:
    """Read-only ocean grid inputs, independent of any ocean-model container.

    ``maskT``, ``maskU`` and ``maskV`` are volume masks; ``dxt`` and ``dxu``
    are x-spacing vectors, and ``dyt`` and ``dyu`` are y-spacing vectors (m).
    ``ht`` is depth (m), ``coriolis_t`` is Coriolis frequency (s-1), and the
    three horizontal cell-area arrays are in m2. Shapes and finite positive
    metrics are checked by the host adapter before any reciprocal is computed.
    """

    maskT: ArrayInput
    maskU: ArrayInput
    maskV: ArrayInput
    ht: ArrayInput
    coriolis_t: ArrayInput
    dxt: ArrayInput
    dxu: ArrayInput
    dyt: ArrayInput
    dyu: ArrayInput
    area_t: ArrayInput
    area_u: ArrayInput
    area_v: ArrayInput


def _validate_geometry(geometry: Geometry, shape: tuple[int, ...]) -> None:
    """Reject mismatched grids and invalid reciprocal inputs on the host."""
    if len(shape) != 2:
        raise ValueError("State.hIceMean must have a two-dimensional storage shape")
    for field in fields(geometry):
        name = field.name
        array = np.asarray(getattr(geometry, name))
        if name.startswith("mask"):
            if array.ndim != 3 or array.shape[:2] != shape or array.shape[2] < 1:
                raise ValueError(f"{name} must have shape {shape} + (nonempty z,)")
        else:
            expected = (
                (shape[0],)
                if name in ("dxt", "dxu")
                else (shape[1],)
                if name in ("dyt", "dyu")
                else shape
            )
            if array.shape != expected:
                raise ValueError(f"{name} has shape {array.shape}; expected {expected}")
        if not np.isfinite(array).all():
            raise ValueError(f"{name} must contain only finite values")
        if name in (
            "dxt",
            "dxu",
            "dyt",
            "dyu",
            "area_t",
            "area_u",
            "area_v",
        ) and np.any(array <= 0):
            raise ValueError(f"{name} must contain positive values")


def set_inits(
    state: State, geometry: Geometry, sett: Settings, phys: PhysicalConstants
) -> State:
    """Return State with surface masks and staggered metrics initialized.

    Input State and Geometry are unchanged; non-geometry fields retain their
    initialized values. ``sett.geometrySurfaceTemperature`` preserves the
    original setup temperature of 273 K. ``phys`` is supplied consistently with
    other setup adapters; these geometric equations need no physical constants.
    This host routine validates inputs and is not a compiled time-step kernel.
    """
    _validate_geometry(geometry, state.hIceMean.shape)
    dtype = state.hIceMean.dtype
    ice_mask = npx.asarray(geometry.maskT[:, :, -1], dtype=dtype)
    ice_mask_u = npx.asarray(geometry.maskU[:, :, -1], dtype=dtype)
    ice_mask_v = npx.asarray(geometry.maskV[:, :, -1], dtype=dtype)
    ones = npx.ones_like(ice_mask)
    dx_c = ones * npx.asarray(geometry.dxt, dtype=dtype)[:, npx.newaxis]
    dy_c = ones * npx.asarray(geometry.dyt, dtype=dtype)
    dx_u = ones * npx.asarray(geometry.dxu, dtype=dtype)[:, npx.newaxis]
    dy_u = ones * npx.asarray(geometry.dyu, dtype=dtype)
    dx_g = 0.5 * (dx_u + npx.roll(dx_u, 1, 1))
    dy_g = 0.5 * (dy_u + npx.roll(dy_u, 1, 0))
    dx_v = 0.5 * (dx_c + npx.roll(dx_c, 1, 1))
    dy_v = 0.5 * (dy_c + npx.roll(dy_c, 1, 0))
    area = npx.asarray(geometry.area_t, dtype=dtype)
    area_z = area + npx.roll(area, 1, 0)
    area_z = 0.25 * (area_z + npx.roll(area_z, 1, 1))
    return replace(
        state,
        iceMask=ice_mask,
        iceMaskU=ice_mask_u,
        iceMaskV=ice_mask_v,
        maskInC=ice_mask,
        maskInU=ice_mask_u,
        maskInV=ice_mask_v,
        R_low=npx.asarray(geometry.ht, dtype=dtype),
        fCori=npx.asarray(geometry.coriolis_t, dtype=dtype),
        dxG=dx_g,
        dyG=dy_g,
        dxU=dx_u,
        dyU=dy_u,
        dxV=dx_v,
        dyV=dy_v,
        recip_dxC=1 / dx_c,
        recip_dyC=1 / dy_c,
        recip_dxU=1 / dx_u,
        recip_dyU=1 / dy_u,
        recip_dxV=1 / dx_v,
        recip_dyV=1 / dy_v,
        rAz=area_z,
        recip_rA=1 / area,
        recip_rAu=1 / npx.asarray(geometry.area_u, dtype=dtype),
        recip_rAv=1 / npx.asarray(geometry.area_v, dtype=dtype),
        TSurf=ones * sett.geometrySurfaceTemperature,
    )

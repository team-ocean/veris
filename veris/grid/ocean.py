"""Initialize immutable sea-ice fields from supplied surface ocean geometry.

This host adapter preserves the original Veris ``set_inits`` metric equations.
Ocean volume masks have shape (x, y, z), with the surface at the last level;
spacing vectors and horizontal arrays include the same halos as State. The
periodic corner area is the mean of four neighboring tracer-cell areas. Ocean
geometry is external initialization input, never an AD leaf in the ice State.
"""

from collections.abc import Mapping
from dataclasses import fields
from typing import Any

import jax.numpy as npx
import numpy as np

from veris.execution.initialization import initialize
from veris.grid.geometry import OceanGeometry
from veris.model.configuration import Configuration
from veris.model.forcing import FORCING_FIELDS, _freeze_fields, _precision
from veris.model.physical_constants import PhysicalConstants
from veris.model.types import State
from veris.model.variables import VARIABLES


def _validate_geometry(geometry: OceanGeometry, shape: tuple[int, ...]) -> None:
    """Reject mismatched grids and invalid reciprocal inputs on the host."""
    if len(shape) != 2:
        raise ValueError("maskT must have a two-dimensional horizontal storage shape")
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
        if name.startswith("mask") and np.any((array < 0) | (array > 1)):
            raise ValueError(f"{name} masks must lie between zero and one")
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


def initialize_from_ocean(
    geometry: OceanGeometry,
    *,
    dtype: str | None = None,
    settings_overrides: Mapping[str, Any] | None = None,
    physical_overrides: Mapping[str, Any] | None = None,
    state_overrides: Mapping[str, Any] | None = None,
) -> tuple[State, Configuration, PhysicalConstants]:
    """Allocate a fresh Veris state and static objects from ocean geometry.

    Geometry includes two halo cells at each boundary. Its horizontal shape
    determines nx and ny, taking precedence over settings overrides. Additional
    state overrides supply ocean forcing and initial ice fields; geometry-derived
    fields take precedence. Unspecified fields use VARIABLES defaults. The shared
    initializer validates overrides and allocates every field at the selected
    precision. No pre-existing State is required or updated.
    """
    shape = np.shape(geometry.maskT)[:2]
    _validate_geometry(geometry, shape)
    options = dict(settings_overrides or {})
    options.update(nx=shape[0] - 4, ny=shape[1] - 4)
    if dtype is not None:
        options["dtype"] = dtype
    conf = Configuration(**options)
    dtype = conf.dtype
    overrides = dict(state_overrides or {})
    unknown = overrides.keys() - VARIABLES.keys()
    if unknown:
        raise ValueError(f"unknown State fields: {', '.join(sorted(unknown))}")
    allowed = FORCING_FIELDS | {"hIceMean", "hSnowMean", "Area", "TSurf"}
    _freeze_fields({k: v for k, v in overrides.items() if k in allowed}, allowed)
    for name, value in overrides.items():
        _precision(name, value, dtype)
    metrics = {"dxt", "dxu", "dyt", "dyu", "area_t", "area_u", "area_v"}
    for item in fields(geometry):
        _precision(
            item.name,
            getattr(geometry, item.name),
            dtype,
            reciprocal=item.name in metrics,
        )
    with np.errstate(over="ignore", invalid="ignore"):
        area = np.asarray(geometry.area_t, dtype=dtype)
        west_sum = area + np.roll(area, 1, axis=0)
        if not np.isfinite(west_sum + np.roll(west_sum, 1, axis=1)).all():
            raise ValueError(f"area_t corner sums are not representable in {dtype}")
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
    overrides.update(
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
        TSurf=ones * conf.geometrySurfaceTemperature,
    )
    return initialize(
        settings_overrides=options,
        physical_overrides=physical_overrides,
        state_overrides=overrides,
    )


__all__ = ["initialize_from_ocean"]

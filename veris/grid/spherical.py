"""Rectilinear spherical geometry from physical degree coordinates.

Ocean lengths use equatorial zonal metres and physical meridional metres.
Latitude centers/edges exclude halos. Native areas remain authoritative: no
spherical-integral correction is applied to a supplied midpoint quadrature.
"""

from typing import Any

import numpy as np
from jax import Array
from numpy.typing import NDArray

from veris.grid.geometry import OceanGeometry


def validate_spherical_geometry(geometry: OceanGeometry, ny: int) -> None:
    """Reject incomplete, nonmonotone or singular host geometry before allocation."""
    values = (geometry.latitude, geometry.latitude_edges, geometry.radius)
    if all(value is None for value in values):
        return
    if any(value is None for value in values):
        raise ValueError(
            "latitude, latitude_edges and radius must be supplied together"
        )
    lat, edges, radius = (np.asarray(value) for value in values)
    if lat.shape != (ny,) or edges.shape != (ny + 1,):
        raise ValueError(
            f"latitude/latitude_edges must have shapes {(ny,)} and {(ny + 1,)}"
        )
    if radius.shape != () or radius.dtype.kind not in "fiu":
        raise ValueError("radius must be a real scalar")
    if not all(
        a.dtype.kind in "fiu" and np.isfinite(a).all() for a in (lat, edges, radius)
    ):
        raise ValueError("spherical coordinates and radius must be finite real values")
    if radius <= 0:
        raise ValueError("radius must be positive")
    if np.any(np.diff(lat) <= 0) or np.any(np.diff(edges) <= 0):
        raise ValueError("spherical latitudes must be strictly increasing")
    if (
        np.any(np.abs(edges) >= 90)
        or np.any(lat <= edges[:-1])
        or np.any(lat >= edges[1:])
    ):
        raise ValueError("latitude centers must lie inside nonsingular latitude edges")


def _host_spherical_metrics(
    geometry: OceanGeometry, *, dtype: str
) -> dict[str, NDArray[Any]]:
    """Return full C-grid metric arrays at their actual tracer/face positions.

    Supplied latitude vectors contain physical cells only. Two stored halos,
    when present, extend tracer positions and retain the actual northern face.
    ``k2AtZ`` is the centered divided difference of log(cos(latitude)); the
    resulting shear annihilates solid-body rotation exactly on this grid.
    """
    latitude = np.asarray(geometry.latitude, dtype=dtype)
    edges = np.asarray(geometry.latitude_edges, dtype=dtype)
    shape = np.shape(geometry.area_t)
    halo = shape[1] - latitude.size
    if halo not in (0, 4):
        raise ValueError("spherical metrics require physical or two-halo storage")
    centers = np.deg2rad(latitude)
    faces = np.deg2rad(edges[:-1])
    if halo:
        centers = np.pad(centers, 2, mode="edge")
        faces = np.pad(np.deg2rad(edges), (2, 1), mode="edge")
    cos_c, cos_f = np.cos(centers), np.cos(faces)
    previous = np.r_[cos_c[0], cos_c[:-1]]
    dy_c = np.broadcast_to(np.asarray(geometry.dyu, dtype=dtype), shape)
    dy_v = np.broadcast_to(np.asarray(geometry.dyt, dtype=dtype), shape)
    dx_c = np.asarray(geometry.dxu, dtype=dtype)[:, None] * cos_c
    dx_u = np.asarray(geometry.dxt, dtype=dtype)[:, None] * cos_c
    dx_v = np.asarray(geometry.dxu, dtype=dtype)[:, None] * cos_f
    dx_g = np.asarray(geometry.dxt, dtype=dtype)[:, None] * cos_f
    k2_c = np.broadcast_to(
        -np.tan(centers) / np.asarray(geometry.radius, dtype=dtype), shape
    )
    k2_z = 2 * (cos_c - previous) / (cos_c + previous) / dy_c
    result = {
        "dxG": dx_g,
        "dyG": dy_v,
        "dxU": dx_u,
        "dyU": dy_c,
        "dxV": dx_v,
        "dyV": dy_v,
        "recip_dxC": 1 / dx_c,
        "recip_dyC": 1 / dy_c,
        "recip_dxU": 1 / dx_u,
        "recip_dyU": 1 / dy_c,
        "recip_dxV": 1 / dx_v,
        "recip_dyV": 1 / dy_v,
        "k1AtC": np.zeros(shape, dtype=dtype),
        "k1AtZ": np.zeros(shape, dtype=dtype),
        "k2AtC": k2_c,
        "k2AtZ": k2_z,
    }
    return result


def spherical_metric_overrides(
    geometry: OceanGeometry, *, dtype: str
) -> dict[str, Array]:
    """Place the spherical C-grid metrics of one supplied geometry on device."""
    import jax.numpy as jnp

    return {
        name: jnp.asarray(value, dtype=dtype)
        for name, value in _host_spherical_metrics(geometry, dtype=dtype).items()
    }


def regional_spherical_metrics(
    geometry: OceanGeometry,
    x: NDArray[np.integer[Any]],
    y: NDArray[np.integer[Any]],
    *,
    dtype: str,
) -> dict[str, Array]:
    """Slice authoritative global metrics without rebuilding irregular local halos.

    Definitions already own global host geometry; no global numerical State is
    constructed. The callbacks transfer only their selected metric regions.
    """
    from dataclasses import fields

    import jax.numpy as jnp

    values = {}
    for item in fields(geometry):
        value = getattr(geometry, item.name)
        if item.name in ("latitude", "latitude_edges", "radius"):
            values[item.name] = value
        else:
            array = np.asarray(value)
            padding = (
                ((2, 2),)
                if array.ndim == 1
                else ((2, 2), (0, 0)) + ((0, 0),) * (array.ndim - 2)
            )
            # Longitude is periodic; latitude halos extend external geometry.
            array = (
                np.pad(array, padding, mode="wrap")
                if item.name in ("dxt", "dxu") or array.ndim > 1
                else array
            )
            if array.ndim > 1:
                array = np.pad(
                    array, ((0, 0), (2, 2)) + ((0, 0),) * (array.ndim - 2), mode="edge"
                )
            elif item.name in ("dyt", "dyu"):
                array = np.pad(array, (2, 2), mode="edge")
            values[item.name] = array
    padded = OceanGeometry(**values)
    yy = y.copy()
    if yy[2] == 0:
        yy[:2] = [-2, -1]
    if yy[-3] == len(np.asarray(geometry.latitude)) - 1:
        yy[-2:] = [
            len(np.asarray(geometry.latitude)),
            len(np.asarray(geometry.latitude)) + 1,
        ]
    metrics = _host_spherical_metrics(padded, dtype=dtype)
    return {
        name: jnp.asarray(np.asarray(value)[np.ix_(x + 2, yy + 2)])
        for name, value in metrics.items()
    }

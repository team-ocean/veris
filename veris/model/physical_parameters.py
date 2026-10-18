"""Physical-domain validation and dependent laws shared by host and JAX trees.

The native PHYSICALCONSTANTS registry remains the sole parameter schema. Host
validation uses NumPy scalars without device discovery; compiled validation
returns a scalar boolean for advance's explicit physics status.
"""

from __future__ import annotations

import math
from itertools import pairwise
from typing import TYPE_CHECKING, Any

import jax
import jax.numpy as jnp
import numpy as np
from jax.core import Tracer

from veris.model.metadata import validate_derived, validate_scalars

if TYPE_CHECKING:
    from veris.model.physical_constants import PhysicalConstants

_POSITIVE_PARAMETERS = frozenset(
    (
        "evpStressRelaxation",
        "evpShearRelaxation",
        "Area_min",
        "basalDragMinArea",
        "basalDragSmoothing",
        "bulkStabilityLimit",
        "deltaMin",
        "hIce_min",
        "hIce_reg",
        "lanlMinWindSpeed",
        "minActualIceThickness",
        "umin_i",
        "umin_o",
        "wSpeedMin",
        "zref",
        "ztref",
        "hCut",
        "rhoIce",
        "rhoFresh",
        "rhoSea",
        "rhoAir",
        "rhoSnow",
        "lhFusion",
        "lhEvap",
        "cpAir",
        "cpWater",
        "stefBoltz",
        "iceConduct",
        "snowConduct",
        "h0",
        "h0_south",
        "gravity",
        "PlasDefCoeff",
        "basalDragU0",
        "basalDragK1",
        "sideDragU0",
        "bolzc",
        "avogad",
        "rgas",
        "mwdair",
        "mwwv",
        "rdair",
        "rwv",
        "zvir",
        "cpdair",
        "cpwv",
        "karman",
        "latvap",
        "p0",
        "zzsice",
        "radius",
        "iceSurfacePressure",
        "McPheeTaperArea",
        "lanlReferencePressure",
    )
)


def _validate_host_constants(instance: PhysicalConstants) -> None:
    """Validate authored scalars and retain the original rounded host defaults."""
    from veris.model.physical_constants import PHYSICALCONSTANTS

    validate_scalars(
        instance,
        PHYSICALCONSTANTS,
        positive=_POSITIVE_PARAMETERS,
    )
    if instance.saltIce_ref < 0:
        raise ValueError("saltIce_ref must be nonnegative")
    if instance.Area_reg < 0:
        raise ValueError("Area_reg must be nonnegative")
    if instance.minTIce > instance.maxTIce:
        raise ValueError("minTIce must not exceed maxTIce")
    _update_dependencies(instance, host=True)
    if instance.tensileStrFac <= -1:
        raise ValueError("tensileStrFac must exceed -1")
    validate_derived(
        instance,
        (
            "recip_rhoFresh",
            "recip_rhoSea",
            "rhoIce2rhoSnow",
            "rhoIce2rhoFresh",
            "rhoFresh2rhoSnow",
            "lhSublim",
            "recip_h0",
            "recip_h0_south",
            "sinWat",
            "cosWat",
        ),
    )
    if len(instance.longwaveCloudLatitudes) < 2:
        raise ValueError("longwaveCloudLatitudes must contain at least two knots")
    if len(instance.longwaveCloudLatitudes) != len(instance.longwaveCloudCoefficients):
        raise ValueError(
            "longwaveCloudCoefficients must match longwaveCloudLatitudes length"
        )
    if any(
        right <= left
        for left, right in zip(
            instance.longwaveCloudLatitudes[:-1],
            instance.longwaveCloudLatitudes[1:],
            strict=True,
        )
    ):
        raise ValueError("longwaveCloudLatitudes must be strictly increasing")


def _require_runtime_precision(dtype: str) -> None:
    """Reject unavailable precision before JAX can silently truncate parameters."""
    if jax.dtypes.canonicalize_dtype(dtype) != np.dtype(dtype):
        raise ValueError(
            f"dtype {dtype} requires jax_enable_x64 before tracing parameters"
        )


def _update_dependencies(instance: PhysicalConstants, *, host: bool = False) -> None:
    """Recompute one set of laws with host NumPy or runtime JAX arithmetic.

    NumPy tree maps/device_get must stay on the host. Placeholder and gradient
    trees do not receive host physical-domain validation; their base leaves are
    the only independent data, even when a derived reciprocal is undefined.
    """
    values = tuple(getattr(instance, name) for name in instance.base_parameter_names())
    runtime = not host and any(
        isinstance(value, (jax.Array, Tracer)) for value in jax.tree.leaves(values)
    )
    if runtime:
        _require_runtime_precision(instance.dtype)
    xp = jnp if runtime else np

    def value(name: str) -> Any:
        parameter = getattr(instance, name)
        return float(parameter) if host else parameter

    angle = value("waterTurnAngle")
    sine = math.sin(math.radians(angle)) if host else xp.sin(xp.deg2rad(angle))
    cosine = math.cos(math.radians(angle)) if host else xp.cos(xp.deg2rad(angle))
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        dependent = {
            "recip_rhoFresh": xp.divide(1.0, value("rhoFresh")),
            "recip_rhoSea": xp.divide(1.0, value("rhoSea")),
            "rhoIce2rhoSnow": xp.divide(value("rhoIce"), value("rhoSnow")),
            "rhoIce2rhoFresh": xp.divide(value("rhoIce"), value("rhoFresh")),
            "rhoFresh2rhoSnow": xp.divide(value("rhoFresh"), value("rhoSnow")),
            "lhSublim": value("lhFusion") + value("lhEvap"),
            "recip_h0": xp.divide(1.0, value("h0")),
            "recip_h0_south": xp.divide(1.0, value("h0_south")),
            "sinWat": sine,
            "cosWat": cosine,
        }
    for name, parameter in dependent.items():
        object.__setattr__(instance, name, parameter)


@jax.jit
def _physical_parameters_valid(instance: PhysicalConstants) -> jax.Array:
    """Return domain validity, including finite dependent values, without raising."""
    from veris.model.physical_constants import PHYSICALCONSTANTS

    valid = jnp.asarray(True)
    for name in PHYSICALCONSTANTS:
        value = jnp.asarray(getattr(instance, name))
        valid = valid & jnp.all(jnp.isfinite(value))
        if name in _POSITIVE_PARAMETERS:
            valid = valid & jnp.all(value > 0)
    valid = valid & (instance.Area_reg >= 0) & (instance.minTIce <= instance.maxTIce)
    valid = valid & (instance.tensileStrFac > -1) & (instance.saltIce_ref >= 0)
    latitudes = instance.longwaveCloudLatitudes
    if len(latitudes) < 2 or len(latitudes) != len(instance.longwaveCloudCoefficients):
        return jnp.asarray(False)
    for left, right in pairwise(latitudes):
        valid = valid & (right > left)
    return valid

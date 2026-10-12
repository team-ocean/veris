"""Compatibility alias for :mod:`veris.model.types`.

Import from the canonical domain module in new code. Both paths share
one module object, including private helpers and patched globals.
"""

import sys

from veris.model import types as _implementation
from veris.model.types import (
    ArrayInput,
    CESMFluxes,
    EVPCarry,
    GrowthResult,
    HeatFluxes,
    Kernel,
    LANLFluxes,
    MaskInput,
    OceanGeometry,
    P,
    Parameter,
    R,
    R_co,
    State,
    SumInput,
    SurfaceFluxResult,
    jit,
)

__all__ = [
    "ArrayInput",
    "CESMFluxes",
    "EVPCarry",
    "GrowthResult",
    "HeatFluxes",
    "Kernel",
    "LANLFluxes",
    "MaskInput",
    "OceanGeometry",
    "P",
    "Parameter",
    "R",
    "R_co",
    "State",
    "SumInput",
    "SurfaceFluxResult",
    "jit",
]

sys.modules[__name__] = _implementation

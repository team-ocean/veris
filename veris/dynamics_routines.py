"""Compatibility alias for :mod:`veris.physics.dynamics.rheology`.

Import from the canonical domain module in new code. Both paths share
one module object, including private helpers and patched globals.
"""

import sys

from veris.physics.dynamics import rheology as _implementation
from veris.physics.dynamics.rheology import (
    SeaIceStrength,
    basal_drag_coeffs,
    ocean_drag_coeffs,
    side_drag,
    strainrates,
    stress,
    stressdiv,
    viscosities,
)

__all__ = [
    "SeaIceStrength",
    "basal_drag_coeffs",
    "ocean_drag_coeffs",
    "side_drag",
    "strainrates",
    "stress",
    "stressdiv",
    "viscosities",
]

sys.modules[__name__] = _implementation

"""Compatibility alias for :mod:`veris.physics.transport.advection`.

Import from the canonical domain module in new code. Both paths share
one module object, including private helpers and patched globals.
"""

import sys

from veris.physics.transport import advection as _implementation
from veris.physics.transport.advection import (
    Advection,
    _calc_flux,
    calc_Advection,
    calc_MeridionalFlux,
    calc_ZonalFlux,
    limiter,
)

__all__ = [
    "Advection",
    "_calc_flux",
    "calc_Advection",
    "calc_MeridionalFlux",
    "calc_ZonalFlux",
    "limiter",
]

sys.modules[__name__] = _implementation

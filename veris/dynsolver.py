"""Compatibility alias for :mod:`veris.numerics.momentum`.

Import from the canonical domain module in new code. Both paths share
one module object, including private helpers and patched globals.
"""

import sys

from veris.numerics import momentum as _implementation
from veris.numerics.momentum import (
    IceVelocities,
    WindForcingXY,
    tauXY,
)

__all__ = ["IceVelocities", "WindForcingXY", "tauXY"]

sys.modules[__name__] = _implementation

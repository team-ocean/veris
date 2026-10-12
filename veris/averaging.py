"""Compatibility alias for :mod:`veris.grid.averaging`.

Import from the canonical domain module in new code. Both paths share
one module object, including private helpers and patched globals.
"""

import sys

from veris.grid import averaging as _implementation
from veris.grid.averaging import (
    c_point_to_z_point,
)

__all__ = ["c_point_to_z_point"]

sys.modules[__name__] = _implementation

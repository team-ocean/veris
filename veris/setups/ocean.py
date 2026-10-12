"""Compatibility alias for :mod:`veris.grid.ocean`.

Import from the canonical domain module in new code. Both paths share
one module object, including private helpers and patched globals.
"""

import sys

from veris.grid import ocean as _implementation
from veris.grid.ocean import (
    OceanGeometry,
    _validate_geometry,
    initialize_from_ocean,
)

__all__ = ["OceanGeometry", "_validate_geometry", "initialize_from_ocean"]

sys.modules[__name__] = _implementation

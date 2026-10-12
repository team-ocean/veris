"""Compatibility alias for :mod:`veris.grid.regions`.

Import from the canonical domain module in new code. Both paths share
one module object, including private helpers and patched globals.
"""

import sys

from veris.grid import regions as _implementation
from veris.grid.regions import (
    _Indices,
    physical_indices,
    regional_forcing,
    regional_geometry,
    regional_state,
)

__all__ = [
    "_Indices",
    "physical_indices",
    "regional_forcing",
    "regional_geometry",
    "regional_state",
]

sys.modules[__name__] = _implementation

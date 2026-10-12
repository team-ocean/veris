"""Compatibility alias for :mod:`veris.model.variables`.

Import from the canonical domain module in new code. Both paths share
one module object, including private helpers and patched globals.
"""

import sys

from veris.model import variables as _implementation
from veris.model.variables import (
    C_GRID,
    U_GRID,
    V_GRID,
    VARIABLES,
    Z_GRID,
    Variable,
)

__all__ = ["C_GRID", "U_GRID", "VARIABLES", "V_GRID", "Z_GRID", "Variable"]

sys.modules[__name__] = _implementation

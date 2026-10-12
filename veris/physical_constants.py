"""Compatibility alias for :mod:`veris.model.physical_constants`.

Import from the canonical domain module in new code. Both paths share
one module object, including private helpers and patched globals.
"""

import sys

from veris.model import physical_constants as _implementation
from veris.model.physical_constants import (
    PHYSICALCONSTANTS,
    Parameter,
    PhysicalConstants,
)

__all__ = ["PHYSICALCONSTANTS", "Parameter", "PhysicalConstants"]

sys.modules[__name__] = _implementation

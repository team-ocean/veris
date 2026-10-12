"""Compatibility alias for :mod:`veris.numerics.surface_temperature`.

Import from the canonical domain module in new code. Both paths share
one module object, including private helpers and patched globals.
"""

import sys

from veris.numerics import surface_temperature as _implementation
from veris.numerics.surface_temperature import (
    solve4temp,
)

__all__ = ["solve4temp"]

sys.modules[__name__] = _implementation

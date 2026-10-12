"""Compatibility alias for :mod:`veris.numerics.freedrift`.

Import from the canonical domain module in new code. Both paths share
one module object, including private helpers and patched globals.
"""

import sys

from veris.numerics import freedrift as _implementation
from veris.numerics.freedrift import (
    freedrift_solver,
)

__all__ = ["freedrift_solver"]

sys.modules[__name__] = _implementation

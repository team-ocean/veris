"""Compatibility alias for :mod:`veris.numerics.evp`.

Import from the canonical domain module in new code. Both paths share
one module object, including private helpers and patched globals.
"""

import sys

from veris.numerics import evp as _implementation
from veris.numerics.evp import (
    evp_solver,
)

__all__ = ["evp_solver"]

sys.modules[__name__] = _implementation

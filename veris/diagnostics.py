"""Compatibility alias for :mod:`veris.model.diagnostics`.

Import from the canonical domain module in new code. Both paths share
one module object, including private helpers and patched globals.
"""

import sys

from veris.model import diagnostics as _implementation
from veris.model.diagnostics import (
    DIAGNOSTICS,
    Diagnostics,
)

__all__ = ["DIAGNOSTICS", "Diagnostics"]

sys.modules[__name__] = _implementation

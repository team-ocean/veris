"""Compatibility alias for :mod:`veris.grid.reductions`.

Import from the canonical domain module in new code. Both paths share
one module object, including private helpers and patched globals.
"""

import sys

from veris.grid import reductions as _implementation
from veris.grid.reductions import (
    global_sum,
)

__all__ = ["global_sum"]

sys.modules[__name__] = _implementation

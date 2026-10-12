"""Compatibility alias for :mod:`veris.execution.initialization`.

Import from the canonical domain module in new code. Both paths share
one module object, including private helpers and patched globals.
"""

import sys

from veris.execution import initialization as _implementation
from veris.execution.initialization import (
    initialize,
)

__all__ = ["initialize"]

sys.modules[__name__] = _implementation

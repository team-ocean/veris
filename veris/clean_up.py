"""Compatibility alias for :mod:`veris.physics.transport.cleanup`.

Import from the canonical domain module in new code. Both paths share
one module object, including private helpers and patched globals.
"""

import sys

from veris.physics.transport import cleanup as _implementation
from veris.physics.transport.cleanup import (
    clean_up_advection,
    ridging,
)

__all__ = ["clean_up_advection", "ridging"]

sys.modules[__name__] = _implementation

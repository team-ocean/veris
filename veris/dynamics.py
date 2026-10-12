"""Compatibility alias for :mod:`veris.physics.dynamics.driver`.

Import from the canonical domain module in new code. Both paths share
one module object, including private helpers and patched globals.
"""

import sys

from veris.physics.dynamics import driver as _implementation
from veris.physics.dynamics.driver import (
    dynamics_transport,
)

__all__ = ["dynamics_transport"]

sys.modules[__name__] = _implementation

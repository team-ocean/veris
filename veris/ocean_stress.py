"""Compatibility alias for :mod:`veris.physics.dynamics.ocean_stress`.

Import from the canonical domain module in new code. Both paths share
one module object, including private helpers and patched globals.
"""

import sys

from veris.physics.dynamics import ocean_stress as _implementation
from veris.physics.dynamics.ocean_stress import (
    OceanStressUV,
)

__all__ = ["OceanStressUV"]

sys.modules[__name__] = _implementation

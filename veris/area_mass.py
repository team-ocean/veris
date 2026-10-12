"""Compatibility alias for :mod:`veris.physics.dynamics.area_mass`.

Import from the canonical domain module in new code. Both paths share
one module object, including private helpers and patched globals.
"""

import sys

from veris.physics.dynamics import area_mass as _implementation
from veris.physics.dynamics.area_mass import (
    AreaWS,
    SeaIceMass,
)

__all__ = ["AreaWS", "SeaIceMass"]

sys.modules[__name__] = _implementation

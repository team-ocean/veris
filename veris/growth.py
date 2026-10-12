"""Compatibility alias for :mod:`veris.physics.thermodynamics.growth`.

Import from the canonical domain module in new code. Both paths share
one module object, including private helpers and patched globals.
"""

import sys

from veris.physics.thermodynamics import growth as _implementation
from veris.physics.thermodynamics.growth import (
    Growth,
)

__all__ = ["Growth"]

sys.modules[__name__] = _implementation

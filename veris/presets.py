"""Compatibility alias for :mod:`veris.setups.presets`.

Import from the canonical domain module in new code. Both paths share
one module object, including private helpers and patched globals.
"""

import sys

from veris.setups import presets as _implementation
from veris.setups.presets import (
    PRESET_PHYSICS,
    preset_controls,
)

__all__ = ["PRESET_PHYSICS", "preset_controls"]

sys.modules[__name__] = _implementation

"""Compatibility alias for :mod:`veris.model.configuration`.

Import from the canonical domain module in new code. Both paths share
one module object, including private helpers and patched globals.
"""

import sys

from veris.model import configuration as _implementation
from veris.model.configuration import (
    SETTINGS,
    Configuration,
    Parameter,
)

__all__ = ["SETTINGS", "Configuration", "Parameter"]

sys.modules[__name__] = _implementation

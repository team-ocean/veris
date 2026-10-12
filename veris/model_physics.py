"""Compatibility alias for :mod:`veris.execution.advance`.

Import from the canonical domain module in new code. Both paths share
one module object, including private helpers and patched globals.
"""

import sys

from veris.execution import advance as _implementation
from veris.execution.advance import (
    _advance_local,
    _apply_forcing,
    _execute_advance,
    _forcing_arrays,
    _validate_state,
    advance,
)

__all__ = [
    "_advance_local",
    "_apply_forcing",
    "_execute_advance",
    "_forcing_arrays",
    "_validate_state",
    "advance",
]

sys.modules[__name__] = _implementation

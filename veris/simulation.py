"""Compatibility alias for :mod:`veris.execution.simulation`.

Import from the canonical domain module in new code. Both paths share
one module object, including private helpers and patched globals.
"""

import sys

from veris.execution import simulation as _implementation
from veris.execution.simulation import (
    RunResult,
    Simulation,
    _Carry,
    _Observer,
)

__all__ = ["RunResult", "Simulation", "_Carry", "_Observer"]

sys.modules[__name__] = _implementation

"""Compatibility alias for :mod:`veris.execution.rollout`.

Import from the canonical domain module in new code. Both paths share
one module object, including private helpers and patched globals.
"""

import sys

from veris.execution import rollout as _implementation
from veris.execution.rollout import (
    _Carry,
    _scan,
    step,
)

__all__ = ["_Carry", "_scan", "step"]

sys.modules[__name__] = _implementation

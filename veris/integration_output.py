"""Compatibility alias for :mod:`veris.execution.rollout_output`.

Import from the canonical domain module in new code. Both paths share
one module object, including private helpers and patched globals.
"""

import sys

from veris.execution import rollout_output as _implementation
from veris.execution.rollout_output import (
    _T,
    output_callbacks,
    run_timed,
)

__all__ = ["_T", "output_callbacks", "run_timed"]

sys.modules[__name__] = _implementation

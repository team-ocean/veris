"""Compatibility alias for :mod:`veris.model.experiment`.

Import from the canonical domain module in new code. Both paths share
one module object, including private helpers and patched globals.
"""

import sys

from veris.model import experiment as _implementation
from veris.model.experiment import (
    INITIAL_FIELDS,
    Execution,
    Experiment,
    Grid,
    _partitions,
)

__all__ = ["INITIAL_FIELDS", "Execution", "Experiment", "Grid", "_partitions"]

sys.modules[__name__] = _implementation

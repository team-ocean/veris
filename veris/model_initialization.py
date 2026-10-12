"""Compatibility alias for :mod:`veris.execution.experiment_initialization`.

Import from the canonical domain module in new code. Both paths share
one module object, including private helpers and patched globals.
"""

import sys

from veris.execution import experiment_initialization as _implementation
from veris.execution.experiment_initialization import (
    _METRICS,
    _apply_inputs,
    _initialize_serial,
    _initialize_sharded,
    _pad_geometry,
    _precision,
    _stored_field,
    _validate_derived_inputs,
    _validate_inputs,
    initialize_experiment,
)

__all__ = [
    "_METRICS",
    "_apply_inputs",
    "_initialize_serial",
    "_initialize_sharded",
    "_pad_geometry",
    "_precision",
    "_stored_field",
    "_validate_derived_inputs",
    "_validate_inputs",
    "initialize_experiment",
]

sys.modules[__name__] = _implementation

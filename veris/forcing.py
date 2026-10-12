"""Compatibility alias for :mod:`veris.model.forcing`.

Import from the canonical domain module in new code. Both paths share
one module object, including private helpers and patched globals.
"""

import sys

from veris.model import forcing as _implementation
from veris.model.forcing import (
    FORCING_FIELDS,
    Forcing,
    _freeze_fields,
    _immutable_array,
    _ImmutableArray,
    _validate_shapes,
)

__all__ = [
    "FORCING_FIELDS",
    "Forcing",
    "_ImmutableArray",
    "_freeze_fields",
    "_immutable_array",
    "_validate_shapes",
]

sys.modules[__name__] = _implementation

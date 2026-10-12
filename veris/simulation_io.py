"""Compatibility alias for :mod:`veris.io.lifecycle`.

Import from the canonical domain module in new code. Both paths share
one module object, including private helpers and patched globals.
"""

import sys

from veris.io import lifecycle as _implementation
from veris.io.lifecycle import (
    _new_path,
    load_fields,
    metadata,
    output_resources,
    snapshot,
    validate_output,
)

__all__ = [
    "_new_path",
    "load_fields",
    "metadata",
    "output_resources",
    "snapshot",
    "validate_output",
]

sys.modules[__name__] = _implementation

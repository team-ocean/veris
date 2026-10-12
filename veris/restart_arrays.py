"""Compatibility alias for :mod:`veris.io.restart_arrays`.

Import from the canonical domain module in new code. Both paths share
one module object, including private helpers and patched globals.
"""

import sys

from veris.io import restart_arrays as _implementation
from veris.io.restart_arrays import (
    _SECTIONS,
    ArrayManifest,
    _allowed,
    _dtype_descriptor,
    _group,
    hydrate_definition,
    validate_links,
)

__all__ = [
    "_SECTIONS",
    "ArrayManifest",
    "_allowed",
    "_dtype_descriptor",
    "_group",
    "hydrate_definition",
    "validate_links",
]

sys.modules[__name__] = _implementation

"""Compatibility alias for :mod:`veris.model.metadata`.

Import from the canonical domain module in new code. Both paths share
one module object, including private helpers and patched globals.
"""

import sys

from veris.model import metadata as _implementation
from veris.model.metadata import (
    FROM_REGISTRY,
    precision_scalar,
    registry_defaults,
    validate_derived,
    validate_scalars,
)

__all__ = [
    "FROM_REGISTRY",
    "precision_scalar",
    "registry_defaults",
    "validate_derived",
    "validate_scalars",
]

sys.modules[__name__] = _implementation

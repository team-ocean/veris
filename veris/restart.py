"""Compatibility alias for :mod:`veris.io.restart`.

Import from the canonical domain module in new code. Both paths share
one module object, including private helpers and patched globals.
"""

import sys

from veris.io import restart as _implementation
from veris.io.restart import (
    _ATTRIBUTES,
    _NAMES,
    _RECORD_NAMES,
    _calendar,
    _clock,
    _decode,
    _json,
    _output,
    _record,
    _serial,
    _state_arrays,
    read_restart,
    save_restart,
)

__all__ = [
    "_ATTRIBUTES",
    "_NAMES",
    "_RECORD_NAMES",
    "_calendar",
    "_clock",
    "_decode",
    "_json",
    "_output",
    "_record",
    "_serial",
    "_state_arrays",
    "read_restart",
    "save_restart",
]

sys.modules[__name__] = _implementation

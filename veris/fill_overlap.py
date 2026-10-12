"""Compatibility alias for :mod:`veris.grid.halos`.

Import from the canonical domain module in new code. Both paths share
one module object, including private helpers and patched globals.
"""

import sys

from veris.grid import halos as _implementation
from veris.grid.halos import (
    Boundary,
    _close_y,
    _validate_mesh,
    fill_circular_overlap,
    fill_overlap,
    fill_overlap_shard,
    fill_overlap_uv,
    fill_state_overlap,
    make_sharded_fill_overlap,
)

__all__ = [
    "Boundary",
    "_close_y",
    "_validate_mesh",
    "fill_circular_overlap",
    "fill_overlap",
    "fill_overlap_shard",
    "fill_overlap_uv",
    "fill_state_overlap",
    "make_sharded_fill_overlap",
]

sys.modules[__name__] = _implementation

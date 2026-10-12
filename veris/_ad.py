"""Compatibility alias for :mod:`veris.numerics.autodiff`.

Import from the canonical domain module in new code. Both paths share
one module object, including private helpers and patched globals.
"""

import sys

from veris.numerics import autodiff as _implementation
from veris.numerics.autodiff import (
    _norm_hypot_jvp,
    norm_hypot,
    norm_sqrt,
)

__all__ = ["_norm_hypot_jvp", "norm_hypot", "norm_sqrt"]

sys.modules[__name__] = _implementation

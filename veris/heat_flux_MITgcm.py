"""Compatibility alias for :mod:`veris.physics.thermodynamics.heat_flux_mitgcm`.

Import from the canonical domain module in new code. Both paths share
one module object, including private helpers and patched globals.
"""

import sys

from veris.physics.thermodynamics import heat_flux_mitgcm as _implementation
from veris.physics.thermodynamics.heat_flux_mitgcm import (
    bulkf_formula_lanl,
)

__all__ = ["bulkf_formula_lanl"]

sys.modules[__name__] = _implementation

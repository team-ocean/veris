"""Compatibility alias for :mod:`veris.physics.thermodynamics.heat_flux_cesm`.

Import from the canonical domain module in new code. Both paths share
one module object, including private helpers and patched globals.
"""

import sys

from veris.physics.thermodynamics import heat_flux_cesm as _implementation
from veris.physics.thermodynamics.heat_flux_cesm import (
    cdn,
    compute_z_level,
    dqnetdt,
    flux_atmOcn,
    flux_atmOcn_simple,
    get_press_levs,
    net_lw_ocn,
    psimhu,
    psixhu,
    qsat,
    qsat_august_eqn,
)

__all__ = [
    "cdn",
    "compute_z_level",
    "dqnetdt",
    "flux_atmOcn",
    "flux_atmOcn_simple",
    "get_press_levs",
    "net_lw_ocn",
    "psimhu",
    "psixhu",
    "qsat",
    "qsat_august_eqn",
]

sys.modules[__name__] = _implementation

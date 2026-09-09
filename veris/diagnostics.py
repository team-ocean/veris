"""Output-only ocean coupling fields from one sea-ice integration step.

These halo-inclusive C-grid arrays are returned separately from State so they
add no permanent calculation or differentiation leaves. Metadata follows the
same Variable schema as state output and can be passed directly to h5netcdf.
The legacy salt-forcing formula combines differently normalized terms when
saltIce_ref is nonzero; its units remain unknown pending a physics correction.
"""

from dataclasses import dataclass

import jax
from jax import Array

from veris.variables import C_GRID, U_GRID, V_GRID, Variable

DIAGNOSTICS: dict[str, Variable] = {
    "IcePenetSW": Variable(
        "Shortwave radiation penetrating the ice",
        C_GRID,
        "W m-2",
        "Grid-cell mean shortwave heat flux transmitted through sea ice",
    ),
    "OceanStressU": Variable(
        "Zonal stress on the ocean surface",
        U_GRID,
        "N m-2",
        "Zonal ice-ocean stress from the momentum step at zonal velocity faces",
    ),
    "OceanStressV": Variable(
        "Meridional stress on the ocean surface",
        V_GRID,
        "N m-2",
        "Meridional ice-ocean stress from the momentum step at meridional velocity faces",
    ),
    "EmPmR": Variable(
        "Evaporation minus precipitation minus runoff",
        C_GRID,
        "kg m-2 s-1",
        "Net freshwater mass loss from the ocean, including ice and snow exchange",
    ),
    "forc_salt_surface": Variable(
        "Legacy surface salt forcing",
        C_GRID,
        "unknown",
        "Combined virtual and actual ice salt forcing; the legacy terms have "
        "different freshwater normalizations when saltIce_ref is nonzero",
    ),
}


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class Diagnostics:
    """Per-step coupling outputs in DIAGNOSTICS registry order."""

    IcePenetSW: Array
    OceanStressU: Array
    OceanStressV: Array
    EmPmR: Array
    forc_salt_surface: Array

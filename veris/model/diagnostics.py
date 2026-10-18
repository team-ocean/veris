"""Output-only ocean coupling fields from one sea-ice integration step.

These halo-inclusive C-grid arrays are returned separately from State so they
add no permanent calculation or differentiation leaves. Metadata follows the
same Variable schema as state output and can be passed directly to h5netcdf.
Actual ocean mass/salt fluxes and independent external boundary inputs use SI
rates, downward positive. The legacy-named salt forcing is now the consistent
local-salinity virtual flux in (g/kg)*m/s.
"""

from dataclasses import dataclass

import jax
from jax import Array

from veris.model.variables import C_GRID, U_GRID, V_GRID, Variable


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class Diagnostics:
    """Per-step coupling outputs in DIAGNOSTICS registry order."""

    IcePenetSW: Array
    OceanStressU: Array
    OceanStressV: Array
    EmPmR: Array
    forc_salt_surface: Array
    physics_status: Array
    ocean_mass_flux: Array
    ocean_salt_mass_flux: Array
    external_mass_flux: Array
    external_heat_flux: Array


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
        "Total ocean mass loss, including ice/snow and external freshwater exchange",
    ),
    "forc_salt_surface": Variable(
        "Surface virtual salt forcing",
        C_GRID,
        "g kg-1 m s-1",
        "Local-salinity virtual forcing: (EmPmR*ocSalt + 1000*ocean_salt_mass_flux)/rhoFresh",
    ),
    "physics_status": Variable(
        "Physics validity status",
        C_GRID,
        "1",
        "Zero for valid physics; one for invalid physical parameters; two for invalid saline-ice formation",
    ),
    "ocean_mass_flux": Variable(
        "Actual mass input to ocean",
        C_GRID,
        "kg m-2 s-1",
        "Downward total bulk mass input, including retained ice salt mass",
    ),
    "ocean_salt_mass_flux": Variable(
        "Actual salt input to ocean",
        C_GRID,
        "kg m-2 s-1",
        "Downward salt mass input for fixed bulk ice salinity in g/kg",
    ),
    "external_mass_flux": Variable(
        "External column mass input",
        C_GRID,
        "kg m-2 s-1",
        "Downward atmospheric/land mass from rain, snow, evaporation, sublimation and runoff",
    ),
    "external_heat_flux": Variable(
        "External column enthalpy input",
        C_GRID,
        "W m-2",
        "Downward zero-layer enthalpy input from surface thermal fluxes and boundary mass events",
    ),
}


__all__ = ["DIAGNOSTICS", "Diagnostics"]

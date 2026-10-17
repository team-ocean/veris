"""Actual ocean mass and salt exchange for the native zero-layer ice column.

For solid volume change dV per horizontal area and external atmospheric/land
mass input B, the ocean input is B - (rho_i*dV_i + rho_s*dV_s)/dt. Uniform
bulk ice salinity S_i in g/kg retains rho_i*dV_i*S_i/1000 kg salt/m2 in ice;
snow and external water are fresh. Both outputs are downward-positive grid
arrays in kg/m2/s. These are physical transfers, before a fixed-volume ocean's
separate virtual-salinity conversion. Transport redistribution is excluded.
"""

from dataclasses import dataclass
from functools import partial

import jax
from jax import Array

from veris.model.configuration import Configuration
from veris.model.physical_constants import PhysicalConstants


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class OceanExchangeFluxes:
    """Grid-shaped downward mass and actual salt input in kg/m2/s."""

    mass_input: Array
    salt_mass_input: Array


@partial(jax.jit, static_argnames=("configuration",))
def ocean_exchange_fluxes(
    *,
    ice_change: Array,
    snow_change: Array,
    external_mass_input: Array,
    cleanup_ice: Array,
    cleanup_snow: Array,
    configuration: Configuration,
    constants: PhysicalConstants,
) -> OceanExchangeFluxes:
    """Account for thermodynamics and signed raw-minus-prepared cleanup.

    Changes are final minus prepared pre-thermodynamic thickness in metres.
    Cleanup is raw post-advection minus prepared thickness, also in metres;
    subtracting it includes both clipped negative stock and removed thin stock.
    The caller supplies finite, zeroed inactive cells and boundary-event mass
    rates independently of final inventories. Numerical timestep is static;
    physical densities and salinity remain differentiable leaves.
    """
    ice_rate = (ice_change - cleanup_ice) * configuration.recip_deltatTherm
    snow_rate = (snow_change - cleanup_snow) * configuration.recip_deltatTherm
    return OceanExchangeFluxes(
        external_mass_input
        - constants.rhoIce * ice_rate
        - constants.rhoSnow * snow_rate,
        -constants.rhoIce * ice_rate * constants.saltIce_ref / 1000,
    )


__all__ = ["OceanExchangeFluxes", "ocean_exchange_fluxes"]

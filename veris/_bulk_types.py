"""Array return contracts shared by the standalone bulk heat-flux kernels.

The CESM and MITgcm LANL kernels take initialized Settings and
PhysicalConstants explicitly; no configuration is stored in array state.
"""

from jax import Array

HeatFluxes = tuple[Array, Array, Array]
CESMFluxes = tuple[
    Array, Array, Array, Array, Array, Array, Array, Array, Array, Array, Array, Array
]
LANLFluxes = tuple[Array, Array, Array, Array, Array, Array, Array, Array, Array]

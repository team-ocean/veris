"""State metadata for the Cartesian Arakawa C grid used by Veris.

Arrays store x on axis zero and y on axis one, including two periodic halo
cells at each boundary. Face and center dimensions have equal storage lengths.
Mass and stress units describe vertically integrated ice equations. Only fields
read by the maintained numerical kernels are allocated in State.
"""

from dataclasses import dataclass
from typing import cast

from veris.physical_constants import PHYSICALCONSTANTS

C_GRID = ("x_center", "y_center")
U_GRID = ("x_face", "y_center")
V_GRID = ("x_center", "y_face")
Z_GRID = ("x_face", "y_face")


@dataclass(frozen=True)
class Variable:
    """Allocation defaults and h5netcdf-compatible variable attributes."""

    long_name: str
    dimensions: tuple[str, str]
    units: str
    description: str
    dtype: str = "float64"
    default: float = 0.0

    def netcdf_attributes(self) -> dict[str, str]:
        """Return text attributes suitable for a netCDF variable."""
        return {
            "long_name": self.long_name,
            "units": self.units,
            "description": self.description,
        }


VARIABLES: dict[str, Variable] = {
    "hIceMean": Variable(
        "Mean ice thickness", C_GRID, "m", "Mean ice thickness", default=0.0
    ),
    "hSnowMean": Variable(
        "Mean snow thickness", C_GRID, "m", "Mean snow thickness", default=0.0
    ),
    "Area": Variable(
        "Sea ice cover fraction", C_GRID, "1", "Sea ice cover fraction", default=0.0
    ),
    "TSurf": Variable(
        "Ice/ snow surface temperature",
        C_GRID,
        "K",
        "Ice/ snow surface temperature",
        default=cast(float, PHYSICALCONSTANTS["celsius2K"].default),
    ),
    "SeaIceMassC": Variable(
        "Sea ice mass centered around c point",
        C_GRID,
        "kg m-2",
        "Sea ice mass centered around c point",
        default=0.0,
    ),
    "SeaIceMassU": Variable(
        "Sea ice mass centered around u point",
        U_GRID,
        "kg m-2",
        "Sea ice mass centered around u point",
        default=0.0,
    ),
    "SeaIceMassV": Variable(
        "Sea ice mass centered around v point",
        V_GRID,
        "kg m-2",
        "Sea ice mass centered around v point",
        default=0.0,
    ),
    "SeaIceStrength": Variable(
        "Ice strength", C_GRID, "N m-1", "Ice strength", default=0.0
    ),
    "os_hIceMean": Variable(
        "Overshoot of ice thickness from advection",
        C_GRID,
        "m",
        "Overshoot of ice thickness from advection",
        default=0.0,
    ),
    "os_hSnowMean": Variable(
        "Overshoot of snow thickness from advection",
        C_GRID,
        "m",
        "Overshoot of snow thickness from advection",
        default=0.0,
    ),
    "AreaW": Variable(
        "Sea ice cover fraction centered around u point",
        U_GRID,
        "1",
        "Sea ice cover fraction centered around u point",
        default=0.0,
    ),
    "AreaS": Variable(
        "Sea ice cover fraction centered around v point",
        V_GRID,
        "1",
        "Sea ice cover fraction centered around v point",
        default=0.0,
    ),
    "uIce": Variable(
        "Zonal ice velocity", U_GRID, "m s-1", "Zonal ice velocity", default=0.0
    ),
    "vIce": Variable(
        "Meridional ice velocity",
        V_GRID,
        "m s-1",
        "Meridional ice velocity",
        default=0.0,
    ),
    "sigma1": Variable(
        "Stress trace sigma11 + sigma22",
        C_GRID,
        "N m-1",
        "Stress trace sigma11 + sigma22",
        default=0.0,
    ),
    "sigma2": Variable(
        "Normal-stress difference sigma11 - sigma22",
        C_GRID,
        "N m-1",
        "Normal-stress difference sigma11 - sigma22",
        default=0.0,
    ),
    "sigma12": Variable(
        "Stress tensor component",
        Z_GRID,
        "N m-1",
        "Stress tensor component",
        default=0.0,
    ),
    "WindForcingX": Variable(
        "Zonal forcing on ice by wind stress",
        U_GRID,
        "Pa",
        "Zonal forcing on ice by wind stress",
        default=0.0,
    ),
    "WindForcingY": Variable(
        "Meridional forcing on ice by wind stress",
        V_GRID,
        "Pa",
        "Meridional forcing on ice by wind stress",
        default=0.0,
    ),
    "recip_hIceMean": Variable(
        "Regularized reciprocal mean ice thickness",
        C_GRID,
        "m-1",
        "Reciprocal square root of hIceMean squared plus hIce_reg; refreshed by initialization or growth",
        default=0.0,
    ),
    "SeaIceLoad": Variable(
        "Load of sea ice on ocean surface",
        C_GRID,
        "kg m-2",
        "Load of sea ice on ocean surface",
        default=0.0,
    ),
    "uOcean": Variable(
        "Zonal ocean surface velocity",
        U_GRID,
        "m s-1",
        "Zonal ocean surface velocity",
        default=0.0,
    ),
    "vOcean": Variable(
        "Meridional ocean surface velocity",
        V_GRID,
        "m s-1",
        "Meridional ocean surface velocity",
        default=0.0,
    ),
    "theta": Variable(
        "Ocean surface temperature",
        C_GRID,
        "K",
        "Ocean surface temperature",
        default=cast(float, PHYSICALCONSTANTS["celsius2K"].default),
    ),
    "ocSalt": Variable(
        "Ocean surface salinity",
        C_GRID,
        "g kg-1",
        "Ocean surface salinity",
        default=0.0,
    ),
    "Qnet": Variable(
        "Net heat flux out of the ocean",
        C_GRID,
        "W m-2",
        "Net heat flux out of the ocean",
        default=0.0,
    ),
    "R_low": Variable(
        "Sea floor depth (<0)", C_GRID, "m", "Sea floor depth (<0)", default=0.0
    ),
    "ssh_an": Variable(
        "Sea surface height anomaly",
        C_GRID,
        "m",
        "Sea surface height anomaly",
        default=0.0,
    ),
    "Qsw": Variable(
        "Surface shortwave heatflux (+ = upwards)",
        C_GRID,
        "W m-2",
        "Surface shortwave heatflux (+ = upwards)",
        default=0.0,
    ),
    "uWind": Variable(
        "Zonal wind velocity", C_GRID, "m s-1", "Zonal wind velocity", default=0.0
    ),
    "vWind": Variable(
        "Meridional wind velocity",
        C_GRID,
        "m s-1",
        "Meridional wind velocity",
        default=0.0,
    ),
    "wSpeed": Variable(
        "Total wind speed", C_GRID, "m s-1", "Total wind speed", default=0.0
    ),
    "surfPress": Variable(
        "Surface pressure", C_GRID, "Pa", "Surface pressure", default=0.0
    ),
    "SWdown": Variable(
        "Downward shortwave radiation",
        C_GRID,
        "W m-2",
        "Downward shortwave radiation",
        default=0.0,
    ),
    "LWdown": Variable(
        "Downward longwave radiation",
        C_GRID,
        "W m-2",
        "Downward longwave radiation",
        default=0.0,
    ),
    "ATemp": Variable(
        "Atmospheric temperature",
        C_GRID,
        "K",
        "Atmospheric temperature",
        default=cast(float, PHYSICALCONSTANTS["celsius2K"].default),
    ),
    "aqh": Variable(
        "Atmospheric specific humidity",
        C_GRID,
        "kg kg-1",
        "Atmospheric specific humidity",
        default=0.0,
    ),
    "precip": Variable(
        "Precipitation rate (freshwater flux)",
        C_GRID,
        "m s-1",
        "Precipitation rate (freshwater flux)",
        default=0.0,
    ),
    "snowfall": Variable(
        "Snowfall rate", C_GRID, "m s-1", "Snowfall rate", default=0.0
    ),
    "evap": Variable(
        "Evaporation rate over open ocean (freshwater flux, <0 increases salinity)",
        C_GRID,
        "m s-1",
        "Evaporation rate over open ocean (freshwater flux, <0 increases salinity)",
        default=0.0,
    ),
    "runoff": Variable(
        "Runoff into ocean", C_GRID, "m s-1", "Runoff into ocean", default=0.0
    ),
    "maskInC": Variable(
        "Mask at c-points, used for open boundaries",
        C_GRID,
        "1",
        "Mask at c-points, used for open boundaries",
        default=1.0,
    ),
    "maskInU": Variable(
        "Mask at u-points, used for open boundaries",
        U_GRID,
        "1",
        "Mask at u-points, used for open boundaries",
        default=1.0,
    ),
    "maskInV": Variable(
        "Mask at v-points, used for open boundaries",
        V_GRID,
        "1",
        "Mask at v-points, used for open boundaries",
        default=1.0,
    ),
    "iceMask": Variable(
        "Mask at c-points", C_GRID, "1", "Mask at c-points", default=1.0
    ),
    "iceMaskU": Variable(
        "Mask at u-points", U_GRID, "1", "Mask at u-points", default=1.0
    ),
    "iceMaskV": Variable(
        "Mask at v-points", V_GRID, "1", "Mask at v-points", default=1.0
    ),
    "k1AtC": Variable(
        "Zonal metric curvature at cell centers",
        C_GRID,
        "m-1",
        "Zonal metric curvature at cell centers",
        default=0.0,
    ),
    "k2AtC": Variable(
        "Meridional metric curvature at cell centers",
        C_GRID,
        "m-1",
        "Meridional metric curvature at cell centers",
        default=0.0,
    ),
    "k1AtZ": Variable(
        "Zonal metric curvature at cell corners",
        Z_GRID,
        "m-1",
        "Zonal metric curvature at cell corners",
        default=0.0,
    ),
    "k2AtZ": Variable(
        "Meridional metric curvature at cell corners",
        Z_GRID,
        "m-1",
        "Meridional metric curvature at cell corners",
        default=0.0,
    ),
    "Fu": Variable(
        "U-component of form factor",
        U_GRID,
        "1",
        "U-component of form factor",
        default=0.0,
    ),
    "Fv": Variable(
        "V-component of form factor",
        V_GRID,
        "1",
        "V-component of form factor",
        default=0.0,
    ),
    "fCori": Variable(
        "Coriolis parameter", C_GRID, "s-1", "Coriolis parameter", default=0.0
    ),
    "dxG": Variable(
        "Zonal spacing of cell faces along southern cell wall",
        V_GRID,
        "m",
        "Zonal spacing of cell faces along southern cell wall",
        default=1.0,
    ),
    "dyG": Variable(
        "Meridional spacing of cell faces along western cell wall",
        U_GRID,
        "m",
        "Meridional spacing of cell faces along western cell wall",
        default=1.0,
    ),
    "dxU": Variable(
        "Zonal spacing of u-points through cell center",
        C_GRID,
        "m",
        "Zonal spacing of u-points through cell center",
        default=1.0,
    ),
    "dyU": Variable(
        "Meridional spacing of u-points through south-west corner of the cel",
        Z_GRID,
        "m",
        "Meridional spacing of u-points through south-west corner of the cel",
        default=1.0,
    ),
    "dxV": Variable(
        "Zonal spacing of v-points through south-west corner of the cell",
        Z_GRID,
        "m",
        "Zonal spacing of v-points through south-west corner of the cell",
        default=1.0,
    ),
    "dyV": Variable(
        "Meridional spacing of v-points through cell center",
        C_GRID,
        "m",
        "Meridional spacing of v-points through cell center",
        default=1.0,
    ),
    "recip_dxC": Variable(
        "Reciprocal of dxC", U_GRID, "m-1", "Reciprocal of dxC", default=1.0
    ),
    "recip_dyC": Variable(
        "Reciprocal of dyC", V_GRID, "m-1", "Reciprocal of dyC", default=1.0
    ),
    "recip_dxU": Variable(
        "Reciprocal of dxU", C_GRID, "m-1", "Reciprocal of dxU", default=1.0
    ),
    "recip_dyU": Variable(
        "Reciprocal of dyU", Z_GRID, "m-1", "Reciprocal of dyU", default=1.0
    ),
    "recip_dxV": Variable(
        "Reciprocal of dxV", Z_GRID, "m-1", "Reciprocal of dxV", default=1.0
    ),
    "recip_dyV": Variable(
        "Reciprocal of dyV", C_GRID, "m-1", "Reciprocal of dyV", default=1.0
    ),
    "rAz": Variable(
        "Grid cell area centered around z-point",
        Z_GRID,
        "m2",
        "Grid cell area centered around z-point",
        default=1.0,
    ),
    "recip_rA": Variable(
        "Reciprocal of rA", C_GRID, "m-2", "Reciprocal of rA", default=1.0
    ),
    "recip_rAu": Variable(
        "Reciprocal of rAu", U_GRID, "m-2", "Reciprocal of rAu", default=1.0
    ),
    "recip_rAv": Variable(
        "Reciprocal of rAv", V_GRID, "m-2", "Reciprocal of rAv", default=1.0
    ),
}

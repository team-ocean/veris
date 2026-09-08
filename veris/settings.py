"""Mutable default configuration with statically checked per-key scalar types."""

from veris.state import SettingsDict

settings: SettingsDict = {
    "deltatTherm": 86400,  # timestep for thermodynamic equations /s
    "recip_deltatTherm": 1 / 86400,  # 1 / deltatTherm /s
    "deltatDyn": 86400,  # timestep for dynamic equations /s
    "recip_deltatDyn": 1 / 86400,  # 1 / deltatTDyn /s
    "nITC": 5,  # number of ice thickness categories /-
    "recip_nITC": 1 / 5,  # 1 / nITC /-
    "noSlip": True,  # flag for using the no-slip condition
    "useRelativeWind": True,  # flag for using the wind-ice velocity difference (True)
    # or just the wind velocity (False) for the wind stress acting on the ice
    "secondOrderBC": False,  # flag for using the second order approximation for boundary conditions
    "extensiveFld": True,  # flag whether the advective fields are extensive
    "useRealFreshWaterFlux": False,  # flag for using the sea ice load in the calculation of the ocean surface height
    "useFreedrift": False,  # flag for using the freedrift solver
    "useEVP": True,  # flag for using the EVP solver
    "evpAlpha": 500,  # EVP parameter /-
    "evpBeta": 500,  # EVP parameter /-
    "useAdaptiveEVP": False,  # flag for using adaptive relaxation parameters
    "aEVPalphaMin": 5,  # lower limit of alpha and beta /-
    "aEvpCoeff": 0.5,  # largest stabilized frequency for adaptive EVP /-
    "explicitDrag": True,  # flag for stepping the momentum equation in a explicit or implicit way
    "nEVPsteps": 400,  # number of sub-cycling iterations of the EVP solver
    "computeEvpResidual": False,  # flag for computing the residual of stress and velocity in the EVP loop
    "use_coastline": False,  # flag for using the coastline data for lateral drag
    "use_sharding": True,  # flag for using parallel execution via sharded arrays
    "rhoIce": 900,  # density of ice /kg/m3
    "rhoFresh": 1000,  # density of fresh water /kg/m3
    "rhoSea": 1026,  # density of sea water /kg/m3
    "rhoAir": 1.3,  # density of air /kg/m3
    "rhoSnow": 330,  # density of snow /kg/m3
    "recip_rhoFresh": 1 / 1000,  # 1 / rhoFresh /m3/kg
    "recip_rhoSea": 1 / 1026,  # 1 / rhoSea /m3/kg
    "rhoIce2rhoSnow": 900 / 330,  # rhoIce / rhoSnow /m3/kg
    "rhoIce2rhoFresh": 900 / 1000,  # rhoIce / rhoFresh /m3/kg
    "rhoFresh2rhoSnow": 1000 / 330,  # rhoFresh / rhoSnow /m3/kg
    ##### constants used in growth and solve4temp #####
    "dryIceAlb": 0.75,  # albedo of dry ice /-
    "dryIceAlb_south": 0.75,  # albedo of dry ice in the southern hemisphere /-
    "wetIceAlb": 0.66,  # albedo of wet ice /-
    "wetIceAlb_south": 0.66,  # albedo of wet ice in the southern hemisphere /-
    "drySnowAlb": 0.84,  # albedo of dry snow /-
    "drySnowAlb_south": 0.84,  # albedo of dry snow in the southern hemisphere /-
    "wetSnowAlb": 0.7,  # albedo of wet snow /-
    "wetSnowAlb_south": 0.7,  # albedo of wet snow in the southern hemisphere /-
    "wetAlbTemp": 0,  # temperature above which the wet albedos are used /°C
    "lhFusion": 3.34e5,  # latent heat of fusion /J/kg
    "lhEvap": 2.5e6,  # latent heat of evaporation /J/kg
    "lhSublim": 3.34e5 + 2.5e6,  # latent heat of sublimation /J/kg
    "cpAir": 1005,  # heat capacity of air /J/kg K
    "cpWater": 3986,  # heat capacity of water /J/kg K
    "stefBoltz": 5.67e-8,  # Stefan-Boltzmann constant /W/m^2/K^4
    "iceEmiss": 0.95,  # longwave ice emissivity /-
    "snowEmiss": 0.95,  # longwave snow emissivity /-
    "iceConduct": 2.1656,  # sea ice conductivity /-
    "snowConduct": 0.31,  # snow conductivity /-
    "hCut": 0.15,  # cut off snow thickness /m
    "shortwave": 0.3,  # shortwave ice penetration factor /-
    "tempFrz": -1.96,  # freezing temperature /°C
    "dtempFrz_dS": 0,  # - 0.0575 # derivative of freezing temperature wrt salinity /°C/(g/kg)
    "saltIce_ref": 0,  # reference salinity of sea ice /g/kg
    "saltOcn_ref": 34.7,  # reference salinity of the ocean /g/kg
    "minLWdown": 60,  # minimum downward longwave radiation /W/m^2
    "maxTIce": 30,  # maximum ice temperature /°C
    "minTIce": -50,  # minimum ice temperature /°C
    "minTAir": -50,  # minimum air temperature /°C
    "dalton": 0.00175,  # dalton number/ sensible and latent heat transfer coefficient /m/s
    "Area_reg": 0.15**2,  # regularization value for the ice concentration /m^2
    "hIce_reg": 0.10**2,  # regularization value for the ice thickness /m^2
    "celsius2K": 273.15,  # conversion from [K] to [°C] /K
    "stantonNr": 0.0056,  # stanton number /-
    "uStarBase": 0.0125,  # typical friction velocity beneath sea ice /m/s
    "McPheeTaperFac": 12.5,  # tapering factor at the ice bottom /-
    "h0": 0.5,  # lead closing parameter
    "recip_h0": 1 / 0.5,  # 1 / h0
    "h0_south": 0.5,  # lead closing parameter in the southern hemisphere
    "recip_h0_south": 1 / 0.5,  # 1 / h0_south
    ##### constants used in advection routines #####
    "airTurnAngle": 0,  # turning angle of air-ice stress /°
    "waterTurnAngle": 0,  # turning angle of water-ice stress /°
    "sinWat": 0,  # sin of waterTurnAngle /-
    "cosWat": 1,  # cos of waterTurnAngle /-
    "wSpeedMin": 1e-10,  # minimum wind speed /m/s
    "hIce_min": 1e-5,  # 'minimum' ice thickness /m
    "Area_min": 1e-5,  # 'minimum' ice cover fraction /-
    "airIceDrag": 0.0012,  # air-ice drag coefficient /-
    "airIceDrag_south": 0.0012,  # air-ice drag coefficient in the southern hemisphere /-
    "waterIceDrag": 0.0055,  # water-ice drag coefficient /-
    "waterIceDrag_south": 0.0055,  # water-ice drag coefficient in the southern hemisphere /-
    "cDragMin": 0.25,  # minimum of linear ice-ocean drag coefficient /-
    "seaIceLoadFac": 1,  # factor to scale sea ice loading /-
    "gravity": 9.81,  # gravitational acceleration /m/s^2
    "PlasDefCoeff": 2,  # axes ratio of the elliptical yield curve /-
    "deltaMin": 2e-9,  # minimum value of delta /-
    "pressReplFac": 1,  # flag whether to use replacement pressure /-
    "pStar": 27.5e3,  # sea ice strength parameter /Pa
    "cStar": 20,  # sea ice strength parameter /-
    "basalDragU0": 5e-5,  # basal drag parameter /m/s
    "basalDragK1": 8,  # basal drag parameter /-
    "basalDragK2": 0,  # basal drag parameter /-
    "cBasalStar": 20,  # basal drag parameter /-
    "tensileStrFac": 0,  # sea ice tensile strength factor /-
    "CrMax": 1e6,  # advective flux parameter /-
    "sideDragCoeff": 0.001,  # side drag coefficient /-
    "sideDragU0": 0.01,  # side drag critical velocity /m/s
    ##### forcing #####
    "umin_o": 0.5,  # minimum atm. wind speed over ocean surface /m/s
    "umin_i": 1.0,  # minimum atm. wind speed over ice surface /m/s
    "zref": 10.0,  # reference height for wind speed /m
    "ztref": 2.0,  # reference height for air temperature /m
    "bolzc": 1.38065e-23,  # Boltzmann's constant /J/K/molecule
    "avogad": 6.02214e26,  # Avogadro number /molecules/kmole
    "rgas": 8314.47,  # avogad * bolzc - Ideal gas constant /J/K/kmole
    "mwdair": 28.966,  # molecular weight of dry air /kg/kmole
    "mwwv": 18.016,  # molecular weight water vapor /kg/kmole
    "rdair": 287.042,  # RGAS / MWDAIR - dry air gas constant /J/K/kg
    "rwv": 461.505,  # RGAS / MWWV - water vapor constant /J/K/kg
    "zvir": 0.608,  # (RWV / RDAIR) - 1.0 - Dry-air water-vapor molecular mass ratio /-
    "cpdair": 1.00464e3,  # specific heat of dry air /J/K/kg
    "cpwv": 1.810e3,  # specific heat of water vapor /J/K/kg
    "cpvir": 0.802,  # - /-
    "karman": 0.4,  # von Karman constant
    "latvap": 2.501e6,  # latent heat of evaporation /J/kg
    "p0": 1e5,  # reference pressure to compute potential temperature /Pa
    "cappa": 0.286,  # R/Cp /-
    "zzsice": 0.0005,  # ice surface roughness /m
    "ch": 1e-3,  # bulk transfer coefficient for sensible heat /-
    "ce": 1.15e-3,  # bulk transfer coefficient for latent heat /-
    "eps2": 1e-20,  # threshold value /-
    "emissivity": 1,  # surface emissivity /-
    "ocean_emissivity": 0.985,  # ocean surface emissivity /-
    "snow_emissivity": 0.98,  # snow surface emissivity /-
    "ice_emissivity": 0.98,  # ice surface emissivity /-
    "tf0kel": 273.15,  # freezing temp of fresh water /K
    "gamma_blk": 0.010,  # adiabatic lapse rate /C/m
    "ocean_albedo": 0.1,  # ocean albedo /-
    "ice_albedo": 0.7,  # ice albedo /-
}

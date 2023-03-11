# constants for the heat flux bulk formulae in setup_flux_CESM and setup_flux_MITgcm

# minimum atm. wind speed over ocean surface ~ (m/s)
UMIN_O = 0.5

# minimum atm. wind speed over ice surface ~ (m/s)
UMIN_I = 1.

# reference height ~ (m)
ZREF = 10.0

# reference height for air T (m)
ZTREF = 2.0

# Boltzmann's constant ~ J/K/molecule
BOLTZ = 1.38065e-23

# avogadro's number ~ molecules/kmole
AVOGAD = 6.02214e26

RGAS = AVOGAD * BOLTZ

# molecular weight dry air ~ kg/kmole
MWDAIR = 28.966

# molecular weight water vapor
MWWV = 18.016

# dry air gas constant     ~ J/K/kg
RDAIR = RGAS / MWDAIR

# water vapor gas constant ~ J/K/kg
RWV = RGAS / MWWV

# Dry-air - water-vapor molecular mass ratio
ZVIR = (RWV / RDAIR) - 1.0

# specific heat of dry air   ~ J/kg/K
CPDAIR = 1.00464e3

# specific heat of water vap ~ J/kg/K
CPWV = 1.810e3

# CPWV / CPDAIR - 1.0
CPVIR = (CPWV / CPDAIR) - 1.0

# von Karman constant
KARMAN = 0.4

# acceleration of gravity ~ m/s^2
G = 9.80616

# latent heat of evaporation ~ J/kg
LATVAP = 2.501e6

# Stefan-Boltzmann constant ~ W/m^2/K^4
STEBOL = 5.67e-8

# reference pressure to compute potential temperature
P0 = 1e5

# R/Cp
CAPPA = (RGAS / MWDAIR) / CPDAIR

# Earth radius
RE = 6371.2290e3

# ice surface roughness
ZZSICE = 0.0005

# latent heat of fusion ~ J/kg
LATICE = 3.337e5

# latent heat of evaporation ~ J/kg
LATVAP = 2.501e6

# latent heat for surface ~ J/kg
LTHEAT = LATICE + LATVAP

# Bulk transfer coefficient for sensible heat
CH = 1e-3

# Bulk transfer coefficient for latent heat
CE = 1.15e-3

# Threshold value
EPS2 = 1.0e-20

# Surface emissivity (usually 0.97-0.98 for sea surface)
EMISSIVITY = 1.0
OCEAN_EMISSIVITY= 0.985
SNOW_EMISSIVITY = 0.98
ICE_EMISSIVITY  = 0.98

# FROM MITgcm
# Freezing temp of fresh water in Kelvin
TF0KEL = 273.15

# adiabatic lapse rate
GAMMA_BLK = 0.010

# reference density of air [kg/m^3]
RHOA = 1.3

# ocean albedo
OCEAN_ALBEDO = 0.1

# ice albedo
ICE_ALBEDO = 0.7

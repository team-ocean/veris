from veros.settings import Setting


SETTINGS = dict(
    deltatTherm=Setting(86400, float, "Timestep for thermodynamic equations [s]"),
    recip_deltatTherm=Setting(1 / 86400, float, "1 / deltatTherm [1/s]"),
    deltatDyn=Setting(86400, float, "Timestep for dynamic equations [s]"),
    olx=Setting(2, int, "Grid points in zonal overlap"),
    oly=Setting(2, int, "Grid points in meridional overlap"),
    gridcellWidth=Setting(444709.893408, float, "Grid cell width [m]"),
    nITC=Setting(5, int, "Number of ice thickness categories"),
    recip_nITC=Setting(1 / 5, float, "1 / nITC"),
    noSlip=Setting(True, bool, "flag for using the no-slip condition"),
    secondOrderBC=Setting(
        False,
        bool,
        "flag for using the second order approximation for boundary conditions",
    ),
    extensiveFld=Setting(True, bool, "flag whether the advective fields are extensive"),
    useRealFreshWaterFlux=Setting(
        False,
        bool,
        "flag for using the sea ice load in the calculation of the ocean surface height",
    ),
    useFreedrift=Setting(False, bool, "flag for using the freedrift solver"),
    useEVP=Setting(True, bool, "flag for using the EVP solver"),
    veros_fill=Setting(True, bool, "flag for using the fill overlap function of Veros"),
    use_coastline=Setting(
        False, bool, "flag for using the coastline data for lateral drag"
    ),
    rhoIce=Setting(900, float, "density of ice [kg/m3]"),
    rhoFresh=Setting(1000, float, "density of fresh water [kg/m3]"),
    rhoSea=Setting(1026, float, "density of sea water [kg/m3]"),
    rhoAir=Setting(1.3, float, "density of air [kg/m3]"),
    rhoSnow=Setting(330, float, "density of snow [kg/m3]"),
    recip_rhoFresh=Setting(1 / 1000, float, "1 / rhoFresh [m3/kg]"),
    recip_rhoSea=Setting(1 / 1026, float, "1 / rhoSea [m3/kg]"),
    rhoIce2rhoSnow=Setting(900 / 330, float, "rhoIce / rhoSnow [m3/kg]"),
    rhoIce2rhoFresh=Setting(900 / 1000, float, "rhoIce / rhoFresh [m3/kg]"),
    rhoFresh2rhoSnow=Setting(1000 / 330, float, "rhoFresh / rhoSnow [m3/kg]"),
    # constants used in growth and solve4temp
    dryIceAlb=Setting(0.75, float, "albedo of dry ice"),
    dryIceAlb_south=Setting(
        0.75, float, "albedo of dry ice in the southern hemisphere"
    ),
    wetIceAlb=Setting(0.66, float, "albedo of wet ice"),
    wetIceAlb_south=Setting(
        0.66, float, "albedo of wet ice in the southern hemisphere"
    ),
    drySnowAlb=Setting(0.84, float, "albedo of dry snow"),
    drySnowAlb_south=Setting(
        0.84, float, "albedo of dry snow in the southern hemisphere"
    ),
    wetSnowAlb=Setting(0.7, float, "albedo of wet snow"),
    wetSnowAlb_south=Setting(
        0.7, float, "albedo of wet snow in the southern hemisphere"
    ),
    wetAlbTemp=Setting(
        0, float, "temperature [°C] above which the wet albedos are used"
    ),
    lhFusion=Setting(3.34e5, float, "latent heat of fusion [J/kg]"),
    lhEvap=Setting(2.5e6, float, "latent heat of evaporation [J/kg]"),
    lhSublim=Setting(3.34e5 + 2.5e6, float, "latent heat of sublimation [J/kg]"),
    cpAir=Setting(1005, float, "heat capacity of air [J/kg K]"),
    cpWater=Setting(3986, float, "heat capacity of water [J/kg K]"),
    stefBoltz=Setting(5.67e-8, float, "stefan boltzmann constant"),
    iceEmiss=Setting(0.95, float, "longwave ice emissivity"),
    snowEmiss=Setting(0.95, float, "longwave snow emissivity"),
    iceConduct=Setting(2.1656, float, "sea ice conductivity"),
    snowConduct=Setting(0.31, float, "snow conductivity"),
    hCut=Setting(0.15, float, "cut off snow thickness [m]"),
    shortwave=Setting(0.3, float, "shortwave ice penetration factor"),
    tempFrz=Setting(-1.96, float, "freezing temperature [°C]"),  # 0.0901
    dtempFrz_dS=Setting(
        0, float, "derivative of freezing temperature wrt salinity [°C/(g/kg)]"
    ),  # - 0.0575
    saltIce_ref=Setting(0, float, "reference salinity of sea ice [g/kg]"),
    saltOcn_ref=Setting(34.7, float, "reference salinity of the ocean [g/kg]"),
    minLWdown=Setting(60, float, "minimum downward longwave radiation"),
    maxTIce=Setting(30, float, "maximum ice temperature"),
    minTIce=Setting(-50, float, "minimum ice temperature"),
    minTAir=Setting(-50, float, "minimum air temperature"),
    dalton=Setting(
        0.00175, float, "dalton number/ sensible and latent heat transfer coefficient"
    ),
    Area_reg=Setting(
        0.15**2, float, "regularization value for the ice concentration"
    ),
    hIce_reg=Setting(0.10**2, float, "regularization value for the ice thickness"),
    celsius2K=Setting(273.15, float, "conversion from [K] to [°C]"),
    stantonNr=Setting(0.0056, float, "stanton number"),
    uStarBase=Setting(0.0125, float, "typical friction velocity beneath sea ice [m/s]"),
    McPheeTaperFac=Setting(12.5, float, "tapering factor at the ice bottom"),
    h0=Setting(0.5, float, "lead closing parameter"),
    recip_h0=Setting(1 / 0.5, float, "1 / h0"),
    h0_south=Setting(0.5, float, "lead closing parameter in the southern hemisphere"),
    recip_h0_south=Setting(1 / 0.5, float, "1 / h0_south"),
    # constants used in advection routines
    airTurnAngle=Setting(0, float, "turning angle of air-ice stress"),
    waterTurnAngle=Setting(0, float, "turning angle of water-ice stress"),
    sinWat=Setting(0, float, "sin of waterTurnAngle"),
    cosWat=Setting(1, float, "cos of waterTurnAngle"),
    wSpeedMin=Setting(1e-10, float, "minimum wind speed [m/s]"),
    hIce_min=Setting(1e-5, float, "'minimum' ice thickness [m]"),
    Area_min=Setting(1e-5, float, "'minimum' ice cover fraction"),
    airIceDrag=Setting(0.0012, float, "air-ice drag coefficient"),
    airIceDrag_south=Setting(
        0.0012, float, "air-ice drag coefficient in the southern hemisphere"
    ),
    waterIceDrag=Setting(0.0055, float, "water-ice drag coefficient"),
    waterIceDrag_south=Setting(
        0.0055, float, "water-ice drag coefficient in the southern hemisphere"
    ),
    cDragMin=Setting(0.25, float, "minimum of linear ice-ocean drag coefficient"),
    seaIceLoadFac=Setting(1, float, "factor to scale sea ice loading"),
    gravity=Setting(9.81, float, "gravitational acceleration"),
    PlasDefCoeff=Setting(2, float, "axes ratio of the elliptical yield curve"),
    deltaMin=Setting(2e-9, float, "minimum value of delta"),
    pressReplFac=Setting(1, int, "flag whether to use replacement pressure"),
    pStar=Setting(27.5e3, float, "sea ice strength parameter"),
    cStar=Setting(20, float, "sea ice strength parameter"),
    basalDragU0=Setting(5e-5, float, "basal drag parameter"),
    basalDragK1=Setting(8, float, "basal drag parameter"),
    basalDragK2=Setting(0, float, "basal drag parameter"),
    cBasalStar=Setting(20, float, "basal drag parameter"),
    tensileStrFac=Setting(0, float, "sea ice tensile strength factor"),
    CrMax=Setting(1e6, float, "advective flux parameter"),
    sideDragCoeff=Setting(0.001, float, "side drag coefficient"),
    sideDragU0=Setting(0.01, float, "side drag critical velocity [m/s]"),
)

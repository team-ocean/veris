
variables = dict(
    ##### sea ice #####
    
    hIceMean = None, # mean ice thickness /m
    hSnowMean = None, # mean snow thickness /m
    Area = None, # sea ice cover fraction
    TSurf = None, # ice/ snow surface temperature /K
    SeaIceMassC = None, # sea ice mass centered around c point /kg
    SeaIceMassU = None, # sea ice mass centered around u point /kg
    SeaIceMassV = None, # sea ice mass centered around v point /kg
    SeaIceStrength = None, # ice strength /N/m
    os_hIceMean = None, # overshoot of ice thickness from advection /m
    os_hSnowMean = None, # overshoot of snow thickness from advection /m/s
    AreaW = None, # sea ice cover fraction centered around u point
    AreaS = None, # sea ice cover fraction centered around v point
    uIce = None, # zonal ice velocity /m/s
    vIce = None, # meridional ice velocity /m/s
    sigma1 = None, # eigenvalue of the stress tensor /N/m2
    sigma2 = None, # eigenvalue of the stress tensor /N/m2
    sigma12 = None, # stress tensor component /N/m2
    WindForcingX = None, # zonal forcing on ice by wind stress /N
    WindForcingY = None, # meridional forcing on ice by wind stress /N
    recip_hIceMean = None, # 1 / hIceMean /1/m
    SeaIceLoad = None, # load of sea ice on ocean surface /kg/m2
    IcePenetSW = None, # shortwave radiation that penetrates through the ice /W/m2
    
    
    ##### ocean, wind, surface forcing #####
    
    uOcean = None, # zonal ocean surface velocity /m/s
    vOcean = None, # meridional ocean surface velocity /m/s
    theta = None, # ocean surface temperature /K
    ocSalt = None, # ocean surface salinity /g/kg
    Qnet = None, # net heat flux out of the ocean /W/m2
    OceanStressU = None, # zonal stress on ocean surface /N/m2
    OceanStressV = None, # meridional stress on ocean surface /N/m2
    saltflux = None, # salt flux into the ocean /m/s
    R_low = None, # sea floor depth (<0) /m
    ssh_an = None, # sea surface height anomaly /m
    # atmosphere
    Qsw = None, # surface shortwave heatflux (+ = upwards) /W/m2
    uWind = None, # zonal wind velocity /m/s
    vWind = None, # merdional wind velocity /m/s
    wSpeed = None, # total wind speed /m/s
    surfPress = None, # surface pressure /P
    SWdown = None, # downward shortwave radiation /W/m2
    LWdown = None, # downward longwave radiation /W/m2
    ATemp = None, # atmospheric temperature /K
    aqh = None, # atmospheric specific humidity /kg/kg
    precip = None, # precipitation rate (freshwater flux) /m/s
    snowfall = None, # snowfall rate /m/s
    evap = None, # evaporation rate over open ocean (freshwater flux, <0 increases salinity) /m/s
    runoff = None, # runoff into ocean /m/s
    EmPmR = None, # evaporation minus precipitation minus runoff /kg/m2 s

    
    ##### masks #####
    
    maskInC = None, # mask at c-points, used for open boundaries
    maskInU = None, # mask at u-points, used for open boundaries
    maskInV = None, # mask at v-points, used for open boundaries
    iceMask = None, # mask at c-points
    iceMaskU = None, # mask at u-points
    iceMaskV = None, # mask at v-points
    k1AtC = None, # 
    k2AtC = None, # 
    k1AtZ = None, # 
    k2AtZ = None, # 
    Fu = None, # u-component of form factor
    Fv = None, # v-component of form factor

    
    ##### grid #####
    
    fCori = None, # coriolis parameter /1/s
    dxC = None, # zonal spacing of cell centers across western cell wall /m
    dyC = None, # meridional spacing of cell centers across southern cell wall /m
    dxG = None, # zonal spacing of cell faces along southern cell wall /m
    dyG = None, # meridional spacing of cell faces along western cell wall /m
    dxU = None, # zonal spacing of u-points through cell center /m
    dyU = None, # meridional spacing of u-points through south-west corner of the cel /m
    dxV = None, # zonal spacing of v-points through south-west corner of the cell /m
    dyV = None, # meridional spacing of v-points through cell center /m
    recip_dxC = None, # 1 / dxC /1/m
    recip_dyC = None, # 1 / dyC /1/m
    recip_dxG = None, # 1 / dxG /1/m
    recip_dyG = None, # 1 / dyG /1/m
    recip_dxU = None, # 1 / dxU /1/m
    recip_dyU = None, # 1 / dyU /1/m
    recip_dxV = None, # 1 / dxV /1/m
    recip_dyV = None, # 1 / dyV /1/m
    rA = None, # Grid cell area centered around c-point /m2
    rAu = None, # Grid cell area centered around u-point /m2
    rAv = None, # Grid cell area centered around v-point /m2
    rAz = None, # Grid cell area centered around z-point /m2
    recip_rA = None, # 1 / rA /1/m2
    recip_rAu = None, # 1 / rAu /1/m2
    recip_rAv = None, # 1 / rAv /1/m2
    recip_rAz = None # 1 / rAz /1/m2
)

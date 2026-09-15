External ocean model
====================

``veris.setups.ocean.initialize_from_ocean`` allocates a fresh State,
Configuration and PhysicalConstants through ``veris.initialization.initialize``.
It accepts ``OceanGeometry`` directly; no existing Veris state is needed.
The geometry arrays include two halo cells at each horizontal boundary and
volume masks use the last vertical level as the surface. Horizontal dimensions
are inferred from the geometry and take precedence over settings overrides.
The metric equations retain the original periodic four-cell corner-area mean.

Supply ocean forcing and initial ice fields as full halo-inclusive arrays::

   from veris.setups.ocean import initialize_from_ocean

   state, conf, phys = initialize_from_ocean(
       geometry,
       dtype="float32",
       settings_overrides={"deltatDyn": 600.0},
       physical_overrides={"rhoIce": 920.0},
       state_overrides={"theta": ocean_temperature, "ocSalt": ocean_salinity},
   )

Unspecified state fields use the VARIABLES registry defaults. Geometry-derived
masks, metrics and surface temperature take precedence over state overrides;
``geometrySurfaceTemperature`` configures the initial surface temperature.
All arrays and physical coefficients use the selected precision. Geometry and
caller override dictionaries are left unchanged. Invalid shapes, nonfinite
geometry and nonpositive spacings or areas are rejected on the host.

The former ``veris.set_inits.set_inits`` entry point has been removed, and all
setup modules now live under ``veris.setups``.

.. autofunction:: veris.setups.ocean.initialize_from_ocean

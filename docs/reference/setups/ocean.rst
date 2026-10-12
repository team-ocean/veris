Prescribed ocean geometry
=========================

Use public ``OceanGeometry`` inside an ``Experiment`` for a complete standalone
lifecycle with physical arrays, initial ice and prescribed ocean/atmospheric
fields. :doc:`/quickstart/custom-ocean` provides a runnable example using only
public imports; it supports serial or explicit x/y meshes and requires no
external ocean package. Geometry is supplied in global physical dimensions;
initialization constructs its halos and derived metrics.

The compatibility adapter below accepts existing halo-inclusive geometry. Its
low-level allocation alone does not define a complete physical experiment.

``veris.grid.ocean.initialize_from_ocean`` allocates a fresh State,
Configuration and PhysicalConstants through ``veris.execution.initialization.initialize``.
It accepts ``OceanGeometry`` directly; no existing Veris state is needed.
The geometry arrays include two halo cells at each horizontal boundary and
volume masks use the last vertical level as the surface. Horizontal dimensions
are inferred from the geometry and take precedence over settings overrides.
The metric equations retain the original periodic four-cell corner-area mean.

Supply ocean forcing and initial ice fields as full halo-inclusive arrays::

   from veris.grid.ocean import initialize_from_ocean

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

The ocean adapter lives in ``veris.grid.ocean``; its input schema is
``veris.grid.geometry.OceanGeometry``. Existing imports from
``veris.setups.ocean`` remain available. The former ``veris.set_inits.set_inits``
entry point has been removed. See :doc:`/reference/package-hierarchy` for the
current module paths.

For a rollout, bind a physics kernel to the initialized model objects::

   from functools import partial
   from veris import step
   from veris.setups import island

   advance = partial(island.step, conf=conf, phys=phys, cooling=100.0)
   final = step(state, advance, 3)

Choose and supply the ice and forcing fields needed by that kernel; geometry
initialization alone does not define the experiment. The example uses the
coupled island physics sequence. See :doc:`../integration` for the common
rollout contract and AD.

See :func:`~veris.grid.ocean.initialize_from_ocean` in the
:doc:`Python API </api/initialization>` for its signature and docstring.

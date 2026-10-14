Prescribed ocean geometry
=========================

Use public ``OceanGeometry`` inside an ``Experiment`` for a complete standalone
run with physical arrays, initial ice and prescribed ocean/atmospheric fields.
Prescribed fields are inputs supplied by you, such as ocean temperature.
:doc:`/quickstart/custom-ocean` provides a runnable example using only public
imports. It supports a single device or explicit x/y device meshes and needs
no external ocean package. Supply geometry on the global physical grid.
Initialization adds halos (extra boundary cells) and calculates grid metrics
(geometric quantities such as areas).

The low-level adapter below accepts geometry that includes halos. Creating
its arrays alone does not define a complete physical experiment.

``veris.grid.ocean.initialize_from_ocean`` creates a fresh State, Configuration
and PhysicalConstants through ``veris.execution.initialization.initialize``.
State holds the model arrays. The adapter accepts ``OceanGeometry`` directly;
no existing Veris state is needed. The geometry arrays include two halo cells
at each horizontal boundary. Volume masks identify ocean and land through
depth and use the last vertical level as the surface. The geometry determines
horizontal dimensions, which take priority over settings overrides. The grid
metric equations keep the original periodic four-cell corner-area mean.

Supply ocean forcing (prescribed inputs) and initial ice fields as full arrays
that include halos::

   from veris.grid.ocean import initialize_from_ocean

   state, conf, phys = initialize_from_ocean(
       geometry,
       dtype="float32",
       settings_overrides={"deltatDyn": 600.0},
       physical_overrides={"rhoIce": 920.0},
       state_overrides={"theta": ocean_temperature, "ocSalt": ocean_salinity},
   )

State fields you do not supply use the VARIABLES registry defaults. Masks,
metrics and surface temperature calculated from geometry take priority over
state overrides. ``geometrySurfaceTemperature`` sets the initial surface
temperature. All arrays and physical coefficients use the selected precision.
Geometry and your override dictionaries stay unchanged. The host (the CPU
running Python) rejects invalid shapes, nonfinite geometry (NaN or infinity)
and spacings or areas that are not positive.

The ocean adapter lives in ``veris.grid.ocean``. Its input structure is
``veris.grid.geometry.OceanGeometry``. See :doc:`/reference/package-hierarchy`
for the modules responsible for each part.

To repeat timesteps, fix the initialized model objects as arguments of the
physics function::

   from functools import partial
   from veris import advance, step

   transition = partial(
       advance, configuration=conf, constants=phys,
       forcing={"Qnet": 100.0, "Qsw": 0.0}, physics="coupled",
   )
   final = step(state, transition, 3, has_aux=True)

Choose and supply the ice and forcing fields needed by that function.
Geometry initialization alone does not define the experiment. The example
uses the coupled island physics sequence, which combines ice motion with
freezing and melting. See :doc:`../integration` for the shared rules for
repeating timesteps and automatic differentiation (AD), which calculates how
results change with inputs.

See :func:`~veris.grid.ocean.initialize_from_ocean` in the
:doc:`Python API </api/initialization>` for its signature and docstring.

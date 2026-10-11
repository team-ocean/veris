Experiment definitions
======================

``Grid``, ``Execution``, ``Forcing`` and ``Experiment`` are immutable host
definitions exported directly from ``veris``. They validate inputs and resolve
the existing registry-backed settings and physical constants, without allocating
model arrays or selecting a JAX backend::

   from veris import Execution, Experiment, Forcing, Grid

   experiment = Experiment(
       grid=Grid(nx=6, ny=8),
       settings={"dtype": "float32", "deltatDyn": 600, "deltatTherm": 600},
       constants={"rhoIce": 910},
       initial_conditions={"hIceMean": 1.0, "hSnowMean": 0.05, "Area": 0.8},
       forcing=Forcing(
           constant={"uWind": 5.0, "ocSalt": 34.7},
           series={"Qnet": [80.0, 100.0, 120.0]},
       ),
       execution=Execution(backend="cpu"),
       steps=3,
   )

Runnable initialization is available through ``initialize_experiment``. For
example, the coupled island preset supplies consistent geometry and prescribed
atmospheric fields::

   from veris import Experiment, Grid, initialize_experiment

   experiment = Experiment.builtin(
       "island", grid=Grid(6, 8), settings={"dtype": "float32"}, steps=3,
   )
   state, configuration, constants = initialize_experiment(experiment)

The public :doc:`simulation` facade initializes and executes successive runs
with absolute forcing position, model time, scheduled output and complete serial
restart. Definition and initialization
are host operations; enable JAX x64 explicitly before requesting float64 allocation.

Declarative files
-----------------

``Experiment.from_file(path, overrides=...)`` reads version 1 JSON. The version
may be omitted; unsupported versions and duplicate or unknown keys fail clearly.
Preset defaults, then file values, then explicit Python overrides determine the
resolved definition. Mappings merge by key; arrays and lists replace whole
values. A runnable definition can be written as:

.. code-block:: json

   {
     "schema_version": 1,
     "preset": "island",
     "grid": {"nx": 6, "ny": 8},
     "settings": {"dtype": "float32", "nEVPsteps": 2},
     "scenario": {"islandWindSpeed": 4.0},
     "forcing": {"series": {"Qnet": [80.0, 100.0, 120.0]}},
     "steps": 3
   }

``experiment.to_dict()`` returns independent JSON-compatible data with resolved
numerical and physical defaults. Derived scalar values are reconstructed from
those inputs. Output durations use ``{"microseconds": integer}``; human-written
files can also use ``{"seconds": number}`` when exactly representable in
microseconds. Gregorian origins use timezone-free ISO dates. Fixed-calendar
origins use objects with integer year, month, day and optional time components.

Dimensions, units and precedence
--------------------------------

``Grid.nx`` and ``Grid.ny`` count **global physical cells**, excluding all
halos. Constant and initial-condition arrays are scalars or ``(nx, ny)`` fields;
forcing series have shape ``(time,)`` or ``(time, nx, ny)``. All staggered
center, face and corner fields use these same extents with ``(x, y)`` ordering.
The numerical State uses two halo cells at every partition boundary.
For example, ``Grid(12, 8)`` on ``partitions=(3, 2)`` has local physical
extents ``(4, 4)`` and packed storage shape ``(24, 16)``.

Settings and constants use their existing metadata defaults and units from
:doc:`/reference/settings` and :doc:`/reference/physical-constants`. Supplied
overrides take precedence over registry defaults. Constants inherit the model
``dtype``; a conflicting constants dtype is rejected. Use ``Grid`` for dimensions
and ``Execution`` for sharding; those controls cannot also appear in ``settings``.
Resolved static objects are available as ``experiment.configuration`` and
``experiment.physical_constants``. Their numerical grid extents remain local.

Initial conditions and forcing use :doc:`/reference/variables` units:
thickness in meters, temperatures in kelvin, concentration between zero and one,
and heat fluxes in W/m² with ``Qnet`` and ``Qsw`` positive upward. Initial
conditions accept evolved ice fields; derived loads, geometry and metrics are
not independent initial conditions. Forcing accepts prescribed atmospheric and
ocean fields, rather than evolved ice. Arrays and mappings are defensively copied
and immutable in a definition.

Accepted field names are generated from the authoritative input sets below.
Inputs must contain finite real values. Ice/snow thickness, wind speed and
salinity are nonnegative; temperatures in kelvin are positive; concentration
``Area`` lies in [0, 1]. Initial conditions and constant forcing are scalar or
physical 2D arrays; series use the leading time dimension described above.
``load_fields`` accepts the union of these names from instantaneous serial
records. Geometry, masks and independent metrics require initialization.

.. exec::

   from veris.experiment import INITIAL_FIELDS
   from veris.forcing import FORCING_FIELDS
   from veris.variables import VARIABLES
   for category, names in (("Initial conditions", INITIAL_FIELDS),
                           ("Prescribed forcing", FORCING_FIELDS)):
       print(f".. list-table:: {category}")
       print("   :header-rows: 1\n")
       print("   * - Name\n     - Units\n     - Description")
       for name in sorted(names):
           metadata = VARIABLES[name]
           print(f"   * - ``{name}``\n     - {metadata.units}")
           print(f"     - {metadata.description}")
       print("")

User-supplied ``OceanGeometry`` uses physical arrays without halos. ``maskT``,
``maskU`` and ``maskV`` have shape ``(nx, ny, z)`` with a nonempty last dimension
and surface masks at the last level. Masks lie in [0, 1]; zero means dry.
``dxt``/``dxu`` are positive x-spacing vectors of length nx;
``dyt``/``dyu`` are positive y-spacing vectors of length ny, all in meters.
``area_t``, ``area_u`` and ``area_v`` are positive ``(nx, ny)`` areas in m².
``ht`` is the existing signed bottom elevation in meters; ``coriolis_t`` is
frequency in s⁻¹. Initialization preserves the established ocean metric
equations, closes blocked velocity faces and computes load and regularized
inverse thickness. Fractional wet masks retain supplied cell-mean ice amounts.
Explicit ``TSurf`` initial conditions override geometry's default temperature.
Geometry is required for a custom runnable experiment: allocation defaults alone
do not define one.

Builtins are ``island``, ``dynamics`` and ``growth``. Their scenario controls use
the maintained setup registries. ``Grid`` supplies dimensions; scenario does not
also accept nx, ny, steps or dtype. Island initialization retains zero initial
``Qnet``/``Qsw`` coupling outputs; prescribed heat is queued for each timestep.
Growth retains its recursive flux policy unless heat is explicitly prescribed.
Explicit winds derive ``wSpeed`` unless it is separately supplied.

Series override constants of the same field name. ``forcing.at(0)`` selects the
first step's prescribed inputs; it does not mutate a cursor. A missing field
retains its current State value, allowing recursive heat-flux policies.
Spatial shapes, series lengths and configured duration are validated before
execution. A requested position beyond the series raises an exhaustion error.

Physics modes are ``coupled``, ``dynamics`` and ``thermodynamics``. Coupled mode
requires equal dynamics and thermodynamic timesteps. Model time uses the active
timestep and microsecond host precision; unsupported sub-microsecond values and
clock overflow fail clearly. ``backend=None`` honors the caller's JAX selection;
``cpu`` and ``gpu`` are resource requests, with enumeration left to initialization.

Explicit meshes
---------------

All builtins and custom geometry/initial/forcing fields use the same public
sharded initializer. For example, start a fresh Python process with
``JAX_NUM_CPU_DEVICES=2`` before importing JAX, then run::

   import jax
   from veris import Execution, Experiment, Grid, Simulation

   mesh = jax.make_mesh((2, 1), ("x", "y"), devices=jax.devices("cpu"))
   experiment = Experiment.builtin(
       "island", grid=Grid(12, 8), steps=2,
       settings={"dtype": "float32", "nEVPsteps": 2},
       execution=Execution(backend="cpu", partitions=(2, 1)),
   )
   with Simulation.initialize(experiment, mesh=mesh) as simulation:
       result = simulation.run()
       simulation.snapshot("sharded-final.nc", variables=("hIceMean", "Area"))

Input arrays retain global physical shapes; only addressable packed State
regions are initialized on each process. Spatial series keep an unpartitioned
time axis. The facade borrows the mesh. For multiple processes, initialize JAX
distributed services before enumerating devices and retain caller ownership;
see :doc:`/reference/setups/parallel-dynamics`. Complete distributed restart is
unsupported. Local CPU sharding validation does not substitute for GPU or
actual multihost verification.

Definition construction is host-only. The pure numerical API consists of
``veris.advance`` for one timestep and :func:`veris.step` for a rollout;
see :doc:`/reference/integration` for JIT and differentiation.
Initial State and supplied JAX forcing arrays are differentiable; static
definitions, settings, constants and host validation are outside that path.

.. autoclass:: veris.experiment.Grid
   :members:

.. autoclass:: veris.experiment.Execution
   :members:

.. autoclass:: veris.experiment.Experiment
   :members:

.. autoclass:: veris.forcing.Forcing
   :members:

.. autofunction:: veris.model_initialization.initialize_experiment

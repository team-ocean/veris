Experiment definitions
======================

``Grid``, ``Execution``, ``Forcing`` and ``Experiment`` are definitions you can
import directly from ``veris``. They run in Python on the host, the computer
that controls the calculation, and cannot change in place. They check inputs
and combine supplied values with defaults from the model's named tables of
settings and physical constants. They do not create model arrays or choose a
JAX backend, the CPU or GPU system that will run the calculation::

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

Use :func:`veris.initialize_experiment` to create model arrays and the
:doc:`simulation` interface to control runs from Python. :doc:`initialization`
describes preset providers, which supply starting fields, and lower-level array
creation. These operations run on the host. Enable JAX x64 before requesting
float64 arrays (see :doc:`/reference/settings`).

Authored and resolved controls
------------------------------

``settings``, ``constants``, ``scenario`` and ``forcing`` keep the inputs you
wrote, without allowing changes in place. ``configuration``,
``physical_constants``, ``resolved_settings``, ``resolved_scenario`` and
``resolved_forcing`` expose the checked values that the model will use after
applying defaults and overrides. Their priority is shared registry defaults <
setup preset/profile < experiment configuration < explicit Python/CLI overrides.
Values later in this order win. Veris keeps track of which values you supplied
and which came from a preset, so it can update inherited defaults when the
experiment changes::

   updated = Experiment.builtin("island").with_overrides({
       "scenario": {"islandTimeStep": 300, "islandCooling": 25},
       "settings": {"dtype": "float64"},
   })

Mappings, or dictionaries, merge by key. Arrays, lists, single values and
allowed ``None`` values replace whole values. Explicit numerical settings or
forcing, the prescribed external inputs, continue to override scenario defaults.
Changing numerical precision re-evaluates inherited defaults from their original
values. A rounded value that you supplied explicitly keeps that value.

Declarative files
-----------------

``Experiment.from_file(path, overrides=...)`` reads version 1 or 2 JSON data
files. If the version is omitted, it means 1. Unsupported versions and duplicate
or unknown keys produce clear errors. The definition starts with preset
defaults, then applies file values and finally explicit Python overrides, using
the merge rules above. You can write a runnable definition as:

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

Output durations use ``{"microseconds": integer}``. Files you write by hand can
also use ``{"seconds": number}`` if the duration is an exact number of
microseconds. A calendar origin is its starting date and time. Gregorian origins
use ISO-format dates without a timezone. Fixed-calendar origins use objects
with integer year, month, day and optional time components. See
:ref:`experiment-export-modes` for exports of supplied inputs and resolved values.

Dimensions, units and precedence
--------------------------------

``Grid.nx`` and ``Grid.ny`` count **global physical cells**, the cells in the
full modeled area, excluding all halos. Halos are extra cells around each grid
piece that hold boundary or neighboring values. Constant and initial-condition
arrays are single values or ``(nx, ny)`` fields. Forcing series have shape
``(time,)`` or ``(time, nx, ny)``. A staggered grid places different fields at
cell centers, faces and corners. All these fields use the same extents with
``(x, y)`` ordering. The numerical State uses two halo cells at every partition
boundary, where a partition is a grid piece. For example, ``Grid(12, 8)`` on
``partitions=(3, 2)`` has local physical extents ``(4, 4)`` and packed storage
shape ``(24, 16)``. Packed storage includes every partition's halos.

Settings and constants use the defaults and units listed in
:doc:`/reference/settings` and :doc:`/reference/physical-constants`. Supplied
overrides take precedence over table defaults. Constants inherit the model
``dtype``, its numerical data type. A conflicting constants dtype is rejected.
Use ``Grid`` for dimensions and ``Execution`` for sharding, which splits arrays
across devices. Those controls cannot also appear in ``settings``. The resolved
objects are available as ``experiment.configuration`` and
``experiment.physical_constants`` and stay fixed during numerical execution.
Their numerical grid extents remain local to each partition.

Initial conditions and forcing use the units in :doc:`/reference/variables`:
thickness in meters, temperatures in kelvin, the absolute temperature scale,
concentration between zero and one, and heat fluxes in W/m² with ``Qnet`` and
``Qsw`` positive upward. Concentration is the fraction of a cell covered by ice.
A heat flux is heat transferred per unit area per unit time. Initial conditions
accept ice fields that evolve during a run. Loads computed from other fields,
grid geometry, distances and areas are not independent initial conditions.
Forcing accepts prescribed atmospheric and ocean fields rather than evolving
ice fields. Definitions copy input arrays and mappings and prevent changes in
place, so editing the original inputs does not change the definition.

The accepted field names below are generated from the input sets used by the
code. Inputs must contain finite real values, with no infinities or undefined
numerical values. Ice/snow thickness, wind speed and salinity are nonnegative.
Temperatures in kelvin are positive, and concentration ``Area`` lies in [0, 1].
Initial conditions and constant forcing are single values or physical 2D arrays.
Series put time in the first dimension, as described above. ``load_fields``
accepts any of these names from serial records sampled at a single instant.
Geometry, masks and independent grid distances and areas require initialization.

.. exec::

   from veris.model.experiment import INITIAL_FIELDS
   from veris.model.forcing import FORCING_FIELDS
   from veris.model.variables import VARIABLES
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
``maskU`` and ``maskV`` have shape ``(nx, ny, z)``. The last dimension must be
nonempty, and the last level holds the surface masks. Masks mark wet cells with
values in [0, 1]; zero means dry. ``dxt``/``dxu`` are positive x-spacing vectors
of length nx. ``dyt``/``dyu`` are positive y-spacing vectors of length ny.
All spacings are in meters. ``area_t``, ``area_u`` and ``area_v`` are positive
``(nx, ny)`` areas in m². ``ht`` is the existing signed bottom elevation in
meters; ``coriolis_t`` is a frequency in s⁻¹. Initialization uses the established
equations for ocean grid distances and areas, closes blocked velocity faces,
and computes load and inverse thickness with regularization to handle thin ice.
Fractional wet masks retain supplied ice amounts averaged over each cell.
Explicit ``TSurf`` initial conditions override the grid's default temperature.
A custom runnable experiment requires geometry. Creating arrays from defaults
alone does not define one.

Builtins are ``island``, ``dynamics`` and ``growth``. Their scenario controls use
the maintained setup tables. ``Grid`` supplies dimensions; scenario does not
also accept nx, ny, steps or dtype. Island initialization keeps initial
``Qnet``/``Qsw`` coupling outputs at zero and queues prescribed heat for each
timestep. Growth feeds computed heat fluxes into later steps unless heat is
explicitly prescribed. Explicit winds determine ``wSpeed`` unless it is
supplied separately.

Series override constants with the same field name. ``forcing.at(0)`` selects
the first step's prescribed inputs without advancing a position tracker through
the series. A missing field keeps its current State value, allowing computed heat
fluxes to feed into later steps. Veris checks spatial shapes, series lengths
and configured duration before execution. Requesting a position beyond the
series raises an exhaustion error.

Physics modes are ``coupled``, ``dynamics`` and ``thermodynamics``. Dynamics
computes ice motion and transport; thermodynamics computes freezing and melting.
Coupled mode combines both and requires equal dynamics and thermodynamic
timesteps. Model time uses the active timestep and microsecond precision on
the host. Unsupported sub-microsecond values and clock overflow produce clear
errors. ``backend=None`` honors the caller's JAX selection. ``cpu`` and ``gpu``
request resources; initialization determines which devices are available.

Explicit meshes
---------------

All builtins and custom geometry, initial fields and forcing use the same public
initializer for arrays split across devices. For example, start a fresh Python
process with ``JAX_NUM_CPU_DEVICES=2`` before importing JAX, then run::

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

Input arrays keep their full-grid physical shapes. Each process initializes only
the packed State regions it can access. Spatial series split only the spatial
dimensions; their time axis stays whole. The caller keeps ownership of the mesh.
For multiple processes, initialize JAX distributed services before listing
devices and keep responsibility for closing those services. See
:doc:`/reference/setups/parallel-dynamics`. Complete distributed restart is
unsupported. Checks of local CPU sharding do not establish correctness on GPUs
or across multiple computers; those need their own verification.

Definitions are constructed only on the host. The pure numerical API consists
of ``veris.advance`` for one timestep and :func:`veris.step` to repeat timesteps
without Python side effects. See :doc:`/reference/integration` for JIT, which
compiles functions for execution, and differentiation, which measures how
results change with inputs. JAX can compute derivatives with respect to initial
State and supplied JAX forcing arrays. Fixed definitions, settings, constants
and host input checks stay outside that path.

.. autoclass:: veris.model.experiment.Grid
   :members:

.. autoclass:: veris.model.experiment.Execution
   :members:

.. autoclass:: veris.model.experiment.Experiment
   :members:

.. autoclass:: veris.model.forcing.Forcing
   :members:

.. _experiment-export-modes:

Export modes
------------

``experiment.to_dict()`` defaults to ``mode="resolved"`` and fixes all values
that the model will use, including the profile identity. ``mode="authored"``
keeps only the caller's supplied inputs, leaving omitted groups and values
omitted. Reading authored data and then changing dtype/scenario recomputes
inherited defaults. Resolved data keeps the deliberately fixed literal values.
Both return independent data suitable for JSON. Derived single values are
reconstructed from the stored inputs. Returned arrays and containers do not
share storage with the definition. Schema version 1 remains supported;
profile/distributed/all-device declarations use version 2. Use ``veris inspect``
to inspect or export from the command line without selecting devices.

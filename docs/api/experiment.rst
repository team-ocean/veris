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

The centralized initialization, execution and restart interfaces are being
implemented incrementally. For a currently runnable example, use
:doc:`/quickstart/user-guide`.

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

Definition construction is host-only. The pure numerical API remains
:func:`veris.step`; see :doc:`/reference/integration` for JIT and differentiation.
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

Prescribed ocean geometry and custom fields
===========================================

To define a custom experiment, supply geometry (the grid's shape and sizes),
masks (values marking ocean and land), starting model values and prescribed
fields (inputs supplied to the model). A grid divides the modeled area into
cells. You can do this without importing internal modules or editing Veris
source. This small CPU (computer processor) example uses arrays (tables of
numbers) for physical cells in ``(x, y)`` order, with the x direction first::

   import numpy as np
   from veris import Execution, Experiment, Forcing, Grid, OceanGeometry, Simulation

   grid = Grid(6, 8)
   shape = (grid.nx, grid.ny)
   wet = np.ones(shape, dtype=np.float32)
   wet[2:4, 3:5] = 0
   dx = np.full(grid.nx, 8000, dtype=np.float32)
   dy = np.full(grid.ny, 9000, dtype=np.float32)
   area = dx[:, None] * dy[None, :]
   geometry = OceanGeometry(
       maskT=wet[..., None],
       maskU=(wet * np.roll(wet, 1, axis=0))[..., None],
       maskV=(wet * np.roll(wet, 1, axis=1))[..., None],
       ht=np.full(shape, -100, dtype=np.float32),
       coriolis_t=np.full(shape, 1e-4, dtype=np.float32),
       dxt=dx, dxu=dx, dyt=dy, dyu=dy,
       area_t=area, area_u=area, area_v=area,
   )
   experiment = Experiment(
       grid=grid, geometry=geometry, physics="coupled", steps=3,
       settings={"dtype": "float32", "deltatDyn": 600,
                 "deltatTherm": 600, "nEVPsteps": 2},
       initial_conditions={"hIceMean": 1.0, "hSnowMean": 0.05,
                           "Area": 0.8, "TSurf": 260},
       forcing=Forcing(
           constant={"uWind": 4, "vWind": 0, "theta": 271.35,
                     "ocSalt": 34.7, "ATemp": 260, "LWdown": 260,
                     "aqh": 0.001, "Qsw": 0},
           series={"Qnet": [80, 100, 120]},
       ),
       execution=Execution(backend="cpu"),
   )
   with Simulation.initialize(experiment) as simulation:
       result = simulation.run()
       simulation.snapshot("custom-final.nc",
                           variables=("hIceMean", "Area", "iceMask"))
   assert result.completed_steps == 3

``maskT``, ``maskU`` and ``maskV`` mark ocean at cell centers, west faces (western
edges) and south faces (southern edges). They have shape ``(nx, ny, z)``; their
last level is the surface. Zero means dry. Fractional masks keep the model's
existing weighting of calculations. The model sets initial ice amounts to zero
on dry cells and ice velocities to zero on blocked faces. Spacing vectors list
distances in meters; cell/face areas use m². Bottom elevation ``ht`` uses meters.
Coriolis frequency, which describes the effect of Earth's rotation, uses s⁻¹.
Supply positive spacings/areas and finite masks (no infinite or undefined values)
in [0, 1]. Initialization
calculates corner metrics (grid measurements at cell corners), reciprocals
(one divided by a value), ice load (mass per area), inverse thickness and wind
magnitude (speed). It also prepares field-specific boundary halos, extra cells
at grid edges set according to each field's boundary rules.

All ice and prescribed spatial fields are scalars (single values) or ``(nx, ny)``
arrays. Time-varying forcing, or supplied inputs that change with time, has shape
``(time,)`` or ``(time, nx, ny)``. The model selects values using the total number
of steps already completed, including earlier runs. Series override constants
with the same name. Thickness is averaged over the whole grid cell and measured
in meters. ``Area`` is concentration, the fraction covered by ice, in [0, 1].
Temperatures are kelvin and salinity is g kg⁻¹. Heat fluxes (heat transferred
per area per time) are W/m², positive upward. Coupled execution, which combines
ice motion and growth, requires equal dynamics/thermodynamic timesteps (intervals
of simulated time). Supported names and more input checks are in
:doc:`/api/experiment`.

``enable_cyclic_y=False`` closes the north and south (meridional) boundaries.
Initialization and execution apply wall rules to staggered fields, which store
values at different grid locations. This includes the physical north-wall
shear traction (force per area along the wall) stored in its first halo column.
With an explicit mesh (a group of devices), internal boundaries between grid
partitions exchange values between neighboring partitions.

To save a definition you can load later, write ``experiment.to_dict()`` as JSON,
a text format, and load it with ``Experiment.from_file``. JSON stores arrays as
lists. A complete restart file stores numerical inputs with their types intact
and preserves the evolved State (the arrays holding the model's current values).
For very large inputs, build the Python definition from your data loader using
the physical shapes above. Use ``Simulation.load_fields`` to load selected fields
from netCDF, a file format for arrays. Changing geometry requires a new
initialization.

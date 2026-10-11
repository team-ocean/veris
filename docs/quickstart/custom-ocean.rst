Prescribed ocean geometry and custom fields
===========================================

A custom experiment requires geometry, masks, initial conditions and prescribed
fields. It can be defined without importing internal modules or editing Veris
source. This small CPU example uses physical arrays in ``(x, y)`` order::

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

``maskT``, ``maskU`` and ``maskV`` are center, west-face and south-face ocean
masks with shape ``(nx, ny, z)``; their last level is the surface. Zero means
dry; fractional masks retain the established numerical weighting. The model
zeros initial ice amounts on dry cells and ice velocities on blocked faces.
Spacing vectors are in meters, cell/face areas in m², bottom elevation ``ht``
in meters and Coriolis frequency in s⁻¹. Supply positive spacings/areas and
finite masks in [0, 1]. Initialization derives corner metrics, reciprocals,
ice load, inverse thickness, wind magnitude and field-specific boundary halos.

All ice and prescribed spatial fields are scalars or ``(nx, ny)`` arrays.
Time-varying forcing is ``(time,)`` or ``(time, nx, ny)`` and is indexed by the
absolute completed-step position. Series override constants with the same name.
Thickness is grid-cell mean meters, ``Area`` is concentration in [0, 1],
temperatures are kelvin, salinity is g kg⁻¹ and heat fluxes are W/m², positive
upward. Coupled execution requires equal dynamics/thermodynamic timesteps.
Supported names and additional validation are in :doc:`/api/experiment`.

``enable_cyclic_y=False`` selects closed meridional boundaries. Initialization
and execution honor staggered wall fields, including the physical north-wall
shear traction stored in its first halo column. Internal partition boundaries
remain exchanges when an explicit mesh is used.

To save a portable definition, serialize ``experiment.to_dict()`` as JSON and
load it with ``Experiment.from_file``. JSON stores arrays as lists; complete
restart stores typed numerical inputs directly and preserves the evolved State.
For very large inputs, construct the Python definition from your data loader
using the physical shapes above. Selected netCDF input is available through
``Simulation.load_fields``; geometry changes require a new initialization.

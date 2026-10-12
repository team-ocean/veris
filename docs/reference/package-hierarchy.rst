Scientific package hierarchy
============================

Veris groups its implementations by scientific and functional responsibility.
The public workflow continues to use ``Experiment``, ``Simulation``, ``advance``
and ``step`` from ``veris``, with output interfaces from ``veris.io``. Use the
canonical modules below when importing individual kernels or host helpers.

.. list-table:: Package responsibilities
   :header-rows: 1
   :widths: 25 75

   * - Package
     - Responsibility
   * - ``veris.model``
     - Experiment definitions, forcing, state and array contracts, numerical
       settings, physical constants, metadata and diagnostic fields.
   * - ``veris.grid``
     - Ocean geometry and initialization adapter, staggered interpolation,
       halos, global reductions and regional inputs.
   * - ``veris.physics``
     - Dynamics and rheology, transport and ridging, thermodynamic growth and
       atmospheric bulk heat fluxes.
   * - ``veris.numerics``
     - Momentum and surface-temperature solvers and differentiation primitives.
   * - ``veris.execution``
     - Allocation, experiment initialization, pure timesteps and rollouts,
       timed output rollouts and the host simulation lifecycle.
   * - ``veris.io``
     - Calendars, output schedules, records, distributed collection, selected
       input, snapshots and complete restart.
   * - ``veris.setups``
     - Island, dynamics, growth and parallel examples and preset controls.

Implementation tree
-------------------

Every directory shown is a Python package with an ``__init__.py``. Those package
initializers are omitted below except for the public facade at the root. Previous
root modules remain alongside these packages as compatibility imports; the next
section lists every migrated module::

   veris/
     __init__.py
     model/
       arrays.py
       configuration.py
       diagnostics.py
       experiment.py
       experiment_file.py
       forcing.py
       metadata.py
       physical_constants.py
       types.py
       variables.py
     grid/
       averaging.py
       geometry.py
       halos.py
       ocean.py
       reductions.py
       regions.py
     physics/
       dynamics/
         area_mass.py
         driver.py
         ocean_stress.py
         rheology.py
       transport/
         advection.py
         cleanup.py
       thermodynamics/
         growth.py
         heat_flux_cesm.py
         heat_flux_mitgcm.py
     numerics/
       autodiff.py
       evp.py
       freedrift.py
       momentum.py
       surface_temperature.py
     execution/
       advance.py
       experiment_initialization.py
       initialization.py
       rollout.py
       rollout_output.py
       simulation.py
     io/
       calendar.py
       cli.py
       configuration.py
       distributed.py
       guard.py
       lifecycle.py
       output.py
       restart.py
       restart_arrays.py
       scan.py
       schedule.py
       storage.py
     setups/
       island.py
       ocean.py                    # compatibility import for grid.ocean
       presets.py
       run_dyn.py
       run_growth.py
       run_parallel.py
       run_parallel.slurm
     settings.py                   # public settings compatibility import
     _ad.py
     _metadata.py
     _typing.py
     advection.py
     area_mass.py
     averaging.py
     clean_up.py
     configuration.py
     diagnostics.py
     dynamics.py
     dynamics_routines.py
     dynsolver.py
     evp_solver.py
     experiment.py
     experiment_file.py
     fill_overlap.py
     forcing.py
     freedrift_solver.py
     global_sum.py
     growth.py
     heat_flux_CESM.py
     heat_flux_MITgcm.py
     initialization.py
     integration.py
     integration_output.py
     model_initialization.py
     model_physics.py
     model_regions.py
     ocean_stress.py
     physical_constants.py
     presets.py
     restart.py
     restart_arrays.py
     simulation.py
     simulation_io.py
     solve4temp.py
     variables.py
     py.typed
     _version.py
     _version.pyi

``model.arrays`` owns ``ArrayInput`` and ``MaskInput``. ``model.types`` owns the
array-only State, Parameter metadata and compiled-kernel contracts.
``grid.geometry`` owns the immutable OceanGeometry input schema; the ocean
adapter in ``grid.ocean`` validates geometry and allocates model fields.
``model.types`` and the previous ``_typing`` import retain their geometry and
array aliases. Geometry construction and model allocation retain their existing
shape conventions; see :doc:`/api/experiment` and :doc:`/api/initialization`.

Migration paths
---------------

Previous module paths remain available and expose the same objects as their
canonical modules. Public function names and call signatures are unchanged.
New imports can follow this complete module mapping:

.. list-table:: Module imports
   :header-rows: 1
   :widths: 40 60

   * - Previous module
     - Canonical module
   * - ``veris._typing``
     - ``veris.model.types``
   * - ``veris._metadata``
     - ``veris.model.metadata``
   * - ``veris.configuration``
     - ``veris.model.configuration``
   * - ``veris.physical_constants``
     - ``veris.model.physical_constants``
   * - ``veris.variables``
     - ``veris.model.variables``
   * - ``veris.diagnostics``
     - ``veris.model.diagnostics``
   * - ``veris.experiment``
     - ``veris.model.experiment``
   * - ``veris.experiment_file``
     - ``veris.model.experiment_file``
   * - ``veris.forcing``
     - ``veris.model.forcing``
   * - ``veris.fill_overlap``
     - ``veris.grid.halos``
   * - ``veris.averaging``
     - ``veris.grid.averaging``
   * - ``veris.global_sum``
     - ``veris.grid.reductions``
   * - ``veris.model_regions``
     - ``veris.grid.regions``
   * - ``veris.setups.ocean``
     - ``veris.grid.ocean``
   * - ``veris.dynamics``
     - ``veris.physics.dynamics.driver``
   * - ``veris.dynamics_routines``
     - ``veris.physics.dynamics.rheology``
   * - ``veris.area_mass``
     - ``veris.physics.dynamics.area_mass``
   * - ``veris.ocean_stress``
     - ``veris.physics.dynamics.ocean_stress``
   * - ``veris.advection``
     - ``veris.physics.transport.advection``
   * - ``veris.clean_up``
     - ``veris.physics.transport.cleanup``
   * - ``veris.growth``
     - ``veris.physics.thermodynamics.growth``
   * - ``veris.heat_flux_CESM``
     - ``veris.physics.thermodynamics.heat_flux_cesm``
   * - ``veris.heat_flux_MITgcm``
     - ``veris.physics.thermodynamics.heat_flux_mitgcm``
   * - ``veris._ad``
     - ``veris.numerics.autodiff``
   * - ``veris.dynsolver``
     - ``veris.numerics.momentum``
   * - ``veris.evp_solver``
     - ``veris.numerics.evp``
   * - ``veris.freedrift_solver``
     - ``veris.numerics.freedrift``
   * - ``veris.solve4temp``
     - ``veris.numerics.surface_temperature``
   * - ``veris.initialization``
     - ``veris.execution.initialization``
   * - ``veris.model_initialization``
     - ``veris.execution.experiment_initialization``
   * - ``veris.model_physics``
     - ``veris.execution.advance``
   * - ``veris.integration``
     - ``veris.execution.rollout``
   * - ``veris.integration_output``
     - ``veris.execution.rollout_output``
   * - ``veris.simulation``
     - ``veris.execution.simulation``
   * - ``veris.simulation_io``
     - ``veris.io.lifecycle``
   * - ``veris.restart``
     - ``veris.io.restart``
   * - ``veris.restart_arrays``
     - ``veris.io.restart_arrays``
   * - ``veris.presets``
     - ``veris.setups.presets``

Several shared interfaces have more specific canonical owners:

.. list-table:: Shared interfaces
   :header-rows: 1
   :widths: 40 60

   * - Previous import
     - Canonical import
   * - ``veris._typing.ArrayInput``
     - ``veris.model.arrays.ArrayInput``
   * - ``veris._typing.MaskInput``
     - ``veris.model.arrays.MaskInput``
   * - ``veris._typing.OceanGeometry``
     - ``veris.grid.geometry.OceanGeometry``
   * - ``veris.setups.ocean.OceanGeometry``
     - ``veris.grid.geometry.OceanGeometry``
   * - ``veris.setups.run_parallel.remove_halos``
     - ``veris.grid.halos.remove_halos``

For example, specialized imports can use::

   from veris.grid.halos import fill_state_overlap, remove_halos
   from veris.model.configuration import Configuration
   from veris.model.physical_constants import PhysicalConstants
   from veris.model.types import State
   from veris.numerics.evp import evp_solver
   from veris.physics.thermodynamics.growth import Growth

The public ``OceanGeometry`` import from ``veris`` remains available.
``veris.settings`` continues to export ``Configuration``, ``SETTINGS`` and
``Parameter``. Existing ``veris.io`` exports and setup functions remain available.
Class introspection reports the canonical owner; previous serialized Python
import paths continue to resolve.

Command-line entry points are unchanged::

   python -m veris.setups.run_dyn --help
   python -m veris.setups.run_growth --help
   python -m veris.setups.run_parallel --help
   sbatch veris/setups/run_parallel.slurm

The reorganization preserves numerical equations, field ordering, masks,
precision controls and JAX transformation contracts. See
:doc:`/quickstart/migration` for the lifecycle migration and
:doc:`automatic-differentiation` for derivative conventions. Historical design
plans and reference-capture manifests retain the paths of their recorded source
revisions.

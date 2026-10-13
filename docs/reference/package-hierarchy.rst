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
   * - ``veris.cli``
     - Central Click commands, setup template discovery/copying and file execution.

Implementation tree
-------------------

Every directory shown is a Python package with an ``__init__.py``, except the
CLI template directories, which are packaged data. Package initializers are
omitted below except for the public facade at the root::

   veris/
     __init__.py
     cli/
       __main__.py
       run.py
       setups.py
       templates/
         dynamics/setup.py
         growth/setup.py
         island/setup.py
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
       presets.py
       run_dyn.py
       run_growth.py
       run_parallel.py
       run_parallel.slurm
     py.typed
     _version.py
     _version.pyi

``model.arrays`` owns ``ArrayInput`` and ``MaskInput``. ``model.types`` owns the
array-only State, Parameter metadata and compiled-kernel contracts.
``grid.geometry`` owns the immutable OceanGeometry input schema; the ocean
adapter in ``grid.ocean`` validates geometry and allocates model fields.
Import array contracts from ``model.arrays`` and geometry from ``grid.geometry``.
See :doc:`/api/experiment` and :doc:`/api/initialization` for physical input and
allocation shape conventions.

Canonical imports
-----------------

Specialized numerical and host code can import objects from their owning modules::

   from veris.grid.geometry import OceanGeometry
   from veris.grid.halos import fill_state_overlap, remove_halos
   from veris.model.arrays import ArrayInput, MaskInput
   from veris.model.configuration import Configuration
   from veris.model.physical_constants import PhysicalConstants
   from veris.model.types import State
   from veris.numerics.evp import evp_solver
   from veris.physics.thermodynamics.growth import Growth

The public ``veris`` facade exports experiment definitions, ``OceanGeometry``,
initialization, ``advance``, ``step`` and ``Simulation``. ``veris.io`` exports the
host output and record interfaces. See :doc:`/api/index` for their contracts.

Command-line examples use the maintained setup drivers::

   python -m veris.setups.run_dyn --help
   python -m veris.setups.run_growth --help
   python -m veris.setups.run_parallel --help
   sbatch veris/setups/run_parallel.slurm

See :doc:`integration` for pure rollout composition and
:doc:`automatic-differentiation` for derivative conventions. Scientific capture
manifests record source paths at their archived revisions.

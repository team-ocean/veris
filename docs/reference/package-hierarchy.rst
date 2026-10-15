Scientific package hierarchy
============================

Veris groups its code by the science or task it handles. The public workflow
uses ``Experiment``, ``Simulation``, ``advance`` and ``step`` from ``veris``,
with output interfaces from ``veris.io``. Use the modules below to import
individual numerical functions or helpers that control runs from Python.

.. list-table:: Package responsibilities
   :header-rows: 1
   :widths: 25 75

   * - Package
     - Responsibility
   * - ``veris.model``
     - Experiment definitions, prescribed external inputs called forcing,
       state and array requirements, numerical settings, physical constants,
       descriptions of data and diagnostic results.
   * - ``veris.grid``
     - Ocean grid information and its input adapter, interpolation between
       cell centers, faces and corners, extra boundary cells called halos,
       whole-grid summaries and inputs for grid regions.
   * - ``veris.physics``
     - Ice motion and its response to forces, transport across cells and ice
       piling into ridges, freezing and melting, and heat exchange estimated
       from near-surface atmospheric conditions.
   * - ``veris.numerics``
     - Functions that solve equations for momentum, a measure of mass in motion,
       and surface temperature, and basic operations for derivatives that
       measure changes with inputs.
   * - ``veris.execution``
     - Creating arrays and initial fields, model steps and repeated steps
       without Python side effects, timed runs with output, and starting,
       running and closing simulations from Python.
   * - ``veris.io``
     - Calendars, output schedules, records, collecting data across processes,
       reading selected fields, snapshots and complete restart.
   * - ``veris.setups``
     - Island, dynamics, growth and parallel examples and preset controls.
   * - ``veris.cli``
     - Central commands built with Click, listing and copying setup templates,
       and running experiment files.

Implementation tree
-------------------

Every directory shown is a Python package with an ``__init__.py``, except the
command-line template directories, which are packaged data. The tree omits
package initializer files except for the public interface at the root::

   veris/
     __init__.py
     cli/
       __main__.py
       overrides.py
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
       experiment_resolution.py
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
       checkpoint.py
       experiment_initialization.py
       initialization.py
       rollout.py
       rollout_output.py
       runner.py
       runtime.py
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

``model.arrays`` defines ``ArrayInput`` and ``MaskInput``. ``model.types`` defines
State, which contains only arrays, Parameter metadata that describes controls,
and the requirements for compiled numerical functions. ``grid.geometry`` defines
the OceanGeometry input format. These inputs cannot change in place. The ocean adapter
in ``grid.ocean`` checks grid information and creates model fields. Import array
requirements from ``model.arrays`` and grid geometry from ``grid.geometry``.
See :doc:`/api/experiment` and :doc:`/api/initialization` for physical input
dimensions and the shapes of model arrays.

Canonical imports
-----------------

Specialized numerical code and Python run controls can import objects from the
modules that define them::

   from veris.grid.geometry import OceanGeometry
   from veris.grid.halos import fill_state_overlap, remove_halos
   from veris.model.arrays import ArrayInput, MaskInput
   from veris.model.configuration import Configuration
   from veris.model.physical_constants import PhysicalConstants
   from veris.model.types import State
   from veris.numerics.evp import evp_solver
   from veris.physics.thermodynamics.growth import Growth

The public ``veris`` interface exports experiment definitions, ``OceanGeometry``,
initialization, ``advance``, ``step`` and ``Simulation``. ``veris.io`` exports
output and record interfaces that run in Python. See :doc:`/api/index` for their
requirements and behavior.

Command-line examples use the maintained setup programs::

   python -m veris.setups.run_dyn --help
   python -m veris.setups.run_growth --help
   python -m veris.setups.run_parallel --help
   sbatch veris/setups/run_parallel.slurm

See :doc:`integration` for combining and repeating model steps without Python
side effects, and :doc:`automatic-differentiation` for derivative rules. The
manifests for archived scientific reference data record source paths from the
archived code revisions.

``veris.execution.runtime`` checks launch requests, closes distributed resources
and discovers device meshes, which arrange devices that share a grid.
``veris.execution.runner`` provides the shared Python runner.
``veris.cli.overrides`` merges grouped command-line inputs using the existing
Python merge rules. ``veris.setups.presets`` holds provider descriptions,
functions that create initial fields, and launch profiles. It contains no code
for advancing timesteps. See :doc:`/quickstart/setup-blueprint` for the full path
from initialization to execution.

Getting started
===============

This guide takes you from installation to a first sea-ice simulation and its
output. It follows the progression of the `Veros getting-started guide
<https://veros.readthedocs.io/en/latest/introduction/get-started.html>`_, using
Veris's standalone JAX model and experiment interface.

Installation
------------

Use Python 3.12 or newer and Git. Install the current ``jax-only`` branch in a
virtual environment; the repository's ``main`` branch predates this interface.
From a directory where you want to keep the source, run::

   git clone --branch jax-only https://github.com/team-ocean/veris.git
   cd veris
   python3 -m venv .venv
   source .venv/bin/activate
   python -m pip install -e .

On Windows, activate the environment with ``.venv\Scripts\activate`` instead.
If you already have this checkout, start with environment creation in the
repository root. The editable installation makes source changes available
without reinstalling. Runtime dependencies, including CPU JAX and netCDF
support, are installed automatically. Veris requires no external ocean model.

Check that the command is available::

   veris --version
   veris --help

Use ``python -m veris.cli`` if you prefer to invoke the CLI through Python.
GPU execution needs a compatible GPU installation of JAX; follow the
`JAX installation instructions <https://docs.jax.dev/en/latest/installation.html>`_
for your hardware before selecting ``--backend gpu``. The steps below use CPU.

Setting up a model
------------------

List the supplied setups and copy the small thermodynamic column into an
editable directory::

   veris show-setups
   veris copy-setup growth --to my-growth
   cd my-growth

The list contains ``dynamics``, ``growth`` and ``island``. The growth setup evolves
ice and snow without horizontal dynamics. Island combines dynamics, transport
and growth with prescribed ocean masks and atmospheric fields. Their physical
assumptions and scenario controls are in the :doc:`/reference/setup-gallery`.

Open ``setup.py`` to adapt the experiment. Its ``build_experiment()`` factory
returns an immutable ``Experiment`` describing the grid, initial fields, forcing
and controls; the runner initializes and advances it. You can supply grouped
overrides from the CLI without editing this file. See :doc:`/cli` for those
options and :doc:`/quickstart/setup-blueprint` for your own setup factory.

Running Veris
-------------

From ``my-growth``, run two daily steps and save the final fields::

   veris run setup.py --backend cpu --steps 2 --float-type float32 --output growth.nc

The command reports ``Saved 2 steps to growth.nc``. Output paths are relative to
your working directory. Copied setup directories and output files must have new
names; choose another destination when repeating the commands. The short run
demonstrates the workflow, rather than a calibrated physical experiment.

For interactive Python use, see :doc:`/quickstart/user-guide`. It runs a coupled
experiment through ``Simulation`` with scheduled output and exact continuation.
The CLI and Python runner use the same model implementation.

Reading output
~~~~~~~~~~~~~~

The final snapshot is a netCDF file with a ``snapshot`` group. In a Python
session started from ``my-growth``, read its physical fields with::

   from veris.io import read_record

   record = read_record("growth.nc")
   print(record.time, record.fields["hIceMean"].shape)
   ice_thickness = record.fields["hIceMean"]

This prints ``172800.0 (2, 2)``: two days in elapsed seconds and the column's
two-by-two physical grid. ``hIceMean`` is grid-cell mean ice thickness in meters.
Arrays use ``(x, y)`` order and saved fields exclude storage halos.

To record the trajectory, add a separate history destination::

   veris run setup.py --backend cpu --steps 2 --float-type float32 --output growth-final.nc --netcdf growth-history.nc --io-variables hIceMean,Area

With this setup, ``growth-history.nc`` contains an ``instantaneous`` stream.
The :doc:`/reference/setups/growth` example shows how to plot its thickness
history. See :doc:`/reference/io` for calendars, averaging and record selection.

Restarting a simulation
~~~~~~~~~~~~~~~~~~~~~~~

A snapshot selects fields for analysis. Exact continuation uses a complete
checkpoint, saved explicitly through ``Simulation.save_restart`` and restored
with ``Simulation.from_restart``. The :doc:`/quickstart/user-guide` demonstrates
both calls and checks the resumed result against uninterrupted execution.
The :ref:`complete-restart` reference describes supported backends and output
continuation. The CLI does not accept checkpoints as setup files.

Running on multiple devices or processes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Once a serial setup works, use :doc:`/reference/setups/parallel-dynamics` for
local CPU/GPU meshes and multi-process SLURM launches. Veris uses JAX sharding
and distributed services; its launcher does not require Veros's MPI packages.
The CLI's mesh and backend options are documented in :doc:`/cli`.

Extending Veris
---------------

Start with a copied setup to change scientific inputs. Use the
:doc:`/quickstart/custom-ocean` example for your own geometry and fields, or the
:doc:`/quickstart/setup-blueprint` to package a reusable factory. Numerical
development starts with the :doc:`/reference/package-hierarchy` and
:doc:`/api/index`. The :doc:`/development` guide describes checks and contribution
workflow.

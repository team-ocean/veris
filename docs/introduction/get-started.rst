Getting started
===============

This guide shows you how to install Veris, run your first sea-ice simulation and
read its results.

Installation
------------

Use Python 3.12 or newer. Install the latest published release in a virtual
environment, a separate place for this project's Python packages::

   python3 -m venv .venv
   source .venv/bin/activate
   python -m pip install --upgrade veris

This documentation describes Veris |release|. See the
`latest release notes <https://github.com/team-ocean/veris/releases/latest>`_
for changes and downloads. The installation also adds the packages Veris needs
to run, including JAX for computer processors (CPU) and support for netCDF files.
Veris requires no external ocean model. For an editable source installation,
follow :doc:`/development`.

Check that the command is available::

   veris --version
   veris --help

Use ``python -m veris.cli`` to run the command-line interface (CLI) through Python.
Running on a graphics processor (GPU) needs a compatible GPU installation of
JAX; follow the
`JAX installation instructions <https://docs.jax.dev/en/latest/installation.html>`_
for your hardware before selecting ``--backend gpu`` in CLI. The steps below use CPU.

Setting up a model
------------------

List the supplied setups and copy the small thermodynamic column into a
directory you can edit. This setup models how ice and snow grow and melt
through heat exchange::

   veris show-setups
   veris copy-setup growth --to my-growth
   cd my-growth

The list contains ``dynamics``, ``growth`` and ``island``. The growth setup changes
ice and snow without horizontal dynamics, or motion across the surface. Island
combines ice motion, transport (movement of ice and snow between cells) and
growth. It supplies ocean masks (arrays marking land and ocean) and atmospheric
fields. The grid divides the modeled area into cells.
Their physical assumptions and scenario controls are in the
:doc:`/reference/setup-gallery`.

Open ``setup.py`` to change the experiment. Its ``build_experiment()`` factory
is a function that returns an immutable ``Experiment``, a definition that cannot
be changed in place. It describes the grid, starting values, forcing (supplied
inputs such as wind and ocean temperature) and settings. The runner prepares
the model and advances it through time. You can override groups of settings
from the CLI without editing this file. See :doc:`/cli` for those options and
:doc:`/quickstart/setup-blueprint` for your own setup factory.

Running Veris
-------------

From ``my-growth``, run two daily steps and save the final fields::

   veris run setup.py --backend cpu --steps 2 --float-type float32 --output-snapshot growth.nc

The command reports ``Saved 2 steps to growth.nc``. Output paths are relative to
your working directory. Copied setup directories and output files must have new
names; choose another destination when repeating the commands. The short run
shows the workflow; its inputs have not been calibrated for a physical study.

For interactive Python use, see :doc:`/quickstart/user-guide`. It runs a coupled
experiment, which combines dynamics and growth, through ``Simulation``. It saves
output on a schedule and shows how to continue a saved run exactly.

Reading output
~~~~~~~~~~~~~~

The final snapshot saves values at one time in a netCDF file. Its ``snapshot``
group holds those values. In a Python session started from ``my-growth``, read
its physical fields with::

   from veris.io import read_record

   record = read_record("growth.nc")
   print(record.time, record.fields["hIceMean"].shape)
   ice_thickness = record.fields["hIceMean"]

This prints ``172800.0 (2, 2)``: two days in elapsed seconds and the column's
two-by-two physical grid. ``hIceMean`` is mean ice thickness across the whole
grid cell, in meters.
Arrays use ``(x, y)`` order, with the x direction first. Saved fields exclude
storage halos, the extra cells around grid edges used for boundary calculations.

To record the trajectory (results through time), add a separate history
destination::

   veris run setup.py --backend cpu --steps 2 --float-type float32 --output-snapshot growth-final.nc --output-history growth-history.nc --io-variables hIceMean,Area

With this setup, ``growth-history.nc`` contains an ``instantaneous`` stream,
a sequence of values saved at specific times.
The :doc:`/reference/setups/growth` example shows how to plot its thickness
history. See :doc:`/reference/io` for calendars, averaging and record selection.

Restarting a simulation
~~~~~~~~~~~~~~~~~~~~~~~

A snapshot saves selected fields for analysis. To continue a run exactly, use a
complete checkpoint, a file with all the model values and inputs needed to
resume. Save it explicitly through ``Simulation.save_restart`` and restore it
with ``Simulation.from_restart``. The :doc:`/quickstart/user-guide` demonstrates
both calls and checks the resumed result against uninterrupted execution.
The :ref:`complete-restart` reference describes supported backends (CPU or GPU
execution) and output
continuation. The CLI does not accept checkpoints as setup files.

Running on multiple devices or processes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Once a serial setup (one without a device mesh) works, use
:doc:`/reference/setups/parallel-dynamics` for local CPU/GPU meshes, which group
devices on one computer, and multi-process SLURM launches. SLURM starts work on
a computing cluster. Veris uses JAX sharding to divide arrays across devices
and distributed services to coordinate separate Python processes.
The CLI's mesh and backend options are documented in :doc:`/cli`.

Extending Veris
---------------

Start with a copied setup to change the model's physical inputs. Use the
:doc:`/quickstart/custom-ocean` example for your own geometry and fields, or the
:doc:`/quickstart/setup-blueprint` to package a reusable function that creates
experiments. To change the calculations themselves, start with the
:doc:`/reference/package-hierarchy` and
:doc:`/api/index`. The :doc:`/development` guide describes checks and contribution
workflow.

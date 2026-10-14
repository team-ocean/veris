Veris: a versatile sea ice simulator in JAX
===========================================

Veris models sea-ice dynamics (how ice moves) and thermodynamics (how ice grows
and melts through heat exchange). It uses JAX, a Python library that calculates
with arrays, or tables of numbers. Automatic differentiation calculates how
changes in inputs affect results. Veris supports both forward and reverse modes,
which carry these calculations from inputs to outputs or from outputs back to
inputs. It can divide work across computer processors (CPUs) and graphics
processors (GPUs). The package includes several setups to show how to use it.

Veris was derived from the `MITgcm SEAICE package
<https://mitgcm.readthedocs.io/en/latest/phys_pkgs/seaice.html>`_.
`Jan Philipp Gärtner <https://github.com/jpgaertner>`_ created the original Veris
implementation as part of his `Master's thesis <https://nbi.ku.dk/teamocean/docs/Jan_Gaertner_MSc_thesis.pdf>`__.

.. toctree::
   :maxdepth: 2
   :caption: Usage

   introduction/get-started
   quickstart/user-guide
   quickstart/custom-ocean
   quickstart/setup-blueprint
   cli

.. toctree::
   :maxdepth: 2
   :caption: Reference

   reference/package-hierarchy
   reference/setup-gallery
   reference/integration
   reference/setups/ocean
   reference/settings
   reference/physical-constants
   reference/variables
   reference/io
   reference/automatic-differentiation

.. toctree::
   :maxdepth: 2
   :caption: Python API

   api/index

.. toctree::
   :maxdepth: 1
   :caption: Development

   development

Source code: `team-ocean/veris <https://github.com/team-ocean/veris>`_.

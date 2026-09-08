Veris: a standalone JAX sea-ice model
=====================================

Veris implements sea-ice dynamics, transport, and thermodynamic growth with JAX
arrays and compiled kernels. The included artificial-island example runs with
prescribed ocean and atmospheric fields.

The physics derives from the `MITgcm SEAICE package
<https://mitgcm.readthedocs.io/en/latest/phys_pkgs/seaice.html>`_.
`Jan Philipp Gärtner <https://github.com/jpgaertner>`_ created the original Veris
implementation as part of his Master's thesis.

.. toctree::
   :maxdepth: 2
   :caption: Usage

   quickstart/user-guide

.. toctree::
   :maxdepth: 2
   :caption: Reference

   reference/setup-gallery
   reference/settings
   reference/variables

Source code: `team-ocean/veris <https://github.com/team-ocean/veris>`_.

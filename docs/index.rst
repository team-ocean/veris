Veris: a versatile sea ice simulator in JAX
===========================================

Veris implements sea-ice dynamics and thermodynamics using JAX
for array operations. It supports forward and reverse modes of
automatic differentiation and parallel excution on CPUs and GPUs.
The package is bundeled with several setups to demonstrate its functionality.

Veris was derived from the `MITgcm SEAICE package
<https://mitgcm.readthedocs.io/en/latest/phys_pkgs/seaice.html>`_.
`Jan Philipp Gärtner <https://github.com/jpgaertner>`_ created the original Veris
implementation as part of his `Master's thesis <https://nbi.ku.dk/teamocean/docs/Jan_Gaertner_MSc_thesis.pdf>`__.

.. toctree::
   :maxdepth: 2
   :caption: Usage

   quickstart/user-guide

.. toctree::
   :maxdepth: 2
   :caption: Reference

   reference/setup-gallery
   reference/integration
   reference/settings
   reference/physical-constants
   reference/variables
   reference/io
   reference/automatic-differentiation

Source code: `team-ocean/veris <https://github.com/team-ocean/veris>`_.

Veris: a versatile sea ice simulator in JAX
===========================================

Veris implements sea ice dynamics and thermodynamics using JAX
for array operations. It supports forward and reverse modes of
automatic differentiation and parallel execution on CPUs and GPUs.
The package includes several setups to show how to use it.

This documentation describes Veris |release|. The
`stable documentation <https://veris.readthedocs.io/en/stable/>`_ follows the
`latest published release <https://github.com/team-ocean/veris/releases/latest>`_.

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

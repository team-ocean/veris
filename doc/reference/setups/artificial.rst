Artificial island
=================

``veris.setup.artificial`` provides a small periodic Cartesian sea surrounding
a two-by-two-cell island. The default interior is 8 by 12 cells at 8 km spacing;
arrays include two halo cells on each side. Face masks prevent transport across
the coastline. Temperatures are in kelvin and thicknesses are grid-cell means.

.. autofunction:: veris.setup.artificial.initialize

.. autofunction:: veris.setup.artificial.step

The step retains all velocity and stress outputs. Its sequence is mass and area
averaging, wind forcing, ice strength, EVP dynamics, ocean stress, advection,
cleanup, ridging, growth, and periodic halo refresh. It restores prescribed
open-water heat forcing each step because growth returns ocean-coupling fluxes
in the same state fields.

Run initialization before importing halo-dependent kernels in a fresh process.
The example selects serial halos and does not initialize a distributed mesh.
The same JAX kernels can execute on a supported CPU or GPU device.

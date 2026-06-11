# Veris

A sea ice model that can be used both as a plugin for Veros and as a standalone model.

### How to use

To use Veris as plugin for Veros, first [install Veros](https://veros.readthedocs.io/en/latest/introduction/get-started.html). Veris can be installed via pip (```pip install veris```) or from this repository (```pip install -e .```) if you want to modify the model code.

Follow the steps below to run the coupled Veris-Veros model:
```bash
$ veros copy-setup seaice_global_4deg --to /tmp/seaice_4deg
$ cd /tmp/seaice_4deg
$ veros run seaice_global_4deg.py
```

To use Veris as a standalone model, see the scripts provided [here](https://github.com/jpgaertner/veris_minimum_working_example) [![DOI](https://zenodo.org/badge/953970891.svg)](https://doi.org/10.5281/zenodo.20642250).


### Credits

Veris was created by [Jan P. Gärtner](https://github.com/jpgaertner) and is based on the [MITgcm sea ice component](https://mitgcm.readthedocs.io/en/latest/phys_pkgs/seaice.html), developed by [Martin Losch](https://www.awi.de/ueber-uns/organisation/mitarbeiter/detailseite/martin-losch.html) and collaborators.

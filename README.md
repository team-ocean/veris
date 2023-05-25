# Veris
Sea ice plugin for Veros, based on SEAICE package of MITgcm model

## Quick usage
In order to start using the sea ice plugin you need to [install Veros](https://veros.readthedocs.io/en/latest/introduction/get-started.html).
Then follow the steps below to install and use Veris: 
```bash
$ pip install veris
$ veros copy-setup seaice_global_4deg --to /tmp/seaice_4deg
$ cd /tmp/seaice_4deg
$ veros run seaice_global_4deg.py
```

## Credits

Veris is based on [SEAICE package](https://mitgcm.readthedocs.io/en/latest/phys_pkgs/seaice.html) by [Martin Losch et al.](https://www.awi.de/ueber-uns/organisation/mitarbeiter/detailseite/martin-losch.html), Alfred Wegener institute.

[Jan Philipp Gärtner](https://github.com/jpgaertner) created Veris plugin as a part of his Master's thesis.

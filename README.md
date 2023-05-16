# Veris
Sea ice plugin for Veros, based on SEAICE package of MITgcm model

## Quick usage

```bash
$ pip install veros 
$ git clone https://github.com/team-ocean/veris.git
$ cd veris
$ pip install -e .
$ veros copy-setup seaice_global_4deg --to /tmp/seaice-4deg
$ cd /tmp/seaice-4deg
$ python seaice_global_4deg.py
```

## Credits

Veris is based on [SEAICE package](https://mitgcm.readthedocs.io/en/latest/phys_pkgs/seaice.html) by [Martin Losch et al.](https://www.awi.de/ueber-uns/organisation/mitarbeiter/detailseite/martin-losch.html), Alfred Wegener institute.

[Jan Philipp Gärtner](https://github.com/jpgaertner) created Veris plugin as a part of his Master's thesis.

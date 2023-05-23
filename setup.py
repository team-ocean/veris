#!/usr/bin/env python
# coding=utf-8

from setuptools import setup, find_packages
from codecs import open
import sys
import os


here = os.path.abspath(os.path.dirname(__file__))
sys.path.append(here)
import versioneer  # noqa: E402


CLASSIFIERS = """
Development Status :: 3 - Alpha
Intended Audience :: Science/Research
License :: OSI Approved :: MIT License
Programming Language :: Python :: 3
Programming Language :: Python :: 3.7
Programming Language :: Python :: 3.8
Programming Language :: Python :: 3.9
Programming Language :: Python :: 3.10
Topic :: Scientific/Engineering
Operating System :: Microsoft :: Windows
Operating System :: POSIX
Operating System :: Unix
Operating System :: MacOS
"""

INSTALL_REQUIRES = [
    "veros>=1.4.4"
]

with open(os.path.join(here, "README.md"), encoding="utf-8") as f:
    long_description = f.read()


setup(
    name="veris",
    license="MIT",
    author="Jan Gärtner (AWI Bremen)",
    author_email="jph.gaertner@gmail.com",
    keywords="oceanography python parallel numpy multi-core geophysics ocean-model mpi4py jax",
    description="Sea-ice plugin for Veros, the versatile ocean simulator",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://veris.readthedocs.io',
    python_requires=">=3.7",
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    packages=find_packages(),
    install_requires=INSTALL_REQUIRES,
    entry_points={
            "veros.setup_dirs": [
                "seaice = veris.setup"
            ]
    },
    package_data={"veris": ["setup/*/assets.json"]},
    classifiers=[c for c in CLASSIFIERS.split("\n") if c],
)

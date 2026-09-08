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
Programming Language :: Python :: 3.12
Programming Language :: Python :: 3.13
Programming Language :: Python :: 3.14
Topic :: Scientific/Engineering
Operating System :: Microsoft :: Windows
Operating System :: POSIX
Operating System :: Unix
Operating System :: MacOS
"""

INSTALL_REQUIRES = [
    "jax>=0.11.1",
    "numpy>=2.0"
]

with open(os.path.join(here, "README.md"), encoding="utf-8") as f:
    long_description = f.read()


setup(
    name="veris",
    license="MIT",
    author="Jan Gärtner (AWI Bremen)",
    author_email="jph.gaertner@gmail.com",
    keywords="oceanography python parallel numpy multi-core geophysics ocean-model mpi4py jax",
    description="Standalone differentiable sea-ice model in JAX",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://veris.readthedocs.io',
    python_requires=">=3.12",
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    packages=find_packages(),
    install_requires=INSTALL_REQUIRES,
    classifiers=[c for c in CLASSIFIERS.split("\n") if c],
)

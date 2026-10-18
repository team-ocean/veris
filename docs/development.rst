Developing Veris
================

Development and release tagging use ``main``. For the latest published package,
follow :doc:`/introduction/get-started`.
To work on the source, use Git and Python 3.14::

   git clone --branch main https://github.com/team-ocean/veris.git
   cd veris
   python3 -m venv .venv
   source .venv/bin/activate
   python -m pip install -e .

The test dependency lock lists the exact package versions for that Python
version, so others can use the same environment. Install the development and
documentation packages from the repository root::

   python -m pip install -r requirements-test.txt -r docs/requirements.txt

The repository's `AGENTS.md`_ defines local development rules. `DESIGN.md`_
explains how the model is organized. `CHANGELOG.md`_ records progress and check
results. Read them before changing the model. For each package's role, see
:doc:`/reference/package-hierarchy`.

Checks
------

During development, use ``JAX_PLATFORMS=cpu pytest tests/ --fast``. Before
committing, run the full tests. Also check test coverage (how much code the tests
run), code style, formatting and types. The commands below come from the
test-harness guide, which explains how to run and check the tests. This keeps
the command list in one place:

.. literalinclude:: ../tests/README.md
   :language: sh
   :start-after: ```sh
   :end-before: ```

Use ``ruff format`` without ``--check`` to apply formatting.

Run one pytest session at a time. Use fewer workers when CPU resources require
it. ``--fast`` selects approximately 10% of cases when pytest collects the tests,
before it does costly test setup. ``VERIS_TEST_SEED`` changes the sample.
Skip the fast run when a full run is already planned. The `test-harness guide`_
explains coverage, the sources of reference results and hardware checks. The
`benchmark guide`_ explains how to measure performance.

Build documentation with warnings treated as errors::

   sphinx-build -E -W -b html docs docs/_build/html

Open ``docs/_build/html/index.html`` locally. CLI help, settings, physical
constants, fields and scenario tables are generated from the implementation;
their source registries or docstrings control the generated text. Make changes
there instead of copying the tables. Continuous integration (CI), the automated
checks run by the repository, also runs the public examples exactly as written
against a clean installation from a wheel, Python's installable package format.

The `stable documentation`_ follows the latest release tag.
Read the Docs' ``latest`` version follows ``main`` and can include unreleased
changes. Versioneer derives the displayed version from Git rather than a fixed
string. The Read the Docs checkout hook fetches tags and completes shallow
history before installing the package, so both package metadata and Sphinx can
resolve the release. After tagging a release, rebuild ``latest`` as well as
``stable`` if ``latest`` was built before the tag existed.

Before committing
-----------------

Before fixing a bug, add a test that shows it. Write tests before adding new
model behavior. Keep the reference equations and rules for calculating
derivatives unless the change explicitly targets them. A derivative describes
how a result changes when an input changes. Complete the correctness and coverage checks described
in the `test-harness guide`_, update `CHANGELOG.md`_, and commit a focused change on
a feature branch based on ``main``. Record failed approaches and remaining work
for the next session.

.. _AGENTS.md: https://github.com/team-ocean/veris/blob/main/AGENTS.md
.. _DESIGN.md: https://github.com/team-ocean/veris/blob/main/DESIGN.md
.. _CHANGELOG.md: https://github.com/team-ocean/veris/blob/main/CHANGELOG.md
.. _test-harness guide: https://github.com/team-ocean/veris/blob/main/tests/README.md
.. _benchmark guide: https://github.com/team-ocean/veris/blob/main/benchmarks/README.md
.. _stable documentation: https://veris.readthedocs.io/en/stable/

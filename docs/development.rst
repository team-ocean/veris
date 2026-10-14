Developing Veris
================

Work from the ``jax-only`` branch in an activated Python 3.14 environment; the
reproducible test dependency lock targets that version. Install development and
documentation dependencies from the repository root::

   python -m pip install -r requirements-test.txt -r docs/requirements.txt

The repository's `AGENTS.md`_ defines local development rules. `DESIGN.md`_
records the architecture, and `CHANGELOG.md`_ records progress and verification
results. Read them before changing the model. Package responsibilities are in
:doc:`/reference/package-hierarchy`.

Checks
------

During development, use ``JAX_PLATFORMS=cpu pytest tests/ --fast``. Before
committing, run full correctness, coverage, lint, format and type checks. These
commands are included from the test-harness guide so there is one maintained
version:

.. literalinclude:: ../tests/README.md
   :language: sh
   :start-after: ```sh
   :end-before: ```

Use ``ruff format`` without ``--check`` to apply formatting.

Run one pytest session at a time. Use fewer workers when CPU resources require
it. ``--fast`` selects approximately 10% of cases at collection, before expensive
fixture setup; ``VERIS_TEST_SEED`` varies that sample. Skip a redundant fast run
when a full run is already planned. The repository's `test-harness guide`_ owns details of coverage, reference
provenance and hardware checks; the `benchmark guide`_ owns performance
measurements.

Build documentation with warnings treated as errors::

   sphinx-build -E -W -b html docs docs/_build/html

Open ``docs/_build/html/index.html`` locally. CLI help, settings, physical
constants, fields and scenario tables are generated from the implementation;
edit their authoritative registries or docstrings instead of copying tables.
CI also executes the literal public examples against a clean installed wheel.

Before committing
-----------------

Add a reproducing test before a bug fix and tests before new model behavior.
Preserve reference equations and derivative conventions unless the change
explicitly targets them. Complete the correctness and coverage gates described
in the `test-harness guide`_, update `CHANGELOG.md`_, and commit a focused change on
``jax-only``. Record failed approaches and remaining work for the next session.

.. _AGENTS.md: https://github.com/team-ocean/veris/blob/jax-only/AGENTS.md
.. _DESIGN.md: https://github.com/team-ocean/veris/blob/jax-only/DESIGN.md
.. _CHANGELOG.md: https://github.com/team-ocean/veris/blob/jax-only/CHANGELOG.md
.. _test-harness guide: https://github.com/team-ocean/veris/blob/jax-only/tests/README.md
.. _benchmark guide: https://github.com/team-ocean/veris/blob/jax-only/benchmarks/README.md

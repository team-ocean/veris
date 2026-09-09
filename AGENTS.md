# Sea ice model

## Goal 

Replace NamedTuples with frozen data classes for State and Settings. 
Settings must be separated into two classes: one for physical constants and another for model settings.  
There are settings and physical constants with their definitions scattered all over the source code.
I need all physical constants in a physical constants object and all settings in a settings object.
The model State shall contain only Veris variables, which are used in calculations to keep the State minimal for AD.
Settings, PhysicalConstants, State objects with their attributes/fields must be defined and allocated at Veris initialization stage with their default values.
For the sake of clarity, I need 2 namedtuple and dictionary objects for Settings and PhysicalConstants, which look something like:

```python
from collections import namedtuple

description = namedtuple("setting", ("default", "type", "description"))

PHYSICALCONSTANTS = {
    "rhoAir": Setting(1.3, float, “Density of air :math:`kg/m^3`"),
    …
}

SETTINGS = {
    …
}
```

from which Settings and PhysicalConstants are initialized.

For the State object, all to be allocated variables must be specified in a dictionary, which looks something like:

```python
VARIABLES = {
    "theta": Variable(“Ocean surface temperature", XT+YT, "K", "Ocean surface temperature"),
}
```
 
where XT+YT are latitude (XT) and longitude (YT) dimensions of a variable and Variable object is built in a way that it can be used as a metadata source for output in a netcdf file with h5netcdf library.
Use more suitable names for variables' dimensions, which reflect the internals of Veris. You can add more attributes to the Variable class for more metadata.

Veris documentation shall use these dicts to describe settings, physical constants, and state variables.   

## CRITICAL: local environment rules

### Filesystem

- **NEVER** use `find /` or scan outside the project directory.
  The filesystem has millions of files and these command will hang forever.
- **Project root**: the current working directory (use `.` or relative paths)
- Use `rg` and `rg --files` for scoped content and file searches; fall back to `grep` when needed.

## What is this?

A fully differentiable sea-ice model in JAX with CPU and GPU as first-class execution targets.

## Quick reference

- **Reference source codes**: `/groups/ocean/nutrik/dev/veris`, `/groups/ocean/nutrik/dev/veris_minimum_working_example`
- **Design document**: `DESIGN.md` (read this first)
- **Progress log**: `CHANGELOG.md`

The `main` branch in both reference codes is very old and shall not be used.
Instead, you need to branch off `jax-only` in `veris` and use `jax_halo_exchange` branch in `veris_minimum_working_example`.
After a certain development stage is completed in `veris` repo, you need to commit it back to `jax-only` branch rather than `main`.
Developments shall be done for `veris` repo, not `veris_minimum_working_example`. Use `veris_minimum_working_example` only for
reference on how to define Veris state, initialize it, and perform sequential and parallel runs. 

## Setup

```bash
# To use uv you need to load it with module utility
module load uv/lates

# Use uv to create and setup the virtual environment at the very beginning of the project, then skip this step in subsequent implementation sessions
uv venv

# Initialize the project at the very beginning of the project, then skip this step in subsequent sessions
uv init

# Initialize git repository at the very beginning of the project and create main branch, then skip this step in subsequent sessions
git init

# Install dependencies at the very beginning of the project, then skip this step in subsequent sessions
uv add pip numpy scipy jax jaxlib diffrax equinox jaxtyping pytest pytest-cov pytest-xdist ruff ty h5py h5netcdf matplotlib PyYAML types-PyYAML build pyproject_hooks flit flit-core

# Always activate venv
source .venv/bin/activate

# Run linters and formatters during development
ruff check .
# Fix ruff errors
ruff check --fix
# Format source code
ruff format .

# Do static type checking during development
ty check .

# Run tests
pytest tests/ -v
pytest tests/ -v -m "not slow"     # skip integration tests
pytest tests/test_gradients.py -v  # gradient checks only

# Commit changes to git only after full suite unit tests pass
git add .

# For example, after implementing module X and its unit tests:
git commit -m "Implement module X with tests"
```

---

## Orientation (read this first when starting a session)

When you start a new session, orient yourself:
1. Read `CHANGELOG.md` to see what's done and what's next.
2. Inspect the last recorded test status; run `pytest tests/ -q --fast` if a fresh check is needed and a full suite is not already planned.
3. Pick the next failing test or unchecked item from `CHANGELOG.md`, prioritizing correctness blockers and measured CPU/GPU bottlenecks.
4. When you finish a unit of work, update `CHANGELOG.md` before stopping.
5. Run only one pytest instance at a time. Use fast mode during ordinary development; if a full suite is already planned, omit redundant fast runs. Keep solver performance measurements in the separate benchmark harness.

---

## Coding best practices

Follow these principles to keep the source code maintainable, readable, and long-term successful.

### 1. Code Organization

- Many small files over few large files
- High cohesion, low coupling
- 200-400 lines typical, 800 max per file
- Organize by feature/domain, not by type

### 2. Code Style

- Use meaningful variable, function and class names (prioritizes readability)
- Follow PEP8 guidelines for Python
- Write docstrings for all public modules, functions, classes, and methods following PEP 257
- Functionality should only be added when deemed necessary, following YAGNI (You Aren't Gonna Need It) principle
- Follow DRY (Don't Repeat Yourself) principle
- Follow SOLID principles: Single Responsibility, Open/Closed, Liskov Substitution, Interface Segregation, and Dependency Inversion
- Validate and raise informative errors in host wrappers. Compiled kernels return JAX-compatible status values; Python try/except is not a device-side numerical error mechanism.

### 3. Testing

- TDD: Write tests first
- Aim for 80%+ minimum code coverage
- Use pytest for testing
- Test edge cases and error conditions, not just happy paths
- Use descriptive test names that clearly indicate what is being tested
- Organize tests in a separate directory (e.g., `tests/`) and mirror the structure of the main codebase
- Use fixtures for setup and teardown of test environments when necessary

---

## Current status of source codes

There are legacy source code parts like `veris/setup/seaice_global_4deg` in Veris repository, which depend on Veros ocean model.
They contain important example of how to perform Veris integration, which includes both sea-ice dynamics and growth and how to use masks to identify grid areas with ocean. So, I need to remove such Veros dependencies but at the same time I need to keep such integration example with artificial ocean mask and fields.
The veris_minimum_working_example repo contains examples of how to define Veris state, initialize it, and perform sequential and parallel runs. These examples are only for stand alone Veris runs.

---

## Principles for autonomous development

### 1. Reference/original source code is the oracle -- tests are everything

Reference source code is our known-good reference. The test harness is the most important part of the project.
If the tests are wrong or incomplete, agents will solve the wrong problem.

**Rules:**
- Never merge or commit code that breaks existing passing tests.
- Every new module must have a corresponding test file BEFORE implementation.
  Write the test first (specifying what source code produces), then make it pass.
- When you find a bug, add a test that reproduces it before fixing it.
- Tests must be nearly perfect. Invest heavily in the test harness: generate
  high-quality source code reference data at many parameter points, write clear
  verifiers, and watch for failure modes so you can add targeted tests.
- When a discrepancy is found, trace upstream through the pipeline to find the
  first module where things diverge. Fix there; downstream improves automatically.

### 2. Concise test output (context window hygiene)

LLMs have finite context windows. Every line of noisy test output displaces
useful information and degrades reasoning quality.

**Rules:**
- Tests print at most 5-10 lines on success, ~20 lines on failure.
- Use `pytest -q` by default. Never dump large arrays to stdout.
- Log verbose diagnostics to `test_logs/` files, not stdout.
- Pre-compute aggregate summary statistics. Print them, not raw data.
- When comparing arrays, print: max relative error, the index/value where
  it occurs, and the overall pass rate. Not the full arrays.
- Error messages should be greppable: put ERROR and the reason on one line
  so `grep ERROR logfile` works.

Good:
```
FAILED test_advection.py::test_uv_advection_momentum_eq - max rel err 0.032% at z=1089.2
  Expected MAXIMUM GROWTH VALUE=8.27146287E-04, got MAXIMUM GROWTH VALUE=9.27244317E-04
  (23/25 quantities pass at <0.01%, 2 at <0.05%)
```

Bad:
```
FAILED - arrays not equal:
  [1.0183e-4, 1.0182e-4, 1.0181e-4, ...]  (500 more lines)
```

### 3. Fast tests to avoid time blindness

LLMs can't tell time and will happily spend hours running full test suites
instead of making progress.

**Rules:**
- Provide a project-wide `--fast` option with selection at collection time, before expensive fixture setup.
- `--fast` runs a deterministic ~10% subsample.
- The subsample should be deterministic per-agent but cover different points
  across agents (use a hash of the agent ID or test name as seed).
- Default development cycle: run `--fast` after every change, full suite
  only before committing.

### 4. Keep CHANGELOG.md current (agent orientation)

CHANGELOG.md is the shared memory. Without it, agents waste time re-discovering what's done and what's broken.

**Rules:**
- Update CHANGELOG.md after every meaningful unit of work.
- Check off completed items with dates.
- Note what worked, what didn't, what's blocked.
- **Record failed approaches** so they aren't re-attempted. If something does not work switch to alternative(s).
- Add new tasks discovered during implementation.
- When stuck, maintain a running doc of attempts in CHANGELOG.md.

### 5. Prevent regressions (CI discipline)

Once the codebase grew, new features frequently broke existing functionality.
Therefore, build a CI pipeline with strict enforcement and discipline.

**Rules:**
- Run the full correctness suite before a code commit; use fast mode during development, without a redundant fast run immediately before or after the full suite.
- If anything regresses, fix it before committing. Never "fix it later."
- If a new feature requires changing behavior in an existing test, update the
  test explicitly (don't just delete or skip it).
- Track test pass rates over time in CHANGELOG.md (e.g., "advection: 25/25,
  growth: 18/20, time_stepping: 142/150, etc.").

### 6. Structure work for parallelism

Parallelism is easy when there are many independent failing tests
(each agent picks a different one), but hard when there's one giant failing task
(all agents hit the same bug and overwrite each other).

**How this applies to source code:**
- Easy to parallelize - many independent tasks;
- Hard to parallelize - one giant task.

**Mitigation for the "one giant task" problem:** Break it into sub-tests.
Test individual equations or components separately.
Then combine. This way, multiple agents can work on different subsystems.

**Task claiming:** When working in parallel, note your task in CHANGELOG.md
(e.g., "IN PROGRESS: diffusion.py (@agent-1)"). Check CHANGELOG.md before
starting in order to avoid duplicate work.

### 7. Small, testable commits

**Rules:**
- Each commit implements one thing (one function, one module, one bug-fix).
- Each commit passes all existing tests.
- Each commit includes or updates tests for the new code.
- Avoid large commits that change multiple modules at once.
- If a refactor touches many files, do it as a separate commit from features.

### 8. Document for the next session, not for users

Documentation is not a nicety; it's a critical coordination mechanism.

**Every module should have a docstring explaining:**
- What physics and numerics it implements (with equation references, scientific papers).
- What it takes as input and produces as output (types, shapes).
- Any non-obvious numerical choices (why this tolerance? why this grid size?).
- Known limitations or accuracy issues.
- What source code function/file this corresponds to.

### 9. Specialized agent roles

Use specialized agents beyond just "write code":
one for deduplication, one for performance, one for code quality review,
one for documentation.

**For source code useful specializations:**
- **Implementer agents**: Write the module code to pass tests.
- **Test quality agent**: Reviews and improves the test harness. Adds edge
  cases, improves error messages, catches gaps in coverage.
- **Gradient validation agent**: Focused solely on testing AD correctness.
  Runs finite-difference checks for every module, every parameter.
- **Performance agent**: Profiles the code, identifies bottlenecks, optimizes
  JIT compilation time, reduces memory usage.
- **Code quality agent**: Looks for duplicated code, inconsistent patterns,
  missing type hints, unclear variable names. Refactors.
- **Documentation agent**: Keeps CHANGELOG.md and docstrings in sync with actual code.


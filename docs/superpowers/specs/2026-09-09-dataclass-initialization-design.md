# Registry-driven Veris initialization

## Objective and current evidence

Implement the Goal in AGENTS.md on `jax-only`. The existing `State` and
`Settings` in `veris/state.py` are named tuples, `settings.py` mixes physical
constants with configuration, and `variables.py` has no allocation or output
metadata. The artificial setup allocates 84 fields, including output-only
diagnostics. Bulk flux functions and `solve4temp` contain additional local
coefficients. `fill_overlap` selects a backend at import time by reading a
mutable dictionary. These are all migration scope, not just container syntax.

## Proposed architecture

Use three frozen dataclasses: `Settings`, `PhysicalConstants`, and JAX PyTree
`State`. Pass settings and constants explicitly to numerical routines; neither
belongs among the array leaves of State. Keep settings and constants hashable
static JIT arguments, preserving current coefficient differentiation behavior.
AD with respect to state and dynamic forcing must continue to work.

Two named-tuple metadata types, `Setting` and `PhysicalConstant`, provide
`default`, `type`, and `description`, plus units where useful. `SETTINGS` and
`PHYSICALCONSTANTS` are authoritative initialization registries. Keep explicit
typed dataclass fields for editor/type-checker support, with exhaustive schema
checks ensuring each field derives its default from its registry. Do not keep
a combined settings facade that hides constants behind settings attributes.

PhysicalConstants holds material properties, physical and empirical law
coefficients, albedos, emissivities, phase-change quantities, gas constants,
reference salinities and physical conversion factors. Settings holds execution
choices, timesteps, iteration counts, solver controls, regularization values,
numerical cutoffs, forcing heights and experiment configuration. Preserve
different established parameterizations even where their constants differ;
do not silently equate rounded gas constants or latent heats. Pure algebraic
numbers and stencil weights remain in equations. Inventory local named values
and empirical coefficient expressions throughout maintained kernels.

Derived values (reciprocal timesteps, density ratios, turning-angle sine/cosine,
and similar dependencies) are computed at construction from their independent
values. Immutable updates reconstruct them, preventing stale combinations.
Historical independently rounded physical defaults remain unchanged unless an
exact dependency is documented in the current source.

## State and metadata

`VARIABLES` maps names to immutable `Variable` metadata containing long name,
description, dimensions, units, dtype, default and grid location. Horizontal
dimensions describe the Cartesian staggered grid: `x_center`, `y_center`,
`x_face`, `y_face`. Array order stays x then y. All local horizontal extents
include the existing two-cell periodic halos; face and center arrays retain
equal storage extents. Surface temperature stays two-dimensional; temporary
ice-thickness category arrays stay local to growth.

Retain only fields consumed by calculations, including geometry, prescribed
forcing and intermediates required by downstream kernels. Remove dead fields.
Return useful output-only diagnostics separately from the stepped State so
they remain available to callers without permanent AD leaves. Document the
field-use inventory and distinguish output diagnostics from dead aliases.
Metadata must supply valid h5netcdf variable dimensions, dtype and attributes;
test a real write/read round trip, including staggered grids and units.

Initial kernel-read audit identifies 14 unread fields: `IcePenetSW`,
`OceanStressU`, `OceanStressV`, `saltflux`, `EmPmR`, `dxC`, `dyC`,
`recip_dxG`, `recip_dyG`, `rA`, `rAu`, `rAv`, `recip_rAz`, and
`forc_salt_surface`. Recheck indirect use during implementation. The first
diagnostic group and salt forcing remain available as outputs; unused direct
metrics can remain initialization-local while their required reciprocals stay
in State. Bulk helpers also reference `grav` and `radius`, which are absent
from the existing defaults; supply documented physical defaults and use a
single canonical gravity field when migrating those helpers.

## Initialization and execution

A public host initializer constructs settings and physical constants from
defaults plus validated overrides, then allocates every State field from
VARIABLES for the requested grid. Reject unknown keys, invalid scalar types,
invalid extents and incompatible array shapes with informative errors.
Apply setup-specific forcing, geometry and masks through immutable updates.
The artificial island example remains executable without Veros.

Make serial/sharded halo selection explicit from initialized configuration
and supplied mesh, rather than mutable registry values at module import.
Preserve the actual exchange algorithm and current distributed reduction
semantics. A mesh is execution context, not a numerical State field.
Migrate the geometry host adapter without introducing ocean-model fields into
the minimal State. Update bulk-flux wrappers, protocols, tests and benchmark
callers to the separated objects. Use dataclass replacement rather than
retaining named-tuple APIs as the main interface.

## Alternatives considered

Generating all dataclass schemas dynamically minimizes duplication but weakens
the explicit typing recently established in this repository. Explicit typed
fields with registry-derived defaults and schema tests are preferred.
Keeping a combined compatibility settings object would make fewer call-site
changes but would not deliver the requested separation. Retaining every
diagnostic in State would preserve its old shape but would not minimize AD.

## Documentation and verification

Build documentation tables directly from the three registries; add the
physical-constants reference to the Sphinx navigation and update quickstart,
artificial setup instructions and DESIGN.md. Metadata is the documentation
source rather than manually duplicated tables.

Write contract tests before implementation: frozen mutation rejection,
registry/schema parity, defaults and overrides, derived-value consistency,
complete allocation, minimal leaves, replacement, JIT and JVP/VJP traversal,
h5netcdf output and generated documentation. Preserve existing numerical
reference data and tolerances. Update tests that explicitly assert named-tuple
behavior to assert the new contract; retain independent physical oracles.
Exercise actual production dataclasses in integration and gradient tests.

Use one pytest process at a time, --fast during development, full CPU
correctness/coverage before each commit, and full GPU validation for the
completed migration. Run maintained Ruff formatting/lint, annotations and ty
checks. Revalidate serial and sharded initialization. Record failures and
verified results in CHANGELOG.md. Commit AGENTS.md and implementation on
jax-only; do not push or alter the reference example repository.

## Work sequence

1. Preserve a full numerical baseline and commit the user instructions/design.
2. Add registry and frozen configuration contracts, implement defaults and
   dependency validation, and audit scattered coefficients.
3. Migrate kernel signatures, protocols and consumers to separate constants.
4. Implement state metadata/allocation, minimal state and output diagnostics;
   migrate artificial and geometry initialization and halo configuration.
5. Generate reference documentation, verify netCDF metadata, run independent
   code/test review and full validation, then commit the completed migration.

Status: proposed design; production migration has not started.

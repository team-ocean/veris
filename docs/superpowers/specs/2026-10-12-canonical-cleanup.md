# Canonical package cleanup

Remove the compatibility layer added by the scientific package reorganization.
The user explicitly requests canonical imports and authorizes breaking obsolete
paths. Work on `jax-only`; preserve scientific equations, JAX execution and
differentiation, prescribed-ocean examples, and frozen numerical reference data.

Delete the 38 old-path modules, settings facade, model.types geometry/array
reexports, setup timestep wrappers, parallel output wrappers and unequal-clock
execution bypass. Keep preset factories and current CLI/root/io APIs that serve
canonical execution. Tests and examples use the responsible canonical modules.
Remove alias/pickle/wrapper-identity tests and obsolete migration documentation;
retain numerical, gradient, import-order, packaging and installed-example checks.

Verification requires lint, formatting, annotation and type checks, strict
Sphinx, clean wheel/sdist installation, the full CPU correctness suite in at
most 1200 seconds with maintained coverage at least 80%, and both native GPU
contracts. Commit only after all checks and independent review pass.

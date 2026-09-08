# Source typing implementation plan

Goal: explicit, useful type hints throughout project-owned code, with typed
interfaces to generated/vendor dependencies and unchanged numerical behavior.

- [x] Inventory public/private functions, registry values, dynamic state builders,
  mutable geometry initialization, and dependency boundaries.
- [x] Define structural read-only state/settings interfaces and exact array
  tuple returns. Keep mutable initializer fields separate. Test static positive
  and negative examples so missing fields and invalid scalar settings are caught.
- [x] Annotate model kernels, loop carries, bulk heat functions and halo callbacks.
- [x] Type standalone setup constructors, registries, tests/probes, documentation
  hooks and packaging; provide interfaces without editing excluded internals.
- [x] Publish typing metadata in wheel; verify runtime imports, JIT/AD and typing
  with valid/invalid usage examples, annotation coverage audit and ty.
- [x] Run full CPU correctness and affected GPU checks, lint/format, wheel and
  strict docs build; review, update changelog, commit to jax-only and verify CI.

No blanket Any or catch-all attribute protocol in numerical source. Dynamic
construction boundaries must be narrow and documented. Array dimensions and
units remain documented alongside JAX Array annotations; no runtime shape
checking is added to compiled kernels.

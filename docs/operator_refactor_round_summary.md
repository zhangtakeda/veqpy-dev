# Operator Refactor Round Summary

## Current Conclusion

This round's target architecture is now clear:

1. `veqpy.operator` owns semantics, layouts, and mutable runtime state.
2. `veqpy.engine` consumes plans/layouts/runtime buffers to build backend-specific `x -> residual` runners.
3. Backend-specific control semantics should not grow back inside `numba_operator.py` / `numpy_operator.py`.

In concrete terms, the operator layer is now organized around:

- Plans
  - `SourcePlan`
  - `ResidualPlan`
- Read-mostly layouts
  - `StaticLayout`
  - `SetupLayout`
  - `ResidualBindingLayout`
- Mutable runtime state
  - `FieldRuntimeState`
  - `SourceRuntimeState`
  - `ExecutionState`
- Thin operator-facing helpers
  - source runtime/materialization helpers
  - source orchestration helpers
  - runner binding helpers
  - stage helpers
  - profile setup helpers
  - runtime allocation helpers

`Operator` is no longer a class hierarchy. It is now effectively a facade/coordinator over:

- case/grid identity
- plans
- layouts
- runtime state
- backend runner bindings

## What Is Considered Done

- Remove `Operator` inheritance and collapse to a single owner object.
- Make execution/source/field runtime state explicit.
- Move source runtime/materialization logic out of `operator.py`.
- Move residual/stage/fused runner binding glue out of `operator.py`.
- Move stage/profile setup glue out of `operator.py`.
- Keep benchmark-critical kernels and JIT signatures unchanged.

## Final 3 Cuts

Only the following items were worth doing in this round:

1. Move remaining layout/runtime allocation/build glue out of `operator.py`.
2. Write the operator/engine boundary down as a stable ABI note for future JAX work.
3. Stop refactoring once `Operator` mainly contains lifecycle/orchestration methods plus lightweight property access.

## Stop Condition

This round should be considered complete once:

- `Operator` is clearly readable as an owner/coordinator.
- layout/state construction no longer dominates the file.
- no benchmark regression is observed on the maintained hotspot set.

Status now:

- Item 1 is complete. The minimal layout/build glue now lives directly in `veqpy/operator/operator.py`, while one-time runtime slab allocation remains in `veqpy/operator/runtime_allocation.py`.
- Item 2 is complete via `docs/operator_engine_abi.md`.
- Item 3 is complete. `Operator` now reads primarily as an owner/coordinator instead of a compatibility-heavy facade.

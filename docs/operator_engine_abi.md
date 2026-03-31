# Operator/Engine ABI Note

## Boundary

The intended Python-side ABI is:

- `operator` owns semantic decisions.
- `engine` owns backend-specific runner construction.

More concretely:

- `Operator` builds:
  - `SourcePlan`
  - `ResidualPlan`
  - `StaticLayout`
  - `SetupLayout`
  - `ResidualBindingLayout`
  - mutable runtime state containers
- `engine` receives:
  - plans
  - layouts
  - mutable runtime arrays
  - a small number of scalar case parameters

and returns backend-specific callables for:

- stage runners
- residual pack/full runners
- fused `x -> residual` runners

## What Should Stay Out Of Engine

The following should remain operator-side semantics:

- route/source strategy selection
- projection policy selection
- fixed-point orchestration policy
- warm-start/invalidation policy
- runtime buffer ownership

## What Can Differ By Backend

Backend implementations are free to differ in:

- fused vs staged binding strategy
- scratch/in-place usage
- metadata predecode strategy
- JIT compilation model

This is especially important for future JAX work:

- NumPy/Numba may prefer explicit mutable buffers.
- JAX may prefer pure-function state transitions over in-place style code.

The shared boundary should therefore be plan/layout/state based, not mutation-pattern based.

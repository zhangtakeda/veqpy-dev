# Repository Overview

## Purpose

This document answers four practical questions:

- What are the main layers in this repository?
- Which files own the `x -> residual -> solve -> snapshot` path?
- Which scripts and tests are the current entry points?
- What should a developer read first before changing runtime code?

For domain formulas, see the files under [`docs/theory/`](./theory).
For implementation constraints and invariants, see [`docs/guardrails.md`](./guardrails.md).

## Repository Map

The active codebase is organized into four layers:

- [`veqpy/model/`](../veqpy/model)
  - Grid definitions, profiles, geometry state, GEQDSK payloads, and equilibrium snapshots.
- [`veqpy/operator/`](../veqpy/operator)
  - Packed layout and codec ownership, case normalization, runtime state, and the main `x -> residual` operator.
- [`veqpy/engine/`](../veqpy/engine)
  - Array-first backend kernels and backend binding surfaces.
- [`veqpy/solver/`](../veqpy/solver)
  - SciPy solve orchestration, fallback behavior, history, and result objects.

## Main Entry Points

The most useful top-level entry points today are:

- [`README.md`](../README.md)
  - User-facing setup and command examples.
- [`tests/demo.py`](../tests/demo.py)
  - Minimal no-argument workflow demo.
- [`tests/demo_geqdsk_workflow.py`](../tests/demo_geqdsk_workflow.py)
  - `GEQDSK -> boundary fit -> solve -> simple flux-surface comparison` workflow.
- [`tests/benchmark.py`](../tests/benchmark.py)
  - Internal benchmark and route-comparison script.
- [`tests/test_model_core_regression.py`](../tests/test_model_core_regression.py)
  - Core model and operator-facing regression coverage.
- [`tests/test_solver_core_regression.py`](../tests/test_solver_core_regression.py)
  - Core solver lifecycle and fallback regression coverage.

## Supported Runtime Surface

The default and supported user-facing backend is `numba`.

There is no supported `jax` runtime surface in the current codebase. Any future
or experimental non-numba path should be documented as development-only until it
has matching tests, packaging, and user-facing guarantees.

## Environment

Repository commands are expected to run inside the project-managed Python environment.

Recommended workflow on Windows PowerShell:

```powershell
uv sync --group dev
```

Recommended command style:

- `uv run python -m pytest ...`
- `uv run python tests/demo.py`
- `uv run python tests/demo_geqdsk_workflow.py`
- `uv run python tests/benchmark.py`
- `uv run python -m compileall veqpy tests`

If an activated environment is already available, the corresponding `python ...`
commands are also acceptable. Avoid relying on a random system interpreter.

## Runtime Path

The main runtime path is centered on [`veqpy/operator/operator.py`](../veqpy/operator/operator.py).

The normal execution order is:

1. `Solver.solve(...)`
2. `scipy.optimize.root(...)` or `scipy.optimize.least_squares(...)`
3. `Operator.__call__(x)`
4. Stage A: `profile`
5. Stage B: `geometry`
6. Stage C: `source`
7. Stage D: `residual`

The stage entry points are:

- `Operator.stage_a_profile(...)`
- `Operator.stage_b_geometry(...)`
- `Operator.stage_c_source(...)`
- `Operator.stage_d_residual(...)`

## Ownership Summary

The core ownership model is:

- `Solver` owns nonlinear iteration strategy and solve lifecycle.
- `Operator` owns the runtime path from packed `x` to packed residual.
- `Equilibrium` is a post-solve materialized snapshot, not a live runtime owner.
- `Grid` and `OperatorCase` define the static problem shape and inputs, but not solver iteration state.

This is the most important mental model in the repository: the packed optimization
vector `x` is the only solver-facing state.

## Packed Layout Surface

Packed layout semantics are owned by:

- [`veqpy/operator/packed_layout.py`](../veqpy/operator/packed_layout.py)

That file defines:

- profile ordering
- coefficient indexing
- packed state topology
- packed residual position semantics
- packed state encoding/decoding helpers

If a change touches layout, indexing, coefficient activation rules, or packed
state codec behavior, this is the first file to inspect.

## Operator Runtime Surface

The operator layer owns the mutable runtime path from packed state to packed
residual. The most relevant implementation files are:

- [`veqpy/operator/operator.py`](../veqpy/operator/operator.py)
  - Main `Operator` facade, stage pipeline, `replace_case(...)`, and
    `build_equilibrium(...)` snapshot materialization.
- [`veqpy/operator/operator_case.py`](../veqpy/operator/operator_case.py)
  - Normalized case inputs, boundary offsets, and source/profile metadata.
- [`veqpy/operator/runtime_layout.py`](../veqpy/operator/runtime_layout.py)
  - Static/runtime/backend state containers and one-time runtime allocation.
- [`veqpy/operator/profile_runtime.py`](../veqpy/operator/profile_runtime.py)
  - Profile construction, profile runtime refresh, Stage-A binding, and Fourier
    family metadata refresh.

## Backend Binding Surface

Numba backend kernels and binding helpers live under [`veqpy/engine/`](../veqpy/engine).

The most relevant files are:

- [`veqpy/engine/backend_abi.py`](../veqpy/engine/backend_abi.py)
- [`veqpy/engine/numba_operator.py`](../veqpy/engine/numba_operator.py)
- [`veqpy/engine/numba_profile.py`](../veqpy/engine/numba_profile.py)
- [`veqpy/engine/numba_geometry.py`](../veqpy/engine/numba_geometry.py)
- [`veqpy/engine/numba_residual.py`](../veqpy/engine/numba_residual.py)
- [`veqpy/engine/numba_source.py`](../veqpy/engine/numba_source.py)
- [`veqpy/orchestration.py`](../veqpy/orchestration.py)

In practice:

- `operator` decides semantics and assembles runtime state.
- `orchestration` resolves route/source/residual metadata and stage runner policy.
- `engine` consumes array bundles and binds numba-specific runners.

## Snapshot Surface

Snapshot and inspection behavior are centered on:

- [`veqpy/model/equilibrium.py`](../veqpy/model/equilibrium.py)

Important public behaviors include:

- `plot(...)`
- `compare(...)`
- `resample(...)`
- `to_geqdsk(...)`

These APIs operate on a materialized equilibrium snapshot, not on the mutable runtime state used during solving.

## Reading Order For Developers

For most runtime changes, the fastest useful reading order is:

1. [`docs/guardrails.md`](./guardrails.md)
2. [`veqpy/operator/operator.py`](../veqpy/operator/operator.py)
3. [`veqpy/operator/packed_layout.py`](../veqpy/operator/packed_layout.py)
4. [`veqpy/operator/runtime_layout.py`](../veqpy/operator/runtime_layout.py)
5. [`veqpy/operator/profile_runtime.py`](../veqpy/operator/profile_runtime.py)
6. [`veqpy/orchestration.py`](../veqpy/orchestration.py)
7. The relevant numba kernel or binding file under [`veqpy/engine/`](../veqpy/engine)
8. The matching regression or script entry under [`tests/`](../tests)

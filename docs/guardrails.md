# Guardrails

## Purpose

This document records the constraints that should be treated as current implementation contracts.

It is not an idealized architecture note. It is a practical list of boundaries,
ownership rules, naming rules, and runtime invariants that are easy to break by accident.

## Language And Output Rules

- Runtime-facing text must be English.
- This applies to:
  - `print(...)`
  - warnings
  - exceptions
  - benchmark and report text
  - generated artifact summaries
- Documentation in this repository should prefer English for shared project docs.

## Layer Boundaries

### Model

[`veqpy/model/`](../veqpy/model) owns:

- grid definitions and precomputed tables
- profile and geometry containers
- GEQDSK payload I/O
- equilibrium snapshots and inspection APIs

It must not own:

- packed layout topology
- solve orchestration
- backend selection policy

### Operator

[`veqpy/operator/`](../veqpy/operator) is the runtime owner.

It owns:

- packed layout and codec semantics
- `OperatorCase` compatibility rules
- runtime buffers
- stage orchestration
- the full `x -> residual` path

It must not delegate packed-layout ownership back into `solver` or `tests`.

Current operator-layer implementation boundaries:

- [`veqpy/operator/packed_layout.py`](../veqpy/operator/packed_layout.py)
  owns packed layout and packed state codec helpers.
- [`veqpy/operator/runtime_layout.py`](../veqpy/operator/runtime_layout.py)
  owns runtime containers and one-time runtime allocation.
- [`veqpy/operator/profile_runtime.py`](../veqpy/operator/profile_runtime.py)
  owns profile setup/refresh rules and Stage-A runtime binding.
- [`veqpy/operator/operator.py`](../veqpy/operator/operator.py)
  owns the public `Operator` facade, stage calls, and snapshot materialization.

### Engine

[`veqpy/engine/`](../veqpy/engine) owns backend-facing numerical kernels and backend runner construction.

It should prefer:

- `ndarray`
- `float`
- `int`
- explicit index arrays
- explicit code arrays
- array bundles with stable slot conventions

It must not depend on rich Python object semantics in the hot path.

### Solver

[`veqpy/solver/`](../veqpy/solver) owns:

- solve policy
- `x0`
- fallback behavior
- history
- result packaging

It must not own:

- packed layout definitions
- backend selection semantics
- stage A/B/C/D implementations

## Core Ownership Model

The packed optimization vector `x` is the only solver-facing state.

From that rule:

- `Operator` is the runtime owner.
- `Solver` owns iteration, not physics state.
- `Equilibrium` is a post-solve snapshot, not a live runtime cache.

If a change creates a second state owner for the same runtime information, it is
probably architectural drift rather than a local refactor.

## Packed ABI Rules

The only authority for packed layout semantics is:

- [`veqpy/operator/packed_layout.py`](../veqpy/operator/packed_layout.py)

Required rules:

- packed state and packed residual position semantics must continue to flow through `coeff_index` / `coeff_indices`
- packed encode/decode helpers must use the same layout authority
- profile ordering must remain layout-driven rather than handwritten in scattered places
- `replace_case(...)` must not change packed topology

Forbidden regressions:

- reintroducing a second packed protocol
- reintroducing `coeff_matrix`
- reintroducing Python-side mirrored row caches as semantic owners

## Replace-Case Contract

`Operator.replace_case(...)` may update compatible case values, but it may not
change the packed topology.

If a change requires different active profile lengths, different coefficient
counts, or a different effective packed layout, the correct action is usually to
rebuild the `Operator`, not to hot-swap the case.

## Field Bundle ABI

At the engine boundary, stable array bundles are preferred over long exploded
argument lists.

Important bundle-style data includes:

- `T_fields`
- `u_fields`
- `rp_fields`
- `env_fields`
- `root_fields`
- geometry workspaces
- residual workspaces

Semantic convenience properties are still acceptable for readability and cold
paths, but the hot path should not expand back into long Python-managed call signatures
when stable bundles already exist.

## Stage Ownership

The stage pipeline is owned by [`veqpy/operator/operator.py`](../veqpy/operator/operator.py).

The current stage terms are fixed:

- `profile`
- `geometry`
- `source`
- `residual`

Preferred lifecycle terms are also fixed:

- `setup`
- `runtime`
- `refresh`
- `snapshot`

Avoid introducing near-synonyms when the existing vocabulary already matches the behavior.

## Hot-Path Rules

Engine hot paths should avoid:

- Python object chasing
- `None`-driven optional semantics inside tight kernels
- repeated Python-side block assembly when array kernels already exist

If a richer public semantic is needed, lower it in the facade or operator layer
before entering backend hot code.

## Snapshot Semantics

[`veqpy/model/equilibrium.py`](../veqpy/model/equilibrium.py) must continue to mean:

- one materialized equilibrium snapshot on one grid

Important semantic constraints:

- `resample(...)` means interpolation onto another grid, not exact parametric reconstruction
- `Equilibrium` must not become a second live runtime state container
- derived values should remain derived when practical instead of being stored as duplicated canonical state

## Fourier-Family Semantics

Do not confuse representational order with active runtime order.

Current meaning:

- `Grid.M_max` is the maximum representable order
- effective runtime order is determined by active coefficients and nonzero boundary content

This means a change that only raises `M_max` is not automatically a change in
what the hot path must compute for every case.

## Backend Support Surface

The supported user-facing backend is `numba`.

No `jax` module is part of the current supported runtime surface. If a future
`jax` or other non-numba path is added experimentally, it should not be
documented or treated as a stable user workflow.

Any documentation or code change that makes a non-numba path look
production-ready without the corresponding testing and packaging guarantees is a
contract drift.

## Script And Test Roles

Keep the distinction between these file types clear:

- `tests/demo.py` and `tests/demo_geqdsk_workflow.py`
  - user-facing workflow examples
- `tests/benchmark.py`
  - internal benchmark and comparison driver
- `tests/test_*.py`
  - regression evidence

Do not treat demo or benchmark scripts as substitutes for regression coverage.

## Comment And Documentation Rules

Use comments to explain:

- ownership
- invariants
- compatibility constraints
- numerical stability assumptions
- why a non-obvious workflow step exists

Avoid:

- line-by-line translation comments
- comments that restate obvious syntax
- long stale architecture prose embedded next to fast-moving implementation details

## Documentation Governance

When changing any of the following, review both [`docs/overview.md`](./overview.md)
and this file:

- packed layout or codec rules
- runtime ownership boundaries
- backend support surface
- stage pipeline semantics
- solver public behavior
- equilibrium snapshot semantics

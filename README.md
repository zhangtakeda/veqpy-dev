# veqpy

`veqpy` is a Python package for VEQ (Veloce/Variational Equilibrium),
a high-performance Python wrapper for plasma equilibrium simulations in magnetic confinement fusion (MCF) devices.

- Author: `rhzhang`
- Updated: `2026-03-24`

This README was prepared with Codex assistance. The source code remains the authoritative reference.

## What veqpy provides

- Packed-state equilibrium solving with a fixed `x -> residual` operator path.
- A layered runtime split into:
  - `engine` for array-first numerical kernels
  - `model` for grids, runtime buffers, and equilibrium snapshots
  - `operator` for packed layout, cases, and the full residual operator
  - `solver` for SciPy-based nonlinear solve orchestration
- Snapshot and post-processing utilities through `Equilibrium`.
- Benchmark and profiling entry points under `tests/`.

## Quick Start

Basic installation:

```bash
py -m pip install -e .
```

Development installation:

```bash
py -m pip install -e .[dev]
```

For a runnable example, see [tests/demo.py](E:/Dev/veqpy-dev/tests/demo.py).

## Project Layout

- `veqpy/engine/`
  - Backend export surface and the `numpy` / `numba` numerical kernels
- `veqpy/model/`
  - `Grid`, `Profile`, `Geometry`, and `Equilibrium`
- `veqpy/operator/`
  - Packed `layout/codec`, `OperatorCase`, and the full `x -> residual` operator path
- `veqpy/solver/`
  - `Solver`, `SolverConfig`, `SolverRecord`, and `SolverResult`
- `tests/`
  - `demo.py` example entry point
  - `benchmark.py` multi-mode benchmark and consistency-check entry point
  - `benchmark/` benchmark artifact directory

## Runtime Model

- `Operator` owns the packed runtime path from `x` to residual.
- `Solver` owns solve orchestration, warmstart state, and history records.
- `Equilibrium` is a snapshot object, not a mutable solver-side runtime.
- `OperatorCase` may change physical inputs and profile values, but may not change packed topology.

## Runtime Boundaries

- `Operator` owns the full packed `x -> residual` runtime path.
- `Operator.__call__(x)` currently executes the Stage A/B/C/D chain directly.
- `Solver` is a nonlinear solve facade. It does not own the packed layout/codec and does not choose the backend.
- `Solver.solve(...)` performs one solve and returns packed `x`.
- Stable post-solve entry points include:
  - `solver.result`
  - `solver.history`
  - `solver.build_equilibrium()`
  - `solver.build_equilibrium_history()`
  - `solver.build_coeffs()`
  - `solver.build_coeffs_history()`
- The backend is controlled only through `VEQPY_BACKEND`, with supported values:
  - `numpy`
  - `numba`

`veqpy.engine` defaults to `numba` when the environment variable is not set.

## Current Runtime ABI

- The authoritative profile order is:
  - `psin`, `F`, `h`, `v`, `k`, `c0`, `c1`, `s1`, `s2`
- Packed state and packed residual both use `coeff_index` / `coeff_indices` as the only layout language.
- Stage A reads profile coefficients directly from packed `x` using `coeff_indices`.
- Stage D writes residual blocks directly into packed residual output using `coeff_indices`.
- `coeff_matrix` is no longer part of the runtime path.
- Engine-facing hot-path ABI prefers packed field bundles over exploded argument lists.

Current packed field bundles include:

- `Grid.T_fields`
- `Profile.u_fields`
- `Profile.rp_fields`
- `Profile.env_fields`
- `Geometry.tb_fields`
- `Geometry.R_fields`
- `Geometry.Z_fields`
- `Geometry.J_fields`
- `Geometry.g_fields`
- `Operator.root_fields`
- `Operator.residual_fields`

Named properties such as `grid.T`, `profile.u`, and `geometry.R` remain valid semantic aliases, but hot operator paths prefer direct `*_fields[...]` access.

## Optional Semantics

- At the public/model layer, `None` remains valid where it expresses real topology or input semantics.
- Example: `coeffs_by_name[name] is None` means that profile is inactive in the packed layout.
- But hot engine kernels are intentionally kept monomorphic whenever possible.
- Therefore, hot Numba kernels prefer plain arrays and scalars over `optional(...)` arguments.
- Packed profile execution uses empty `coeff_indices` arrays instead of `None`.
- Source optional constraints (`Ip`, `beta`) remain a special case: semantic “optional” meaning is valid at the facade level, but hot kernels still use the current scalar ABI because direct `None` in Numba caused a measurable regression.

## Solver Capabilities

- `SolverConfig.method` currently supports:
  - Root-based methods: `hybr`, `krylov`, `root-lm`, `broyden1`, `broyden2`
  - Least-squares methods: `trf`, `dogbox`, `lm`
- Root-based methods use `scipy.optimize.root(...)`.
- `lm`, `trf`, and `dogbox` use `scipy.optimize.least_squares(...)` directly.
- If the primary method fails, `Solver` automatically retries in this order:
  - `least_squares/lm`
  - `least_squares/trf`
- When `enable_homotopy=True`, the staged solve expands the active set level by level in profile order.
- Homotopy also supports freezing higher-order shape coefficients through `homotopy_truncation_tol` and `homotopy_truncation_patience`.

## Key Files

- `veqpy/engine/__init__.py`
  - Backend control surface and stable exports
- `veqpy/operator/operator.py`
  - Main runtime path from packed state to residual
- `veqpy/operator/layout.py`
  - Packed layout definition
- `veqpy/operator/codec.py`
  - Packed state / residual encoding and decoding
- `veqpy/solver/solver.py`
  - Solve lifecycle entry point, root / least-squares / fallback / homotopy
- `veqpy/solver/solver_config.py`
  - Solver method configuration and staged-solve options
- `veqpy/model/equilibrium.py`
  - Snapshots, diagnostics, plotting, comparison, and resampling
- `tests/demo.py`
  - Minimal example and demo artifact entry point
- `tests/benchmark.py`
  - Multi-mode benchmark, delta checks, and benchmark artifact entry point

## Notes

- `replace_case(...)` only supports `OperatorCase` instances compatible with the packed layout.
- `SolverRecord` copies `OperatorCase` snapshots to prevent later in-place updates from contaminating history.
- `Grid` is immutable.
- `Equilibrium.resample(...)` is snapshot interpolation, not strict parametric reconstruction.
- If you change the packed layout, packed codec, operator contract, solver control flow, or engine exports, update `docs/` as well.

## Related Documentation

- [`docs/overview.md`](docs/overview.md)
- [`docs/conventions.md`](docs/conventions.md)
- [`docs/guardrails.md`](docs/guardrails.md)

### Runtime/Passive Engines

[docs/theory/profile.md](docs/theory/profile.md): Parameterized profile computation

- `veqpy/engine/numpy_profile.py`
- `veqpy/engine/numba_profile.py`

[docs/theory/geometry.md](docs/theory/geometry.md): Metric and geometry computation

- `veqpy/engine/numpy_geometry.py`
- `veqpy/engine/numba_geometry.py`

[docs/theory/source.md](docs/theory/source.md): Source term computation and source families definition

- `veqpy/engine/numpy_source.py`
- `veqpy/engine/numba_source.py`

[docs/theory/residual.md](docs/theory/residual.md): Residual field and packed residual assembly

- `veqpy/engine/numpy_residual.py`
- `veqpy/engine/numba_residual.py`

### Facade Models

[docs/theory/equilibrium.md](docs/theory/equilibrium.md): Data-driven equilibrium snapshot modeling and diagnostics

- `veqpy/model/equilibrium.py`

## Profiling Notes

- There are two useful solve timing metrics:
  - full wall-clock around `solver.solve(...)`
  - `SolverResult.elapsed`, which measures only the core solve-attempt region around `_solve_with_fallbacks(...)`
- `SolverResult.elapsed` does not include later `SolverRecord` construction or `history.append(...)`.
- Therefore:
  - use full wall-clock when you want user-visible end-to-end `solve(...)` latency
  - use `SolverResult.elapsed` when you want solve-core latency

The tables below come from a warm single solve of the demo case with internal source-level probes enabled and `enable_history=True`.

### Timing Scope

PF mode, 9 parameters, 12x12, all coefficients set to zero.

### Solve Breakdown

| Item                                 |       Time | Share of full solve |
| ------------------------------------ | ---------: | ------------------: |
| full `solver.solve(...)` wall time   | `0.595 ms` |            `100.0%` |
| solve core (`_solve_with_fallbacks`) | `0.550 ms` |             `92.4%` |
| SciPy solve call(s)                  | `0.505 ms` |             `84.9%` |
| `SolverResult` construction          | `0.005 ms` |              `0.8%` |
| `SolverRecord` construction          | `0.020 ms` |              `3.4%` |
| `history.append(...)`                | `0.000 ms` |              `0.0%` |
| history bookkeeping total            | `0.021 ms` |              `3.5%` |

### Residual Summary

| Item                         |      Value |
| ---------------------------- | ---------: |
| solve success                |     `True` |
| `nfev`                       |       `25` |
| packed state size (`x_size`) |        `9` |
| residual calls per solve     |       `25` |
| residual total               | `0.443 ms` |
| average per residual         | `0.018 ms` |

### Stage Totals

| Stage            | Total time | Share of full solve | Average per residual |
| ---------------- | ---------: | ------------------: | -------------------: |
| Stage-A profile  | `0.038 ms` |              `6.4%` |           `0.002 ms` |
| Stage-B geometry | `0.145 ms` |             `24.3%` |           `0.006 ms` |
| Stage-C source   | `0.121 ms` |             `20.4%` |           `0.005 ms` |
| Stage-D residual | `0.103 ms` |             `17.4%` |           `0.004 ms` |

### Stage Engine Share

| Stage   | Stage total | Pure engine time | Engine share of stage | Non-engine remainder | Non-engine share of stage |
| ------- | ----------: | ---------------: | --------------------: | -------------------: | ------------------------: |
| Stage-A |  `0.038 ms` |       `0.027 ms` |               `71.8%` |           `0.011 ms` |                   `28.2%` |
| Stage-B |  `0.145 ms` |       `0.128 ms` |               `88.5%` |           `0.017 ms` |                   `11.5%` |
| Stage-C |  `0.121 ms` |       `0.096 ms` |               `79.6%` |           `0.025 ms` |                   `20.4%` |
| Stage-D |  `0.103 ms` |       `0.072 ms` |               `69.8%` |           `0.031 ms` |                   `30.2%` |

### Current Non-Engine Sources

| Stage   | Main non-engine sources                                                                                                               |
| ------- | ------------------------------------------------------------------------------------------------------------------------------------- |
| Stage-A | operator-side stage call boundary and remaining profile bulk wrapper overhead around `update_profiles_packed_bulk(...)`               |
| Stage-B | `Geometry.update(...)` wrapper, argument passing, and Python-to-engine call boundary                                                  |
| Stage-C | `source_runner(...)` wrapper, scalar/array argument passing, and Python-to-engine call boundary                                       |
| Stage-D | `stage_d_residual()` / `_assemble_residual()` Python shell plus fresh packed residual allocation before residual blocks write into it |

## Reference

[1] Huasheng Xie and Yueyan Li, "What Is the Minimum Number of Parameters Required to Represent Solutions of the Grad-Shafranov Equation?," arXiv:2601.02942, 2026. [https://arxiv.org/abs/2601.02942](https://arxiv.org/abs/2601.02942)
[2] Xingyu Li, Huasheng Xie, Lai Wei, and Zhengxiong Wang, "Investigation of Toroidal Rotation Effects on Spherical Torus Equilibria using the Fast Spectral Solver VEQ-R," arXiv:2602.11422, 2026. [https://arxiv.org/abs/2602.11422](https://arxiv.org/abs/2602.11422)

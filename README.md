# veqpy

`veqpy` is a Python package for VEQ (Veloce/Variational Equilibrium),
a high-performance Python wrapper for plasma equilibrium simulations in magnetic confinement fusion (MCF) devices.

- Author: `rhzhang`
- Updated: `2026-04-15`
- Version: `0.3.0`

## Code Structure

- `veqpy/engine/`
  - Array-first numerical kernels.
  - `numba` is the supported user-facing backend for `profile`, `geometry`, `source`, and `residual`.
  - `jax` backend work is experimental and still under development; it is not intended for users.
- `veqpy/model/`
  - Passive or snapshot-oriented objects: `Grid`, `Profile`, `Geometry`, `Equilibrium`.
- `veqpy/operator/`
  - Packed layout and the main `x -> residual` runtime path.
  - Owns `OperatorCase`, packed `layout/codec`, and `Operator`.
- `veqpy/solver/`
  - SciPy solve orchestration, fallback logic, history, and result objects.
- `tests/`
  - Focused regression tests.

## Environment

All scripts, benchmarks, compile checks, and `pytest` runs should be executed inside the project `uv`-managed virtual environment.

Recommended workflow on Windows PowerShell:

```powershell
uv sync --group dev
```

If you want JAX inside a Linux devcontainer:

```powershell
uv sync --group dev --extra jax-gpu
```

The repository devcontainer is based on `python:3.12-slim`, passes through `--gpus=all`, and installs dependencies into the container's system Python with `uv sync --system --extra jax-gpu`. This matches JAX's pip-based NVIDIA GPU installation model, which supports the CUDA wheels on Linux containers rather than native Windows Python.

JAX is currently a backend-development path only. It is not part of the supported end-user workflow, and the current implementation covers only a narrow experimental route.

Recommended command style:

- `python -m pytest ...`
- `python tests/demo.py`
- `python tests/demo_geqdsk_workflow.py`
- `python tests/benchmark.py`
- `python -m compileall veqpy tests`

Prefer running them through `uv`:

- `uv run python -m pytest ...`
- `uv run python tests/demo.py`
- `uv run python tests/demo_geqdsk_workflow.py`
- `uv run python tests/benchmark.py`
- `uv run python -m compileall veqpy tests`

If you prefer an activated shell, use:

```powershell
.\.venv\Scripts\Activate.ps1
```

Then run the same `python ...` commands inside that environment.  
Avoid running repository commands from system Python or a non-project interpreter.  
If you do not activate the environment, use `uv run ...` or `.\.venv\Scripts\python.exe ...` explicitly.

## Regression Suites

Core regressions are now organized by submodule instead of by temporary refactor phase:

- [tests/test_model_core_regression.py](tests/test_model_core_regression.py)
  - `Grid`, `Boundary.from_geqdsk`, `Equilibrium` snapshot/serialization/comparison semantics
- [tests/test_solver_core_regression.py](tests/test_solver_core_regression.py)
  - solve facade, fallback/reset behavior, and solver/operator state lifecycle semantics

The main runtime path is:

1. `Solver.solve(...)`
2. SciPy root or least-squares entry
3. `Operator.__call__(x)`
4. Stage-A `profile`
5. Stage-B `geometry`
6. Stage-C `source`
7. Stage-D `residual`

## Performance Snapshot

Current Fourier-family runtime is driven by `Grid.M_max`, but hot-path kernels only compute up to the current effective active order.

- Low-order case with the same active profiles:
  - `M_max=4` vs `M_max=2` currently gives about `1.05x` full residual time in the `numba` microbenchmark.
  - Geometry alone is about `1.03x`.
- When higher-order terms are actually active:
  - `M_max=4` high-order case is about `1.23x` full residual time relative to the low-order `M_max=2` baseline.

This means the remaining cost from the more general `M_max` architecture is small in steady-state hot paths; most extra cost appears only when higher-order Fourier terms are really in use.

Benchmark entry points:

- [benchmark.py](tests/benchmark.py)

## Project Understanding

The project is organized around one core idea: packed optimization variables are the only solver-facing state, and everything else is derived from them through a fixed operator pipeline.

- `Grid` defines the discretization and precomputed tables.
- `OperatorCase` defines physical inputs and boundary/profile metadata, but not solver state.
- `Operator` owns the runtime buffers and transforms packed `x` into residuals.
- `Solver` owns nonlinear iteration strategy.
- `Equilibrium` is a post-solve materialized snapshot, not a live runtime owner.

Current shape parameterization is a dynamic Fourier family:

- Packed profile order is generated from `Grid.M_max`.
- Boundary offsets are stored canonically in:
  - `OperatorCase.c_offsets`
  - `OperatorCase.s_offsets`
- Runtime geometry/residual kernels consume Fourier family stacks rather than hard-coded `c1/s2` slots.

## Collaboration Notes

For future development, keep these boundaries stable:

- Do not put grid-construction logic into hot runtime code.
- Do not reintroduce fixed low-order special cases such as dedicated `c1/s2` runtime paths unless a benchmark proves they are necessary.
- Keep packed layout authority in `veqpy/operator/layout.py`.
- Keep `OperatorCase` as normalized input state, not layout owner and not solver owner.
- Keep `Equilibrium` snapshot-oriented; it should not become a second runtime state container.
- Prefer extending family-based kernels over adding more named one-off profile fields.

When changing performance-sensitive code:

- Benchmark before and after.
- Check low-order cases separately from truly high-order active cases.
- Treat initialization-path cost and steady-state residual cost as different problems.

## Related Docs

- [docs/overview.md](docs/overview.md)
- [docs/guardrails.md](docs/guardrails.md)
- [docs/theory/profile.md](docs/theory/profile.md)
- [docs/theory/geometry.md](docs/theory/geometry.md)
- [docs/theory/residual.md](docs/theory/residual.md)
- [docs/theory/equilibrium.md](docs/theory/equilibrium.md)
- Huasheng Xie and Yueyan Li, "What Is the Minimum Number of Parameters Required to Represent Solutions of the Grad-Shafranov Equation?," arXiv:2601.02942, 2026. [https://arxiv.org/abs/2601.02942](https://arxiv.org/abs/2601.02942)
- Xingyu Li, Huasheng Xie, Lai Wei, and Zhengxiong Wang, "Investigation of Toroidal Rotation Effects on Spherical Torus Equilibria using the Fast Spectral Solver VEQ-R," arXiv:2602.11422, 2026. [https://arxiv.org/abs/2602.11422](https://arxiv.org/abs/2602.11422)

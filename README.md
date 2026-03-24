# veqpy

`veqpy` is a Python package for VEQ (Veloce/Variational Equilibrium),
a high-performance Python wrapper for plasma equilibrium simulations in magnetic confinement fusion (MCF) devices.

- Author: `rhzhang`
- Updated: `2026-03-24`

This README was prepared with Codex assistance. The source code remains the authoritative reference.

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

## Installation

Basic installation:

```bash
py -m pip install -e .
```

Development installation:

```bash
py -m pip install -e .[dev]
```

## Minimal Example

```python
import numpy as np

from veqpy.model import Grid
from veqpy.operator import Operator, OperatorCase
from veqpy.solver import Solver, SolverConfig

grid = Grid(Nr=12, Nt=12, scheme="legendre")

coeffs = {
    "h": [0.0, 0.0, 0.0],
    "v": None,
    "k": [0.0, 0.0, 0.0],
    "c0": None,
    "c1": None,
    "s1": [0.0, 0.0, 0.0],
    "s2": None,
}

rho = grid.rho
psin = rho**2
psin_r = 2.0 * rho
beta0 = 0.75
alpha_p, alpha_f = 5.0, 3.32
exp_ap, exp_af = np.exp(alpha_p), np.exp(alpha_f)
den_p = 1.0 + exp_ap * (alpha_p - 1.0)
den_f = 1.0 + exp_af * (alpha_f - 1.0)

current_input = (1.0 - beta0) * alpha_f * (np.exp(alpha_f * psin) - exp_af) / den_f * psin_r
heat_input = beta0 * alpha_p * (np.exp(alpha_p * psin) - exp_ap) / den_p * psin_r

case = OperatorCase(
    coeffs_by_name=coeffs,
    a=1.05 / 1.85,
    R0=1.05,
    Z0=0.0,
    B0=3.0,
    ka=2.2,
    s1a=float(np.arcsin(0.5)),
    heat_input=heat_input,
    current_input=current_input,
    Ip=3.0e6,
)

operator = Operator(grid=grid, case=case, name="PF", derivative="rho")
solver = Solver(operator=operator, config=SolverConfig(method="hybr", enable_warmstart=False))

x = solver.solve()
eq = solver.build_equilibrium()

print(solver.result.success, x.shape)
print(float(eq.Ip), float(eq.beta_t))
```

For a more complete runnable example, see `tests/demo.py`.

## Common Commands

Syntax check:

```bash
py -m compileall veqpy tests
```

Run the demo and generate demo artifacts:

```bash
py tests/demo.py
```

Run the multi-mode benchmark and delta checks:

```bash
py tests/benchmark.py
```

## Generated Artifacts

After running `py tests/demo.py`, the following files are generated under `tests/`:

- `demo-1.json` / `demo-1.png`
- `demo-2.json` / `demo-2.png`
- `demo-3.json` / `demo-3.png`
- `demo-4.json` / `demo-4.png`
- `demo-coeffs-comparison.png`
- `demo-grid-comparison.png`
- `demo-homo-comparison.png`

After running `py tests/benchmark.py`, the default outputs are:

- `tests/benchmark/cold-<backend>/pf_reference_summary.png`
- `tests/benchmark/cold-<backend>/pf_reference_summary.txt`
- `tests/benchmark/cold-<backend>/benchmark_compare.txt`
- `tests/benchmark/cold-<backend>/benchmark_notes.txt`
- `tests/benchmark/cold-<backend>/plots/`

If `WARMSTART` in `tests/benchmark.py` is switched to `True`, the artifact root changes to `tests/benchmark/warm-<backend>/`.

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
- `OperatorCase` is a mutable runtime case and is suitable for live updates to `Ip`, `beta`, `heat_input`, and `current_input`.
- `SolverRecord` copies `OperatorCase` snapshots to prevent later in-place updates from contaminating history.
- `Grid` is immutable.
- `Equilibrium` is a single-grid snapshot, not a solver-side mutable state object.
- `Equilibrium.resample(...)` is snapshot interpolation, not strict parametric reconstruction.
- If you change the packed layout, packed codec, operator contract, solver control flow, or engine exports, update `docs/` as well.

## Related Documentation

- [`docs/overview.md`](docs/overview.md)
- [`docs/conventions.md`](docs/conventions.md)
- [`docs/guardrails.md`](docs/guardrails.md)
- [`docs/veqpy_operators.md`](docs/veqpy_operators.md)
- [`docs/veqpy_equilibrium.md`](docs/veqpy_equilibrium.md)

## Reference

[1] Huasheng Xie and Yueyan Li, "What Is the Minimum Number of Parameters Required to Represent Solutions of the Grad-Shafranov Equation?," arXiv:2601.02942, 2026. [https://arxiv.org/abs/2601.02942](https://arxiv.org/abs/2601.02942)
[2] Xingyu Li, Huasheng Xie, Lai Wei, and Zhengxiong Wang, "Investigation of Toroidal Rotation Effects on Spherical Torus Equilibria using the Fast Spectral Solver VEQ-R," arXiv:2602.11422, 2026. [https://arxiv.org/abs/2602.11422](https://arxiv.org/abs/2602.11422)

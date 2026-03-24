# Repository Overview

- `veqpy` 是 Python 3.12+ 的磁流体平衡仓库.
- 当前只支持 `numpy` 和 `numba` 两种后端, 不再维护 native C++ 构建链路.
- 当前核心层次是:
  - `veqpy/engine/`: backend-facing array kernels 和导出面
  - `veqpy/model/`: `Grid`, `Profile`, `Geometry`, `Equilibrium`
  - `veqpy/operator/`: `OperatorCase`, packed `layout/codec`, `Operator`
  - `veqpy/solver/`: `Solver`, `SolverConfig`, `SolverRecord`, `SolverResult`
  - `tests/`: `demo.py`, `benchmark.py`, `benchmark/<startup>-<backend>/`

# Current Facts

- `Operator` 是完整的 packed `x -> residual` runtime owner.
- Stage A/B/C/D 当前仍由 [`veqpy/operator/operator.py`](../veqpy/operator/operator.py) 组织.
- packed `layout/codec` 归 `veqpy/operator/`.
- `Solver` 是 solve facade, 不持有 backend 选择逻辑.
- `Solver.solve(...)` 只执行求解并返回 packed `x`; `SolverResult` / `Equilibrium` / coeff 重建由 `solver.result`、`solver.build_equilibrium()`、`solver.build_coeffs()` 提供.
- `Solver` 还提供 `build_equilibrium_history()` 和 `build_coeffs_history()` 作为 history 重建入口.
- `OperatorCase` 当前是可变 runtime case; `SolverRecord` 会复制 case snapshot.
- `Equilibrium` 是单网格 materialized diagnostic snapshot, 不是 solver-side parametric state.
- 当前运行时浮点基线固定为 `np.float64`.

# Backend Surface

- backend control surface 只有 [`veqpy/engine/__init__.py`](../veqpy/engine/__init__.py).
- 当前真实后端文件是:
  - `numpy_profile.py`
  - `numpy_geometry.py`
  - `numpy_source.py`
  - `numpy_residual.py`
  - `numba_profile.py`
  - `numba_geometry.py`
  - `numba_source.py`
  - `numba_residual.py`
- `VEQPY_BACKEND` 可接受值只有:
  - `numpy`
  - `numba`
- 环境变量未设置时, 默认后端是 `numba`.

# Developer Workflows

- 安装:
  - `py -m pip install -e .`
  - `py -m pip install -e .[dev]`
- 语法快速检查:
  - `py -m compileall veqpy tests`
- 运行最小示例并生成 demo 产物:
  - `py tests/demo.py`
- 运行多模式 benchmark:
  - `py tests/benchmark.py`

# Solver Surface

- `SolverConfig.method` 当前支持:
  - root 路径: `hybr`, `krylov`, `root-lm`, `broyden1`, `broyden2`
  - least-squares 路径: `trf`, `dogbox`, `lm`
- `Solver.solve(...)` 对 root 方法使用 `scipy.optimize.root(...)`.
- `Solver.solve(...)` 对 `trf` / `dogbox` / `lm` 使用 `scipy.optimize.least_squares(...)`.
- 当主方法失败时, 当前 fallback 顺序是:
  - `least_squares/lm`
  - `least_squares/trf`
- `SolverConfig` 当前主要包含三类字段:
  - 求解方法和收敛阈值: `method`, `rtol`, `atol`
  - SciPy 限制字段: `root_maxiter`, `root_maxfev`
  - solve 行为开关: `enable_warmstart`, `enable_homotopy`, `enable_verbose`, `enable_history`
- homotopy 相关策略字段还包括:
  - `homotopy_truncation_tol`
  - `homotopy_truncation_patience`
- `Solver.solve(...)` 支持对上述字段做单次覆盖, 但不接收 `case`.
- 长期替换物理 case 用 `replace_case(...)`.
- 长期替换默认求解配置用 `replace_config(...)`.

# Suggested Checks

- 只改 `README.md` 或 `doc/`:
  - 不强制跑数值脚本.
  - 至少核对路径、命令、产物目录仍真实存在.
- 改任意 `veqpy/*.py` 或 `tests/*.py`:
  - 建议先跑 `py -m compileall veqpy tests`
- 改 `veqpy/engine/`, `veqpy/model/`, `veqpy/operator/`, `veqpy/solver/`, 包级 `__init__.py`, packed `layout/codec`, `Operator`, `OperatorCase`, `Solver`, `Equilibrium`:
  - 建议跑 `py -m compileall veqpy tests`
  - 建议跑 `py tests/demo.py`
- 改 source route, residual assembly, solver fallback, homotopy stage policy, benchmark 口径:
  - 额外建议跑 `py tests/benchmark.py`

# High-Risk Areas

- `veqpy/engine/__init__.py`
  - 错误导出会直接破坏 backend surface.
- `veqpy/operator/operator.py`
  - packed runtime ownership, Stage A/B/C/D, residual assembly 当前都在这里收束.
- `veqpy/operator/layout.py`
  - packed index 变化会影响 `x0`, residual, `replace_case(...)`, benchmark 产物和文档.
- `veqpy/operator/codec.py`
  - packed state 和 packed residual 的边界转码都依赖这里.
- `veqpy/solver/solver.py`
  - root / least-squares 路径、fallback、homotopy stage policy 都在这里.
- `veqpy/model/equilibrium.py`
  - snapshot semantics, resample semantics, plotting/comparison 在这里定义.
- `tests/demo.py`
  - 当前最直接的端到端示例入口.
- `tests/benchmark.py`
  - 当前多模式 benchmark 和 physics delta 观察口径入口.

# Quick File Map

- `README.md`
- `TODO.md`
- `pyproject.toml`
- `veqpy/engine/__init__.py`
- `veqpy/operator/operator.py`
- `veqpy/operator/layout.py`
- `veqpy/operator/codec.py`
- `veqpy/solver/solver.py`
- `veqpy/solver/solver_config.py`
- `veqpy/model/equilibrium.py`
- `tests/demo.py`
- `tests/benchmark.py`
- `doc/conventions.md`
- `doc/guardrails.md`
- `doc/veqpy_operators.md`
- `doc/veqpy_equilibrium.md`

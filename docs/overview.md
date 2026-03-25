# Repository Overview

## Scope

这份文档只回答三个问题:

- 当前仓库的代码层次是什么.
- 当前 `x -> residual -> solve -> snapshot` 链路由哪些文件负责.
- 修改时优先看哪些入口和风险点.

如果要看风格和注释规则, 读 [`docs/conventions.md`](./conventions.md).  
如果要看边界合同和不能打破的约束, 读 [`docs/guardrails.md`](./guardrails.md).

## Repository Map

当前主代码只分四层:

- [`veqpy/engine/`](../veqpy/engine)
  - 数组导向的 backend kernels 和 backend 导出面.
- [`veqpy/model/`](../veqpy/model)
  - `Grid`, `Profile`, `Geometry`, `Equilibrium`.
- [`veqpy/operator/`](../veqpy/operator)
  - packed `layout/codec`, `OperatorCase`, `Operator`.
- [`veqpy/solver/`](../veqpy/solver)
  - `Solver`, `SolverConfig`, `SolverRecord`, `SolverResult`.

配套入口:

- [`tests/demo.py`](../tests/demo.py)
  - 最小示例和 demo 产物入口.
- [`tests/benchmark.py`](../tests/benchmark.py)
  - 多模式 benchmark 和差异检查入口.

## Runtime Path

当前 runtime 主路径以 [`veqpy/operator/operator.py`](../veqpy/operator/operator.py) 为中心.

执行顺序是:

1. `Solver.solve(...)`
2. `scipy.optimize.root(...)` 或 `scipy.optimize.least_squares(...)`
3. `Operator.__call__(x)`
4. Stage-A `profile`
5. Stage-B `geometry`
6. Stage-C `source`
7. Stage-D `residual`

当前四阶段的真实入口在 [`veqpy/operator/operator.py`](../veqpy/operator/operator.py):

- `stage_a_profile(...)`
- `stage_b_geometry(...)`
- `stage_c_source(...)`
- `stage_d_residual(...)`

## Packed ABI

packed runtime 的权威定义只在 [`veqpy/operator/layout.py`](../veqpy/operator/layout.py) 和 [`veqpy/operator/codec.py`](../veqpy/operator/codec.py).

当前必须知道的事实:

- profile 权威顺序固定为:
  - `psin`, `F`, `h`, `v`, `k`, `c0`, `c1`, `s1`, `s2`
- packed state 和 packed residual 的唯一位置语义都是:
  - `coeff_index`
  - `coeff_indices`
- runtime 路径里已经不再维护 `coeff_matrix`

这意味着:

- Stage-A 直接从 packed `x` 读取 profile coefficients
- Stage-D 直接向 packed residual 写 block 输出

## Engine ABI

engine 边界当前优先使用 field bundles, 不优先使用 exploded argument lists.

当前主要 bundles 是:

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

热路径可以直接使用 `*_fields[...]`.  
语义化 property 仍然保留给阅读和冷路径使用, 例如:

- `grid.T`
- `profile.u`
- `geometry.R`

## Backend Surface

backend control surface 只有 [`veqpy/engine/__init__.py`](../veqpy/engine/__init__.py).

当前真实 backend 文件是:

- `numpy_profile.py`
- `numpy_geometry.py`
- `numpy_source.py`
- `numpy_residual.py`
- `numba_profile.py`
- `numba_geometry.py`
- `numba_source.py`
- `numba_residual.py`

环境变量 `VEQPY_BACKEND` 当前只接受:

- `numpy`
- `numba`

未设置时默认使用 `numba`.

## Solver Surface

[`veqpy/solver/solver.py`](../veqpy/solver/solver.py) 是 solve facade.

它负责:

- 持有 `x0`
- 调用 SciPy solve API
- fallback 编排
- history 记录
- 从结果重建 coeff 和 `Equilibrium`

它不负责:

- packed `layout/codec`
- backend 选择
- Stage A/B/C/D 数值核

当前 `SolverConfig.method` 支持:

- root 路径:
  - `hybr`, `krylov`, `root-lm`, `broyden1`, `broyden2`
- least-squares 路径:
  - `trf`, `dogbox`, `lm`

## Snapshot Surface

[`veqpy/model/equilibrium.py`](../veqpy/model/equilibrium.py) 是 snapshot 和 inspection 层入口.

当前要点:

- `Equilibrium` 表示单网格 materialized snapshot
- `Equilibrium.resample(...)` 表示 snapshot 插值后重建
- `Equilibrium.compare(...)` 和 `plot_comparison(...)` 现在只做 1D 比较, 不再承担 resample 接口

这层不是 solver runtime owner, 也不是 packed state owner.

## Current Stage Ownership

当前四阶段的 owner 如下:

- Stage-A `profile`
  - `Operator.stage_a_profile(...)`
  - `update_profiles_packed_bulk(...)`
- Stage-B `geometry`
  - `Operator.stage_b_geometry(...)`
  - `Geometry.update(...)`
  - `update_geometry(...)`
- Stage-C `source`
  - `Operator.source_stage_runner(...)`
- Stage-D `residual`
  - `Operator._build_G_inplace()`
  - `Operator.residual_stage_runner(...)`

当前代码已经是:

- Stage-A bulk profile update
- Stage-D engine-level residual runner

不再是旧的:

- `coeff_matrix` 中转
- per-block Python residual assembly

## Key Files

建议优先熟悉这些文件:

- [`veqpy/engine/__init__.py`](../veqpy/engine/__init__.py)
- [`veqpy/operator/operator.py`](../veqpy/operator/operator.py)
- [`veqpy/operator/layout.py`](../veqpy/operator/layout.py)
- [`veqpy/operator/codec.py`](../veqpy/operator/codec.py)
- [`veqpy/solver/solver.py`](../veqpy/solver/solver.py)
- [`veqpy/solver/solver_config.py`](../veqpy/solver/solver_config.py)
- [`veqpy/model/equilibrium.py`](../veqpy/model/equilibrium.py)
- [`tests/demo.py`](../tests/demo.py)
- [`tests/benchmark.py`](../tests/benchmark.py)

## Suggested Checks

只改文档:

- 至少核对路径, 命令, 环境变量, 入口文件仍存在.

改任意 Python 源码:

- `py -m compileall veqpy tests`

改 runtime 主链:

- `py -m compileall veqpy tests`
- `py tests/demo.py`

改 solver 策略, benchmark 口径, source route, residual runner:

- `py -m compileall veqpy tests`
- `py tests/demo.py`
- `py tests/benchmark.py`

## High-Risk Files

- [`veqpy/engine/__init__.py`](../veqpy/engine/__init__.py)
  - backend control surface
- [`veqpy/operator/operator.py`](../veqpy/operator/operator.py)
  - runtime owner, stage orchestration, residual path
- [`veqpy/operator/layout.py`](../veqpy/operator/layout.py)
  - packed ABI definition
- [`veqpy/operator/codec.py`](../veqpy/operator/codec.py)
  - packed state / residual codec
- [`veqpy/solver/solver.py`](../veqpy/solver/solver.py)
  - SciPy solve path, fallback, homotopy
- [`veqpy/model/equilibrium.py`](../veqpy/model/equilibrium.py)
  - snapshot semantics, plotting, comparison, resample

## Open Questions

这份文档不记录未来方案, 只记录当前代码事实。  
如果某个提案还没落地到代码, 不要先写进这里。

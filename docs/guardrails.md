# Guardrails

## Purpose

这份文档只记录当前代码必须守住的边界合同。  
它不描述理想架构, 只描述当前实现中不能随意打破的事实。

## Layer Boundaries

### Engine

位置:

- [`veqpy/engine/`](../veqpy/engine)

必须:

- 只放 backend-facing 数值核和 backend 导出面.
- hot-path ABI 优先使用:
  - `ndarray`
  - `float`
  - `int`
  - 显式长度数组 / 索引数组 / code 数组
- 通过 [`veqpy/engine/__init__.py`](../veqpy/engine/__init__.py) 暴露稳定入口.

不得:

- 不持有 packed layout owner.
- 不持有 `OperatorCase`, `Solver`, `Equilibrium`.
- 不把 Python 对象语义直接拖进 hot kernels.

### Model

位置:

- [`veqpy/model/`](../veqpy/model)

必须:

- `Grid` 表示网格和 basis tables.
- `Profile` / `Geometry` 表示 runtime buffers.
- `Equilibrium` 表示 snapshot 和 inspection surface.

不得:

- 不持有 packed layout owner.
- 不承担 SciPy solve orchestration.
- 不把 snapshot 和 runtime owner 混成一个对象.

### Operator

位置:

- [`veqpy/operator/`](../veqpy/operator)

必须:

- 持有 packed `profile_L`, `coeff_index`, `order_offsets`.
- 持有完整的 `x -> residual` runtime 路径.
- 持有四阶段组织:
  - `profile`
  - `geometry`
  - `source`
  - `residual`
- 持有 `OperatorCase` 兼容性检查和 `replace_case(...)` 逻辑.

不得:

- 不把 packed ABI owner 拆回 `solver`.
- 不把 stage owner 分散到 `tests`.

### Solver

位置:

- [`veqpy/solver/`](../veqpy/solver)

必须:

- 只负责 solve policy 和 solve lifecycle.
- 持有:
  - `x0`
  - `config`
  - `result`
  - `history`
- 通过 `Operator` 调 residual.

不得:

- 不持有 packed layout/codec owner.
- 不持有 backend 选择逻辑.
- 不重写 Stage A/B/C/D.

## Packed ABI

当前 packed ABI 的唯一权威位置在:

- [`veqpy/operator/layout.py`](../veqpy/operator/layout.py)
- [`veqpy/operator/codec.py`](../veqpy/operator/codec.py)

必须:

- packed state 和 packed residual 都只用 `coeff_index` / `coeff_indices` 表示位置语义.
- profile 权威顺序固定为:
  - `psin`, `F`, `h`, `v`, `k`, `c0`, `c1`, `s1`, `s2`

不得:

- 不重新引入第二套 packed 协议.
- 不重新引入 `coeff_matrix`.
- 不允许 `replace_case(...)` 改 packed topology.

## Field Bundle ABI

当前 engine 边界优先使用 field bundles.

必须优先使用的 bundles:

- `T_fields`
- `u_fields`
- `rp_fields`
- `env_fields`
- `tb_fields`
- `R_fields`
- `Z_fields`
- `J_fields`
- `g_fields`
- `root_fields`
- `residual_fields`

允许:

- 保留语义化 property 作为冷路径和阅读入口

不得:

- 不在 hot path 中重新扩回长参数表, 前提是稳定 bundle 已存在.

## Stage Ownership

当前四阶段由 [`veqpy/operator/operator.py`](../veqpy/operator/operator.py) 组织.

### Stage-A

当前事实:

- `Stage-A` 已经是 bulk packed profile update.
- 入口是 `stage_a_profile(...)`.
- 主要 engine 入口是 `update_profiles_packed_bulk(...)`.

不得:

- 不回退到 `coeff_matrix` 中转.

### Stage-B

当前事实:

- `Stage-B` 通过 `Geometry.update(...)` 和 backend `update_geometry(...)` 刷 geometry fields.

### Stage-C

当前事实:

- `Stage-C` 通过 `source_stage_runner` 绑定 source route.

### Stage-D

当前事实:

- `Stage-D` 先构造 `G`, 再通过 `residual_stage_runner` 生成 packed residual.
- residual block registry 仍留在 engine 内部.
- 当前已经不是 per-block Python assembly.

不得:

- 不回退到 Python-side row buffer + scatter.

## Optional Semantics

public/model 层允许 richer 语义, hot engine ABI 保持单态。

必须:

- 在 facade/operator 层把 richer 语义 lower 成 engine-friendly ABI.
- profile packed 路径使用空 `coeff_indices` 数组表达 offset-only.

当前例外:

- `Ip` / `beta` 在 facade 语义上仍可理解为 optional constraints.
- 但 hot source kernels 继续使用当前 scalar ABI, 因为 direct `None` in Numba 已证明有性能回退.

不得:

- 不把 `None` 直接塞进 hot Numba kernels, 除非有新的 benchmark 证据.

## Snapshot Semantics

[`veqpy/model/equilibrium.py`](../veqpy/model/equilibrium.py) 的语义必须保持清楚.

必须:

- `Equilibrium` 表示单网格 materialized snapshot.
- `Equilibrium.resample(...)` 表示 snapshot 插值后重建.
- `compare(...)` / `plot_comparison(...)` 表示比较接口, 不再承担 resample 参数.

不得:

- 不把 `Equilibrium` 当成 solver runtime owner.
- 不把 `resample(...)` 描述成严格参数化重建.

## Backend Control Surface

backend control surface 只有 [`veqpy/engine/__init__.py`](../veqpy/engine/__init__.py).

必须:

- 只在这里读取 `VEQPY_BACKEND`.
- 当前只支持:
  - `numpy`
  - `numba`

不得:

- 不在 `SolverConfig` 或 `Solver` 里加入 instance-level backend 开关.
- 不在文档里写不存在的 backend 文件或旧 build 链路.

## Runtime Output

必须:

- runtime 输出保持纯英文.

适用范围:

- `print(...)`
- rich 输出
- warnings
- exceptions
- benchmark / artifact / report 文本

## Documentation Governance

修改下列内容后, 必须同步检查文档:

- packed layout / codec
- field bundle ABI
- `Operator` runtime owner 语义
- `Solver` public surface
- `Equilibrium` snapshot 语义
- backend exports

至少需要同步核对:

- [`README.md`](../README.md)
- [`docs/overview.md`](./overview.md)
- [`docs/conventions.md`](./conventions.md)
- [`docs/guardrails.md`](./guardrails.md)

## Minimum Validation

只改文档:

- 至少核对路径, 文件名, 入口命名仍然存在.

改任意源码:

- `py -m compileall veqpy tests`

改 runtime 主链或 solver:

- `py -m compileall veqpy tests`
- `py tests/demo.py`

改 benchmark 口径, source route, residual runner:

- `py -m compileall veqpy tests`
- `py tests/demo.py`
- `py tests/benchmark.py`

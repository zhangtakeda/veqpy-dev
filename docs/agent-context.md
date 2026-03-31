# Agent Context

## Purpose

这份文档是给后续 coding agent 的仓库交接说明.

目标不是重复 README, 而是把这些内容说清楚:

- 当前仓库真正的权威状态是什么.
- 哪些对象只是派生物, 看起来重要但不是 owner.
- 哪些约束写在代码结构里, 不写出来就容易被误改.
- 改某一层代码后, 应该联动看哪些地方.

默认前提:

- 所有脚本, `pytest`, `tests/demo.py`, `tests/benchmark.py`, `compileall` 都在项目 `uv` 管理的虚拟环境里运行.
- 如果 shell 没激活虚拟环境, 优先使用 `uv run ...`, 或显式使用 `.\.venv\Scripts\python.exe ...`.

## Repository Reality

当前仓库主路径只有四层:

- `veqpy/model/`
  - 网格, profile, geometry, equilibrium snapshot.
- `veqpy/operator/`
  - packed layout/codec, case normalization, runtime owner.
- `veqpy/engine/`
  - `numpy` / `numba` 数值核和 backend 导出面.
- `veqpy/solver/`
  - SciPy solve facade, history, result, fallback.

关键入口:

- `tests/demo.py`
- `tests/benchmark.py`
- `tests/test_model_core_regression.py`
- `tests/test_operator_core_regression.py`
- `tests/test_engine_core_regression.py`
- `tests/test_solver_core_regression.py`

## Core Mental Model

最重要的事实只有一条:

- packed optimization vector `x` 是 solver-facing 的唯一状态.

由此推出:

- `Operator` 是 runtime owner.
- `Solver` 只负责迭代和 lifecycle, 不拥有 runtime physics state.
- `Equilibrium` 是 post-solve snapshot, 不是 live runtime container.
- `model` 层对象可以参与 runtime 派生, 但不拥有 packed state topology.

当前主链是:

1. `Solver.solve(...)`
2. SciPy root / least-squares
3. `Operator.__call__(x)`
4. Stage-A `profile`
5. Stage-B `geometry`
6. Stage-C `source`
7. Stage-D `residual`

如果一个改动破坏了这条 owner 链, 它大概率不是局部优化, 而是架构回退.

## Ownership And Boundaries

### Model

`veqpy/model/` 负责:

- `Grid`: 离散化和预计算 tables.
- `Profile`: 单个 profile 的 root parameters 和 runtime fields.
- `Geometry`: 几何 fields.
- `Equilibrium`: 单网格 snapshot 和 inspection surface.

`veqpy/model/` 不负责:

- packed layout owner
- SciPy solve orchestration
- backend 选择

### Operator

`veqpy/operator/` 是当前最关键的一层.

它负责:

- packed ABI
- `profile_L`, `coeff_index`, `order_offsets`
- `OperatorCase` compatibility
- `replace_case(...)`
- Stage A/B/C/D runtime orchestration
- runtime buffers 和 `x -> residual`

它不应该负责:

- solve policy
- backend registry 之外的数值核实现
- snapshot 持久化策略以外的展示逻辑

### Engine

`veqpy/engine/` 只应持有 array-first kernels.

隐含要求:

- 热路径 ABI 优先 `ndarray`, `float`, `int`
- 不把 Python object semantics 直接送进 hot kernels
- backend control surface 只在 `veqpy/engine/__init__.py`

### Solver

`veqpy/solver/` 只负责:

- `x0`
- solve config
- fallback
- history
- `SolverResult`
- 从结果重建 coeff / `Equilibrium`

这里的语义是:

- `x0` 是下一次 solve 的初值.
- warmstart 路径若主方法异常, solver 只复核 `x0` residual, 不引入新的 packed state owner.

它不应该重新拥有:

- packed layout
- stage 实现
- backend 开关

## Hidden Invariants

这些是不写出来最容易被误伤的点.

### Packed ABI Is Singular

packed ABI 的唯一权威位置是:

- `veqpy/operator/layout.py`
- `veqpy/operator/codec.py`

不要重新引入:

- 第二套 packed 协议
- `coeff_matrix`
- Python side row-cache mirror

### `replace_case(...)` Cannot Change Topology

`Operator.replace_case(...)` 的隐含合同是:

- 可以替换 case 数值
- 不可以改变 packed topology

如果你需要变 profile activation / coefficient counts / `M_max`, 通常意味着应该重建 `Operator`, 不是热替换 case.

### `Equilibrium` Is Canonical Snapshot, Not Runtime Cache

当前 `Equilibrium` 的权威 shape state 是:

- `active_profiles: dict[str, Profile]`

这意味着:

- 不再持有 legacy `shape_profile_names` / `shape_profiles` snapshot 状态
- 不再以 list `active_profiles` 作为 canonical 输入
- 零 shape 项默认不持久化
- 缺失的默认形状项按名字补默认 `Profile`

对后续 agent 最重要的判断规则:

- `Equilibrium` 里能算出来的东西, 优先视为 derived, 不要再存成第二份状态

### `M_max` And `effective_order` Are Different

当前 Fourier family 语义里:

- `Grid.M_max` 表示可表示的最大阶数
- `Operator.c_effective_order`
- `Operator.s_effective_order`
  表示当前一次 runtime 真正需要算到几阶

决定 `effective_order` 的不是 `M_max` 本身, 而是:

- active high-order coeffs
- fixed nonzero boundary offsets

所以不要把 “支持到几阶” 和 “本次求值算到几阶” 混成一个概念.

### `resample(...)` Is Interpolation, Not Exact Parametric Reconstruction

`Equilibrium.resample(...)` 的语义是:

- snapshot 插值后在新 grid 上重建

它不是:

- 从物理参数严格反演回原始 parametric model

如果后续 agent 把它当作严格参数化 roundtrip, 很容易写出看似合理但语义错误的代码或测试.

## Runtime-Specific Knowledge

### Stage-A

当前已经是 bulk packed profile update.

关键入口:

- `Operator.stage_a_profile(...)`
- `update_profiles_packed_bulk(...)`

风险:

- 不要退回逐 profile Python 拼装

### Stage-B

当前 geometry 路径消费 family stacks:

- `Operator.c_family_fields`
- `Operator.s_family_fields`

风险:

- 不要重新写死 `c1/s2` 这类低阶 special-case runtime path
- geometry kernel 是否正确, 不能只看低阶 demo

### Stage-C

source route 通过 bound runner 绑定.

风险:

- 改 source route 时, 很容易只看 solve success, 却漏掉物理量定义偏移

### Stage-D

当前 residual 不是 per-block Python assembly.

关键判断:

- residual runner 是 engine-level 执行入口
- Python 侧只负责 orchestration 和少量 glue

风险:

- 性能回退往往来自把 row/block 语义拉回 Python

## Demo And Benchmark Semantics

`tests/demo.py` 的用途:

- 展示 solve / snapshot / plotting 行为
- 生成 demo artifacts

`tests/benchmark.py` 的用途:

- 比较性能口径
- 观察 `M_max` 和 active high-order terms 的成本变化

不要把它们当成正确性回归的替代品.

原因:

- `demo` 可能只覆盖少数 happy path
- `benchmark` 关注性能和趋势, 不会主动指出语义回归
- 某些 serialization / snapshot / compare regression 在 benchmark 中完全不会暴露

当前 demo 已做 warmup, 因为:

- 首次 solve 会吃 backend warmup / JIT 成本
- 首轮耗时不能直接拿来比较算法本体

## Regression Map

当前核心回归按子模块组织:

### `tests/test_model_core_regression.py`

守住:

- `Grid` 预计算表
- `Boundary.from_geqdsk`
- `Equilibrium` canonical snapshot
- serialization roundtrip
- legacy payload rejection
- `compare()` 的主 shape 误差语义

### `tests/test_operator_core_regression.py`

守住:

- `OperatorCase` 归一化
- layout naming/order
- active profile metadata
- effective Fourier order 行为

### `tests/test_engine_core_regression.py`

守住:

- `numpy` / `numba` geometry 一致性
- `numpy` / `numba` residual 一致性
- high-order runtime propagation

### `tests/test_solver_core_regression.py`

守住:

- solve facade
- history
- reset / clear
- equilibrium-history rebuild

隐含规则:

- 改一层时优先跑对应子模块回归
- 改 runtime 主链时跑四个核心回归全套

## Common Misreads

后续 agent 最常见的误判通常是这些:

- “`Equilibrium` 有这些 property, 所以它应该是 runtime owner”
- “`M_max` 提高了, 所以 runtime 一定会完整算到高阶”
- “demo 跑通了, 所以 serialization / snapshot / compare 没问题”
- “benchmark 正常, 所以 residual 语义没变”
- “某个字段在对象里存在, 所以它一定是 canonical state”
- “某段旧兼容代码看起来冗余, 所以可以直接删”

正确的处理方式是:

- 先问这个状态的 owner 在哪一层
- 再问这个值是 source state 还是 derived state
- 再看对应核心回归是否真的覆盖了你要改的语义

## Change Map

### 改 `veqpy/operator/layout.py` 或 `veqpy/operator/codec.py`

至少联动检查:

- `tests/test_operator_core_regression.py`
- `tests/test_engine_core_regression.py`
- `tests/test_solver_core_regression.py`

因为 packed ABI 会向上影响 solver, 向下影响 residual runner.

### 改 `veqpy/operator/operator.py`

至少联动检查:

- 四个核心回归全套
- `tests/demo.py`

因为它是 runtime owner 和 stage glue.

### 改 `veqpy/model/equilibrium.py`

至少联动检查:

- `tests/test_model_core_regression.py`
- `tests/demo.py`

重点看:

- canonical snapshot state
- `resample(...)`
- `compare(...)`
- plotting
- JSON roundtrip

### 改 `veqpy/engine/*`

至少联动检查:

- `tests/test_engine_core_regression.py`
- `tests/test_operator_core_regression.py`
- `tests/benchmark.py`

### 改 `veqpy/solver/*`

至少联动检查:

- `tests/test_solver_core_regression.py`
- `tests/demo.py`

重点看:

- history
- warmstart
- fallback
- result snapshot semantics

## Safe Working Rules For Agents

- 始终在项目 `uv` 虚拟环境中运行仓库命令.
- 优先跑最贴近改动边界的核心回归.
- 不要在没有证据前删除看似多余的 state 或 compatibility logic.
- 不要把测试脚本变成生产逻辑 owner.
- 不要在热路径重新引入 Python object semantics.
- 不要把 `Equilibrium` 变成第二个 runtime state container.
- 不要修改 packed ABI 却不同步检查 solver/operator/engine.

## Suggested Commands

推荐先同步环境:

```powershell
uv sync --group dev
```

直接运行:

```powershell
uv run python -m pytest tests\test_model_core_regression.py -q
```

如果更喜欢激活 shell, 再执行:

```powershell
.\.venv\Scripts\Activate.ps1
```

日常最常用命令:

```powershell
uv run python -m compileall veqpy tests
uv run python -m pytest tests\test_model_core_regression.py tests\test_operator_core_regression.py tests\test_engine_core_regression.py tests\test_solver_core_regression.py -q
uv run python tests/demo.py
uv run python tests/benchmark.py
```

## Quick File Map

- `README.md`
  - 面向仓库整体的简明说明
- `docs/overview.md`
  - 当前架构和执行链路概览
- `docs/guardrails.md`
  - 当前必须守住的边界合同
- `docs/theory/equilibrium.md`
  - `Equilibrium` 相关物理量和 snapshot 语义
- `veqpy/engine/__init__.py`
  - backend control surface
- `veqpy/operator/layout.py`
  - packed ABI 权威定义
- `veqpy/operator/codec.py`
  - packed state encode/decode
- `veqpy/operator/operator_case.py`
  - case normalization
- `veqpy/operator/operator.py`
  - runtime owner 和四阶段 glue
- `veqpy/model/grid.py`
  - 网格和预计算 tables
- `veqpy/model/profile.py`
  - profile root parameters 和 runtime fields
- `veqpy/model/equilibrium.py`
  - snapshot / compare / resample / plotting
- `veqpy/solver/solver.py`
  - solve facade
- `tests/demo.py`
  - demo, plotting, warmup 口径
- `tests/benchmark.py`
  - benchmark 和多模式比较
- `tests/test_model_core_regression.py`
  - `model` 核心回归
- `tests/test_operator_core_regression.py`
  - `operator` 核心回归
- `tests/test_engine_core_regression.py`
  - `engine` 核心回归
- `tests/test_solver_core_regression.py`
  - `solver` 核心回归

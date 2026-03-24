# Solver Boundary

## Contract

`Solver` 是一个 fixed-layout nonlinear solve facade. 它围绕一个 `Operator` 组织 `x0`, `config`, `result`, `history`, 并调用 `scipy.optimize.root` 或 `scipy.optimize.least_squares`.

## Must

- 持有 `operator`, `config`, `x0`, `result`, `history`.
- 通过 `self.operator(x)` 执行当前 residual hot path.
- 把 `replace_case(...)` 作为对 `Operator.replace_case(...)` 的 facade.
- 把 `build_equilibrium()` 作为从当前 operator runtime 重建 snapshot 的 facade.

## Must Not

- 不持有 packed `layout/codec` owner.
- 不持有 runtime `Profile` 或 runtime `Geometry`.
- 不持有 backend 选择逻辑.
- 不把 Stage A/B/C/D 重新拆回 `Solver`.

## Rationale

这样 `Solver` 只负责 solve policy, `Operator` 只负责 residual runtime. 边界清晰后, `x0`, 历史记录, 结果对象不会和 packed runtime owner 纠缠.

## Validation

- `replace_case(...)` 仅转发到 `Operator`.
- `SolverConfig` 中没有 backend 字段.
- `Solver` 只负责编排求解尝试和 fallback, 不重新持有 packed layout/codec.

# Operator Boundary

## Contract

`Operator` 是完整 packed `x -> residual` runtime owner.

## Must

- 持有 packed `profile_L`, `coeff_index`, `order_offsets`.
- 持有 packed-state runtime arrays 和 residual assembly slots.
- 持有 runtime `Profile` buffers, runtime `Geometry`, `psin_r`, `psin_rr`, `FFn_r`, `Pn_r`, `alpha1`, `alpha2`, `G`.
- 组织 Stage A/B/C/D.
- 绑定 runner, 负责 source route 和 residual assembly.
- 从当前 runtime 拷贝出 `Equilibrium`.

## Must Not

- 不把 packed ownership 拆回 `Solver`.
- 不把 runtime profile/geometry owner 拆到 `model`.
- 不把 physical input owner 混进 `Equilibrium`.

## Rationale

packed state, stage orchestration, source scaling, residual assembly 必须在一个 owner 里闭合. 否则 layout 漂移, slot 漂移, runtime duplication 和文档漂移会一起出现.

## Validation

- `veqpy/operator/operator.py` 同时定义 `profile_L`, `coeff_index`, runtime buffers, `stage_a_profile`, `stage_b_geometry`, `stage_c_source`, `stage_d_residual`.
- `operator(x)` 直接串联 Stage A/B/C/D.

## Current Fact

- 当前代码里, `Operator` 仍然直接组织 packed `x -> residual` 的完整执行路径.

## Unfinished Migration

- 当前分支没有正在维护的 engine-side residual ABI 迁移。
- 现阶段的 current fact 就是:
  - `Operator` 继续持有 packed ABI, metadata construction, solver-facing API;
  - `Operator` 直接组织完整 residual path。

# Packed Layout / Codec Ownership

## Contract

packed `layout` 和 packed `codec` 的所有权固定在 `veqpy/operator/`.

## Must

- 继续由 `veqpy/operator/layout.py` 定义 `PROFILE_NAMES`, `PROFILE_INDEX`, `build_profile_layout(...)`, `packed_size(...)`.
- 继续由 `veqpy/operator/codec.py` 定义 packed state 和 packed residual 的编解码.
- 把跨层使用限制在 `veqpy.operator` 包级接口.

## Must Not

- 不把 packed `layout/codec` 放回 `veqpy/solver/`.
- 不把 packed `layout/codec` 放进 `veqpy/engine/`.
- 不让测试直接发明第二套 packed ABI.

## Rationale

只要 packed ABI 分裂, `x0`, residual slot, `replace_case(...)`, regression artifact 和 README/doc 就会同时失真.

## Validation

- `veqpy/operator/__init__.py` 导出 `build_profile_layout`, `decode_packed_state_inplace`, `Operator`, `OperatorCase`, `PROFILE_INDEX`, `PROFILE_NAMES`.

# Equilibrium Semantic Boundary

## Contract

`Equilibrium` 是单网格 materialized diagnostic snapshot. 它属于 model/control plane, 不是 solver runtime state.

## Must

- 持有 root snapshot fields 的拷贝.
- 把几何和诊断作为从 snapshot root fields 重新派生的结果.
- 让 `resample(...)` 表示 snapshot 插值到目标网格后再重建 geometry 和 diagnostics.

## Must Not

- 不把 `Equilibrium` 当成参数化 profile spec.
- 不把 `Equilibrium` 当成 packed state owner.
- 不把 `Equilibrium.resample(...)` 描述成严格保持同一参数化平衡态的重建.

## Rationale

这保证 plotting, comparison, serialization 和 inspection 都基于 snapshot 语义, 不会反向污染求解器 runtime.

## Validation

- `veqpy/model/equilibrium.py` 类说明明确写了 snapshot 语义.
- `Operator._build_equilibrium_from_runtime()` 会复制 runtime arrays.
- `resample_equilibrium(...)` 先插值 root fields, 再重建目标网格上的对象.

# Backend Control Surface

## Contract

backend control surface 只有 `veqpy/engine/__init__.py`.

## Must

- 只在 `veqpy/engine/__init__.py` 读取 `VEQPY_BACKEND`.
- 让 `veqpy.engine` 的包级导出成为所有 backend-facing imports 的唯一入口.
- 把 profile, geometry, source, residual 的真实文件分拆保持为:
  - `numpy_profile.py`
  - `numpy_geometry.py`
  - `numpy_source.py`
  - `numpy_residual.py`

## Must Not

- 不在 `SolverConfig` 或 `Solver` 里重新引入 instance-level backend 开关.
- 不在文档中继续引用不存在的 `numpy_operator.py`.
- 不把未接线的 backend 写成当前可工作的现实.

## Rationale

backend 入口只保留一个控制面, 才能避免导出漂移和运行时选择逻辑分裂.

## Validation

- `veqpy/engine/__init__.py` 读取 `VEQPY_BACKEND`.
- `SolverConfig` 没有 backend 字段.

## Current Fact

- 当前模块只导入 `numpy` 和 `numba` 两个真实可用分支.
- 当前 runtime 浮点基线固定为 `np.float64`, 没有 `VEQPY_REAL` 或 runtime dtype 切换入口.

## Unfinished Migration

- 如果未来重启 residual ABI 迁移，应保持 metadata 以整数编码为主，不把 packed ABI 语义散落成第二套 Python-side 协议。

# Stage A/B/C/D Ownership

## Contract

Stage A/B/C/D 当前由 `veqpy/operator/operator.py` 组织.

## Must

- Stage A: 解包 packed state 并填充 active profiles.
- Stage B: 用当前 profiles 更新 `Geometry`.
- Stage C: 调用 bound runner 计算 `alpha1`, `alpha2`.
- Stage D: 构造 `G`, 组装 fresh packed residual.

## Must Not

- 不把这些 stage 拆散到 `Solver`, `Equilibrium`, 或测试脚本.
- 不在测试里维护第二份 stage 实现.

## Rationale

stage owner 分散后, 同一 residual path 会出现多份口径, 直接破坏 benchmark 和文档口径的一致性.

## Validation

- `Operator.__call__(x)` 按顺序调用 `stage_a_profile`, `stage_b_geometry`, `stage_c_source`, `stage_d_residual`.
- `tests/demo.py` 和 `tests/benchmark.py` 都通过 `Operator` / `Solver` 的公共接口走这条运行时路径, 没有维护第二份 stage 实现.

## Current Fact

- Stage A/B/C/D 目前既是实现标签, 也是当前 residual hot path 的真实组织方式.

## Unfinished Migration

- 当前没有正在落地的 stage ownership 迁移。
- 如果未来重启迁移，Stage A/B/C/D 也可以继续作为内部实现与调试标签存在。

# replace_case(...) Compatibility

## Contract

`replace_case(...)` 只替换兼容的物理工况, 不改变 packed layout.

## Must

- 保持 `profile_L` 不变.
- 保持 `coeff_index` 不变.
- 保持 `order_offsets` 不变.
- 保持 `heat_input` 和 `current_input` shape 与当前 `grid.Nr` 兼容.

## Must Not

- 不把 `replace_case(...)` 用成动态改 layout 的入口.
- 不把它用成 profile 激活集合变更的入口.

## Rationale

一旦允许换 layout, `x0`, residual slots, history snapshots 和 artifact 口径都会失效.

## Validation

- `Operator._validate_case_compatibility(...)` 显式检查 `profile_L`, `coeff_index`, `order_offsets`, `heat_input/current_input` shape.

# Runtime Containers

## Contract

只有真正承载 runtime 数组并依赖原地刷新的对象, 才是 runtime container.

## Must

- 把 `Profile` 视为 1D runtime container.
- 把 `Geometry` 视为 geometry runtime container.
- 把 `Operator` 视为 packed runtime owner.
- 把 `Solver` 视为 solve runtime facade, 围绕 `x0` 和 history 组织状态.
- 对 `Profile` 和 `Geometry` 允许 `frozen=True + update()/inplace`.

## Must Not

- 不把 `OperatorCase` 视为 runtime container.
- 不把 `Equilibrium` 视为 runtime container.
- 不把 `Grid` 视为可变 runtime container.

## Rationale

只有明确哪些对象允许原地更新, 哪些对象必须保持 snapshot 或 input-only 语义, 才能避免 owner 漂移.

## Validation

- `Profile.update(...)` 和 `Geometry.update(...)` 都是原地刷新.
- `OperatorCase` 在 `__post_init__()` 中复制输入数组.
- `Equilibrium` 通过 copy-built root fields 构建.

# Numerical Semantics

## Contract

下面这些数值语义不得在无文档, 无测试, 无 ADR 的情况下漂移:

- `psin_r`, `psin_rr`, `FFn_r`, `Pn_r` 是当前 grid 上的 root fields.
- `alpha1`, `alpha2` 是 Stage C runner 返回的缩放系数.
- `stage_d_residual()` 返回 fresh packed residual vector.
- `Equilibrium.FF_r = alpha1 * alpha2 * FFn_r`.
- `Equilibrium.P_r = alpha1 * alpha2 * Pn_r / MU0`.
- `q`, `Itor`, `jtor`, `jpara` 是从 snapshot 和 geometry 派生的诊断量, 不是 solve control inputs.

## Must

- 保持 normalized root fields 与 physical diagnostics 的分层.
- 保持 `Equilibrium.resample(...)` 的 snapshot 插值语义.
- 保持 artifact 和文档对这些量的命名一致.

## Must Not

- 不把 normalized quantity 和 physical quantity 混写成同一个字段语义.
- 不把 snapshot diagnostics 重新当成 runtime control inputs.

## Rationale

数值语义一旦漂移, regression artifact 变化会失去解释力, 文档也会立刻失效.

## Validation

- `veqpy/model/equilibrium.py` 中的 property 定义是当前诊断口径.
- `tests/benchmark.py` 当前把 `q`, `Itor`, `jtor`, `jpara` 作为 physics delta 指标.

# Engineering Governance

## Contract

UTF-8, LF, 注释语言规则, runtime 输出语言规则属于项目治理合同的一部分, 不是可选样式.

## Must

- 新增或修改代码文件保持 UTF-8 + LF.
- 新增或修改注释和 `docstring` 使用中文字符 + 英文标点.
- 所有 runtime 输出保持纯英文.
- 重要边界改动后同步更新 `README.md` 与 `docs/` 治理文档.

## Must Not

- 不把这些规则只留在口头约定里.
- 不让 README, `docs/overview.md`, `docs/conventions.md`, `docs/guardrails.md` 互相矛盾.

## Rationale

这批规则直接影响 agent 可编辑性, 终端可读性, artifact 可比对性, 以及文档与代码的一致性.

## Validation

- 代码风格和注释约束看 `docs/conventions.md`.
- 当前仓库入口总览看 `docs/overview.md`.
- 架构边界和治理约束看 `docs/guardrails.md`.

## Current Fact

- 当前仓库已经有明确的治理文档入口, 但源码内历史英文注释尚未完全清理.

## Unfinished Migration

- 触碰历史文件时, 仍需逐步把注释和文件头收敛到当前规则.

# Open Decisions

- `SolverConfig.use_jacobian` 未来若接线, 是否能在不打破当前 `Solver` / `Operator` 边界的前提下完成.
- 如果未来重启 backend ABI 迁移, metadata fields, shape, dtype, 以及 workspace contract 何时冻结为正式 backend contract.

# Code Style

## Scope

适用于:

- `veqpy/engine/`
- `veqpy/model/`
- `veqpy/operator/`
- `veqpy/solver/`
- `tests/`

目标:

- 统一编码和注释风格
- 固定模块边界
- 降低 public 接口漂移
- 约束运行时输出和数值语义说明

## Encoding

- 代码和文档统一使用 UTF-8 + LF.
- 不提交 BOM.
- 不提交 CRLF.

## Comment Rules

- 注释和 docstring 统一使用中文字符 + 英文标点.
- 注释禁止使用中文全角标点.
- 注释和 docstring 中禁止使用 Markdown 反引号包裹代码片段.
- 注释优先写职责, 边界, shape, 单位, 变量域, 不变量, 兼容约束.
- 禁止低价值翻译式注释.

文件头规则:

- 每个 Python 文件都要有顶层 docstring.
- 文件头应该说明:
  - 属于哪一层
  - 负责什么
  - 不负责什么(不必须, 易引发歧义时说明)
  - 主要的实现或者接口

public 接口规则:

- 重要 public 接口必须有 docstring.
- 至少覆盖:
  - 职责边界
  - 输入
  - 输出
  - 兼容约束
  - 是否会更新内部状态
- 输入和返回统一用:
  - `Args:`
  - `Returns:`
- 不使用:
  - `输入语义:`
  - `输出语义:`
  - `返回值依次为:`

私有符号规则:

- 私有函数, 私有成员, 私有局部变量默认不加注释.
- 只有在意图不能被第一时间理解时才加.

行内注释只用于:

- 模块边界
- packed `layout/codec` 规则
- 数值稳定性
- 单位, shape, 变量域
- 不变量和 fallback 语义

## Runtime Output

- 所有运行时输出必须是纯英文.
- 适用于:
  - `print(...)`
  - rich 输出
  - CLI 输出
  - warning
  - 异常消息
  - report / artifact / log 文本

## Module Boundaries

- `veqpy/engine/` 只放数组导向数值核和 backend exports.
- `veqpy/model/` 只放 grid, runtime model, snapshot model.
- `veqpy/operator/` 只放 packed `layout/codec`, `OperatorCase`, 完整 residual operator.
- `veqpy/solver/` 只放 solve facade, config, record, result.
- `tests/` 只放回归, 热点, 脚本入口, 不承载生产逻辑.

## Imports

- 导入顺序: 标准库, 第三方, 项目内.
- 同子模块内部可以直接文件导入.
- 跨子模块导入优先走包级接口.

## Ownership

- 一个 runtime 状态只能有一个 owner.
- 不保留镜像字段和重复状态, 除非有实测收益.
- packed `layout/codec` 归 `veqpy/operator/`.
- solve facade 归 `veqpy/solver/`.
- backend control surface 归 `veqpy/engine/__init__.py`.

## Runtime Containers

- `Profile` 和 `Geometry` 可以使用 frozen + inplace update 模式.
- `OperatorCase`, `Grid`, `SolverConfig`, `SolverResult`, `SolverRecord` 不使用这种 runtime buffer 语义.

## Numerical Code

数值代码优先说明:

- 物理含义
- 变量域
- 单位
- shape
- 归一化量还是物理量
- 不变量
- 边界条件

典型必须说清的内容:

- `u / u_r / u_rr` 是 profile 值和导数
- `psin_r / psin_rr / FFn_r / Pn_r` 是当前 grid 上的 root fields
- `G / psin_R / psin_Z` 是 residual 相关场
- `replace_case(...)` 要求 packed layout 不变
- `Equilibrium.resample(...)` 是 snapshot 插值, 不是严格参数化重建

## Review Checklist

- 是否为 UTF-8 + LF
- 是否有文件头注释
- 注释是否使用中文字符 + 英文标点
- 注释是否含中文全角标点
- 注释是否使用反引号
- 重要 public 接口是否有 docstring
- public docstring 是否使用 `Args:` / `Returns:`
- runtime 输出是否为纯英文
- 是否引入重复 owner 或重复状态
- 跨子模块导入是否经过包级接口
- packed `layout/codec` 是否仍留在 `veqpy/operator/`

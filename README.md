# veqpy

veqpy 是一个用于托卡马克磁流体平衡计算的 Python 包. 服务于 VEQ 项目.

VEQ: Veloce/Variational EQuilbrium code, a high-performance Python wrapper for plasma equilibrium simulations in magnetic confinement devices.

- 作者: rhzhang
- 更新日期: 2026-03-24

文档由 Codex 生成, 以源码实现为准.

## 项目结构

- `veqpy/engine/`
  - backend 导出面, 以及 `numpy` / `numba` 数值 kernel
- `veqpy/model/`
  - `Grid`、`Profile`、`Geometry`、`Equilibrium`
- `veqpy/operator/`
  - packed `layout/codec`、`OperatorCase`、完整的 `x -> residual` 算子
- `veqpy/solver/`
  - `Solver`、`SolverConfig`、`SolverRecord`、`SolverResult`
- `tests/`
  - `demo.py` 示例入口
  - `benchmark.py` 多模式基准与一致性检查入口
  - `benchmark/` 基准产物目录

## 当前运行时边界

- `Operator` 是完整 packed `x -> residual` 路径的 owner
- `Operator.__call__(x)` 当前直接串联 Stage A/B/C/D
- `Solver` 是 nonlinear solve facade, 不持有 packed layout/codec, 也不负责 backend 选择
- `Solver.solve(...)` 只执行一次求解并返回 packed `x`
- 求解后的稳定入口包括:
  - `solver.result`
  - `solver.history`
  - `solver.build_equilibrium()`
  - `solver.build_equilibrium_history()`
  - `solver.build_coeffs()`
  - `solver.build_coeffs_history()`
- backend 只通过 `VEQPY_BACKEND` 控制, 可选值:
  - `numpy`
  - `numba`

`veqpy.engine` 在环境变量未设置时默认使用 `numba`.

## Solver 能力概览

- `SolverConfig.method` 当前支持:
  - root 路径: `hybr`, `krylov`, `root-lm`, `broyden1`, `broyden2`
  - least-squares 路径: `trf`, `dogbox`, `lm`
- root 方法走 `scipy.optimize.root(...)`
- `lm` / `trf` / `dogbox` 直接走 `scipy.optimize.least_squares(...)`
- 主方法失败时, `Solver` 会按顺序自动尝试:
  - `least_squares/lm`
  - `least_squares/trf`
- `enable_homotopy=True` 时, staged solve 会按 profile 阶次逐层扩展 active set
- homotopy 还支持按 `homotopy_truncation_tol` 和 `homotopy_truncation_patience` 冻结后续高阶 shape 系数

## 安装

基础安装:

```bash
py -m pip install -e .
```

开发安装:

```bash
py -m pip install -e .[dev]
```

## 最小示例

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

更完整的可运行示例见 `tests/demo.py`.

## 常用命令

语法检查:

```bash
py -m compileall veqpy tests
```

运行示例并生成演示产物:

```bash
py tests/demo.py
```

运行多模式 benchmark 与物理量 delta 检查:

```bash
py tests/benchmark.py
```

## 产物目录

运行 `py tests/demo.py` 后, 会在 `tests/` 下生成:

- `demo-1.json` / `demo-1.png`
- `demo-2.json` / `demo-2.png`
- `demo-3.json` / `demo-3.png`
- `demo-4.json` / `demo-4.png`
- `demo-coeffs-comparison.png`
- `demo-grid-comparison.png`
- `demo-homo-comparison.png`

运行 `py tests/benchmark.py` 后, 默认会生成:

- `tests/benchmark/cold-<backend>/pf_reference_summary.png`
- `tests/benchmark/cold-<backend>/pf_reference_summary.txt`
- `tests/benchmark/cold-<backend>/benchmark_compare.txt`
- `tests/benchmark/cold-<backend>/benchmark_notes.txt`
- `tests/benchmark/cold-<backend>/plots/`

如果把 `tests/benchmark.py` 中的 `WARMSTART` 切到 `True`, 产物目录会切换到 `tests/benchmark/warm-<backend>/`.

## 关键文件

- `veqpy/engine/__init__.py`
  - backend 控制面和稳定导出面
- `veqpy/operator/operator.py`
  - 完整 packed-state 到 residual 的运行时主路径
- `veqpy/operator/layout.py`
  - packed layout 定义
- `veqpy/operator/codec.py`
  - packed state / residual 编解码
- `veqpy/solver/solver.py`
  - 求解生命周期入口, root / least-squares / fallback / homotopy
- `veqpy/solver/solver_config.py`
  - solver 方法和 staged-solve 配置
- `veqpy/model/equilibrium.py`
  - snapshot、诊断、绘图、comparison、resample
- `tests/demo.py`
  - 最小示例和演示产物入口
- `tests/benchmark.py`
  - 多模式 benchmark、delta 检查和 benchmark 产物入口

## 当前注意事项

- `replace_case(...)` 只支持 packed layout 兼容的 `OperatorCase`
- `OperatorCase` 是可变 runtime case, 适合实时更新 `Ip` / `beta` / `heat_input` / `current_input`
- `SolverRecord` 会复制 `OperatorCase` snapshot, 避免 live case 的后续原位修改污染历史
- `Grid` 是不可变的
- `Equilibrium` 是单网格 snapshot, 不是 solver-side 可回写状态
- `Equilibrium.resample(...)` 的语义是 snapshot 插值, 不是严格参数化重建
- 改 packed layout、packed codec、operator contract、solver 控制流、engine exports 后, 建议同步更新 `doc/`

## 相关文档

- [`doc/overview.md`](doc/overview.md)
- [`doc/conventions.md`](doc/conventions.md)
- [`doc/guardrails.md`](doc/guardrails.md)
- [`doc/veqpy_operators.md`](doc/veqpy_operators.md)
- [`doc/veqpy_equilibrium.md`](doc/veqpy_equilibrium.md)

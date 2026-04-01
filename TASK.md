# 论文任务清单

> 项目: `veqpy`  
> 目标期刊: `Nuclear Fusion` 或同类聚变数值/软件论文期刊  
> 当前版本: 基于 2026-04-01 仓库状态整理  
> 文档定位: 中文版工作稿, 直接面向“怎么推进到能投稿”

---

## 0. 先定论文主线

### 推荐主线

当前仓库最适合写成一篇“方法 + 软件实现 + 内部一致性验证 + 性能画像”的文章, 而不是一上来就宣称“全面优于 EFIT/CHEASE”.

更稳妥的核心主张建议写成:

- `veqpy` 把多种输入路由统一到同一个 `packed x -> residual -> solve -> equilibrium` 框架里.
- 同一参考平衡在不同模式/坐标/输入采样方式下可以重建出彼此一致的结果.
- 该统一框架已经具备可重复的图形输出、基准测试和理论文档支撑.
- 在当前基准设置下, `numba` 后端已经达到毫秒级求解.

### 当前代码最能支撑的卖点

- 统一入口明确:
  - `veqpy/solver/solver.py`
  - `veqpy/operator/operator.py`
  - `veqpy/engine/__init__.py`
- 输入模式丰富:
  - `PF`, `PP`, `PI`, `PJ1`, `PJ2`, `PQ`
  - `rho` / `psin`
  - `uniform` / `grid`
- 理论映射已经写出来:
  - `docs/theory/source.md`
  - `docs/theory/residual.md`
- 图和产物已经能落盘:
  - `tests/demo.py`
  - `tests/benchmark.py`
  - `tests/fitting.py`
  - `tests/hotspot.py`

### 当前不建议直接主打的主张

- “全面替代 EFIT/CHEASE”
- “已完成跨装置、跨放电大规模实验验证”
- “对真实实验反演精度显著优于成熟代码”

原因很简单: 当前仓库已经有很强的内部验证和方法闭环, 但还没有看到成体系的外部基线对比数据管线.

### 经验建议

如果你想尽快形成一篇能投的稿子, 建议分两步:

1. 先完成“方法/软件论文”版本, 把统一路由、理论到实现映射、内部一致性和性能讲扎实.
2. 再在下一篇或扩展版里补“真实 GEQDSK/EFIT/CHEASE 大规模交叉验证”.

---

## 1. 当前仓库已经有的证据

这部分很重要, 因为后面的图和章节都应该优先吃现成资产, 而不是重新造故事.

### 1.1 基准矩阵已经成型

`tests/benchmark.py` 当前已经做了 92 个 route-specific case 的比较, 产物在:

- `tests/benchmark-numba/benchmark_compare.txt`
- `tests/benchmark-numba/reference_summary.json`
- `tests/benchmark-numba/plots/*.png`

当前可直接写进文中的事实:

- 参考算例: `PF_rho + Ip`
- 参考网格: `32x32`
- 测试网格: `12x12`
- 总案例数: `92`
- `shape_tol = 1e-2`
- 当前 `failure_count = 0/92`
- 最坏 shape error: `PQ_psin_uniform_Ip_beta = 9.423975e-03`
- 最慢案例: `PQ_psin_uniform_beta = 3.867 ms`

这说明你已经有了一个很好的“统一路由内部一致性”结果雏形.

### 1.2 参考平衡摘要已经能导出

`tests/benchmark-numba/reference_summary.json` 已经给出了:

- 外边界闭合曲线
- `R0`, `a`, `B0`, `Ip`
- 拉长比 `elongation = 2.2`
- 三角度 `delta_average = 0.5`
- `psin`, `P_psi`, `q` 剖面

这很适合做论文里的参考基准图和表 1.

### 1.3 演示图已经具备“论文图原型”

`tests/demo.py` 当前已经能生成:

- `tests/demo/demo-1.png`
- `tests/demo/demo-2.png`
- `tests/demo/demo-comparison.png`
- `tests/demo/grid-shape-error.png`

它们分别对应:

- 单个平衡六联图摘要
- 高低精度配置对比
- 解之间的 overlay 比较
- 网格收敛/形状误差热图

### 1.4 边界拟合故事已经有入口

`tests/fitting.py` 当前已经能从 `GEQDSK` 读取边界并生成:

- `tests/fitting/geqdsk-boundary.png`

这可以支撑“外部边界输入 -> 参数化边界 -> 嵌套磁面族”的方法图.

### 1.5 性能分解也已经有材料

`tests/hotspot.py` 当前产物:

- `tests/hotspot/full46-numba/hotspot_report.txt`

里面已经能看到一些很适合写进论文的观察:

- 若干慢例中 `solve` 时间的 `~71% - 93%` 在 residual 调用上
- `PQ` / `PJ2` 类 route 更慢, 且 `psin + uniform` 组合明显更吃迭代

这很适合做“性能瓶颈和 route 差异”的一节.

---

## 2. 文章结构建议

### 最推荐的文章结构

1. 引言
2. 理论与参数化
3. 软件架构与统一求解链路
4. 验证协议与基准设置
5. 内部一致性与路由对比结果
6. 性能画像与瓶颈分析
7. 局限性与下一步工作

### 每节应该吃哪些代码资产

#### 第 2 节: 理论与参数化

优先引用/整理:

- `docs/theory/source.md`
- `docs/theory/residual.md`
- `veqpy/model/equilibrium.py`

这里要把以下量的关系讲清楚:

- `alpha1`, `alpha2`
- `psin_r`
- `FFn_psin`, `Pn_psin`
- `Ip`
- `beta_t`
- `q`
- `Itor`, `jtor`, `jpara`

#### 第 3 节: 软件架构

优先围绕以下主链写:

1. `Solver.solve(...)`
2. `Operator.__call__(x)`
3. Stage-A `profile`
4. Stage-B `geometry`
5. Stage-C `source`
6. Stage-D `residual`
7. `Equilibrium` snapshot / plotting / compare

核心代码锚点:

- `veqpy/solver/solver.py`
- `veqpy/operator/operator.py`
- `veqpy/operator/layout.py`
- `veqpy/operator/operator_case.py`
- `veqpy/engine/__init__.py`
- `veqpy/model/equilibrium.py`

#### 第 4-6 节: 结果

优先使用:

- `tests/benchmark.py`
- `tests/demo.py`
- `tests/fitting.py`
- `tests/hotspot.py`

---

## 3. 图表计划

这里我不只列名字, 也把“图类型”和“为什么值得画”写清楚.

### Fig 1. 方法架构图

图类型:

- 流程图 / block diagram

建议内容:

- `OperatorCase` 输入
- `layout/codec`
- `Solver`
- `Operator`
- Stage-A/B/C/D
- `Equilibrium`
- plot / compare / benchmark outputs

为什么必须有:

- 这是你整篇文章的骨架图.
- 读者如果先看懂这个图, 后面的 route 和 benchmark 就不会乱.

### Fig 2. 边界参数化与嵌套磁面图

图类型:

- `R-Z` 平面 overlay 图

直接复用/改造来源:

- `tests/fitting.py`
- `tests/fitting/geqdsk-boundary.png`

建议内容:

- GEQDSK 拟合边界
- 拟合后的闭合外边界
- 若干内部 `rho` 面
- 图旁边列 `a`, `R0`, `Z0`, `ka`, `c_offsets`, `s_offsets`

为什么值得画:

- 这张图把“输入边界不是抽象变量, 而是能落到真实几何”的故事讲明白.

### Fig 3. 参考平衡六联图

图类型:

- multi-panel summary figure

直接来源:

- `Equilibrium.plot(...)`
- `tests/demo/demo-1.png`
- `tests/demo/demo-2.png`

建议作为主图展示:

- Flux surfaces
- Shape profiles
- Source profiles
- `j_phi(R, Z)` contour
- `Itor/jtor/jpara`
- `q/s`

为什么值得画:

- 这张图非常像“方法论文里的标准结果总览图”.
- 它能把几何、源项、电流和安全因子一次讲完整.

### Fig 4. 路由一致性热图或矩阵图

图类型:

- heatmap
- 或 grouped heatmap / dot heatmap

数据来源:

- `tests/benchmark-numba/benchmark_compare.txt`
- 最好后续把 `tests/benchmark.py` 改成直接导出 `csv/json`

推荐横纵轴:

- 横轴: `mode`
- 纵轴: `coordinate + nodes + constraint`

推荐指标:

- `shape_error`
- `avg_ms`
- `nfev`

为什么这张图很关键:

- 它能最直接证明“同一参考平衡在不同 route 下是可复现且稳定的”.
- 比单纯列大表更像论文结果.

### Fig 5. 代表性解的 overlay 对比图

图类型:

- surface overlay
- 1D profile overlay

直接来源:

- `tests/benchmark-numba/plots/*_compare.png`
- `Equilibrium.compare(...)`

建议选 3 组代表性 case:

- 最稳的: `PF_rho_uniform_Ip` 或类似 PF 基线
- 中等难度: `PP` 或 `PI`
- 最难的: `PQ_psin_uniform_Ip_beta`

为什么值得画:

- 不能只给统计图, 还要给“长什么样”的例子.
- 最好包含一个最坏例, 这样文章显得诚实且更可信.

### Fig 6. 速度-误差 Pareto 图

图类型:

- scatter / Pareto front

横纵轴建议:

- x 轴: `avg_ms`
- y 轴: `shape_error`
- 颜色: `mode`
- marker: `coordinate` 或 `nodes`

数据来源:

- `tests/benchmark.py` 现有结果

为什么值得画:

- 一张图同时回答“多快”和“多准”.
- 很适合摘要和结论页重复引用.

### Fig 7. 性能瓶颈分解图

图类型:

- stacked bar chart

数据来源:

- `tests/hotspot/full46-numba/hotspot_report.txt`

建议内容:

- `solve total`
- `residual`
- `other`
- 挑 6 到 10 个代表 case

为什么值得画:

- 软件论文如果只报总时间, 说服力通常不够.
- 这张图能说明“慢在哪里”, 也给后续优化留出口.

### Supplementary Figures

- `tests/demo/grid-shape-error.png`
  - 适合做补充材料中的网格收敛图.
- `reference_summary.json` 导出的 `q`, `P_psi`, `psin`
  - 适合做补充剖面图.
- 所有 benchmark overlay 图
  - 可以做补充材料图库.

---

## 4. 表格计划

### Tab 1. 参考算例参数表

直接可填字段:

- `R0 = 1.05`
- `a = 0.567567...`
- `B0 = 3.0`
- `Ip = 3.0e6`
- `aspect_ratio = 1.85`
- `elongation = 2.2`
- `delta_average = 0.5`

来源:

- `tests/benchmark-numba/reference_summary.json`

### Tab 2. 统一路由结果总表

建议字段:

- `mode`
- `coordinate`
- `nodes`
- `constraint`
- `shape_error`
- `avg_ms`
- `nfev`
- `residual_norm_final`

建议不要整张主文都塞满 92 行, 可以:

- 主文只给分组统计
- 完整表放补充材料

### Tab 3. 分组统计表

建议按以下维度聚合:

- 按 `mode`
- 按 `coordinate`
- 按 `nodes`

统计量建议:

- median
- P90
- worst

### Tab 4. 局限性/失败边缘案例表

即使当前 `0/92` failure, 也建议单独列“最差 5 个 case”:

- 这比只报“全成功”更可信.
- 也能自然过渡到局限性讨论.

---

## 5. 工作流拆分

## WS-A: 固化 benchmark 结果导出

**目标:** 把现在偏“报告文本”的 benchmark 结果变成论文友好的结构化数据.

### 必做任务

- [ ] 在 `tests/benchmark.py` 中增加 `csv/json` 明细导出
- [ ] 为每个 case 保存统一字段:
  - `mode`
  - `coordinate`
  - `nodes`
  - `constraint`
  - `avg_ms`
  - `std_ms`
  - `shape_error`
  - `nfev`
  - `nit`
  - `residual_norm_final`
- [ ] 输出分组统计表
- [ ] 让图脚本直接吃结构化结果而不是手抄文本

### 交付物

- `results/benchmark_cases.csv`
- `results/benchmark_cases.json`
- `results/benchmark_summary.csv`

### 经验建议

这一步优先级非常高, 因为一旦结构化数据出来, 后面的论文图基本都能半自动生成.

## WS-B: 图表生产脚本

**目标:** 从“已有 demo 图”升级到“论文定稿图”.

### 必做任务

- [ ] 写 `analysis/plot_route_matrix.py`
- [ ] 写 `analysis/plot_pareto.py`
- [ ] 写 `analysis/plot_hotspot_breakdown.py`
- [ ] 写 `analysis/plot_reference_case.py`
- [ ] 固定配色、字号、输出分辨率

### 建议图风格

- 主图统一使用白底
- 坐标轴单位写全
- 所有颜色在全文中语义固定
  - 例如 `PF/PP/PI/PJ1/PJ2/PQ` 各自固定颜色
- 对比图中 reference 永远用黑色, comparison 永远用红色或蓝色

### 交付物

- `figures/fig1_pipeline.pdf`
- `figures/fig2_boundary_fit.pdf`
- `figures/fig3_reference_equilibrium.pdf`
- `figures/fig4_route_matrix.pdf`
- `figures/fig5_case_overlays.pdf`
- `figures/fig6_pareto.pdf`
- `figures/fig7_hotspot.pdf`

## WS-C: 论文文字主线

**目标:** 让文字和图一一对应, 不空喊口号.

### 必做任务

- [ ] 写出摘要的 3 句主结论
- [ ] 每个结论至少绑定 1 个图或 1 个表
- [ ] 方法部分明确区分:
  - 理论符号
  - 代码实现
  - 数值 route
- [ ] 局限性单独成段

### 强烈建议加入的段落

- [ ] “为什么要统一多种 profile route”
- [ ] “为什么 `alpha1/alpha2` 的归一化处理重要”
- [ ] “为什么 `packed layout` 能让 solver 接口稳定”
- [ ] “为什么 `Equilibrium` 被设计成 snapshot 而不是 runtime owner”

## WS-D: 外部基线对比

**目标:** 决定文章是不是要上升到“对外部成熟代码的定量对比”.

### 选项 A: 本文先不做外部基线

适用场景:

- 想尽快先投软件/方法文章
- 当前没有干净的 EFIT/CHEASE 数据管线

那就应该把标题和摘要改成:

- unified equilibrium framework
- route consistency
- reproducible benchmark

### 选项 B: 本文补做外部基线

适用场景:

- 目标就是冲更强的“物理结果比较”

那必须补:

- [ ] GEQDSK 数据集定义
- [ ] EFIT/CHEASE 输出对齐协议
- [ ] 对齐指标:
  - LCFS 几何误差
  - `q` profile 误差
  - `Ip`
  - `beta_t`
  - 若可能再加 `jtor`

### 经验建议

没有外部基线时, 不要在摘要里写 `relative to EFIT/CHEASE`.

## WS-E: 消融与边界案例

**目标:** 让文章更像“认真验证过”, 而不是“跑通了”.

### 推荐消融

- [ ] `rho` vs `psin`
- [ ] `uniform` vs `grid`
- [ ] `PF/PP/PI/PJ1/PJ2/PQ`
- [ ] 低阶 shape vs 高阶 shape
- [ ] 网格尺寸: `8,12,16,24,32,48,64`

### 当前仓库已经有基础的消融入口

- `tests/demo.py` 的 `GRID_SIZES`
- `tests/benchmark.py` 的多 mode / coordinate / nodes / constraint 枚举

### 特别建议

不要只写“最优案例”; 一定要保留:

- 最慢案例
- 最差误差案例
- 一个典型中位案例

## WS-F: 可复现性打包

**目标:** 让审稿人感到这个仓库是“可运行、可检查、可追踪”的.

### 必做任务

- [ ] 固定命令入口
- [ ] 固定结果目录结构
- [ ] 输出 git hash / config snapshot
- [ ] 主图全部由脚本生成

### 推荐命令

```powershell
uv run python tests/fitting.py
uv run python tests/demo.py
uv run python tests/benchmark.py
uv run python tests/hotspot.py
```

### 建议补的产物

- `results/manifest.json`
- `results/environment.txt`
- `results/git_rev.txt`

---

## 6. 具体推进顺序

这是我最推荐的落地顺序.

### 第 1 周

- [ ] 明确文章定位: 先做方法/软件论文, 还是同时冲外部 baseline
- [ ] 给 `tests/benchmark.py` 增加结构化导出
- [ ] 固化主图 4, 5, 6 的数据接口
- [ ] 用当前数据先写出摘要初稿

### 第 2 周

- [ ] 产出主图初版
- [ ] 把 `demo/fitting/benchmark/hotspot` 的图风格统一
- [ ] 写 Methods 和 Results 初稿
- [ ] 从当前 benchmark 中挑 3 个代表案例做详细图注

### 第 3 周

- [ ] 决定是否补外部 baseline
- [ ] 补局限性和失败边缘案例讨论
- [ ] 完成补充材料
- [ ] 统一表格和符号

### 第 4 周

- [ ] 内部 freeze
- [ ] 重跑结果
- [ ] 校对图表编号、命令、路径、单位
- [ ] 投稿前自检

---

## 7. Claim-to-Evidence 矩阵

| Claim ID | 主张 | 最直接证据 | 图/表建议 | 当前状态 |
| --- | --- | --- | --- | --- |
| C1 | 同一框架可统一处理 `PF/PP/PI/PJ1/PJ2/PQ` 多路输入 | `tests/benchmark.py` 的 92 case 枚举 | Fig 1, Fig 4 | 可写 |
| C2 | 不同 route 对同一参考平衡能给出一致重建 | `failure_count = 0/92`, 最坏误差 `< 1e-2` | Fig 4, Tab 2 | 可写 |
| C3 | 当前 `numba` 后端已实现毫秒级求解 | `benchmark_compare.txt` 中 `avg_ms` | Fig 6, Tab 3 | 可写 |
| C4 | 框架支持从边界参数化到完整平衡快照输出 | `tests/fitting.py`, `tests/demo.py` | Fig 2, Fig 3 | 可写 |
| C5 | 不同 route 的性能压力和难度不同 | `hotspot_report.txt`, `nfev` 排名 | Fig 6, Fig 7 | 可写 |
| C6 | 对外部成熟代码更优 | 需要外部基线 | 未来工作或扩展版 | 暂不可写 |

---

## 8. 文章充实建议

这部分不是“必须做”, 但很可能显著提高文章质量.

### 建议 1: 加一段“为什么多 route 不是堆功能”

如果只列出 `PF/PP/PI/PJ1/PJ2/PQ`, 审稿人可能会觉得只是接口很多.

更好的写法是:

- 不同实验/建模条件下, 已知输入并不总是同一组物理量.
- `veqpy` 的价值不是“多几个 mode”, 而是把它们统一落在同一个 residual machinery 上.

### 建议 2: 把 `alpha1/alpha2` 讲成文章亮点

当前理论文档已经有这个基础.  
这两个量既是归一化桥梁, 又是 route 之间比较的共同语言.  
如果讲清楚, 文章会从“工程实现”上升到“方法设计”.

### 建议 3: 专门讨论 `rho` vs `psin`

从当前 benchmark 看, `psin + uniform` 在若干模式下明显更难.  
这非常值得成为一个小节, 因为它不是坏消息, 而是很有价值的数值观察.

可以写的角度:

- 为什么某些 route 对采样坐标更敏感
- 为什么 `grid` 输入在当前实现里更稳
- 这对未来实验数据接入意味着什么

### 建议 4: 诚实展示“最差但仍成功”的案例

当前最差不是失败, 这是好事.  
但论文里最好直接拿 `PQ_psin_uniform_Ip_beta` 这类案例出来讲:

- 它为什么更难
- 误差大在哪里
- 但仍然保持在容差内

这会明显提高说服力.

### 建议 5: 增加“实现选择背后的理由”

例如:

- 为什么 `Equilibrium` 是 snapshot
- 为什么 `layout` 权威集中在 `veqpy/operator/layout.py`
- 为什么 Stage-A/B/C/D 这样切分
- 为什么 backend dispatch 放在 `veqpy/engine/__init__.py`

这些内容很适合软件论文读者.

---

## 9. 投稿前闸门

- [ ] 论文主张与当前证据严格匹配
- [ ] 每个主结论都有图或表支撑
- [ ] 所有主图都能通过脚本重生
- [ ] 主文不夸大外部基线结论
- [ ] 局限性单独写出
- [ ] 至少包含一个“最差案例”分析
- [ ] 命令、路径、参数、单位全部核对

---

## 10. 一句话总结

以当前代码基础, 最值得推进的不是“赶紧找更大的口号”, 而是把现有的统一 route 框架、内部一致性、性能画像和可复现图表先做成一篇很扎实的方法/软件论文; 这条路最短, 也最符合仓库现在已经具备的证据强度.

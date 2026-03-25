"""
operator 层主协调器.
负责把 case, grid, model profile, engine kernels 和 packed codec 连接起来, 暴露稳定 residual 求值接口.
不负责 solver 迭代策略, backend 选择, benchmark 编排.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from veqpy.engine import (
    bind_residual_runner,
    bind_runner,
    update_profiles_packed_bulk,
    update_residual,
    validate_operator,
)
from veqpy.model import Equilibrium, Geometry, Grid, Profile
from veqpy.model.profile import (
    ProfileRuntimeView,
    fill_profile_runtime_view,
    fill_profile_runtime_view_from_packed,
)
from veqpy.operator.codec import decode_packed_blocks, encode_packed_state
from veqpy.operator.layout import (
    PROFILE_INDEX,
    PROFILE_NAMES,
    PREFIX_PROFILE_NAMES,
    SHAPE_PROFILE_NAMES,
    build_active_profile_metadata,
    build_profile_layout,
    packed_size,
)
from veqpy.operator.operator_case import OperatorCase

@dataclass(slots=True, frozen=True)
class HomotopyStageGroup:
    """记录 homotopy 分阶段展开时某一阶次对应的 packed 索引集合."""

    order: int
    indices: np.ndarray
    shape_profile_ids: np.ndarray


@dataclass(slots=True)
class Operator:
    """
    封装一个固定算子名, 导数变量域, grid 与 case 的 residual 求值器.

    Args:
        name: 算子名, 例如 PF, PP, PI, PJ1, PJ2, PQ.
        derivative: 导数变量域字符串, 只允许 rho 或 psi.
        grid: 当前求值使用的离散网格与谱矩阵容器.
        case: 当前算例输入, 包含 profile 系数, 几何常数和 source 输入.
    """

    name: str
    derivative: str
    grid: Grid = field(repr=False)
    case: OperatorCase = field(repr=False)
    geometry: Geometry = field(init=False)

    h_profile: Profile = field(init=False)
    v_profile: Profile = field(init=False)
    k_profile: Profile = field(init=False)
    c0_profile: Profile = field(init=False)
    c1_profile: Profile = field(init=False)
    s1_profile: Profile = field(init=False)
    s2_profile: Profile = field(init=False)
    psin_profile: Profile = field(init=False)
    F_profile: Profile = field(init=False)

    psin_R: np.ndarray = field(init=False)
    psin_Z: np.ndarray = field(init=False)
    G: np.ndarray = field(init=False)
    psin_r: np.ndarray = field(init=False)
    psin_rr: np.ndarray = field(init=False)
    FFn_r: np.ndarray = field(init=False)
    Pn_r: np.ndarray = field(init=False)
    residual_fields: np.ndarray = field(init=False, repr=False)
    root_fields: np.ndarray = field(init=False, repr=False)
    alpha1: float = field(init=False)
    alpha2: float = field(init=False)

    profile_L: np.ndarray = field(init=False, repr=False)
    coeff_index: np.ndarray = field(init=False, repr=False)
    order_offsets: np.ndarray = field(init=False, repr=False)
    active_profile_mask: np.ndarray = field(init=False, repr=False)
    active_profile_ids: np.ndarray = field(init=False, repr=False)
    x_size: int = field(init=False, repr=False)
    profile_runtime_views: tuple[ProfileRuntimeView, ...] = field(init=False, repr=False)
    active_profile_runtime_views: tuple[ProfileRuntimeView, ...] = field(init=False, repr=False)
    fixed_profile_runtime_views: tuple[ProfileRuntimeView, ...] = field(init=False, repr=False)
    active_u_fields: np.ndarray = field(init=False, repr=False)
    active_rp_fields: np.ndarray = field(init=False, repr=False)
    active_env_fields: np.ndarray = field(init=False, repr=False)
    active_offsets: np.ndarray = field(init=False, repr=False)
    active_scales: np.ndarray = field(init=False, repr=False)
    active_lengths: np.ndarray = field(init=False, repr=False)
    active_coeff_index_rows: np.ndarray = field(init=False, repr=False)
    residual_runner: Callable = field(init=False, repr=False)
    source_runner: Callable = field(init=False, repr=False)

    def __post_init__(self) -> None:
        """完成 layout 构造, 运行时缓冲区分配和 case 绑定."""
        spec = validate_operator(self.name, self.derivative)

        self.name = spec.name
        self.geometry = Geometry(grid=self.grid)

        self.profile_L, self.coeff_index, self.order_offsets = build_profile_layout(self.case.coeffs_by_name)
        self.active_profile_mask, self.active_profile_ids = build_active_profile_metadata(self.profile_L)
        self.x_size = packed_size(self.coeff_index)

        self._allocate_runtime_arrays()
        self.source_runner = bind_runner(spec.name, self.derivative)
        self.profile_runtime_views = ()
        self.active_profile_runtime_views = ()
        self.fixed_profile_runtime_views = ()
        self.residual_runner = lambda *args, **kwargs: np.zeros(self.x_size, dtype=np.float64)
        self._refresh_case_runtime()

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """
        调用 residual 求值主入口.

        Args:
            x: 一维 packed 状态向量.

        Returns:
            返回与 x 同 layout 的 packed residual 向量.
        """
        return self.residual(x)

    def replace_case(self, case: OperatorCase) -> None:
        """
        在不改变 packed layout 的前提下替换当前 case.

        Args:
            case: 新的 OperatorCase. 它必须与当前 layout 完全兼容.

        Returns:
            返回 None. 运行时 profile 偏移, 输入数组和 residual 槽位会同步更新.
        """
        self._validate_case_compatibility(case)
        self.case = case
        self._refresh_case_runtime()

    def encode_initial_state(self) -> np.ndarray:
        """
        把当前 case 中的 profile 系数编码成 packed 初值.

        Returns:
            返回与当前 layout 匹配的一维 packed 状态向量.
        """
        return encode_packed_state(self.case.coeffs_by_name, self.profile_L, self.coeff_index)

    def residual(self, x: np.ndarray) -> np.ndarray:
        """
        完整执行 profile, geometry, source, residual 四阶段求值.

        Args:
            x: 一维 packed 状态向量.

        Returns:
            返回 packed residual 向量.
        """
        x_eval = self.coerce_x(x)
        self.stage_a_profile(x_eval)
        self.stage_b_geometry()
        self.stage_c_source()
        return self.stage_d_residual()

    def residual_prefix(
        self,
        x_prefix: np.ndarray,
        *,
        active_len: int,
        x_template: np.ndarray,
    ) -> np.ndarray:
        """
        只激活 packed 向量前缀后求值对应 residual 前缀.

        Args:
            x_prefix: 前缀状态向量.
            active_len: 参与求值的前缀长度.
            x_template: 用于补全其余自由度的完整模板向量.

        Returns:
            返回长度为 active_len 的 residual 前缀副本.
        """
        x_full = self._compose_active_state(x_prefix, active_len=active_len, x_template=x_template)
        return self.residual(x_full)[:active_len].copy()

    def residual_masked(
        self,
        x_active: np.ndarray,
        *,
        active_indices: np.ndarray,
        x_template: np.ndarray,
    ) -> np.ndarray:
        """
        只替换指定索引集合后求值对应 masked residual.

        Args:
            x_active: 活跃索引上的状态值.
            active_indices: 参与求值的 packed 索引集合.
            x_template: 用于补全其余自由度的完整模板向量.

        Returns:
            返回与 active_indices 同长度的 residual 副本.
        """
        x_full = self._compose_masked_state(x_active, active_indices=active_indices, x_template=x_template)
        return self.residual(x_full)[active_indices].copy()

    def homotopy_frontiers(self) -> np.ndarray:
        """
        返回 packed 向量按阶次展开时的前缀边界.

        Returns:
            返回升序一维整数数组. 每个元素都表示一个可用于前缀 homotopy 的 frontier.
        """
        if self.x_size == 0:
            return np.zeros(0, dtype=np.int64)
        frontiers = np.asarray(self.order_offsets[1:], dtype=np.int64)
        frontiers = np.unique(frontiers)
        frontiers = frontiers[(frontiers > 0) & (frontiers <= self.x_size)]
        return frontiers.astype(np.int64, copy=False)

    def homotopy_stage_groups(self) -> tuple[HomotopyStageGroup, ...]:
        """
        构造按阶次分组的 homotopy 元数据.

        Returns:
            返回 HomotopyStageGroup 的 tuple.
            每组包含该阶次需要放开的 packed 索引, 以及对应的 shape profile 编号集合.
        """
        if self.x_size == 0:
            return ()

        prefix_indices: list[int] = []
        for name in PREFIX_PROFILE_NAMES:
            p = PROFILE_INDEX[name]
            L = int(self.profile_L[p])
            if L < 0:
                continue
            for k in range(L + 1):
                idx = int(self.coeff_index[p, k])
                if idx >= 0:
                    prefix_indices.append(idx)

        shape_profile_ids = [int(PROFILE_INDEX[name]) for name in SHAPE_PROFILE_NAMES]
        active_shape_ids = [p for p in shape_profile_ids if int(self.profile_L[p]) >= 0]
        if not active_shape_ids:
            if prefix_indices:
                return (
                    HomotopyStageGroup(
                        order=0,
                        indices=np.asarray(prefix_indices, dtype=np.int64),
                        shape_profile_ids=np.zeros(0, dtype=np.int64),
                    ),
                )
            return ()

        max_shape_order = max(int(self.profile_L[p]) for p in active_shape_ids)
        groups: list[HomotopyStageGroup] = []
        for order in range(max_shape_order + 1):
            shape_ids = [p for p in active_shape_ids if int(self.profile_L[p]) >= order]
            indices = list(prefix_indices) if order == 0 else []
            for p in shape_ids:
                idx = int(self.coeff_index[p, order])
                if idx >= 0:
                    indices.append(idx)
            if not indices:
                continue
            groups.append(
                HomotopyStageGroup(
                    order=order,
                    indices=np.asarray(indices, dtype=np.int64),
                    shape_profile_ids=np.asarray(shape_ids, dtype=np.int64),
                )
            )
        return tuple(groups)

    def homotopy_truncation_profile_ids(self) -> np.ndarray:
        """
        返回参与 shape truncation 的 active profile 编号.

        Returns:
            返回一维整数数组, 顺序与 PROFILE_NAMES 中的 shape profiles 一致.
        """
        profile_ids = [int(PROFILE_INDEX[name]) for name in SHAPE_PROFILE_NAMES if int(self.profile_L[PROFILE_INDEX[name]]) >= 0]
        return np.asarray(profile_ids, dtype=np.int64)

    def build_coeffs(self, x: np.ndarray, *, include_none: bool = True) -> dict[str, list[float] | None]:
        """
        把 packed 状态向量还原成 profile 系数字典.

        Args:
            x: 一维 packed 状态向量.
            include_none: 是否保留 inactive profile 的 None 条目.

        Returns:
            返回 profile 名到系数列表的字典表示.
        """
        blocks = decode_packed_blocks(x, self.profile_L, self.coeff_index)
        coeffs: dict[str, list[float] | None] = {}
        for name, block in zip(PROFILE_NAMES, blocks, strict=True):
            if include_none or block is not None:
                coeffs[name] = None if block is None else block.tolist()
        return coeffs

    def build_equilibrium(self, x: np.ndarray) -> Equilibrium:
        """
        从 packed 状态向量构造完整 Equilibrium 快照.

        Args:
            x: 一维 packed 状态向量.

        Returns:
            返回基于当前 grid 和 case 生成的 Equilibrium 副本.
        """
        x_eval = self.coerce_x(x)
        self.stage_a_profile(x_eval)
        self.stage_b_geometry()
        self.stage_c_source()
        return self._build_equilibrium_from_runtime(x_eval)

    def stage_a_profile(self, x: np.ndarray) -> None:
        """
        执行 profile 阶段, 把 packed 系数写入各个 Profile 运行时缓存.

        Args:
            x: 一维 packed 状态向量.

        Returns:
        返回 None. 结果会原地写入各 profile 缓冲区.
        """
        self._fill_active_profile_views_from_packed_bulk(x)

    def stage_b_geometry(self) -> None:
        """
        执行 geometry 阶段, 用当前 shape profiles 刷新几何场.

        Returns:
            返回 None. 所有几何量都会原地写入 self.geometry.
        """
        self.geometry.update(
            float(self.case.a),
            float(self.case.R0),
            float(self.case.Z0),
            self.grid,
            self.h_profile,
            self.v_profile,
            self.k_profile,
            self.c0_profile,
            self.c1_profile,
            self.s1_profile,
            self.s2_profile,
        )

    def stage_c_source(self) -> None:
        """
        执行 source 阶段, 更新 psin_r, psin_rr, FFn_r, Pn_r 与归一化系数.

        Returns:
            返回 None. source 结果会原地写入相应运行时缓冲区.
        """
        alpha1, alpha2 = self.source_runner(
            self.psin_r,
            self.psin_rr,
            self.FFn_r,
            self.Pn_r,
            self.case.heat_input,
            self.case.current_input,
            float(self.case.R0),
            float(self.case.B0),
            self.grid.weights,
            self.grid.differentiation_matrix,
            self.grid.integration_matrix,
            self.grid.rho,
            self.geometry.V_r,
            self.geometry.Kn,
            self.geometry.Kn_r,
            self.geometry.Ln_r,
            self.geometry.S_r,
            self.geometry.R,
            self.geometry.JdivR,
            self.F_profile.u,
            float(self.case.Ip),
            float(self.case.beta),
        )
        self.alpha1 = float(alpha1)
        self.alpha2 = float(alpha2)

    def stage_d_residual(self) -> np.ndarray:
        """
        执行 residual 阶段并编码 packed 残差.

        Returns:
            返回 packed residual 向量.
        """
        self._build_G_inplace()
        return self._assemble_residual()

    def coerce_x(self, x: np.ndarray) -> np.ndarray:
        """
        校验完整 packed 状态向量形状.

        Args:
            x: 待校验的状态向量.

        Returns:
            返回通过校验的 float64 一维数组视图.
        """
        arr = np.asarray(x, dtype=np.float64)
        if arr.ndim != 1 or arr.shape[0] != self.x_size:
            raise ValueError(f"Expected x to have shape ({self.x_size},), got {arr.shape}")
        return arr

    def _coerce_active_x(self, x: np.ndarray, active_len: int) -> np.ndarray:
        active_len = self._validate_active_len(active_len)
        arr = np.asarray(x, dtype=np.float64)
        if arr.ndim != 1 or arr.shape[0] != active_len:
            raise ValueError(f"Expected active x to have shape ({active_len},), got {arr.shape}")
        return arr

    def _validate_active_len(self, active_len: int) -> int:
        length = int(active_len)
        if length < 0 or length > self.x_size:
            raise ValueError(f"active_len must be in [0, {self.x_size}], got {active_len!r}")
        return length

    def _coerce_active_indices(self, active_indices: np.ndarray) -> np.ndarray:
        indices = np.asarray(active_indices, dtype=np.int64)
        if indices.ndim != 1:
            raise ValueError(f"Expected active indices to be 1D, got {indices.shape}")
        if np.any(indices < 0) or np.any(indices >= self.x_size):
            raise ValueError(f"Active indices must be within [0, {self.x_size})")
        if indices.size != np.unique(indices).size:
            raise ValueError("Active indices must be unique")
        return indices

    def _coerce_masked_x(self, x: np.ndarray, active_indices: np.ndarray) -> np.ndarray:
        indices = self._coerce_active_indices(active_indices)
        arr = np.asarray(x, dtype=np.float64)
        if arr.ndim != 1 or arr.shape[0] != indices.shape[0]:
            raise ValueError(f"Expected masked x to have shape ({indices.shape[0]},), got {arr.shape}")
        return arr

    def _compose_active_state(
        self,
        x_prefix: np.ndarray,
        *,
        active_len: int,
        x_template: np.ndarray,
    ) -> np.ndarray:
        active_len = self._validate_active_len(active_len)
        x_active = self._coerce_active_x(x_prefix, active_len)
        x_full = self.coerce_x(x_template).copy()
        x_full[:active_len] = x_active
        return x_full

    def _compose_masked_state(
        self,
        x_active: np.ndarray,
        *,
        active_indices: np.ndarray,
        x_template: np.ndarray,
    ) -> np.ndarray:
        indices = self._coerce_active_indices(active_indices)
        x_values = self._coerce_masked_x(x_active, indices)
        x_full = self.coerce_x(x_template).copy()
        x_full[indices] = x_values
        return x_full

    def _allocate_runtime_arrays(self) -> None:
        nr = self.grid.Nr
        nt = self.grid.Nt
        n_active = int(self.active_profile_ids.size)
        max_active_len = 0
        if n_active > 0:
            max_active_len = max(int(self.profile_L[int(p)]) + 1 for p in self.active_profile_ids)

        self.h_profile = Profile(offset=0.0, coeff=self._profile_coeff_from_case(PROFILE_INDEX["h"]))
        self.v_profile = Profile(offset=0.0, coeff=self._profile_coeff_from_case(PROFILE_INDEX["v"]))
        self.k_profile = Profile(offset=float(self.case.ka), coeff=self._profile_coeff_from_case(PROFILE_INDEX["k"]))
        self.c0_profile = Profile(offset=float(self.case.c0a), coeff=self._profile_coeff_from_case(PROFILE_INDEX["c0"]))
        self.c1_profile = Profile(
            power=1,
            offset=float(self.case.c1a),
            coeff=self._profile_coeff_from_case(PROFILE_INDEX["c1"]),
        )
        self.s1_profile = Profile(
            power=1,
            offset=float(self.case.s1a),
            coeff=self._profile_coeff_from_case(PROFILE_INDEX["s1"]),
        )
        self.s2_profile = Profile(
            power=2,
            offset=float(self.case.s2a),
            coeff=self._profile_coeff_from_case(PROFILE_INDEX["s2"]),
        )
        self.psin_profile = Profile(power=2, offset=1.0, coeff=self._profile_coeff_from_case(PROFILE_INDEX["psin"]))
        self.F_profile = Profile(
            scale=self.case.R0 * self.case.B0,
            envelope_power=2,
            offset=1.0,
            coeff=self._profile_coeff_from_case(PROFILE_INDEX["F"]),
        )

        self.residual_fields = np.zeros((3, nr, nt), dtype=np.float64)
        self.psin_R = self.residual_fields[0]
        self.psin_Z = self.residual_fields[1]
        self.G = self.residual_fields[2]
        self.root_fields = np.empty((4, nr), dtype=np.float64)
        self.psin_r = self.root_fields[0]
        self.psin_rr = self.root_fields[1]
        self.FFn_r = self.root_fields[2]
        self.Pn_r = self.root_fields[3]
        self.alpha1 = 0.0
        self.alpha2 = 0.0
        self.active_u_fields = np.empty((n_active, 3, nr), dtype=np.float64)
        self.active_rp_fields = np.empty((n_active, 3, nr), dtype=np.float64)
        self.active_env_fields = np.empty((n_active, 3, nr), dtype=np.float64)
        self.active_offsets = np.empty(n_active, dtype=np.float64)
        self.active_scales = np.empty(n_active, dtype=np.float64)
        self.active_lengths = np.empty(n_active, dtype=np.int64)
        self.active_coeff_index_rows = np.full((n_active, max_active_len), -1, dtype=np.int64)

    def _refresh_case_runtime(self) -> None:
        self._sync_profile_specs()
        self._prepare_active_stage_a_plan()
        self.profile_runtime_views = self._build_profile_runtime_views()
        self._rebind_runtime()

    def _sync_profile_specs(self) -> None:
        self.h_profile.offset = 0.0
        self.v_profile.offset = 0.0
        self.k_profile.offset = float(self.case.ka)
        self.c0_profile.offset = float(self.case.c0a)
        self.c1_profile.offset = float(self.case.c1a)
        self.s1_profile.offset = float(self.case.s1a)
        self.s2_profile.offset = float(self.case.s2a)
        self.psin_profile.offset = 1.0
        self.F_profile.offset = 1.0
        self.F_profile.scale = self.case.R0 * self.case.B0
        self.h_profile.coeff = self._profile_coeff_from_case(PROFILE_INDEX["h"])
        self.v_profile.coeff = self._profile_coeff_from_case(PROFILE_INDEX["v"])
        self.k_profile.coeff = self._profile_coeff_from_case(PROFILE_INDEX["k"])
        self.c0_profile.coeff = self._profile_coeff_from_case(PROFILE_INDEX["c0"])
        self.c1_profile.coeff = self._profile_coeff_from_case(PROFILE_INDEX["c1"])
        self.s1_profile.coeff = self._profile_coeff_from_case(PROFILE_INDEX["s1"])
        self.s2_profile.coeff = self._profile_coeff_from_case(PROFILE_INDEX["s2"])
        self.psin_profile.coeff = self._profile_coeff_from_case(PROFILE_INDEX["psin"])
        self.F_profile.coeff = self._profile_coeff_from_case(PROFILE_INDEX["F"])

        for profile in (
            self.h_profile,
            self.v_profile,
            self.k_profile,
            self.c0_profile,
            self.c1_profile,
            self.s1_profile,
            self.s2_profile,
            self.psin_profile,
            self.F_profile,
        ):
            profile._prepare_runtime_cache(self.grid)

    def _profile_coeff_from_case(self, p: int) -> np.ndarray | None:
        L = int(self.profile_L[p])
        if L < 0:
            return None
        coeff = self.case.coeffs_by_name.get(PROFILE_NAMES[p])
        if coeff is None:
            return None
        arr = np.asarray(coeff, dtype=np.float64)
        return arr[: L + 1].copy()

    def _validate_case_compatibility(self, case: OperatorCase) -> None:
        profile_L, coeff_index, order_offsets = build_profile_layout(case.coeffs_by_name)
        if not np.array_equal(profile_L, self.profile_L):
            raise ValueError("Replacement case changes the active profile layout")
        if not np.array_equal(coeff_index, self.coeff_index):
            raise ValueError("Replacement case changes the packed coefficient layout")
        if not np.array_equal(order_offsets, self.order_offsets):
            raise ValueError("Replacement case changes the degree ordering layout")
        if case.heat_input.shape != (self.grid.Nr,) or case.current_input.shape != (self.grid.Nr,):
            raise ValueError(f"Expected heat_input/current_input to have shape ({self.grid.Nr},)")

    def _rebind_runtime(self) -> None:
        self.residual_runner = self._build_residual_runner()
        self.active_profile_runtime_views = self._profile_views_from_ids(self.active_profile_ids)
        fixed_profile_ids = np.flatnonzero(~self.active_profile_mask).astype(np.int64, copy=False)
        self.fixed_profile_runtime_views = self._profile_views_from_ids(fixed_profile_ids)
        self._fill_profile_views(self.fixed_profile_runtime_views)

    def _prepare_active_stage_a_plan(self) -> None:
        if self.active_profile_ids.size == 0:
            return

        for slot, p in enumerate(self.active_profile_ids):
            p_int = int(p)
            profile_name = PROFILE_NAMES[p_int]
            profile = getattr(self, f"{profile_name}_profile")
            L = int(self.profile_L[p_int])
            coeff_indices = self.coeff_index[p_int, : L + 1]

            profile.u_fields = self.active_u_fields[slot]
            self.active_rp_fields[slot] = profile.rp_fields
            self.active_env_fields[slot] = profile.env_fields
            self.active_offsets[slot] = profile.offset
            self.active_scales[slot] = profile.scale
            self.active_lengths[slot] = coeff_indices.size
            if self.active_coeff_index_rows.shape[1] > 0:
                self.active_coeff_index_rows[slot].fill(-1)
                self.active_coeff_index_rows[slot, : coeff_indices.size] = coeff_indices

    def _build_profile_runtime_views(self) -> tuple[ProfileRuntimeView, ...]:
        views: list[ProfileRuntimeView] = []
        for p, profile_name in enumerate(PROFILE_NAMES):
            views.append(self._build_profile_runtime_view(p, profile_name))
        return tuple(views)

    def _build_profile_runtime_view(self, p: int, profile_name: str) -> ProfileRuntimeView:
        profile = getattr(self, f"{profile_name}_profile")
        L = int(self.profile_L[p])
        coeff_indices = (
            np.empty(0, dtype=np.int64) if L < 0 else self.coeff_index[p, : L + 1]
        )
        return profile._runtime_view(coeff_indices=coeff_indices)

    def _build_residual_runner(self) -> Callable:
        profile_names = tuple(PROFILE_NAMES[int(p)] for p in self.active_profile_ids)
        try:
            return bind_residual_runner(
                profile_names,
                self.active_coeff_index_rows,
                self.active_lengths,
                self.x_size,
            )
        except KeyError as exc:
            raise ValueError(
                f"Unsupported active residual block set {profile_names!r}"
            ) from exc

    def _profile_views_from_ids(self, profile_ids: np.ndarray) -> tuple[ProfileRuntimeView, ...]:
        return tuple(self.profile_runtime_views[int(p)] for p in profile_ids)

    def _fill_profile_views(self, views: tuple[ProfileRuntimeView, ...]) -> None:
        for view in views:
            fill_profile_runtime_view(view)

    def _fill_profile_views_from_packed(self, x: np.ndarray, views: tuple[ProfileRuntimeView, ...]) -> None:
        for view in views:
            fill_profile_runtime_view_from_packed(view, x)

    def _fill_active_profile_views_from_packed_bulk(self, x: np.ndarray) -> None:
        if self.active_profile_ids.size == 0:
            return
        update_profiles_packed_bulk(
            self.active_u_fields,
            self.grid.T_fields,
            self.active_rp_fields,
            self.active_env_fields,
            self.active_offsets,
            self.active_scales,
            x,
            self.active_coeff_index_rows,
            self.active_lengths,
        )

    def _build_G_inplace(self) -> None:
        update_residual(
            self.residual_fields,
            self.alpha1,
            self.alpha2,
            self.root_fields,
            self.geometry.R_fields,
            self.geometry.Z_fields,
            self.geometry.J_fields,
            self.geometry.g_fields,
        )

    def _assemble_residual(self) -> np.ndarray:
        return self.residual_runner(
            self.G,
            self.psin_R,
            self.psin_Z,
            self.geometry.sin_tb,
            self.grid.sin_theta,
            self.grid.cos_theta,
            self.grid.sin_2theta,
            self.grid.rho,
            self.grid.rho2,
            self.grid.y,
            self.grid.T_fields[0],
            self.grid.weights,
            float(self.case.a),
            float(self.case.R0),
            float(self.case.B0),
        )

    def _build_equilibrium_from_runtime(self, x: np.ndarray) -> Equilibrium:
        coeff_blocks = decode_packed_blocks(x, self.profile_L, self.coeff_index)
        snapshot_profiles = self._snapshot_profiles(coeff_blocks)
        case = self.case
        return Equilibrium(
            R0=case.R0,
            Z0=case.Z0,
            B0=case.B0,
            a=case.a,
            grid=self.grid,
            active_profiles=[name for name in SHAPE_PROFILE_NAMES if case.coeffs_by_name[name] is not None],
            h_profile=snapshot_profiles["h"],
            v_profile=snapshot_profiles["v"],
            k_profile=snapshot_profiles["k"],
            c0_profile=snapshot_profiles["c0"],
            c1_profile=snapshot_profiles["c1"],
            s1_profile=snapshot_profiles["s1"],
            s2_profile=snapshot_profiles["s2"],
            FFn_r=self.FFn_r.copy(),
            Pn_r=self.Pn_r.copy(),
            psin_r=self.psin_r.copy(),
            psin_rr=self.psin_rr.copy(),
            alpha1=float(self.alpha1),
            alpha2=float(self.alpha2),
        )

    def _snapshot_profiles(self, coeff_blocks: tuple[np.ndarray | None, ...]) -> dict[str, Profile]:
        snapshots: dict[str, Profile] = {}
        for name in PROFILE_NAMES:
            profile = getattr(self, f"{name}_profile")
            copied = profile.copy()
            copied.coeff = None if coeff_blocks[PROFILE_INDEX[name]] is None else coeff_blocks[PROFILE_INDEX[name]].copy()
            snapshots[name] = copied
        return snapshots

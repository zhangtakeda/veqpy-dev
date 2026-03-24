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
    assemble_c0_residual_block,
    assemble_c1_residual_block,
    assemble_F_residual_block,
    assemble_h_residual_block,
    assemble_k_residual_block,
    assemble_psin_residual_block,
    assemble_s1_residual_block,
    assemble_s2_residual_block,
    assemble_v_residual_block,
    bind_runner,
    update_residual,
    validate_operator,
)
from veqpy.model import Equilibrium, Geometry, Grid, Profile
from veqpy.model.profile import ProfileRuntimeView, fill_profile_runtime_view
from veqpy.operator.codec import (
    decode_packed_state_active_trusted,
    decode_packed_blocks,
    decode_packed_state_inplace,
    encode_packed_residual,
    encode_packed_state,
)
from veqpy.operator.layout import (
    PROFILE_INDEX,
    PROFILE_NAMES,
    build_active_profile_metadata,
    build_profile_layout,
    packed_size,
)
from veqpy.operator.operator_case import OperatorCase

@dataclass(slots=True, frozen=True)
class ResidualAssembleSlot:
    """描述一个 active profile 对应的 residual 写回槽位."""

    coeff_row: np.ndarray
    coeff_indices: np.ndarray
    assemble: Callable[[], None]


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

    coeff_matrix: np.ndarray = field(init=False)
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
    residual_slots: tuple[ResidualAssembleSlot, ...] = field(init=False, repr=False)
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
        self.residual_slots = ()
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
        for name in ("F", "psin"):
            p = PROFILE_INDEX[name]
            L = int(self.profile_L[p])
            if L < 0:
                continue
            for k in range(L + 1):
                idx = int(self.coeff_index[p, k])
                if idx >= 0:
                    prefix_indices.append(idx)

        shape_profile_ids = [int(PROFILE_INDEX[name]) for name in ("h", "v", "k", "c0", "c1", "s1", "s2")]
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
        profile_ids = [
            int(PROFILE_INDEX[name]) for name in ("h", "v", "k", "c0", "c1", "s1", "s2") if int(self.profile_L[PROFILE_INDEX[name]]) >= 0
        ]
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
        return self._build_equilibrium_from_runtime()

    def stage_a_profile(self, x: np.ndarray) -> None:
        """
        执行 profile 阶段, 把 packed 系数写入各个 Profile 运行时缓存.

        Args:
            x: 一维 packed 状态向量.

        Returns:
            返回 None. 结果会原地写入 coeff_matrix 与各 profile 缓冲区.
        """
        decode_packed_state_active_trusted(
            x,
            self.active_profile_ids,
            self.profile_L,
            self.coeff_index,
            self.coeff_matrix,
        )
        self._fill_profile_views(self.active_profile_runtime_views)

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

        self.coeff_matrix = np.zeros_like(self.coeff_index, dtype=np.float64)
        self._load_case_coeff_matrix()
        self.h_profile = Profile(offset=0.0, coeff=self._profile_coeff_row(PROFILE_INDEX["h"]))
        self.v_profile = Profile(offset=0.0, coeff=self._profile_coeff_row(PROFILE_INDEX["v"]))
        self.k_profile = Profile(offset=float(self.case.ka), coeff=self._profile_coeff_row(PROFILE_INDEX["k"]))
        self.c0_profile = Profile(offset=float(self.case.c0a), coeff=self._profile_coeff_row(PROFILE_INDEX["c0"]))
        self.c1_profile = Profile(
            power=1,
            offset=float(self.case.c1a),
            coeff=self._profile_coeff_row(PROFILE_INDEX["c1"]),
        )
        self.s1_profile = Profile(
            power=1,
            offset=float(self.case.s1a),
            coeff=self._profile_coeff_row(PROFILE_INDEX["s1"]),
        )
        self.s2_profile = Profile(
            power=2,
            offset=float(self.case.s2a),
            coeff=self._profile_coeff_row(PROFILE_INDEX["s2"]),
        )
        self.psin_profile = Profile(power=2, offset=1.0, coeff=self._profile_coeff_row(PROFILE_INDEX["psin"]))
        self.F_profile = Profile(
            scale=self.case.R0 * self.case.B0,
            envelope_power=2,
            offset=1.0,
            coeff=self._profile_coeff_row(PROFILE_INDEX["F"]),
        )

        self.psin_R = np.zeros((nr, nt), dtype=np.float64)
        self.psin_Z = np.zeros((nr, nt), dtype=np.float64)
        self.G = np.zeros((nr, nt), dtype=np.float64)
        self.psin_r = np.empty(nr, dtype=np.float64)
        self.psin_rr = np.empty(nr, dtype=np.float64)
        self.FFn_r = np.empty(nr, dtype=np.float64)
        self.Pn_r = np.empty(nr, dtype=np.float64)
        self.alpha1 = 0.0
        self.alpha2 = 0.0

    def _refresh_case_runtime(self) -> None:
        self._load_case_coeff_matrix()
        self._sync_profile_specs()
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

    def _load_case_coeff_matrix(self) -> None:
        self.coeff_matrix.fill(0.0)
        coeff_dict = self.case.coeffs_by_name
        for p, name in enumerate(PROFILE_NAMES):
            L = int(self.profile_L[p])
            if L < 0:
                continue
            coeff = coeff_dict.get(name)
            if coeff is None:
                continue
            self.coeff_matrix[p, : L + 1] = np.asarray(coeff, dtype=np.float64)

    def _profile_coeff_row(self, p: int) -> np.ndarray | None:
        L = int(self.profile_L[p])
        if L < 0:
            return None
        return self.coeff_matrix[p, : L + 1]

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
        self.residual_slots = self._build_residual_slots()
        self.active_profile_runtime_views = self._profile_views_from_ids(self.active_profile_ids)
        fixed_profile_ids = np.flatnonzero(~self.active_profile_mask).astype(np.int64, copy=False)
        self.fixed_profile_runtime_views = self._profile_views_from_ids(fixed_profile_ids)
        self._fill_profile_views(self.fixed_profile_runtime_views)

    def _build_profile_runtime_views(self) -> tuple[ProfileRuntimeView, ...]:
        views: list[ProfileRuntimeView] = []
        for p, profile_name in enumerate(PROFILE_NAMES):
            views.append(self._build_profile_runtime_view(p, profile_name))
        return tuple(views)

    def _build_profile_runtime_view(self, p: int, profile_name: str) -> ProfileRuntimeView:
        profile = getattr(self, f"{profile_name}_profile")
        return profile._runtime_view()

    def _build_residual_slots(self) -> tuple[ResidualAssembleSlot, ...]:
        slots: list[ResidualAssembleSlot] = []
        for p in self.active_profile_ids:
            slots.append(self._build_residual_slot(int(p)))
        return tuple(slots)

    def _build_residual_slot(self, p: int) -> ResidualAssembleSlot:
        L = int(self.profile_L[p])
        profile_name = PROFILE_NAMES[p]
        coeff_row = self.coeff_matrix[p, : L + 1]
        coeff_indices = self.coeff_index[p, : L + 1]
        a = float(self.case.a)
        R0 = float(self.case.R0)
        B0 = float(self.case.B0)

        match profile_name:
            case "h":

                def assemble() -> None:
                    assemble_h_residual_block(
                        coeff_row,
                        self.G,
                        self.psin_R,
                        self.grid.y,
                        self.grid.T,
                        self.grid.weights,
                        a,
                    )
            case "v":

                def assemble() -> None:
                    assemble_v_residual_block(
                        coeff_row,
                        self.G,
                        self.psin_Z,
                        self.grid.y,
                        self.grid.T,
                        self.grid.weights,
                        a,
                    )
            case "k":

                def assemble() -> None:
                    assemble_k_residual_block(
                        coeff_row,
                        self.G,
                        self.psin_Z,
                        self.grid.sin_theta,
                        self.grid.rho,
                        self.grid.y,
                        self.grid.T,
                        self.grid.weights,
                        a,
                    )
            case "c0":

                def assemble() -> None:
                    assemble_c0_residual_block(
                        coeff_row,
                        self.G,
                        self.psin_R,
                        self.geometry.sin_tb,
                        self.grid.rho,
                        self.grid.y,
                        self.grid.T,
                        self.grid.weights,
                        a,
                    )
            case "c1":

                def assemble() -> None:
                    assemble_c1_residual_block(
                        coeff_row,
                        self.G,
                        self.psin_R,
                        self.geometry.sin_tb,
                        self.grid.cos_theta,
                        self.grid.rho2,
                        self.grid.y,
                        self.grid.T,
                        self.grid.weights,
                        a,
                    )
            case "s1":

                def assemble() -> None:
                    assemble_s1_residual_block(
                        coeff_row,
                        self.G,
                        self.psin_R,
                        self.geometry.sin_tb,
                        self.grid.sin_theta,
                        self.grid.rho2,
                        self.grid.y,
                        self.grid.T,
                        self.grid.weights,
                        a,
                    )
            case "s2":

                def assemble() -> None:
                    assemble_s2_residual_block(
                        coeff_row,
                        self.G,
                        self.psin_R,
                        self.geometry.sin_tb,
                        self.grid.sin_2theta,
                        self.grid.rho,
                        self.grid.rho2,
                        self.grid.y,
                        self.grid.T,
                        self.grid.weights,
                        a,
                    )
            case "psin":

                def assemble() -> None:
                    assemble_psin_residual_block(
                        coeff_row,
                        self.G,
                        self.grid.rho2,
                        self.grid.y,
                        self.grid.T,
                        self.grid.weights,
                    )
            case "F":

                def assemble() -> None:
                    assemble_F_residual_block(
                        coeff_row,
                        self.G,
                        self.grid.y,
                        self.grid.T,
                        self.grid.weights,
                        R0,
                        B0,
                    )
            case _:
                raise ValueError(f"Unsupported active profile {profile_name!r}")

        return ResidualAssembleSlot(
            coeff_row=coeff_row,
            coeff_indices=coeff_indices,
            assemble=assemble,
        )

    def _profile_views_from_ids(self, profile_ids: np.ndarray) -> tuple[ProfileRuntimeView, ...]:
        return tuple(self.profile_runtime_views[int(p)] for p in profile_ids)

    def _fill_profile_views(self, views: tuple[ProfileRuntimeView, ...]) -> None:
        for view in views:
            fill_profile_runtime_view(view)

    def _build_G_inplace(self) -> None:
        update_residual(
            self.psin_R,
            self.psin_Z,
            self.G,
            self.alpha1,
            self.alpha2,
            self.psin_r,
            self.psin_rr,
            self.FFn_r,
            self.Pn_r,
            self.geometry.R,
            self.geometry.R_t,
            self.geometry.Z_t,
            self.geometry.J,
            self.geometry.JdivR,
            self.geometry.gttdivJR,
            self.geometry.grtdivJR_t,
            self.geometry.gttdivJR_r,
        )

    def _assemble_residual(self) -> np.ndarray:
        return encode_packed_residual(self.residual_slots, self.x_size)

    def _build_equilibrium_from_runtime(self) -> Equilibrium:
        case = self.case
        return Equilibrium(
            R0=case.R0,
            Z0=case.Z0,
            B0=case.B0,
            a=case.a,
            grid=self.grid,
            active_profiles=[
                name for name in ("h", "v", "k", "c0", "c1", "s1", "s2") if case.coeffs_by_name[name] is not None
            ],
            h_profile=self.h_profile.copy(),
            v_profile=self.v_profile.copy(),
            k_profile=self.k_profile.copy(),
            c0_profile=self.c0_profile.copy(),
            c1_profile=self.c1_profile.copy(),
            s1_profile=self.s1_profile.copy(),
            s2_profile=self.s2_profile.copy(),
            FFn_r=self.FFn_r.copy(),
            Pn_r=self.Pn_r.copy(),
            psin_r=self.psin_r.copy(),
            psin_rr=self.psin_rr.copy(),
            alpha1=float(self.alpha1),
            alpha2=float(self.alpha2),
        )

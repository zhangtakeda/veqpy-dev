"""
Module: operator.operator

Role:
- 负责连接 case, grid, model runtime, engine kernels 与 packed layout.
- 负责暴露稳定的 residual 求值接口.

Public API:
- HomotopyStageGroup
- Operator

Notes:
- `Operator` 是 operator 层主协调器.
- 不负责 solver 迭代策略, backend 选择, 或 benchmark 编排.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from veqpy.engine import (
    bind_residual_runner,
    bind_source_runner,
    update_profiles_packed_bulk,
    update_residual,
    validate_operator,
)
from veqpy.model import Equilibrium, Geometry, Grid, Profile
from veqpy.operator.codec import decode_packed_blocks, encode_packed_state
from veqpy.operator.layout import (
    PREFIX_PROFILE_NAMES,
    build_active_profile_metadata,
    build_fourier_profile_names,
    build_profile_index,
    build_profile_layout,
    build_profile_names,
    build_shape_profile_names,
    packed_size,
)
from veqpy.operator.operator_case import OperatorCase

_PROFILE_STATIC_KWARGS: dict[str, dict[str, int]] = {
    "psin": {"power": 2},
    "F": {"envelope_power": 2},
}
_PROFILE_OFFSET_SPECS: dict[str, float | str] = {
    "h": 0.0,
    "v": 0.0,
    "k": "ka",
    "psin": 1.0,
    "F": 1.0,
}
_PROFILE_SCALE_SPECS: dict[str, tuple[str, ...]] = {
    "F": ("R0", "B0"),
}


@dataclass(slots=True, frozen=True)
class HomotopyStageGroup:
    """记录 homotopy 某一阶次对应的 packed 索引集合."""

    order: int
    indices: np.ndarray
    shape_profile_ids: np.ndarray


@dataclass(slots=True)
class Operator:
    """封装固定算子名, 导数变量域, grid 与 case 的 residual 求值器."""

    name: str
    derivative: str
    grid: Grid = field(repr=False)
    case: OperatorCase = field(repr=False)
    geometry: Geometry = field(init=False)

    h_profile: Profile = field(init=False)
    v_profile: Profile = field(init=False)
    k_profile: Profile = field(init=False)
    psin_profile: Profile = field(init=False)
    F_profile: Profile = field(init=False)
    profiles_by_name: dict[str, Profile] = field(init=False, repr=False)

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

    prefix_profile_names: tuple[str, ...] = field(init=False, repr=False)
    shape_profile_names: tuple[str, ...] = field(init=False, repr=False)
    profile_names: tuple[str, ...] = field(init=False, repr=False)
    profile_index: dict[str, int] = field(init=False, repr=False)
    c_profile_names: tuple[str, ...] = field(init=False, repr=False)
    s_profile_names: tuple[str, ...] = field(init=False, repr=False)

    profile_L: np.ndarray = field(init=False, repr=False)
    coeff_index: np.ndarray = field(init=False, repr=False)
    order_offsets: np.ndarray = field(init=False, repr=False)
    active_profile_mask: np.ndarray = field(init=False, repr=False)
    active_profile_ids: np.ndarray = field(init=False, repr=False)
    x_size: int = field(init=False, repr=False)
    active_u_fields: np.ndarray = field(init=False, repr=False)
    active_rp_fields: np.ndarray = field(init=False, repr=False)
    active_env_fields: np.ndarray = field(init=False, repr=False)
    active_offsets: np.ndarray = field(init=False, repr=False)
    active_scales: np.ndarray = field(init=False, repr=False)
    active_lengths: np.ndarray = field(init=False, repr=False)
    active_coeff_index_rows: np.ndarray = field(init=False, repr=False)
    c_family_fields: np.ndarray = field(init=False, repr=False)
    s_family_fields: np.ndarray = field(init=False, repr=False)
    c_effective_order: int = field(init=False, repr=False)
    s_effective_order: int = field(init=False, repr=False)
    residual_stage_runner: Callable = field(init=False, repr=False)
    source_stage_runner: Callable = field(init=False, repr=False)

    def __post_init__(self) -> None:
        """完成 layout 构造, 运行时缓冲区分配和 case 绑定."""
        spec = validate_operator(self.name, self.derivative)

        self.name = spec.name
        self.geometry = Geometry(grid=self.grid)
        self.prefix_profile_names = PREFIX_PROFILE_NAMES
        self.shape_profile_names = build_shape_profile_names(self.grid.K_max)
        self.profile_names = build_profile_names(self.grid.K_max)
        self.profile_index = build_profile_index(self.profile_names)
        fourier_profile_names = build_fourier_profile_names(self.grid.K_max)
        self.c_profile_names = tuple(name for name in fourier_profile_names if name.startswith("c"))
        self.s_profile_names = tuple(name for name in fourier_profile_names if name.startswith("s"))

        self.profile_L, self.coeff_index, self.order_offsets = build_profile_layout(
            self.case.profile_coeffs,
            profile_names=self.profile_names,
            prefix_profile_names=self.prefix_profile_names,
        )
        self.active_profile_mask, self.active_profile_ids = build_active_profile_metadata(
            self.profile_L,
            profile_names=self.profile_names,
        )
        self.x_size = packed_size(self.coeff_index)
        self._validate_runtime_profile_support()

        self._setup_runtime_state()
        self.source_stage_runner = bind_source_runner(spec.name, self.derivative)
        self.residual_stage_runner = lambda *args, **kwargs: np.zeros(self.x_size, dtype=np.float64)
        self._refresh_runtime_state()

    def __call__(self, x: np.ndarray, *args, **kwargs) -> np.ndarray:
        """调用 residual 求值主入口."""
        return self.residual(x, *args, **kwargs)

    def _validate_runtime_profile_support(self) -> None:
        """geometry/residual runtime 已支持动态 Fourier family；保留接口以承接阶段性校验."""
        return None

    def replace_case(self, case: OperatorCase) -> None:
        """在不改变 packed layout 的前提下替换当前 case."""
        self._validate_case_compatibility(case)
        self.case = case
        self._refresh_runtime_state()

    def encode_initial_state(self) -> np.ndarray:
        """把当前 case 中的 profile 系数编码成 packed 初值."""
        return encode_packed_state(
            self.case.profile_coeffs,
            self.profile_L,
            self.coeff_index,
            profile_names=self.profile_names,
        )

    def residual(
        self,
        x: np.ndarray,
    ) -> np.ndarray:
        """完整执行 profile, geometry, source, residual 四阶段求值."""
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
        """只激活 packed 向量前缀后求值对应 residual 前缀."""
        x_full = self._compose_active_state(x_prefix, active_len=active_len, x_template=x_template)
        return self.residual(x_full)[:active_len].copy()

    def residual_masked(
        self,
        x_active: np.ndarray,
        *,
        active_indices: np.ndarray,
        x_template: np.ndarray,
    ) -> np.ndarray:
        """只替换指定索引集合后求值对应 masked residual."""
        x_full = self._compose_masked_state(x_active, active_indices=active_indices, x_template=x_template)
        return self.residual(x_full)[active_indices].copy()

    def homotopy_frontiers(self) -> np.ndarray:
        """返回 packed 向量按阶次展开时的前缀边界."""
        if self.x_size == 0:
            return np.zeros(0, dtype=np.int64)
        frontiers = np.asarray(self.order_offsets[1:], dtype=np.int64)
        frontiers = np.unique(frontiers)
        frontiers = frontiers[(frontiers > 0) & (frontiers <= self.x_size)]
        return frontiers.astype(np.int64, copy=False)

    def homotopy_stage_groups(self) -> tuple[HomotopyStageGroup, ...]:
        """构造按阶次分组的 homotopy 元数据."""
        if self.x_size == 0:
            return ()

        prefix_indices: list[int] = []
        for name in self.prefix_profile_names:
            p = self.profile_index[name]
            L = int(self.profile_L[p])
            if L < 0:
                continue
            for k in range(L + 1):
                idx = int(self.coeff_index[p, k])
                if idx >= 0:
                    prefix_indices.append(idx)

        shape_profile_ids = [int(self.profile_index[name]) for name in self.shape_profile_names]
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
        """返回参与 shape truncation 的 active profile 编号."""
        profile_ids = [
            int(self.profile_index[name])
            for name in self.shape_profile_names
            if int(self.profile_L[self.profile_index[name]]) >= 0
        ]
        return np.asarray(profile_ids, dtype=np.int64)

    def build_coeffs(self, x: np.ndarray, *, include_none: bool = True) -> dict[str, list[float] | None]:
        """把 packed 状态向量还原成 profile 系数字典."""
        blocks = decode_packed_blocks(x, self.profile_L, self.coeff_index, profile_names=self.profile_names)
        coeffs: dict[str, list[float] | None] = {}
        for name, block in zip(self.profile_names, blocks, strict=True):
            if include_none or block is not None:
                coeffs[name] = None if block is None else block.tolist()
        return coeffs

    def build_equilibrium(self, x: np.ndarray) -> Equilibrium:
        """从 packed 状态向量构造完整 Equilibrium 快照."""
        x_eval = self.coerce_x(x)
        self.stage_a_profile(x_eval)
        self.stage_b_geometry()
        self.stage_c_source()
        return self._snapshot_equilibrium_from_runtime(x_eval)

    def stage_a_profile(self, x: np.ndarray) -> None:
        """执行 profile 阶段并刷新 active profile fields."""
        self._fill_active_profile_views_from_packed_bulk(x)

    def stage_b_geometry(self) -> None:
        """执行 geometry 阶段并刷新 geometry fields."""
        self._refresh_fourier_family_fields()
        self.geometry.update(
            float(self.case.a),
            float(self.case.R0),
            float(self.case.Z0),
            self.grid,
            self.h_profile.u_fields,
            self.v_profile.u_fields,
            self.k_profile.u_fields,
            self.c_family_fields,
            self.s_family_fields,
            c_active_order=self.c_effective_order,
            s_active_order=self.s_effective_order,
        )

    def stage_c_source(self) -> None:
        """执行 source 阶段并刷新 root fields 与缩放系数."""
        alpha1, alpha2 = self.source_stage_runner(
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
        """执行 residual 阶段并返回 packed 残差."""
        self._build_G_inplace()
        return self._assemble_residual()

    def coerce_x(self, x: np.ndarray) -> np.ndarray:
        """校验完整 packed 状态向量形状."""
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

    def _setup_runtime_state(self) -> None:
        nr = self.grid.Nr
        nt = self.grid.Nt
        n_active = int(self.active_profile_ids.size)
        max_active_len = 0
        if n_active > 0:
            max_active_len = max(int(self.profile_L[int(p)]) + 1 for p in self.active_profile_ids)

        profiles_by_name: dict[str, Profile] = {}
        for name in self.profile_names:
            profile = self._make_profile(name)
            profiles_by_name[name] = profile
            if hasattr(type(self), f"{name}_profile"):
                setattr(self, f"{name}_profile", profile)
        self.profiles_by_name = profiles_by_name

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
        self.c_family_fields = np.empty((self.grid.K_max + 1, 3, nr), dtype=np.float64)
        self.s_family_fields = np.zeros((self.grid.K_max + 1, 3, nr), dtype=np.float64)

    def _refresh_runtime_state(self) -> None:
        self._refresh_profile_runtime()
        self._refresh_fourier_family_metadata()
        self._refresh_stage_a_runtime()
        self._refresh_runtime_bindings()

    def _refresh_profile_runtime(self) -> None:
        for name in self.profile_names:
            profile = self._profile_by_name(name)
            profile.offset = self._profile_offset_from_case(name)
            profile.scale = self._profile_scale_from_case(name)
            profile.coeff = self._profile_coeff_from_case(self.profile_index[name])
            profile._prepare_runtime_cache(self.grid)

    def _make_profile(self, name: str) -> Profile:
        kwargs: dict[str, float | int | np.ndarray | None] = dict(self._profile_static_kwargs(name))
        kwargs["offset"] = self._profile_offset_from_case(name)
        kwargs["coeff"] = self._profile_coeff_from_case(self.profile_index[name])
        kwargs["scale"] = self._profile_scale_from_case(name)
        return Profile(**kwargs)

    def _profile_static_kwargs(self, name: str) -> dict[str, int]:
        if name in _PROFILE_STATIC_KWARGS:
            return _PROFILE_STATIC_KWARGS[name]
        if name.startswith(("c", "s")) and name[1:].isdigit():
            order = int(name[1:])
            return {} if order == 0 else {"power": order}
        return {}

    def _profile_offset_from_case(self, name: str) -> float:
        if name.startswith("c") and name[1:].isdigit():
            return self._offset_from_array(self.case.c_offsets, int(name[1:]))
        if name.startswith("s") and name[1:].isdigit():
            return self._offset_from_array(self.case.s_offsets, int(name[1:]))
        try:
            spec = _PROFILE_OFFSET_SPECS[name]
        except KeyError as exc:
            raise KeyError(f"Unknown profile name {name!r}") from exc
        if isinstance(spec, str):
            return float(getattr(self.case, spec))
        return float(spec)

    def _profile_scale_from_case(self, name: str) -> float:
        attrs = _PROFILE_SCALE_SPECS.get(name)
        if attrs is None:
            return 1.0
        scale = 1.0
        for attr in attrs:
            scale *= float(getattr(self.case, attr))
        return scale

    def _profile_coeff_from_case(self, p: int) -> np.ndarray | None:
        L = int(self.profile_L[p])
        if L < 0:
            return None
        coeff = self.case.profile_coeffs.get(self.profile_names[p])
        if coeff is None:
            return None
        arr = np.asarray(coeff, dtype=np.float64)
        return arr[: L + 1].copy()

    def _offset_from_array(self, offsets: np.ndarray | None, order: int) -> float:
        if offsets is None or order >= offsets.shape[0]:
            return 0.0
        return float(offsets[order])

    def _validate_case_compatibility(self, case: OperatorCase) -> None:
        profile_L, coeff_index, order_offsets = build_profile_layout(
            case.profile_coeffs,
            profile_names=self.profile_names,
            prefix_profile_names=self.prefix_profile_names,
        )
        if not np.array_equal(profile_L, self.profile_L):
            raise ValueError("Replacement case changes the active profile layout")
        if not np.array_equal(coeff_index, self.coeff_index):
            raise ValueError("Replacement case changes the packed coefficient layout")
        if not np.array_equal(order_offsets, self.order_offsets):
            raise ValueError("Replacement case changes the degree ordering layout")
        if case.heat_input.shape != (self.grid.Nr,) or case.current_input.shape != (self.grid.Nr,):
            raise ValueError(f"Expected heat_input/current_input to have shape ({self.grid.Nr},)")

    def _refresh_runtime_bindings(self) -> None:
        self.residual_stage_runner = self._build_residual_stage_runner()
        fixed_profile_ids = np.flatnonzero(~self.active_profile_mask).astype(np.int64, copy=False)
        for p in fixed_profile_ids:
            self._profile_by_name(self.profile_names[int(p)]).update()

    def _refresh_stage_a_runtime(self) -> None:
        if self.active_profile_ids.size == 0:
            return

        for slot, p in enumerate(self.active_profile_ids):
            p_int = int(p)
            profile_name = self.profile_names[p_int]
            profile = self._profile_by_name(profile_name)
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

    def _build_residual_stage_runner(self) -> Callable:
        profile_names = tuple(self.profile_names[int(p)] for p in self.active_profile_ids)
        try:
            return bind_residual_runner(
                profile_names,
                self.active_coeff_index_rows,
                self.active_lengths,
                self.x_size,
            )
        except KeyError as exc:
            raise ValueError(f"Unsupported active residual block set {profile_names!r}") from exc

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

    def _refresh_fourier_family_fields(self) -> None:
        for name in self.c_profile_names[: self.c_effective_order + 1]:
            order = int(name[1:])
            self.c_family_fields[order] = self._profile_by_name(name).u_fields

        self.s_family_fields[0].fill(0.0)
        for name in self.s_profile_names[: self.s_effective_order]:
            order = int(name[1:])
            self.s_family_fields[order] = self._profile_by_name(name).u_fields

    def _refresh_fourier_family_metadata(self) -> None:
        self.c_effective_order = self._effective_family_order(
            self.c_profile_names, self.case.c_offsets, minimum_order=0
        )
        self.s_effective_order = self._effective_family_order(
            self.s_profile_names, self.case.s_offsets, minimum_order=0
        )

        if self.c_effective_order + 1 < self.c_family_fields.shape[0]:
            self.c_family_fields[self.c_effective_order + 1 :].fill(0.0)
        if self.s_effective_order + 1 < self.s_family_fields.shape[0]:
            self.s_family_fields[self.s_effective_order + 1 :].fill(0.0)

    def _effective_family_order(
        self,
        profile_names: tuple[str, ...],
        offsets: np.ndarray | None,
        *,
        minimum_order: int,
    ) -> int:
        effective_order = int(minimum_order)
        for name in profile_names:
            order = int(name[1:])
            if self.case.profile_coeffs.get(name) is not None:
                effective_order = max(effective_order, order)
                continue
            if abs(self._offset_from_array(offsets, order)) > 1e-14:
                effective_order = max(effective_order, order)
        return effective_order

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
        return self.residual_stage_runner(
            self.G,
            self.psin_R,
            self.psin_Z,
            self.geometry.sin_tb,
            self.grid.sin_ktheta,
            self.grid.cos_ktheta,
            self.grid.rho_powers,
            self.grid.y,
            self.grid.T_fields[0],
            self.grid.weights,
            float(self.case.a),
            float(self.case.R0),
            float(self.case.B0),
        )

    def _snapshot_equilibrium_from_runtime(self, x: np.ndarray) -> Equilibrium:
        coeff_blocks = decode_packed_blocks(x, self.profile_L, self.coeff_index, profile_names=self.profile_names)
        snapshot_profiles = self._snapshot_equilibrium_profiles(coeff_blocks)
        case = self.case
        return Equilibrium(
            R0=case.R0,
            Z0=case.Z0,
            B0=case.B0,
            a=case.a,
            grid=self.grid,
            active_profiles=snapshot_profiles,
            FFn_r=self.FFn_r.copy(),
            Pn_r=self.Pn_r.copy(),
            psin_r=self.psin_r.copy(),
            psin_rr=self.psin_rr.copy(),
            alpha1=float(self.alpha1),
            alpha2=float(self.alpha2),
        )

    def _profile_by_name(self, name: str) -> Profile:
        return self.profiles_by_name[name]

    def _snapshot_profile(self, name: str, coeff_block: np.ndarray | None) -> Profile:
        copied = self._profile_by_name(name).copy()
        copied.coeff = None if coeff_block is None else coeff_block.copy()
        return copied

    def _snapshot_equilibrium_profiles(self, coeff_blocks: tuple[np.ndarray | None, ...]) -> dict[str, Profile]:
        return {
            name: self._snapshot_profile(name, coeff_blocks[self.profile_index[name]])
            for name in self.shape_profile_names
        }

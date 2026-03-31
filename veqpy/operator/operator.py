"""
Module: operator.operator

Role:
- 负责连接 case, grid, model runtime, engine kernels 与 packed layout.
- 负责暴露稳定的 residual 求值接口.

Public API:
- Operator

Notes:
- `Operator` 是默认 fused 算子.
- 不负责 solver 迭代策略, backend 选择, 或 benchmark 编排.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from numpy.polynomial.chebyshev import chebval, chebvander
from typing import Callable

import numpy as np

from veqpy.engine import (
    bind_fused_fixed_point_psin_residual_runner,
    bind_fused_profile_owned_psin_residual_runner,
    bind_fused_single_pass_residual_runner,
    bind_residual_runner,
    bind_residual_stage_runner,
    build_source_remap_cache,
    materialize_profile_owned_psin_source,
    materialize_projected_source_inputs,
    resolve_source_inputs,
    update_geometry,
    update_fourier_family_fields,
    update_fixed_point_psin_query,
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
_PROFILE_SCALE_SPECS: dict[str, tuple[str, ...]] = {"F": ("R0", "B0")}


@dataclass(frozen=True, slots=True)
class _SourceProjectionPolicy:
    domain: str
    heat_degree: int
    current_degree: int
    current_ip_endpoint_policy: str = "none"
    current_other_endpoint_policy: str = "none"


_SOURCE_PROJECTION_POLICIES: dict[tuple[str, str, str], _SourceProjectionPolicy] = {
    ("PQ", "psin", "uniform"): _SourceProjectionPolicy(
        domain="sqrt_psin",
        heat_degree=5,
        current_degree=6,
        current_ip_endpoint_policy="affine_both",
        current_other_endpoint_policy="none",
    )
}

_PROJECTION_DOMAIN_CODES = {
    "psin": 0,
    "sqrt_psin": 1,
}

_ENDPOINT_POLICY_CODES = {
    "none": 0,
    "right": 1,
    "both": 2,
    "affine_both": 3,
}

_SOURCE_PARAMETERIZATION_CODES = {
    "identity": 0,
    "sqrt_psin": 1,
}


@dataclass(slots=True)
class _OperatorCore:
    """封装固定 case, grid 与 runtime 的 residual 求值器."""

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
    psin: np.ndarray = field(init=False)
    psin_r: np.ndarray = field(init=False)
    psin_rr: np.ndarray = field(init=False)
    FFn_psin: np.ndarray = field(init=False)
    Pn_psin: np.ndarray = field(init=False)
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
    c_family_base_fields: np.ndarray = field(init=False, repr=False)
    s_family_base_fields: np.ndarray = field(init=False, repr=False)
    active_slot_by_profile_id: np.ndarray = field(init=False, repr=False)
    c_family_source_slots: np.ndarray = field(init=False, repr=False)
    s_family_source_slots: np.ndarray = field(init=False, repr=False)
    c_effective_order: int = field(init=False, repr=False)
    s_effective_order: int = field(init=False, repr=False)
    _source_route_spec: object = field(init=False, repr=False)
    _source_cache_key: tuple[str, str, int] | None = field(init=False, repr=False)
    _source_runner: Callable = field(init=False, repr=False)
    profile_stage_runner: Callable = field(init=False, repr=False)
    geometry_stage_runner: Callable = field(init=False, repr=False)
    source_stage_runner: Callable = field(init=False, repr=False)
    residual_pack_stage_runner: Callable = field(init=False, repr=False)
    residual_full_stage_runner: Callable = field(init=False, repr=False)
    residual_pack_runner: Callable = field(init=False, repr=False)
    residual_stage_runner: Callable = field(init=False, repr=False)
    packed_residual: np.ndarray = field(init=False, repr=False)
    source_barycentric_weights: np.ndarray = field(init=False, repr=False)
    source_fixed_remap_matrix: np.ndarray = field(init=False, repr=False)
    source_psin_query: np.ndarray = field(init=False, repr=False)
    source_parameter_query: np.ndarray = field(init=False, repr=False)
    source_heat_projection_fit_matrix: np.ndarray = field(init=False, repr=False)
    source_current_projection_fit_matrix: np.ndarray = field(init=False, repr=False)
    source_heat_projection_coeff: np.ndarray = field(init=False, repr=False)
    source_current_projection_coeff: np.ndarray = field(init=False, repr=False)
    source_projection_query: np.ndarray = field(init=False, repr=False)
    source_endpoint_blend: np.ndarray = field(init=False, repr=False)
    materialized_heat_input: np.ndarray = field(init=False, repr=False)
    materialized_current_input: np.ndarray = field(init=False, repr=False)
    source_target_root_fields: np.ndarray = field(init=False, repr=False)
    _source_projection_policy: _SourceProjectionPolicy | None = field(init=False, repr=False)

    def __post_init__(self) -> None:
        """完成 layout 构造, 运行时缓冲区分配和 case 绑定."""
        self._refresh_operator_identity()
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
        self.profile_stage_runner = lambda x: None
        self.geometry_stage_runner = lambda: None
        self.source_stage_runner = lambda: (0.0, 0.0)
        self.residual_pack_stage_runner = lambda: np.zeros(self.x_size, dtype=np.float64)
        self.residual_full_stage_runner = lambda: np.zeros(self.x_size, dtype=np.float64)
        self.residual_pack_runner = lambda *args, **kwargs: np.zeros(self.x_size, dtype=np.float64)
        self.residual_stage_runner = lambda *args, **kwargs: np.zeros(self.x_size, dtype=np.float64)
        self._refresh_runtime_state()

    def __call__(self, x: np.ndarray, *args, **kwargs) -> np.ndarray:
        """调用 residual 求值主入口."""
        return self.residual(x, *args, **kwargs)

    @property
    def source_strategy(self) -> str:
        return self._source_route_spec.source_strategy

    @property
    def source_coordinate(self) -> str:
        return self.case.coordinate

    @property
    def source_nodes(self) -> str:
        return self.case.nodes

    @property
    def source_parameterization(self) -> str:
        return self._source_route_spec.source_parameterization

    @property
    def source_n_src(self) -> int:
        return int(self.case.heat_input.shape[0])

    def _validate_runtime_profile_support(self) -> None:
        """校验当前 source route 对 psin profile ownership 的要求."""
        has_active_psin = int(self.profile_L[self.profile_index["psin"]]) >= 0
        if self.source_strategy == "profile_owned_psin" and not has_active_psin:
            raise ValueError(f"{self.case.name} requires an active psin profile because psin is optimized externally")
        if self.case.coordinate == "psin" and self.source_strategy != "profile_owned_psin" and has_active_psin:
            raise ValueError(f"{self.case.name} does not accept an active psin profile because psin is source-owned")
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
        return self._evaluate_residual(x_eval)

    def residual_fast(
        self,
        x: np.ndarray,
    ) -> np.ndarray:
        """供 solver 热路径使用的免校验 residual 求值入口."""
        if isinstance(x, np.ndarray) and x.dtype == np.float64 and x.ndim == 1 and x.shape[0] == self.x_size:
            return self._evaluate_residual(x)
        return self.residual(x)

    def _evaluate_residual(self, x_eval: np.ndarray) -> np.ndarray:
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
        self.profile_stage_runner(x)

    def stage_b_geometry(self) -> None:
        """执行 geometry 阶段并刷新 geometry fields."""
        self.geometry_stage_runner()

    def stage_c_source(self) -> None:
        """执行 source 阶段并刷新 root fields 与缩放系数."""
        alpha1, alpha2 = self.source_stage_runner()
        self.alpha1 = float(alpha1)
        self.alpha2 = float(alpha2)

    def stage_d_residual(self) -> np.ndarray:
        """执行 residual 阶段并返回 packed 残差."""
        packed = self.residual_full_stage_runner()
        return packed.copy()

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
        self.root_fields = np.empty((5, nr), dtype=np.float64)
        self.psin = self.root_fields[0]
        self.psin_r = self.root_fields[1]
        self.psin_rr = self.root_fields[2]
        self.FFn_psin = self.root_fields[3]
        self.Pn_psin = self.root_fields[4]
        self._source_cache_key = None
        self.source_barycentric_weights = np.empty(0, dtype=np.float64)
        self.source_fixed_remap_matrix = np.empty((0, 0), dtype=np.float64)
        self.materialized_heat_input = np.empty(nr, dtype=np.float64)
        self.materialized_current_input = np.empty(nr, dtype=np.float64)
        self.source_psin_query = np.empty(nr, dtype=np.float64)
        self.source_parameter_query = np.empty(nr, dtype=np.float64)
        self.source_projection_query = np.empty(nr, dtype=np.float64)
        self.source_heat_projection_fit_matrix = np.empty((0, 0), dtype=np.float64)
        self.source_current_projection_fit_matrix = np.empty((0, 0), dtype=np.float64)
        self.source_heat_projection_coeff = np.empty(0, dtype=np.float64)
        self.source_current_projection_coeff = np.empty(0, dtype=np.float64)
        self.source_endpoint_blend = np.linspace(0.0, 1.0, nr, dtype=np.float64)
        self.source_target_root_fields = np.empty((3, nr), dtype=np.float64)
        self.alpha1 = 0.0
        self.alpha2 = 0.0
        self.packed_residual = np.empty(self.x_size, dtype=np.float64)
        self.active_u_fields = np.empty((n_active, 3, nr), dtype=np.float64)
        self.active_rp_fields = np.empty((n_active, 3, nr), dtype=np.float64)
        self.active_env_fields = np.empty((n_active, 3, nr), dtype=np.float64)
        self.active_offsets = np.empty(n_active, dtype=np.float64)
        self.active_scales = np.empty(n_active, dtype=np.float64)
        self.active_lengths = np.empty(n_active, dtype=np.int64)
        self.active_coeff_index_rows = np.full((n_active, max_active_len), -1, dtype=np.int64)
        self.c_family_fields = np.empty((self.grid.K_max + 1, 3, nr), dtype=np.float64)
        self.s_family_fields = np.zeros((self.grid.K_max + 1, 3, nr), dtype=np.float64)
        self.c_family_base_fields = np.zeros((self.grid.K_max + 1, 3, nr), dtype=np.float64)
        self.s_family_base_fields = np.zeros((self.grid.K_max + 1, 3, nr), dtype=np.float64)
        self.active_slot_by_profile_id = np.full(len(self.profile_names), -1, dtype=np.int64)
        for slot, p in enumerate(self.active_profile_ids):
            self.active_slot_by_profile_id[int(p)] = int(slot)
        self.c_family_source_slots = np.full(self.grid.K_max + 1, -1, dtype=np.int64)
        self.s_family_source_slots = np.full(self.grid.K_max + 1, -1, dtype=np.int64)
        for order in range(self.grid.K_max + 1):
            c_name = f"c{order}"
            if c_name in self.profile_index:
                self.c_family_source_slots[order] = self.active_slot_by_profile_id[self.profile_index[c_name]]
            if order == 0:
                continue
            s_name = f"s{order}"
            if s_name in self.profile_index:
                self.s_family_source_slots[order] = self.active_slot_by_profile_id[self.profile_index[s_name]]

    def _refresh_runtime_state(self) -> None:
        self._refresh_operator_identity()
        self._validate_runtime_profile_support()
        self._refresh_profile_runtime()
        self._refresh_fourier_family_metadata()
        self._refresh_source_runtime()
        self._refresh_stage_a_runtime()
        self._refresh_runtime_bindings()

    def _refresh_operator_identity(self) -> None:
        spec = validate_operator(self.case.name, self.case.coordinate, self.case.nodes)
        self._source_route_spec = spec
        self._source_runner = _build_source_stage_runner(spec)
        self._source_projection_policy = _SOURCE_PROJECTION_POLICIES.get(
            (self.case.name, self.case.coordinate, self.case.nodes)
        )

    def _refresh_profile_runtime(self) -> None:
        for name in self.profile_names:
            profile = self._profile_by_name(name)
            profile.offset = self._profile_offset_from_case(name)
            profile.scale = self._profile_scale_from_case(name)
            profile.coeff = self._profile_coeff_from_case(self.profile_index[name])
            profile._prepare_runtime_cache(self.grid)
            profile.update()
        self._refresh_fourier_family_base_fields()

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
        validate_operator(case.name, case.coordinate, case.nodes)
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
        self._validate_source_inputs(case)

    def _refresh_runtime_bindings(self) -> None:
        self.profile_stage_runner = self._build_profile_stage_runner()
        self.geometry_stage_runner = self._build_geometry_stage_runner()
        self.source_stage_runner = self._build_bound_source_stage_runner()
        self.residual_pack_runner = self._build_residual_pack_runner()
        self.residual_stage_runner = self._build_residual_stage_runner()
        self.residual_pack_stage_runner = self._build_bound_residual_pack_stage_runner()
        self.residual_full_stage_runner = self._build_bound_residual_full_stage_runner()
        fixed_profile_ids = np.flatnonzero(~self.active_profile_mask).astype(np.int64, copy=False)
        for p in fixed_profile_ids:
            self._profile_by_name(self.profile_names[int(p)]).update()

    def _refresh_source_runtime(self) -> None:
        self._validate_source_inputs(self.case)
        case_key = (self.case.coordinate, self.case.nodes, int(self.case.heat_input.shape[0]))
        if self._source_cache_key != case_key:
            if self.case.nodes == "grid":
                self.source_barycentric_weights = np.empty(0, dtype=np.float64)
                self.source_fixed_remap_matrix = np.empty((0, 0), dtype=np.float64)
                self.source_heat_projection_fit_matrix = np.empty((0, 0), dtype=np.float64)
                self.source_current_projection_fit_matrix = np.empty((0, 0), dtype=np.float64)
                self.source_heat_projection_coeff = np.empty(0, dtype=np.float64)
                self.source_current_projection_coeff = np.empty(0, dtype=np.float64)
            else:
                (
                    _,
                    self.source_barycentric_weights,
                    self.source_fixed_remap_matrix,
                ) = build_source_remap_cache(
                    self.case.coordinate,
                    self.source_n_src,
                    rho=self.grid.rho,
                )
                if self._source_projection_policy is None:
                    self.source_heat_projection_fit_matrix = np.empty((0, 0), dtype=np.float64)
                    self.source_current_projection_fit_matrix = np.empty((0, 0), dtype=np.float64)
                else:
                    self.source_heat_projection_fit_matrix = _build_source_projection_fit_matrix(
                        self.source_n_src,
                        degree=self._source_projection_policy.heat_degree,
                        domain=self._source_projection_policy.domain,
                    )
                    self.source_current_projection_fit_matrix = _build_source_projection_fit_matrix(
                        self.source_n_src,
                        degree=self._source_projection_policy.current_degree,
                        domain=self._source_projection_policy.domain,
                    )
            self._source_cache_key = case_key
        if self.case.nodes == "grid" or self._source_projection_policy is None:
            self.source_heat_projection_coeff = np.empty(0, dtype=np.float64)
            self.source_current_projection_coeff = np.empty(0, dtype=np.float64)
        else:
            self.source_heat_projection_coeff = self.source_heat_projection_fit_matrix @ np.asarray(
                self.case.heat_input,
                dtype=np.float64,
            )
            self.source_current_projection_coeff = self.source_current_projection_fit_matrix @ np.asarray(
                self.case.current_input,
                dtype=np.float64,
            )
        if self.case.nodes == "grid" or self.case.coordinate != "psin":
            self._materialize_source_inputs(self.psin)

    def _validate_source_inputs(self, case: OperatorCase) -> None:
        if case.heat_input.shape != case.current_input.shape:
            raise ValueError(
                f"Expected heat_input/current_input to share a shape, got {case.heat_input.shape} and {case.current_input.shape}"
            )
        if case.nodes == "grid" and case.heat_input.shape[0] != self.grid.Nr:
            raise ValueError(f"Expected grid inputs to have shape ({self.grid.Nr},), got {case.heat_input.shape}")
        if case.heat_input.shape[0] < 1:
            raise ValueError(f"Expected {case.coordinate}-coordinate inputs to contain at least one sample")

    def _materialize_source_inputs(self, psin_query: np.ndarray, *, enable_projection: bool = True) -> None:
        if self.case.nodes == "grid":
            np.copyto(self.materialized_heat_input, self.case.heat_input)
            np.copyto(self.materialized_current_input, self.case.current_input)
            return
        if enable_projection and self.case.coordinate == "psin" and self._source_projection_policy is not None:
            self._materialize_projected_psin_source_inputs(psin_query)
            return
        query = psin_query
        if self.case.coordinate == "psin":
            _parameterize_psin_query_inplace(self.source_parameter_query, psin_query, self.source_parameterization)
            query = self.source_parameter_query
        resolve_source_inputs(
            self.materialized_heat_input,
            self.materialized_current_input,
            self.case.heat_input,
            self.case.current_input,
            route_coordinate_code(self.case.coordinate),
            self.source_n_src,
            self.source_barycentric_weights,
            self.source_fixed_remap_matrix,
            query,
        )

    def _materialize_projected_psin_source_inputs(self, psin_query: np.ndarray) -> None:
        policy = self._source_projection_policy
        if policy is None:
            raise RuntimeError("Projected psin source materialization requested without a projection policy")

        current_endpoint_policy = (
            policy.current_ip_endpoint_policy if np.isfinite(self.case.Ip) else policy.current_other_endpoint_policy
        )
        materialize_projected_source_inputs(
            self.materialized_heat_input,
            self.materialized_current_input,
            self.source_heat_projection_coeff,
            self.source_current_projection_coeff,
            self.case.current_input,
            psin_query,
            _PROJECTION_DOMAIN_CODES[policy.domain],
            _ENDPOINT_POLICY_CODES[current_endpoint_policy],
            self.source_endpoint_blend,
        )

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

    def _build_profile_stage_runner(self) -> Callable:
        if self.active_profile_ids.size == 0:
            return lambda x: None

        active_u_fields = self.active_u_fields
        T_fields = self.grid.T_fields
        active_rp_fields = self.active_rp_fields
        active_env_fields = self.active_env_fields
        active_offsets = self.active_offsets
        active_scales = self.active_scales
        active_coeff_index_rows = self.active_coeff_index_rows
        active_lengths = self.active_lengths

        def runner(x: np.ndarray) -> None:
            update_profiles_packed_bulk(
                active_u_fields,
                T_fields,
                active_rp_fields,
                active_env_fields,
                active_offsets,
                active_scales,
                x,
                active_coeff_index_rows,
                active_lengths,
            )

        return runner

    def _build_geometry_stage_runner(self) -> Callable:
        grid = self.grid
        c_family_fields = self.c_family_fields
        s_family_fields = self.s_family_fields
        c_family_base_fields = self.c_family_base_fields
        s_family_base_fields = self.s_family_base_fields
        active_u_fields = self.active_u_fields
        c_family_source_slots = self.c_family_source_slots
        s_family_source_slots = self.s_family_source_slots
        c_effective_order = int(self.c_effective_order)
        s_effective_order = int(self.s_effective_order)
        h_fields = self.h_profile.u_fields
        v_fields = self.v_profile.u_fields
        k_fields = self.k_profile.u_fields
        case_a = float(self.case.a)
        case_R0 = float(self.case.R0)
        case_Z0 = float(self.case.Z0)
        geometry = self.geometry
        tb_fields = geometry.tb_fields
        R_fields = geometry.R_fields
        Z_fields = geometry.Z_fields
        J_fields = geometry.J_fields
        g_fields = geometry.g_fields
        S_r = geometry.S_r
        V_r = geometry.V_r
        Kn = geometry.Kn
        Kn_r = geometry.Kn_r
        Ln_r = geometry.Ln_r
        rho = grid.rho
        theta = grid.theta
        cos_ktheta = grid.cos_ktheta
        sin_ktheta = grid.sin_ktheta
        k_cos_ktheta = grid.k_cos_ktheta
        k_sin_ktheta = grid.k_sin_ktheta
        k2_cos_ktheta = grid.k2_cos_ktheta
        k2_sin_ktheta = grid.k2_sin_ktheta
        weights = grid.weights

        def runner() -> None:
            update_fourier_family_fields(
                c_family_fields,
                s_family_fields,
                c_family_base_fields,
                s_family_base_fields,
                active_u_fields,
                c_family_source_slots,
                s_family_source_slots,
                c_effective_order,
                s_effective_order,
            )
            update_geometry(
                tb_fields,
                R_fields,
                Z_fields,
                J_fields,
                g_fields,
                S_r,
                V_r,
                Kn,
                Kn_r,
                Ln_r,
                case_a,
                case_R0,
                case_Z0,
                rho,
                theta,
                cos_ktheta,
                sin_ktheta,
                k_cos_ktheta,
                k_sin_ktheta,
                k2_cos_ktheta,
                k2_sin_ktheta,
                weights,
                h_fields,
                v_fields,
                k_fields,
                c_family_fields,
                s_family_fields,
                c_effective_order,
                s_effective_order,
            )

        return runner

    def _build_residual_pack_runner(self) -> Callable:
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

    def _build_residual_stage_runner(self) -> Callable:
        profile_names = tuple(self.profile_names[int(p)] for p in self.active_profile_ids)
        try:
            return bind_residual_stage_runner(
                profile_names,
                self.active_coeff_index_rows,
                self.active_lengths,
                self.x_size,
            )
        except KeyError as exc:
            raise ValueError(f"Unsupported active residual block set {profile_names!r}") from exc

    def _build_bound_source_stage_runner(self) -> Callable:
        psin = self.psin
        psin_r = self.psin_r
        psin_rr = self.psin_rr
        FFn_psin = self.FFn_psin
        Pn_psin = self.Pn_psin
        materialized_heat_input = self.materialized_heat_input
        materialized_current_input = self.materialized_current_input
        source_target_root_fields = self.source_target_root_fields
        grid = self.grid
        geometry = self.geometry
        F_profile_u = self.F_profile.u
        case_R0 = float(self.case.R0)
        case_B0 = float(self.case.B0)
        case_Ip = float(self.case.Ip)
        case_beta = float(self.case.beta)

        if self.source_strategy == "profile_owned_psin":
            if self.case.coordinate == "psin" and self.case.nodes != "grid":
                source_psin_query = self.source_psin_query
                source_parameter_query = self.source_parameter_query
                psin_profile_fields = self.psin_profile.u_fields
                heat_input = self.case.heat_input
                current_input = self.case.current_input
                parameterization_code = _SOURCE_PARAMETERIZATION_CODES[self.source_parameterization]

                def runner() -> tuple[float, float]:
                    if psin_profile_fields is None:
                        raise RuntimeError("psin_profile runtime fields are not initialized")
                    materialize_profile_owned_psin_source(
                        psin,
                        psin_r,
                        psin_rr,
                        source_psin_query,
                        source_parameter_query,
                        materialized_heat_input,
                        materialized_current_input,
                        psin_profile_fields,
                        heat_input,
                        current_input,
                        parameterization_code,
                    )
                    return self._source_runner(
                        source_target_root_fields[0],
                        source_target_root_fields[1],
                        source_target_root_fields[2],
                        FFn_psin,
                        Pn_psin,
                        materialized_heat_input,
                        materialized_current_input,
                        case_R0,
                        case_B0,
                        grid.weights,
                        grid.differentiation_matrix,
                        grid.integration_matrix,
                        grid.rho,
                        geometry.V_r,
                        geometry.Kn,
                        geometry.Kn_r,
                        geometry.Ln_r,
                        geometry.S_r,
                        geometry.R,
                        geometry.JdivR,
                        F_profile_u,
                        case_Ip,
                        case_beta,
                    )

                return runner

            source_psin_query = self.source_psin_query

            def runner() -> tuple[float, float]:
                self._copy_psin_profile_to_root_fields()
                np.copyto(source_psin_query, psin)
                self._materialize_source_inputs(source_psin_query)
                return self._source_runner(
                    source_target_root_fields[0],
                    source_target_root_fields[1],
                    source_target_root_fields[2],
                    FFn_psin,
                    Pn_psin,
                    materialized_heat_input,
                    materialized_current_input,
                    case_R0,
                    case_B0,
                    grid.weights,
                    grid.differentiation_matrix,
                    grid.integration_matrix,
                    grid.rho,
                    geometry.V_r,
                    geometry.Kn,
                    geometry.Kn_r,
                    geometry.Ln_r,
                    geometry.S_r,
                    geometry.R,
                    geometry.JdivR,
                    F_profile_u,
                    case_Ip,
                    case_beta,
                )

            return runner

        if self.source_strategy == "fixed_point_psin":
            source_psin_query = self.source_psin_query
            psin_profile_u = self.psin_profile.u

            def runner() -> tuple[float, float]:
                _normalize_psin_query_inplace(source_psin_query, psin_profile_u)
                return self._run_psin_source_fixed_point()

            return runner

        def runner() -> tuple[float, float]:
            return self._source_runner(
                psin,
                psin_r,
                psin_rr,
                FFn_psin,
                Pn_psin,
                materialized_heat_input,
                materialized_current_input,
                case_R0,
                case_B0,
                grid.weights,
                grid.differentiation_matrix,
                grid.integration_matrix,
                grid.rho,
                geometry.V_r,
                geometry.Kn,
                geometry.Kn_r,
                geometry.Ln_r,
                geometry.S_r,
                geometry.R,
                geometry.JdivR,
                F_profile_u,
                case_Ip,
                case_beta,
            )

        return runner

    def _build_bound_residual_pack_stage_runner(self) -> Callable:
        G = self.G
        psin_R = self.psin_R
        psin_Z = self.psin_Z
        sin_tb = self.geometry.sin_tb
        sin_ktheta = self.grid.sin_ktheta
        cos_ktheta = self.grid.cos_ktheta
        rho_powers = self.grid.rho_powers
        y = self.grid.y
        T = self.grid.T_fields[0]
        weights = self.grid.weights
        case_a = float(self.case.a)
        case_R0 = float(self.case.R0)
        case_B0 = float(self.case.B0)

        def runner() -> np.ndarray:
            return self.residual_pack_runner(
                G,
                psin_R,
                psin_Z,
                sin_tb,
                sin_ktheta,
                cos_ktheta,
                rho_powers,
                y,
                T,
                weights,
                case_a,
                case_R0,
                case_B0,
            )

        return runner

    def _build_bound_residual_full_stage_runner(self) -> Callable:
        packed_residual = self.packed_residual
        residual_fields = self.residual_fields
        root_fields = self.root_fields
        geometry = self.geometry
        grid = self.grid
        case_a = float(self.case.a)
        case_R0 = float(self.case.R0)
        case_B0 = float(self.case.B0)

        def runner() -> np.ndarray:
            return self.residual_stage_runner(
                packed_residual,
                residual_fields,
                self.alpha1,
                self.alpha2,
                root_fields,
                geometry.R_fields,
                geometry.Z_fields,
                geometry.J_fields,
                geometry.g_fields,
                geometry.sin_tb,
                grid.sin_ktheta,
                grid.cos_ktheta,
                grid.rho_powers,
                grid.y,
                grid.T_fields[0],
                grid.weights,
                case_a,
                case_R0,
                case_B0,
            )

        return runner

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
        update_fourier_family_fields(
            self.c_family_fields,
            self.s_family_fields,
            self.c_family_base_fields,
            self.s_family_base_fields,
            self.active_u_fields,
            self.c_family_source_slots,
            self.s_family_source_slots,
            self.c_effective_order,
            self.s_effective_order,
        )

    def _refresh_fourier_family_base_fields(self) -> None:
        self.c_family_base_fields.fill(0.0)
        self.s_family_base_fields.fill(0.0)
        for order in range(self.grid.K_max + 1):
            c_name = f"c{order}"
            if c_name in self.profile_index:
                np.copyto(self.c_family_base_fields[order], self._profile_by_name(c_name).u_fields)
            if order == 0:
                continue
            s_name = f"s{order}"
            if s_name in self.profile_index:
                np.copyto(self.s_family_base_fields[order], self._profile_by_name(s_name).u_fields)

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

    def _copy_psin_profile_to_root_fields(self) -> None:
        if self.psin_profile.u_fields is None:
            raise RuntimeError("psin_profile runtime fields are not initialized")
        np.copyto(self.psin, self.psin_profile.u)
        np.copyto(self.psin_r, self.psin_profile.u_r)
        np.copyto(self.psin_rr, self.psin_profile.u_rr)

    def _run_profile_owned_psin_source(self) -> tuple[float, float]:
        if self.case.coordinate == "psin" and self.case.nodes != "grid":
            if self.psin_profile.u_fields is None:
                raise RuntimeError("psin_profile runtime fields are not initialized")
            materialize_profile_owned_psin_source(
                self.psin,
                self.psin_r,
                self.psin_rr,
                self.source_psin_query,
                self.source_parameter_query,
                self.materialized_heat_input,
                self.materialized_current_input,
                self.psin_profile.u_fields,
                self.case.heat_input,
                self.case.current_input,
                _SOURCE_PARAMETERIZATION_CODES[self.source_parameterization],
            )
        else:
            self._copy_psin_profile_to_root_fields()
            np.copyto(self.source_psin_query, self.psin)
            self._materialize_source_inputs(self.source_psin_query)
        return self._source_runner(
            self.source_target_root_fields[0],
            self.source_target_root_fields[1],
            self.source_target_root_fields[2],
            self.FFn_psin,
            self.Pn_psin,
            self.materialized_heat_input,
            self.materialized_current_input,
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

    def _run_psin_source_fixed_point(self) -> tuple[float, float]:
        alpha1 = float("nan")
        alpha2 = float("nan")
        use_projected_finalize = self.case.coordinate == "psin" and self._source_projection_policy is not None
        for _ in range(8):
            self._materialize_source_inputs(self.source_psin_query, enable_projection=not use_projected_finalize)
            alpha1, alpha2 = self._source_runner(
                self.psin,
                self.psin_r,
                self.psin_rr,
                self.FFn_psin,
                self.Pn_psin,
                self.materialized_heat_input,
                self.materialized_current_input,
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
            if update_fixed_point_psin_query(self.source_psin_query, self.psin, 1e-10):
                break
        if use_projected_finalize:
            np.copyto(self.source_psin_query, self.psin)
            self._materialize_source_inputs(self.source_psin_query, enable_projection=True)
            alpha1, alpha2 = self._source_runner(
                self.psin,
                self.psin_r,
                self.psin_rr,
                self.FFn_psin,
                self.Pn_psin,
                self.materialized_heat_input,
                self.materialized_current_input,
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
        return alpha1, alpha2

    def _assemble_residual(self) -> np.ndarray:
        return self.residual_pack_stage_runner()

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
            psin=self.psin.copy(),
            FFn_psin=self.FFn_psin.copy(),
            Pn_psin=self.Pn_psin.copy(),
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


class Operator(_OperatorCore):
    """默认 fused residual 算子。"""

    def __post_init__(self) -> None:
        self.fused_alpha_state = np.zeros(2, dtype=np.float64)
        self.fused_residual_runner = lambda x_eval: _OperatorCore._evaluate_residual(self, x_eval)
        self.supports_fused_residual = False
        super().__post_init__()

    def _refresh_runtime_bindings(self) -> None:
        super()._refresh_runtime_bindings()
        self.fused_residual_runner = self._build_fused_residual_runner()

    def _build_fused_common_kwargs(self) -> dict[str, object]:
        profile_names = tuple(self.profile_names[int(p)] for p in self.active_profile_ids)
        return dict(
            source_kernel=self._source_route_spec.implementation,
            coordinate_code=int(self._source_route_spec.coordinate_code),
            profile_names=profile_names,
            coeff_index_rows=self.active_coeff_index_rows,
            lengths=self.active_lengths,
            residual_size=self.x_size,
            alpha_state=self.fused_alpha_state,
            active_u_fields=self.active_u_fields,
            T_fields=self.grid.T_fields,
            active_rp_fields=self.active_rp_fields,
            active_env_fields=self.active_env_fields,
            active_offsets=self.active_offsets,
            active_scales=self.active_scales,
            c_family_fields=self.c_family_fields,
            s_family_fields=self.s_family_fields,
            c_family_base_fields=self.c_family_base_fields,
            s_family_base_fields=self.s_family_base_fields,
            c_source_slots=self.c_family_source_slots,
            s_source_slots=self.s_family_source_slots,
            c_active_order=int(self.c_effective_order),
            s_active_order=int(self.s_effective_order),
            h_fields=self.h_profile.u_fields,
            v_fields=self.v_profile.u_fields,
            k_fields=self.k_profile.u_fields,
            tb_fields=self.geometry.tb_fields,
            R_fields=self.geometry.R_fields,
            Z_fields=self.geometry.Z_fields,
            J_fields=self.geometry.J_fields,
            g_fields=self.geometry.g_fields,
            S_r=self.geometry.S_r,
            V_r=self.geometry.V_r,
            Kn=self.geometry.Kn,
            Kn_r=self.geometry.Kn_r,
            Ln_r=self.geometry.Ln_r,
            a=float(self.case.a),
            R0=float(self.case.R0),
            Z0=float(self.case.Z0),
            B0=float(self.case.B0),
            rho=self.grid.rho,
            theta=self.grid.theta,
            cos_ktheta=self.grid.cos_ktheta,
            sin_ktheta=self.grid.sin_ktheta,
            k_cos_ktheta=self.grid.k_cos_ktheta,
            k_sin_ktheta=self.grid.k_sin_ktheta,
            k2_cos_ktheta=self.grid.k2_cos_ktheta,
            k2_sin_ktheta=self.grid.k2_sin_ktheta,
            weights=self.grid.weights,
            differentiation_matrix=self.grid.differentiation_matrix,
            integration_matrix=self.grid.integration_matrix,
            rho_powers=self.grid.rho_powers,
            y=self.grid.y,
            root_fields=self.root_fields,
            residual_fields=self.residual_fields,
            packed_residual=self.packed_residual,
            materialized_heat_input=self.materialized_heat_input,
            materialized_current_input=self.materialized_current_input,
            F_profile_u=self.F_profile.u,
            Ip=float(self.case.Ip),
            beta=float(self.case.beta),
        )

    def _build_single_pass_fused_runner(self) -> Callable[[np.ndarray], np.ndarray]:
        self.supports_fused_residual = True
        return bind_fused_single_pass_residual_runner(**self._build_fused_common_kwargs())

    def _build_profile_owned_psin_fused_runner(self) -> Callable[[np.ndarray], np.ndarray]:
        psin_profile_fields = self.psin_profile.u_fields
        if psin_profile_fields is None:
            self.supports_fused_residual = False
            return lambda x_eval: _OperatorCore._evaluate_residual(self, x_eval)
        self.supports_fused_residual = True
        return bind_fused_profile_owned_psin_residual_runner(
            **self._build_fused_common_kwargs(),
            parameterization_code=_SOURCE_PARAMETERIZATION_CODES[self.source_parameterization],
            source_target_root_fields=self.source_target_root_fields,
            source_psin_query=self.source_psin_query,
            source_parameter_query=self.source_parameter_query,
            psin_profile_fields=psin_profile_fields,
            heat_input=self.case.heat_input,
            current_input=self.case.current_input,
        )

    def _build_fixed_point_psin_fused_runner(self) -> Callable[[np.ndarray], np.ndarray]:
        policy = self._source_projection_policy
        if policy is None:
            use_projected_finalize = False
            projection_domain_code = 0
            endpoint_policy_code = 0
        else:
            endpoint_policy = (
                policy.current_ip_endpoint_policy
                if np.isfinite(self.case.Ip)
                else policy.current_other_endpoint_policy
            )
            use_projected_finalize = self.case.coordinate == "psin"
            projection_domain_code = _PROJECTION_DOMAIN_CODES[policy.domain]
            endpoint_policy_code = _ENDPOINT_POLICY_CODES[endpoint_policy]
        self.supports_fused_residual = True
        return bind_fused_fixed_point_psin_residual_runner(
            **self._build_fused_common_kwargs(),
            source_psin_query=self.source_psin_query,
            psin_seed=self.psin_profile.u,
            heat_input=self.case.heat_input,
            current_input=self.case.current_input,
            source_n_src=self.source_n_src,
            source_barycentric_weights=self.source_barycentric_weights,
            source_fixed_remap_matrix=self.source_fixed_remap_matrix,
            use_projected_finalize=use_projected_finalize,
            heat_projection_coeff=self.source_heat_projection_coeff,
            current_projection_coeff=self.source_current_projection_coeff,
            endpoint_blend=self.source_endpoint_blend,
            projection_domain_code=projection_domain_code,
            endpoint_policy_code=endpoint_policy_code,
        )

    def _build_fused_residual_runner(self) -> Callable[[np.ndarray], np.ndarray]:
        if self.source_strategy == "single_pass":
            return self._build_single_pass_fused_runner()
        if self.source_strategy == "profile_owned_psin":
            return self._build_profile_owned_psin_fused_runner()
        if self.source_strategy == "fixed_point_psin":
            return self._build_fixed_point_psin_fused_runner()
        self.supports_fused_residual = False
        return lambda x_eval: _OperatorCore._evaluate_residual(self, x_eval)

    def residual(
        self,
        x: np.ndarray,
    ) -> np.ndarray:
        x_eval = self.coerce_x(x)
        residual = self.fused_residual_runner(x_eval)
        self.alpha1 = float(self.fused_alpha_state[0])
        self.alpha2 = float(self.fused_alpha_state[1])
        return residual

    def residual_fast(
        self,
        x: np.ndarray,
    ) -> np.ndarray:
        if isinstance(x, np.ndarray) and x.dtype == np.float64 and x.ndim == 1 and x.shape[0] == self.x_size:
            residual = self.fused_residual_runner(x)
            self.alpha1 = float(self.fused_alpha_state[0])
            self.alpha2 = float(self.fused_alpha_state[1])
            return residual
        return self.residual(x)


def _normalize_psin_query_inplace(out: np.ndarray, source: np.ndarray | None) -> np.ndarray:
    if source is None:
        raise RuntimeError("psin_profile.u is not initialized")

    np.copyto(out, np.asarray(source, dtype=np.float64))
    if out.ndim != 1 or out.size < 2:
        raise ValueError(f"Expected psin query to be 1D with at least two points, got {out.shape}")

    offset = float(out[0])
    scale = float(out[-1] - offset)
    if abs(scale) < 1e-12:
        raise ValueError("psin query does not span a valid normalized flux interval")

    out -= offset
    out /= scale
    out[0] = 0.0
    out[-1] = 1.0
    return out


def _parameterize_psin_query_inplace(out: np.ndarray, source: np.ndarray, parameterization: str) -> np.ndarray:
    np.copyto(out, np.asarray(source, dtype=np.float64))
    if parameterization == "identity":
        return out
    if parameterization == "sqrt_psin":
        np.maximum(out, 0.0, out=out)
        np.sqrt(out, out=out)
        return out
    raise ValueError(f"Unsupported source parameterization {parameterization!r}")


def _parameterize_projection_query_inplace(out: np.ndarray, source: np.ndarray, domain: str) -> np.ndarray:
    np.copyto(out, np.asarray(source, dtype=np.float64))
    np.clip(out, 0.0, 1.0, out=out)
    if domain == "psin":
        out *= 2.0
        out -= 1.0
        return out
    if domain == "sqrt_psin":
        np.sqrt(out, out=out)
        out *= 2.0
        out -= 1.0
        return out
    raise ValueError(f"Unsupported source projection domain {domain!r}")


def _build_source_projection_fit_matrix(n_src: int, *, degree: int, domain: str) -> np.ndarray:
    if degree < 0:
        raise ValueError(f"Projection degree must be non-negative, got {degree}")
    source_axis = np.linspace(0.0, 1.0, int(n_src), dtype=np.float64)
    source_query = np.empty_like(source_axis)
    _parameterize_projection_query_inplace(source_query, source_axis, domain)
    vandermonde = chebvander(source_query, degree)
    return np.linalg.pinv(vandermonde)


def _evaluate_source_projection_inplace(
    out: np.ndarray,
    source_values: np.ndarray,
    fit_matrix: np.ndarray,
    projected_query: np.ndarray,
) -> np.ndarray:
    coeff = fit_matrix @ np.asarray(source_values, dtype=np.float64)
    out[:] = chebval(projected_query, coeff)
    return out


def _evaluate_projection_coeff_inplace(
    out: np.ndarray,
    coeff: np.ndarray,
    projected_query: np.ndarray,
) -> np.ndarray:
    out[:] = chebval(projected_query, coeff)
    return out


def _apply_endpoint_policy_inplace(
    out: np.ndarray,
    source_values: np.ndarray,
    *,
    policy: str,
    blend: np.ndarray,
) -> np.ndarray:
    if policy == "none":
        return out
    if policy == "right":
        out[-1] = float(source_values[-1])
        return out
    if policy == "both":
        out[0] = float(source_values[0])
        out[-1] = float(source_values[-1])
        return out
    if policy == "affine_both":
        delta_left = float(source_values[0]) - float(out[0])
        delta_right = float(source_values[-1]) - float(out[-1])
        out += (1.0 - blend) * delta_left + blend * delta_right
        return out
    raise ValueError(f"Unsupported endpoint policy {policy!r}")


def _build_source_stage_runner(route_spec) -> Callable:
    coordinate_code = int(route_spec.coordinate_code)
    operator_kernel = route_spec.implementation

    def runner(
        out_psin: np.ndarray,
        out_psin_r: np.ndarray,
        out_psin_rr: np.ndarray,
        out_FFn_psin: np.ndarray,
        out_Pn_psin: np.ndarray,
        heat_input: np.ndarray,
        current_input: np.ndarray,
        R0: float,
        B0: float,
        weights: np.ndarray,
        differentiation_matrix: np.ndarray,
        integration_matrix: np.ndarray,
        rho: np.ndarray,
        V_r: np.ndarray,
        Kn: np.ndarray,
        Kn_r: np.ndarray,
        Ln_r: np.ndarray,
        S_r: np.ndarray,
        R: np.ndarray,
        JdivR: np.ndarray,
        F: np.ndarray,
        Ip: float,
        beta: float,
    ) -> tuple[float, float]:
        return operator_kernel(
            out_psin,
            out_psin_r,
            out_psin_rr,
            out_FFn_psin,
            out_Pn_psin,
            heat_input,
            current_input,
            coordinate_code,
            R0,
            B0,
            weights,
            differentiation_matrix,
            integration_matrix,
            rho,
            V_r,
            Kn,
            Kn_r,
            Ln_r,
            S_r,
            R,
            JdivR,
            F,
            Ip,
            beta,
        )

    return runner


def route_coordinate_code(coordinate: str) -> int:
    return 1 if coordinate == "psin" else 0

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
from typing import Callable

import numpy as np

from veqpy.engine import operator_ops, orchestration, residual_ops, validate_route
from veqpy.model.equilibrium import Equilibrium
from veqpy.model.grid import Grid
from veqpy.model.profile import Profile
from veqpy.operator.codec import decode_packed_blocks, encode_packed_state
from veqpy.operator.execution_helpers import snapshot_equilibrium_from_runtime
from veqpy.operator.layout import (
    build_active_profile_metadata,
    build_fourier_profile_names,
    build_profile_index,
    build_profile_layout,
    build_profile_names,
    build_shape_profile_names,
    get_prefix_profile_names,
    packed_size,
)
from veqpy.operator.layouts import (
    BackendState,
    ExecutionState,
    FieldRuntimeState,
    ResidualBindingLayout,
    RuntimeLayout,
    SourceRuntimeState,
    StaticLayout,
)
from veqpy.operator.operator_case import OperatorCase
from veqpy.operator.profile_setup import (
    build_profile_stage_runner,
    make_profile,
    refresh_fourier_family_base_fields,
    refresh_fourier_family_metadata,
    refresh_profile_runtime,
    refresh_stage_a_runtime,
    validate_case_compatibility,
)
from veqpy.operator.runtime_allocation import allocate_runtime_state

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
_PROFILE_SCALE_SPECS: dict[str, tuple[str, ...]] = {"F": ("R0", "B0", "R0", "B0")}


@dataclass(slots=True)
class Operator:
    """封装固定 case, grid 与 runtime 的 residual 求值器."""

    grid: Grid = field(repr=False)
    case: OperatorCase = field(repr=False)
    static_layout: StaticLayout = field(init=False, repr=False)
    residual_binding_layout: ResidualBindingLayout = field(init=False, repr=False)
    runtime_layout: RuntimeLayout = field(init=False, repr=False)
    backend_state: BackendState = field(init=False, repr=False)

    h_profile: Profile = field(init=False)
    v_profile: Profile = field(init=False)
    k_profile: Profile = field(init=False)
    psin_profile: Profile = field(init=False)
    F_profile: Profile = field(init=False)
    profiles_by_name: dict[str, Profile] = field(init=False, repr=False)

    psin: np.ndarray = field(init=False)
    psin_r: np.ndarray = field(init=False)
    psin_rr: np.ndarray = field(init=False)
    FFn_psin: np.ndarray = field(init=False)
    Pn_psin: np.ndarray = field(init=False)
    root_fields: np.ndarray = field(init=False, repr=False)

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
    active_profile_slab: np.ndarray = field(init=False, repr=False)
    active_offsets: np.ndarray = field(init=False, repr=False)
    active_scales: np.ndarray = field(init=False, repr=False)
    active_lengths: np.ndarray = field(init=False, repr=False)
    active_coeff_index_rows: np.ndarray = field(init=False, repr=False)
    c_family_fields: np.ndarray = field(init=False, repr=False)
    s_family_fields: np.ndarray = field(init=False, repr=False)
    c_family_base_fields: np.ndarray = field(init=False, repr=False)
    s_family_base_fields: np.ndarray = field(init=False, repr=False)
    family_field_slab: np.ndarray = field(init=False, repr=False)
    active_slot_by_profile_id: np.ndarray = field(init=False, repr=False)
    c_family_source_slots: np.ndarray = field(init=False, repr=False)
    s_family_source_slots: np.ndarray = field(init=False, repr=False)
    geometry_surface_workspace: np.ndarray = field(init=False, repr=False)
    geometry_radial_workspace: np.ndarray = field(init=False, repr=False)
    residual_surface_workspace: np.ndarray = field(init=False, repr=False)
    c_effective_order: int = field(init=False, repr=False)
    s_effective_order: int = field(init=False, repr=False)
    _source_route_spec: object = field(init=False, repr=False)
    source_plan: orchestration.SourcePlan = field(init=False, repr=False)
    field_runtime_state: FieldRuntimeState = field(init=False, repr=False)
    execution_state: ExecutionState = field(init=False, repr=False)
    source_runtime_state: SourceRuntimeState = field(init=False, repr=False)
    packed_residual: np.ndarray = field(init=False, repr=False)
    profile_static_kwargs_by_name: dict[str, dict[str, int]] = field(init=False, repr=False)
    profile_offset_specs: dict[str, float | str] = field(init=False, repr=False)
    profile_scale_specs: dict[str, tuple[str, ...]] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        """完成 layout 构造, 运行时缓冲区分配和 case 绑定."""
        self._refresh_operator_identity()
        self.prefix_profile_names = get_prefix_profile_names()
        self.shape_profile_names = build_shape_profile_names(self.grid.M_max)
        self.profile_names = build_profile_names(self.grid.M_max)
        self.profile_index = build_profile_index(self.profile_names)
        fourier_profile_names = build_fourier_profile_names(self.grid.M_max)
        self.c_profile_names = tuple(name for name in fourier_profile_names if name.startswith("c"))
        self.s_profile_names = tuple(name for name in fourier_profile_names if name.startswith("s"))
        self.static_layout = self._build_static_layout()

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
        self._refresh_profile_config()
        self.residual_binding_layout = self._build_residual_binding_layout()
        self._validate_runtime_profile_support()

        self.execution_state = ExecutionState(
            profile_stage_runner=lambda x: None,
            profile_postprocess_runner=lambda: None,
            geometry_stage_runner=lambda: None,
            source_eval_runner=lambda *args: (0.0, 0.0),
            source_stage_runner=lambda: (0.0, 0.0),
            residual_pack_stage_runner=lambda: np.zeros(self.x_size, dtype=np.float64),
            residual_full_stage_runner=lambda: np.zeros(self.x_size, dtype=np.float64),
            fused_residual_runner=lambda x_eval: self._evaluate_residual(x_eval),
            fused_alpha_state=np.zeros(2, dtype=np.float64),
        )
        self._setup_runtime_state()
        self._refresh_runtime_state()

    def __call__(self, x: np.ndarray, *args, **kwargs) -> np.ndarray:
        """调用 residual 求值主入口."""
        return self.residual(x, *args, **kwargs)

    @property
    def alpha1(self) -> float:
        return float(self.execution_state.fused_alpha_state[0])

    @alpha1.setter
    def alpha1(self, value: float) -> None:
        self.execution_state.fused_alpha_state[0] = float(value)

    @property
    def alpha2(self) -> float:
        return float(self.execution_state.fused_alpha_state[1])

    @alpha2.setter
    def alpha2(self, value: float) -> None:
        self.execution_state.fused_alpha_state[1] = float(value)

    def _validate_runtime_profile_support(self) -> None:
        """校验当前 source route 对 psin profile ownership 的要求."""
        orchestration.validate_source_plan_profile_support(
            source_plan=self.source_plan,
            profile_L=self.profile_L,
            profile_index=self.profile_index,
            case=self.case,
        )
        return None

    def replace_case(self, case: OperatorCase) -> None:
        """在不改变 packed layout 的前提下替换当前 case."""
        validate_case_compatibility(
            case,
            profile_names=self.profile_names,
            prefix_profile_names=self.prefix_profile_names,
            profile_L=self.profile_L,
            coeff_index=self.coeff_index,
            order_offsets=self.order_offsets,
            validate_source_inputs=lambda next_case: orchestration.validate_source_inputs(next_case, self.grid.Nr),
        )
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
        return self.execution_state.fused_residual_runner(x_eval)

    def _evaluate_residual(self, x_eval: np.ndarray) -> np.ndarray:
        self.stage_a_profile(x_eval)
        self.stage_b_geometry()
        self.stage_c_source()
        return self.stage_d_residual()

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
        self.execution_state.profile_stage_runner(x)
        self.execution_state.profile_postprocess_runner()

    def stage_b_geometry(self) -> None:
        """执行 geometry 阶段并刷新 geometry fields."""
        self.execution_state.geometry_stage_runner()

    def stage_c_source(self) -> None:
        """执行 source 阶段并刷新 root fields 与缩放系数."""
        alpha1, alpha2 = self.execution_state.source_stage_runner()
        self.alpha1 = float(alpha1)
        self.alpha2 = float(alpha2)

    def stage_d_residual(self) -> np.ndarray:
        """执行 residual 阶段并返回 packed 残差."""
        packed = self.execution_state.residual_full_stage_runner()
        return packed.copy()

    def coerce_x(self, x: np.ndarray) -> np.ndarray:
        """校验完整 packed 状态向量形状."""
        arr = np.asarray(x, dtype=np.float64)
        if arr.ndim != 1 or arr.shape[0] != self.x_size:
            raise ValueError(f"Expected x to have shape ({self.x_size},), got {arr.shape}")
        return arr

    def _build_static_layout(self) -> StaticLayout:
        return StaticLayout(
            Nr=int(self.grid.Nr),
            Nt=int(self.grid.Nt),
            M_max=int(self.grid.M_max),
            T_fields=self.grid.T_fields,
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
        )

    def _build_residual_binding_layout(self) -> ResidualBindingLayout:
        active_profile_names = tuple(self.profile_names[int(p)] for p in self.active_profile_ids)
        active_residual_block_codes, active_residual_block_orders = orchestration.build_residual_block_metadata(
            active_profile_names
        )
        return ResidualBindingLayout(
            active_profile_names=active_profile_names,
            active_residual_block_codes=active_residual_block_codes,
            active_residual_block_orders=active_residual_block_orders,
        )

    def _refresh_runtime_layout_views(self) -> None:
        runtime = self.runtime_layout
        runtime.active_profile_slab = self.active_profile_slab
        runtime.family_field_slab = self.family_field_slab
        runtime.source_runtime_state = self.source_runtime_state
        runtime.root_fields = self.root_fields
        runtime.packed_residual = self.packed_residual
        runtime.active_u_fields = self.active_u_fields
        runtime.active_rp_fields = self.active_rp_fields
        runtime.active_env_fields = self.active_env_fields
        runtime.active_offsets = self.active_offsets
        runtime.active_scales = self.active_scales
        runtime.active_lengths = self.active_lengths
        runtime.active_coeff_index_rows = self.active_coeff_index_rows
        runtime.c_family_fields = self.c_family_fields
        runtime.s_family_fields = self.s_family_fields
        runtime.c_family_base_fields = self.c_family_base_fields
        runtime.s_family_base_fields = self.s_family_base_fields
        runtime.active_slot_by_profile_id = self.active_slot_by_profile_id
        runtime.c_family_source_slots = self.c_family_source_slots
        runtime.s_family_source_slots = self.s_family_source_slots
        runtime.h_fields = self.h_profile.u_fields
        runtime.v_fields = self.v_profile.u_fields
        runtime.k_fields = self.k_profile.u_fields
        runtime.F_profile_u = self.F_profile.u
        runtime.F_profile_fields = self.F_profile.u_fields
        runtime.psin_profile_u = self.psin_profile.u
        runtime.psin_profile_fields = self.psin_profile.u_fields

    def _setup_runtime_state(self) -> None:
        bundle = allocate_runtime_state(
            grid=self.grid,
            static_layout=self.static_layout,
            source_plan=self.source_plan,
            profile_names=self.profile_names,
            profile_index=self.profile_index,
            active_profile_ids=self.active_profile_ids,
            profile_L=self.profile_L,
            x_size=self.x_size,
            make_profile=lambda name: make_profile(
                case=self.case,
                name=name,
                profile_L=self.profile_L,
                profile_names=self.profile_names,
                profile_index=self.profile_index,
                profile_static_kwargs_by_name=self.profile_static_kwargs_by_name,
                profile_offset_specs=self.profile_offset_specs,
                profile_scale_specs=self.profile_scale_specs,
            ),
        )
        self.profiles_by_name = bundle.profiles_by_name
        for name, profile in self.profiles_by_name.items():
            if hasattr(type(self), f"{name}_profile"):
                setattr(self, f"{name}_profile", profile)
        self.field_runtime_state = bundle.field_runtime_state
        self.root_fields = bundle.field_runtime_state.root_fields
        self.packed_residual = bundle.field_runtime_state.packed_residual
        self.psin = bundle.field_runtime_state.psin
        self.psin_r = bundle.field_runtime_state.psin_r
        self.psin_rr = bundle.field_runtime_state.psin_rr
        self.FFn_psin = bundle.field_runtime_state.FFn_psin
        self.Pn_psin = bundle.field_runtime_state.Pn_psin
        self.source_runtime_state = bundle.source_runtime_state
        self.active_profile_slab = bundle.active_profile_slab
        self.active_u_fields = bundle.active_u_fields
        self.active_rp_fields = bundle.active_rp_fields
        self.active_env_fields = bundle.active_env_fields
        self.active_offsets = bundle.active_offsets
        self.active_scales = bundle.active_scales
        self.active_lengths = bundle.active_lengths
        self.active_coeff_index_rows = bundle.active_coeff_index_rows
        self.family_field_slab = bundle.family_field_slab
        self.c_family_fields = bundle.c_family_fields
        self.s_family_fields = bundle.s_family_fields
        self.c_family_base_fields = bundle.c_family_base_fields
        self.s_family_base_fields = bundle.s_family_base_fields
        self.active_slot_by_profile_id = bundle.active_slot_by_profile_id
        self.c_family_source_slots = bundle.c_family_source_slots
        self.s_family_source_slots = bundle.s_family_source_slots
        self.runtime_layout = bundle.runtime_layout
        self.geometry_surface_workspace = bundle.runtime_layout.geometry_surface_workspace
        self.geometry_radial_workspace = bundle.runtime_layout.geometry_radial_workspace
        self.residual_surface_workspace = bundle.runtime_layout.residual_surface_workspace

    def _refresh_runtime_state(self) -> None:
        self._refresh_operator_identity()
        self._refresh_profile_config()
        self.residual_binding_layout = self._build_residual_binding_layout()
        self._validate_runtime_profile_support()
        self._refresh_profile_runtime()
        self._refresh_fourier_family_metadata()
        orchestration.refresh_source_runtime(
            case=self.case,
            grid_rho=self.grid.rho,
            source_plan=self.source_plan,
            source_runtime_state=self.source_runtime_state,
            psin=self.psin,
        )
        self._refresh_stage_a_runtime()
        self._refresh_runtime_layout_views()
        self._refresh_runtime_bindings()
        self._refresh_backend_state()

    def _refresh_operator_identity(self) -> None:
        spec = validate_route(self.case.route, self.case.coordinate, self.case.nodes)
        self._source_route_spec = spec
        self.source_plan = orchestration.build_source_plan(
            case=self.case,
            source_route_spec=self._source_route_spec,
        )

    def _refresh_profile_config(self) -> None:
        self.profile_static_kwargs_by_name = {name: dict(kwargs) for name, kwargs in _PROFILE_STATIC_KWARGS.items()}
        self.profile_offset_specs = dict(_PROFILE_OFFSET_SPECS)
        self.profile_scale_specs = dict(_PROFILE_SCALE_SPECS)

    def _build_profile_postprocess_runner(self) -> Callable[[], None]:
        eps = 1.0e-10

        def runner() -> None:
            operator_ops.apply_f2_profile_fields(self.runtime_layout.F_profile_fields, eps=eps)

        return runner

    def _refresh_profile_runtime(self) -> None:
        refresh_profile_runtime(
            case=self.case,
            grid=self.grid,
            profile_names=self.profile_names,
            profile_index=self.profile_index,
            profile_L=self.profile_L,
            profiles_by_name=self.profiles_by_name,
            profile_static_kwargs_by_name=self.profile_static_kwargs_by_name,
            profile_offset_specs=self.profile_offset_specs,
            profile_scale_specs=self.profile_scale_specs,
            refresh_fourier_family_base_fields=lambda: refresh_fourier_family_base_fields(
                M_max=self.grid.M_max,
                profile_index=self.profile_index,
                profiles_by_name=self.profiles_by_name,
                c_family_base_fields=self.c_family_base_fields,
                s_family_base_fields=self.s_family_base_fields,
            ),
        )

    def _refresh_runtime_bindings(self) -> None:
        self.execution_state.profile_stage_runner = self._build_profile_stage_runner()
        self.execution_state.profile_postprocess_runner = self._build_profile_postprocess_runner()
        self.execution_state.geometry_stage_runner = self._build_geometry_stage_runner()
        self.execution_state.source_eval_runner = self._build_source_eval_runner()
        self.execution_state.source_stage_runner = self._build_bound_source_stage_runner()
        self.execution_state.residual_pack_stage_runner = self._build_bound_residual_pack_stage_runner()
        self.execution_state.residual_full_stage_runner = self._build_bound_residual_full_stage_runner()
        self.execution_state.fused_residual_runner = self._build_fused_residual_runner()
        fixed_profile_ids = np.flatnonzero(~self.active_profile_mask).astype(np.int64, copy=False)
        for p in fixed_profile_ids:
            self.profiles_by_name[self.profile_names[int(p)]].update()
        f_profile_id = self.profile_index.get("F", -1)
        if f_profile_id >= 0 and not bool(self.active_profile_mask[f_profile_id]):
            self.execution_state.profile_postprocess_runner()

    def _refresh_backend_state(self) -> None:
        self.backend_state = BackendState(
            static_layout=self.static_layout,
            residual_binding_layout=self.residual_binding_layout,
            runtime_layout=self.runtime_layout,
            field_runtime_state=self.field_runtime_state,
            source_runtime_state=self.source_runtime_state,
        )

    def _refresh_stage_a_runtime(self) -> None:
        refresh_stage_a_runtime(
            active_profile_ids=self.active_profile_ids,
            profile_names=self.profile_names,
            profiles_by_name=self.profiles_by_name,
            profile_L=self.profile_L,
            coeff_index=self.coeff_index,
            active_u_fields=self.active_u_fields,
            active_rp_fields=self.active_rp_fields,
            active_env_fields=self.active_env_fields,
            active_offsets=self.active_offsets,
            active_scales=self.active_scales,
            active_lengths=self.active_lengths,
            active_coeff_index_rows=self.active_coeff_index_rows,
        )

    def _build_profile_stage_runner(self) -> Callable:
        return build_profile_stage_runner(
            active_profile_ids=self.active_profile_ids,
            active_profile_slab=self.active_profile_slab,
            T_fields=self.grid.T_fields,
            active_offsets=self.active_offsets,
            active_scales=self.active_scales,
            active_coeff_index_rows=self.active_coeff_index_rows,
            active_lengths=self.active_lengths,
        )

    def _build_geometry_stage_runner(self) -> Callable:
        return orchestration.build_geometry_stage_runner(
            c_family_fields=self.c_family_fields,
            s_family_fields=self.s_family_fields,
            c_family_base_fields=self.c_family_base_fields,
            s_family_base_fields=self.s_family_base_fields,
            active_u_fields=self.active_u_fields,
            c_family_source_slots=self.c_family_source_slots,
            s_family_source_slots=self.s_family_source_slots,
            c_effective_order=self.c_effective_order,
            s_effective_order=self.s_effective_order,
            h_fields=self.runtime_layout.h_fields,
            v_fields=self.runtime_layout.v_fields,
            k_fields=self.runtime_layout.k_fields,
            a=self.case.a,
            R0=self.case.R0,
            Z0=self.case.Z0,
            surface_workspace=self.geometry_surface_workspace,
            radial_workspace=self.geometry_radial_workspace,
            rho=self.grid.rho,
            theta=self.grid.theta,
            cos_ktheta=self.grid.cos_ktheta,
            sin_ktheta=self.grid.sin_ktheta,
            k_cos_ktheta=self.grid.k_cos_ktheta,
            k_sin_ktheta=self.grid.k_sin_ktheta,
            k2_cos_ktheta=self.grid.k2_cos_ktheta,
            k2_sin_ktheta=self.grid.k2_sin_ktheta,
        )

    def _build_bound_source_stage_runner(self) -> Callable:
        return orchestration.build_bound_source_stage_runner(self)

    def _build_source_eval_runner(self) -> Callable:
        return operator_ops.bind_source_eval_runner(
            source_plan=self.source_plan,
            static_layout=self.static_layout,
            runtime_layout=self.runtime_layout,
            B0=self.case.B0,
        )

    def _build_bound_residual_pack_stage_runner(self) -> Callable:
        alpha_state = self.execution_state.fused_alpha_state
        root_fields = self.root_fields
        surface_workspace = self.geometry_surface_workspace
        residual_workspace = self.residual_surface_workspace
        sin_ktheta = self.grid.sin_ktheta
        cos_ktheta = self.grid.cos_ktheta
        rho_powers = self.grid.rho_powers
        y = self.grid.y
        T_fields = self.grid.T_fields
        weights = self.grid.weights
        a = self.case.a
        R0 = self.case.R0
        B0 = self.case.B0

        def runner() -> np.ndarray:
            residual_ops.update_residual_compact(
                residual_workspace,
                float(alpha_state[0]),
                float(alpha_state[1]),
                root_fields,
                surface_workspace,
            )
            packed = np.zeros(self.x_size, dtype=np.float64)
            scratch = np.empty(self.grid.Nr, dtype=np.float64)
            residual_ops._run_residual_blocks_packed_precomputed(
                packed,
                scratch,
                self.residual_binding_layout.active_residual_block_codes,
                self.residual_binding_layout.active_residual_block_orders,
                self.active_coeff_index_rows,
                self.active_lengths,
                residual_workspace,
                sin_ktheta,
                cos_ktheta,
                rho_powers,
                y,
                T_fields,
                weights,
                a,
                R0,
                B0,
            )
            return packed

        return runner

    def _build_bound_residual_full_stage_runner(self) -> Callable:
        packed_residual = self.packed_residual
        alpha_state = self.execution_state.fused_alpha_state
        root_fields = self.root_fields
        surface_workspace = self.geometry_surface_workspace
        residual_workspace = self.residual_surface_workspace
        sin_ktheta = self.grid.sin_ktheta
        cos_ktheta = self.grid.cos_ktheta
        rho_powers = self.grid.rho_powers
        y = self.grid.y
        T_fields = self.grid.T_fields
        weights = self.grid.weights
        a = self.case.a
        R0 = self.case.R0
        B0 = self.case.B0

        def runner() -> np.ndarray:
            residual_ops.update_residual_compact(
                residual_workspace,
                float(alpha_state[0]),
                float(alpha_state[1]),
                root_fields,
                surface_workspace,
            )
            packed_residual.fill(0.0)
            scratch = np.empty(self.grid.Nr, dtype=np.float64)
            residual_ops._run_residual_blocks_packed_precomputed(
                packed_residual,
                scratch,
                self.residual_binding_layout.active_residual_block_codes,
                self.residual_binding_layout.active_residual_block_orders,
                self.active_coeff_index_rows,
                self.active_lengths,
                residual_workspace,
                sin_ktheta,
                cos_ktheta,
                rho_powers,
                y,
                T_fields,
                weights,
                a,
                R0,
                B0,
            )
            return packed_residual

        return runner

    def _refresh_fourier_family_metadata(self) -> None:
        self.c_effective_order, self.s_effective_order = refresh_fourier_family_metadata(
            c_profile_names=self.c_profile_names,
            s_profile_names=self.s_profile_names,
            profile_coeffs=self.case.profile_coeffs,
            c_offsets=self.case.c_offsets,
            s_offsets=self.case.s_offsets,
            c_family_fields=self.c_family_fields,
            s_family_fields=self.s_family_fields,
        )

    def invalidate_source_state(self) -> None:
        if (self.source_plan.route, self.source_plan.coordinate, self.source_plan.nodes) in {
            ("PJ2", "psin", "uniform"),
            ("PQ", "psin", "uniform"),
        }:
            self.source_runtime_state.work_state.psin_query.fill(-1.0)

    def _build_fused_residual_runner(self) -> Callable[[np.ndarray], np.ndarray]:
        if self.source_plan.requires_psin_profile_fields and self.psin_profile.u_fields is None:
            return self._evaluate_residual
        if not self.source_plan.supports_fused_residual:
            return self._evaluate_residual
        return operator_ops.bind_fused_residual_runner(
            source_plan=self.source_plan,
            static_layout=self.static_layout,
            residual_binding_layout=self.residual_binding_layout,
            runtime_layout=self.runtime_layout,
            alpha_state=self.execution_state.fused_alpha_state,
            c_active_order=int(self.c_effective_order),
            s_active_order=int(self.s_effective_order),
            a=float(self.case.a),
            R0=float(self.case.R0),
            Z0=float(self.case.Z0),
            B0=float(self.case.B0),
        )

    def _snapshot_equilibrium_from_runtime(self, x: np.ndarray) -> Equilibrium:
        return snapshot_equilibrium_from_runtime(
            x,
            case=self.case,
            grid=self.grid,
            profile_L=self.profile_L,
            coeff_index=self.coeff_index,
            profile_names=self.profile_names,
            shape_profile_names=self.shape_profile_names,
            profile_index=self.profile_index,
            profiles_by_name=self.profiles_by_name,
            psin=self.psin,
            FFn_psin=self.FFn_psin,
            Pn_psin=self.Pn_psin,
            psin_r=self.psin_r,
            psin_rr=self.psin_rr,
            alpha1=self.alpha1,
            alpha2=self.alpha2,
        )

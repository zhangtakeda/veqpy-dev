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

from veqpy.engine import (
    bind_residual_runner,
    bind_residual_stage_runner,
    validate_route,
)
from veqpy.model import Equilibrium, Geometry, Grid, Profile
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
    ExecutionState,
    FieldRuntimeState,
    ResidualBindingLayout,
    RuntimeLayout,
    SetupLayout,
    SourceRuntimeState,
    StaticLayout,
)
from veqpy.operator.operator_case import OperatorCase
from veqpy.operator.plans import (
    ResidualPlan,
    SourcePlan,
    build_residual_plan,
)
from veqpy.operator.profile_setup import (
    make_profile,
    refresh_profile_runtime,
    validate_case_compatibility,
)
from veqpy.operator.runner_binding import (
    build_bound_residual_full_stage_runner,
    build_bound_residual_pack_stage_runner,
    build_fused_residual_runner,
)
from veqpy.operator.runtime_allocation import allocate_runtime_state
from veqpy.operator.source_orchestration import (
    build_bound_source_stage_runner,
)
from veqpy.operator.source_runtime import (
    ENDPOINT_POLICY_CODES,
    PROJECTION_DOMAIN_CODES,
    SOURCE_PARAMETERIZATION_CODES,
    SOURCE_PROJECTION_POLICIES,
    SourceProjectionPolicy,
    build_source_stage_runner,
    resolve_source_projection_policy,
    refresh_source_runtime,
    validate_source_inputs,
)
from veqpy.operator.stage_helpers import (
    build_geometry_stage_runner,
    build_profile_stage_runner,
    refresh_fourier_family_base_fields,
    refresh_fourier_family_metadata,
    refresh_stage_a_runtime,
)
from veqpy.residual_blocks import build_residual_block_metadata

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


@dataclass(slots=True)
class Operator:
    """封装固定 case, grid 与 runtime 的 residual 求值器."""

    grid: Grid = field(repr=False)
    case: OperatorCase = field(repr=False)
    static_layout: StaticLayout = field(init=False, repr=False)
    residual_binding_layout: ResidualBindingLayout = field(init=False, repr=False)
    setup_layout: SetupLayout = field(init=False, repr=False)
    runtime_layout: RuntimeLayout = field(init=False, repr=False)
    geometry: Geometry = field(init=False)
    geometry_surface_slab: np.ndarray = field(init=False, repr=False)
    geometry_radial_slab: np.ndarray = field(init=False, repr=False)

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
    c_effective_order: int = field(init=False, repr=False)
    s_effective_order: int = field(init=False, repr=False)
    _source_route_spec: object = field(init=False, repr=False)
    _source_runner: Callable = field(init=False, repr=False)
    residual_plan: ResidualPlan = field(init=False, repr=False)
    source_plan: SourcePlan = field(init=False, repr=False)
    field_runtime_state: FieldRuntimeState = field(init=False, repr=False)
    execution_state: ExecutionState = field(init=False, repr=False)
    source_runtime_state: SourceRuntimeState = field(init=False, repr=False)
    packed_residual: np.ndarray = field(init=False, repr=False)
    source_vector_slab: np.ndarray = field(init=False, repr=False)
    _source_projection_policy: SourceProjectionPolicy | None = field(init=False, repr=False)

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
        self.setup_layout = self._build_setup_layout()
        self.residual_binding_layout = self._build_residual_binding_layout()
        self._validate_runtime_profile_support()

        self.execution_state = ExecutionState(
            profile_stage_runner=lambda x: None,
            geometry_stage_runner=lambda: None,
            source_stage_runner=lambda: (0.0, 0.0),
            residual_pack_stage_runner=lambda: np.zeros(self.x_size, dtype=np.float64),
            residual_full_stage_runner=lambda: np.zeros(self.x_size, dtype=np.float64),
            residual_pack_runner=lambda *args, **kwargs: np.zeros(self.x_size, dtype=np.float64),
            residual_stage_runner=lambda *args, **kwargs: np.zeros(self.x_size, dtype=np.float64),
            fused_residual_runner=lambda x_eval: self._evaluate_residual(x_eval),
            fused_alpha_state=np.zeros(2, dtype=np.float64),
            supports_fused_residual=False,
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

    def _build_source_plan(self) -> SourcePlan:
        policy = self._source_projection_policy
        use_projected_finalize = False
        has_projection_policy = policy is not None
        has_ip_constraint = bool(np.isfinite(self.case.Ip))
        projection_domain = "psin"
        heat_projection_degree = 0
        current_projection_degree = 0
        projection_domain_code = 0
        endpoint_policy_code = 0
        allow_query_warmstart = True
        if policy is not None:
            endpoint_policy = policy.current_ip_endpoint_policy if has_ip_constraint else policy.current_other_endpoint_policy
            projection_domain = policy.domain
            heat_projection_degree = int(policy.heat_degree)
            current_projection_degree = int(
                policy.ip_current_degree if has_ip_constraint and policy.ip_current_degree is not None else policy.current_degree
            )
            endpoint_policy_code = ENDPOINT_POLICY_CODES[endpoint_policy]
            projection_domain_code = PROJECTION_DOMAIN_CODES[policy.domain]
            use_projected_finalize = self.case.coordinate == "psin"
            allow_query_warmstart = (not use_projected_finalize) or (
                endpoint_policy_code != ENDPOINT_POLICY_CODES["none"]
            )
        return SourcePlan(
            kernel=self._source_route_spec.implementation,
            coordinate=self.case.coordinate,
            nodes=self.case.nodes,
            coordinate_code=int(self._source_route_spec.coordinate_code),
            strategy=self._source_route_spec.source_strategy,
            parameterization=self._source_route_spec.source_parameterization,
            parameterization_code=SOURCE_PARAMETERIZATION_CODES[self._source_route_spec.source_parameterization],
            n_src=int(self.case.heat_input.shape[0]),
            heat_input=self.case.heat_input,
            current_input=self.case.current_input,
            Ip=float(self.case.Ip),
            beta=float(self.case.beta),
            has_projection_policy=has_projection_policy,
            projection_domain=projection_domain,
            use_projected_finalize=use_projected_finalize,
            heat_projection_degree=heat_projection_degree,
            current_projection_degree=current_projection_degree,
            projection_domain_code=projection_domain_code,
            endpoint_policy_code=endpoint_policy_code,
            allow_query_warmstart=allow_query_warmstart,
        )

    def _build_residual_plan(self) -> ResidualPlan:
        return build_residual_plan(self.source_plan)

    def _validate_runtime_profile_support(self) -> None:
        """校验当前 source route 对 psin profile ownership 的要求."""
        has_active_psin = int(self.profile_L[self.profile_index["psin"]]) >= 0
        if self.source_plan.strategy == "profile_owned_psin" and not has_active_psin:
            raise ValueError(f"{self.case.route} requires an active psin profile because psin is optimized externally")
        if self.case.coordinate == "psin" and self.source_plan.strategy != "profile_owned_psin" and has_active_psin:
            raise ValueError(f"{self.case.route} does not accept an active psin profile because psin is source-owned")
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
            validate_source_inputs=lambda next_case: validate_source_inputs(next_case, self.grid.Nr),
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

    def _build_setup_layout(self) -> SetupLayout:
        return SetupLayout(
            route=self.case.route,
            coordinate=self.case.coordinate,
            nodes=self.case.nodes,
            prefix_profile_names=self.prefix_profile_names,
            shape_profile_names=self.shape_profile_names,
            profile_names=self.profile_names,
            profile_index=self.profile_index,
            c_profile_names=self.c_profile_names,
            s_profile_names=self.s_profile_names,
            profile_L=self.profile_L,
            coeff_index=self.coeff_index,
            order_offsets=self.order_offsets,
            active_profile_mask=self.active_profile_mask,
            active_profile_ids=self.active_profile_ids,
            x_size=self.x_size,
        )

    def _build_residual_binding_layout(self) -> ResidualBindingLayout:
        active_profile_names = tuple(self.profile_names[int(p)] for p in self.active_profile_ids)
        active_residual_block_codes, active_residual_block_orders = build_residual_block_metadata(
            active_profile_names
        )
        return ResidualBindingLayout(
            active_profile_names=active_profile_names,
            active_residual_block_codes=active_residual_block_codes,
            active_residual_block_orders=active_residual_block_orders,
        )

    def _refresh_setup_layout(self) -> None:
        self.setup_layout = self._build_setup_layout()
        self.residual_binding_layout = self._build_residual_binding_layout()

    def _refresh_runtime_layout_views(self) -> None:
        runtime = self.runtime_layout
        runtime.geometry = self.geometry
        runtime.profiles_by_name = self.profiles_by_name
        runtime.active_profile_slab = self.active_profile_slab
        runtime.family_field_slab = self.family_field_slab
        runtime.source_vector_slab = self.source_vector_slab
        runtime.geometry_surface_slab = self.geometry_surface_slab
        runtime.geometry_radial_slab = self.geometry_radial_slab
        runtime.residual_fields = self.residual_fields
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
        runtime.source_barycentric_weights = self.source_runtime_state.barycentric_weights
        runtime.source_fixed_remap_matrix = self.source_runtime_state.fixed_remap_matrix
        runtime.source_psin_query = self.source_runtime_state.psin_query
        runtime.source_parameter_query = self.source_runtime_state.parameter_query
        runtime.source_heat_projection_fit_matrix = self.source_runtime_state.heat_projection_fit_matrix
        runtime.source_current_projection_fit_matrix = self.source_runtime_state.current_projection_fit_matrix
        runtime.source_heat_projection_coeff = self.source_runtime_state.heat_projection_coeff
        runtime.source_current_projection_coeff = self.source_runtime_state.current_projection_coeff
        runtime.source_projection_query = self.source_runtime_state.projection_query
        runtime.source_endpoint_blend = self.source_runtime_state.endpoint_blend
        runtime.materialized_heat_input = self.source_runtime_state.materialized_heat_input
        runtime.materialized_current_input = self.source_runtime_state.materialized_current_input
        runtime.source_scratch_1d = self.source_runtime_state.scratch_1d
        runtime.source_target_root_fields = self.source_runtime_state.target_root_fields

    def _setup_runtime_state(self) -> None:
        bundle = allocate_runtime_state(
            grid=self.grid,
            static_layout=self.static_layout,
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
                profile_static_kwargs_by_name=_PROFILE_STATIC_KWARGS,
                profile_offset_specs=_PROFILE_OFFSET_SPECS,
                profile_scale_specs=_PROFILE_SCALE_SPECS,
            ),
        )
        self.geometry = bundle.geometry
        self.geometry_surface_slab = bundle.geometry_surface_slab
        self.geometry_radial_slab = bundle.geometry_radial_slab
        self.profiles_by_name = bundle.profiles_by_name
        for name, profile in self.profiles_by_name.items():
            if hasattr(type(self), f"{name}_profile"):
                setattr(self, f"{name}_profile", profile)
        self.field_runtime_state = bundle.field_runtime_state
        self.residual_fields = bundle.field_runtime_state.residual_fields
        self.root_fields = bundle.field_runtime_state.root_fields
        self.packed_residual = bundle.field_runtime_state.packed_residual
        self.psin_R = bundle.field_runtime_state.psin_R
        self.psin_Z = bundle.field_runtime_state.psin_Z
        self.G = bundle.field_runtime_state.G
        self.psin = bundle.field_runtime_state.psin
        self.psin_r = bundle.field_runtime_state.psin_r
        self.psin_rr = bundle.field_runtime_state.psin_rr
        self.FFn_psin = bundle.field_runtime_state.FFn_psin
        self.Pn_psin = bundle.field_runtime_state.Pn_psin
        self.source_vector_slab = bundle.source_vector_slab
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

    def _refresh_runtime_state(self) -> None:
        self._refresh_operator_identity()
        self._refresh_setup_layout()
        self._validate_runtime_profile_support()
        self._refresh_profile_runtime()
        self._refresh_fourier_family_metadata()
        refresh_source_runtime(
            case=self.case,
            grid_rho=self.grid.rho,
            source_plan=self.source_plan,
            source_runtime_state=self.source_runtime_state,
            psin=self.psin,
        )
        self._refresh_runtime_layout_views()
        self._refresh_stage_a_runtime()
        self._refresh_runtime_bindings()

    def _refresh_operator_identity(self) -> None:
        spec = validate_route(self.case.route, self.case.coordinate, self.case.nodes)
        self._source_route_spec = spec
        self._source_runner = build_source_stage_runner(spec)
        self._source_projection_policy = resolve_source_projection_policy(
            self.case.route,
            self.case.coordinate,
            self.case.nodes,
            has_ip_constraint=bool(np.isfinite(self.case.Ip)),
            has_beta_constraint=bool(np.isfinite(self.case.beta)),
        )
        self.source_plan = self._build_source_plan()
        self.residual_plan = self._build_residual_plan()

    def _refresh_profile_runtime(self) -> None:
        refresh_profile_runtime(
            case=self.case,
            grid=self.grid,
            profile_names=self.profile_names,
            profile_index=self.profile_index,
            profile_L=self.profile_L,
            profiles_by_name=self.profiles_by_name,
            profile_offset_specs=_PROFILE_OFFSET_SPECS,
            profile_scale_specs=_PROFILE_SCALE_SPECS,
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
        self.execution_state.geometry_stage_runner = self._build_geometry_stage_runner()
        self.execution_state.source_stage_runner = self._build_bound_source_stage_runner()
        self.execution_state.residual_pack_runner = self._build_residual_pack_runner()
        self.execution_state.residual_stage_runner = self._build_residual_stage_runner()
        self.execution_state.residual_pack_stage_runner = self._build_bound_residual_pack_stage_runner()
        self.execution_state.residual_full_stage_runner = self._build_bound_residual_full_stage_runner()
        self.execution_state.fused_residual_runner = self._build_fused_residual_runner()
        fixed_profile_ids = np.flatnonzero(~self.active_profile_mask).astype(np.int64, copy=False)
        for p in fixed_profile_ids:
            self.profiles_by_name[self.profile_names[int(p)]].update()

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
            active_u_fields=self.active_u_fields,
            T_fields=self.grid.T_fields,
            active_rp_fields=self.active_rp_fields,
            active_env_fields=self.active_env_fields,
            active_offsets=self.active_offsets,
            active_scales=self.active_scales,
            active_coeff_index_rows=self.active_coeff_index_rows,
            active_lengths=self.active_lengths,
        )

    def _build_geometry_stage_runner(self) -> Callable:
        return build_geometry_stage_runner(
            c_family_fields=self.c_family_fields,
            s_family_fields=self.s_family_fields,
            c_family_base_fields=self.c_family_base_fields,
            s_family_base_fields=self.s_family_base_fields,
            active_u_fields=self.active_u_fields,
            c_family_source_slots=self.c_family_source_slots,
            s_family_source_slots=self.s_family_source_slots,
            c_effective_order=self.c_effective_order,
            s_effective_order=self.s_effective_order,
            h_fields=self.h_profile.u_fields,
            v_fields=self.v_profile.u_fields,
            k_fields=self.k_profile.u_fields,
            a=self.case.a,
            R0=self.case.R0,
            Z0=self.case.Z0,
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
            rho=self.grid.rho,
            theta=self.grid.theta,
            cos_ktheta=self.grid.cos_ktheta,
            sin_ktheta=self.grid.sin_ktheta,
            k_cos_ktheta=self.grid.k_cos_ktheta,
            k_sin_ktheta=self.grid.k_sin_ktheta,
            k2_cos_ktheta=self.grid.k2_cos_ktheta,
            k2_sin_ktheta=self.grid.k2_sin_ktheta,
            weights=self.grid.weights,
        )

    def _build_residual_pack_runner(self) -> Callable:
        profile_names = self.residual_binding_layout.active_profile_names
        try:
            return bind_residual_runner(
                profile_names,
                self.active_coeff_index_rows,
                self.active_lengths,
                self.x_size,
                block_codes=self.residual_binding_layout.active_residual_block_codes,
                block_orders=self.residual_binding_layout.active_residual_block_orders,
            )
        except KeyError as exc:
            raise ValueError(f"Unsupported active residual block set {profile_names!r}") from exc

    def _build_residual_stage_runner(self) -> Callable:
        profile_names = self.residual_binding_layout.active_profile_names
        try:
            return bind_residual_stage_runner(
                profile_names,
                self.active_coeff_index_rows,
                self.active_lengths,
                self.x_size,
                block_codes=self.residual_binding_layout.active_residual_block_codes,
                block_orders=self.residual_binding_layout.active_residual_block_orders,
            )
        except KeyError as exc:
            raise ValueError(f"Unsupported active residual block set {profile_names!r}") from exc

    def _build_bound_source_stage_runner(self) -> Callable:
        return build_bound_source_stage_runner(self)

    def _build_bound_residual_pack_stage_runner(self) -> Callable:
        return build_bound_residual_pack_stage_runner(
            residual_pack_runner=self.execution_state.residual_pack_runner,
            G=self.G,
            psin_R=self.psin_R,
            psin_Z=self.psin_Z,
            sin_tb=self.geometry.sin_tb,
            sin_ktheta=self.grid.sin_ktheta,
            cos_ktheta=self.grid.cos_ktheta,
            rho_powers=self.grid.rho_powers,
            y=self.grid.y,
            T=self.grid.T_fields[0],
            weights=self.grid.weights,
            a=self.case.a,
            R0=self.case.R0,
            B0=self.case.B0,
        )

    def _build_bound_residual_full_stage_runner(self) -> Callable:
        return build_bound_residual_full_stage_runner(
            residual_stage_runner=self.execution_state.residual_stage_runner,
            packed_residual=self.packed_residual,
            residual_fields=self.residual_fields,
            alpha_state=self.execution_state.fused_alpha_state,
            root_fields=self.root_fields,
            R_fields=self.geometry.R_fields,
            Z_fields=self.geometry.Z_fields,
            J_fields=self.geometry.J_fields,
            g_fields=self.geometry.g_fields,
            sin_tb=self.geometry.sin_tb,
            sin_ktheta=self.grid.sin_ktheta,
            cos_ktheta=self.grid.cos_ktheta,
            rho_powers=self.grid.rho_powers,
            y=self.grid.y,
            T=self.grid.T_fields[0],
            weights=self.grid.weights,
            a=self.case.a,
            R0=self.case.R0,
            B0=self.case.B0,
        )

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
        if self.source_plan.strategy == "fixed_point_psin":
            self.source_runtime_state.psin_query.fill(-1.0)

    def _build_fused_residual_runner(self) -> Callable[[np.ndarray], np.ndarray]:
        runner, supports_fused = build_fused_residual_runner(
            residual_plan=self.residual_plan,
            psin_profile_u_fields=self.psin_profile.u_fields,
            evaluate_residual=self._evaluate_residual,
            fused_common_kwargs=dict(
                residual_plan=self.residual_plan,
                static_layout=self.static_layout,
                residual_binding_layout=self.residual_binding_layout,
                setup_layout=self.setup_layout,
                runtime_layout=self.runtime_layout,
                alpha_state=self.execution_state.fused_alpha_state,
                c_active_order=int(self.c_effective_order),
                s_active_order=int(self.s_effective_order),
                a=float(self.case.a),
                R0=float(self.case.R0),
                Z0=float(self.case.Z0),
                B0=float(self.case.B0),
            ),
        )
        self.execution_state.supports_fused_residual = supports_fused
        return runner

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

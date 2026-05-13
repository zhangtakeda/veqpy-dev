"""
Module: engine.backend_abi

Role:
- 定义 numba fused backend 使用的显式 ABI binding 规范.
- 把 bind-time 数据选择从 numba 实现中剥离到 engine 层 ABI 模块.

Public API:
- SourceExecutionABI
- FusedHotRuntimeABI
- FusedResidualPackABI
- FusedSourceEvalABI
- build_source_execution_abi
- build_fused_hot_runtime_abi
- build_fused_residual_pack_abi
- build_fused_source_eval_abi
- build_profile_owned_psin_source_abi
- build_pj2_fixed_point_psin_source_abi
- build_pq_fixed_point_psin_source_abi
"""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import TYPE_CHECKING, Callable

import numpy as np

from veqpy.engine.numba_source import resolve_source_scratch_kernel, uniform_barycentric_weights

if TYPE_CHECKING:
    from veqpy.operator.runtime_layout import BackendState
    from veqpy.orchestration import SourcePlan


RouteKey = tuple[str, str, str]

PROFILE_OWNED_PSIN_ROUTE_KEYS: frozenset[RouteKey] = frozenset(
    {
        ("PF", "psin", "uniform"),
        ("PP", "psin", "uniform"),
        ("PI", "psin", "uniform"),
        ("PJ1", "psin", "uniform"),
    }
)

FIXED_POINT_PSIN_ROUTE_KEYS: frozenset[RouteKey] = frozenset(
    {
        ("PJ2", "psin", "uniform"),
        ("PQ", "psin", "uniform"),
    }
)

SUPPORTED_FUSED_SOURCE_ROUTE_KEYS: frozenset[RouteKey] = frozenset(
    {
        ("PF", "rho", "uniform"),
        ("PF", "rho", "grid"),
        ("PF", "psin", "uniform"),
        ("PF", "psin", "grid"),
        ("PP", "rho", "uniform"),
        ("PP", "rho", "grid"),
        ("PP", "psin", "uniform"),
        ("PP", "psin", "grid"),
        ("PI", "rho", "uniform"),
        ("PI", "rho", "grid"),
        ("PI", "psin", "uniform"),
        ("PI", "psin", "grid"),
        ("PJ1", "rho", "uniform"),
        ("PJ1", "rho", "grid"),
        ("PJ1", "psin", "uniform"),
        ("PJ1", "psin", "grid"),
        ("PJ2", "rho", "uniform"),
        ("PJ2", "rho", "grid"),
        ("PJ2", "psin", "uniform"),
        ("PJ2", "psin", "grid"),
        ("PQ", "rho", "uniform"),
        ("PQ", "rho", "grid"),
        ("PQ", "psin", "uniform"),
        ("PQ", "psin", "grid"),
    }
)

PSIN_SKIP_PROJECTION_FINALIZE_ROUTE_KEYS: frozenset[RouteKey] = frozenset(
    {("PI", "psin", "uniform")}
)


@dataclass(frozen=True, slots=True)
class SourceExecutionABI:
    route_key: RouteKey
    psin_active_slot: int
    psin_active_length: int
    psin_coeff_start: int
    F_active_slot: int
    F_active_length: int
    F_coeff_start: int
    route_requires_optimized_psin_profile: bool
    requires_optimized_psin_profile: bool
    requires_fixed_point_psin_materialization: bool
    requires_psin_profile_fields: bool
    requires_psin_query_workspace: bool
    requires_source_parameter_query: bool
    requires_target_root_fields: bool
    supports_fused_residual: bool
    skip_projection_finalize: bool

    @property
    def requires_fixed_point_psin(self) -> bool:
        return self.requires_fixed_point_psin_materialization

    @property
    def uses_active_F_profile(self) -> bool:
        return self.F_active_length > 0


def _active_profile_abi_fields(
    name: str,
    *,
    profile_index: dict[str, int],
    profile_L: np.ndarray,
    coeff_index: np.ndarray,
    active_profile_ids: np.ndarray,
) -> tuple[int, int, int]:
    profile_id = int(profile_index.get(name, -1))
    if profile_id < 0:
        return -1, 0, -1
    length = int(profile_L[profile_id]) + 1
    if length <= 0:
        return -1, 0, -1

    active_slot = -1
    for slot, active_profile_id in enumerate(active_profile_ids):
        if int(active_profile_id) == profile_id:
            active_slot = int(slot)
            break

    coeff_start = -1
    if coeff_index.ndim == 2 and coeff_index.shape[1] > 0:
        candidate = int(coeff_index[profile_id, 0])
        if candidate >= 0:
            coeff_start = candidate
    return active_slot, length, coeff_start


def build_source_execution_abi(
    *,
    source_plan: "SourcePlan",
    profile_index: dict[str, int],
    profile_L: np.ndarray,
    coeff_index: np.ndarray,
    active_profile_ids: np.ndarray,
) -> SourceExecutionABI:
    route_key = (source_plan.route, source_plan.coordinate, source_plan.nodes)
    psin_active_slot, psin_active_length, psin_coeff_start = _active_profile_abi_fields(
        "psin",
        profile_index=profile_index,
        profile_L=profile_L,
        coeff_index=coeff_index,
        active_profile_ids=active_profile_ids,
    )
    F_active_slot, F_active_length, F_coeff_start = _active_profile_abi_fields(
        "F",
        profile_index=profile_index,
        profile_L=profile_L,
        coeff_index=coeff_index,
        active_profile_ids=active_profile_ids,
    )

    if route_key not in SUPPORTED_FUSED_SOURCE_ROUTE_KEYS:
        raise ValueError(f"Unsupported source route key {route_key!r}")
    if psin_active_length > 0 and psin_active_slot < 0:
        raise ValueError("psin is active but has no active profile slot")
    if F_active_length > 0 and F_active_slot < 0:
        raise ValueError("F is active but has no active profile slot")

    route_requires_optimized_psin_profile = route_key in PROFILE_OWNED_PSIN_ROUTE_KEYS
    if route_requires_optimized_psin_profile and psin_active_length <= 0:
        raise ValueError(
            f"{route_key[0]} {route_key[1]}/{route_key[2]} requires an active psin profile"
        )
    if (
        source_plan.coordinate == "psin"
        and not route_requires_optimized_psin_profile
        and psin_active_length > 0
    ):
        raise ValueError(
            f"{route_key[0]} {route_key[1]}/{route_key[2]} does not accept an active psin "
            "profile because psin is source-owned"
        )

    requires_optimized_psin_profile = route_requires_optimized_psin_profile
    requires_fixed_point_psin_materialization = route_key in FIXED_POINT_PSIN_ROUTE_KEYS
    return SourceExecutionABI(
        route_key=route_key,
        psin_active_slot=psin_active_slot,
        psin_active_length=psin_active_length,
        psin_coeff_start=psin_coeff_start,
        F_active_slot=F_active_slot,
        F_active_length=F_active_length,
        F_coeff_start=F_coeff_start,
        route_requires_optimized_psin_profile=route_requires_optimized_psin_profile,
        requires_optimized_psin_profile=requires_optimized_psin_profile,
        requires_fixed_point_psin_materialization=requires_fixed_point_psin_materialization,
        requires_psin_profile_fields=requires_optimized_psin_profile,
        requires_psin_query_workspace=(
            requires_optimized_psin_profile or requires_fixed_point_psin_materialization
        ),
        requires_source_parameter_query=bool(
            source_plan.coordinate == "psin" and source_plan.parameterization != "identity"
        ),
        requires_target_root_fields=(
            requires_optimized_psin_profile or requires_fixed_point_psin_materialization
        ),
        supports_fused_residual=route_key in SUPPORTED_FUSED_SOURCE_ROUTE_KEYS,
        skip_projection_finalize=route_key in PSIN_SKIP_PROJECTION_FINALIZE_ROUTE_KEYS,
    )


@dataclass(frozen=True, slots=True)
class FusedHotRuntimeABI:
    active_profile_slab: np.ndarray
    T: np.ndarray
    T_r: np.ndarray
    T_rr: np.ndarray
    active_offsets: np.ndarray
    active_scales: np.ndarray
    active_coeff_index_rows: np.ndarray
    active_lengths: np.ndarray
    c_family_fields: np.ndarray
    s_family_fields: np.ndarray
    c_family_base_fields: np.ndarray
    s_family_base_fields: np.ndarray
    active_u_fields: np.ndarray
    c_family_source_slots: np.ndarray
    s_family_source_slots: np.ndarray
    geometry_surface_workspace: np.ndarray
    geometry_radial_workspace: np.ndarray
    rho: np.ndarray
    theta: np.ndarray
    cos_mtheta: np.ndarray
    sin_mtheta: np.ndarray
    m_cos_mtheta: np.ndarray
    m_sin_mtheta: np.ndarray
    m2_cos_mtheta: np.ndarray
    m2_sin_mtheta: np.ndarray
    h_fields: np.ndarray
    v_fields: np.ndarray
    k_fields: np.ndarray
    F_profile_fields: np.ndarray
    psin_active_slot: int
    psin_active_length: int
    psin_coeff_start: int
    F_active_slot: int
    F_active_length: int
    F_coeff_start: int
    c_active_order: int
    s_active_order: int
    a: float
    R0: float
    Z0: float


@dataclass(frozen=True, slots=True)
class FusedResidualPackABI:
    packed_residual: np.ndarray
    residual_surface_workspace: np.ndarray
    active_residual_block_codes: np.ndarray
    active_residual_block_orders: np.ndarray
    active_residual_block_radial_powers: np.ndarray
    active_coeff_index_rows: np.ndarray
    active_lengths: np.ndarray
    sin_mtheta: np.ndarray
    cos_mtheta: np.ndarray
    rho_powers: np.ndarray
    y: np.ndarray
    T: np.ndarray
    quadrature: np.ndarray
    a: float
    R0: float
    B0: float


@dataclass(frozen=True, slots=True)
class FusedSourceEvalABI:
    source_kernel: Callable
    scratch_source_kernel: Callable | None
    coordinate_code: int
    quadrature: np.ndarray
    differentiator: np.ndarray
    accumulator: np.ndarray
    rho: np.ndarray
    n_axis_fix: int
    radial_workspace: np.ndarray
    surface_workspace: np.ndarray
    F_profile_u: np.ndarray
    Ip: float
    beta: float
    source_scratch_1d: np.ndarray
    source_scratch_2d: np.ndarray
    B0: float


def build_fused_hot_runtime_abi(
    *,
    backend_state: "BackendState",
    source_execution: SourceExecutionABI,
    c_active_order: int,
    s_active_order: int,
    a: float,
    R0: float,
    Z0: float,
) -> FusedHotRuntimeABI:
    static_layout = backend_state.static_layout
    runtime_layout = backend_state.runtime_layout
    return FusedHotRuntimeABI(
        active_profile_slab=runtime_layout.active_profile_slab,
        T=static_layout.T,
        T_r=static_layout.T_r,
        T_rr=static_layout.T_rr,
        active_offsets=runtime_layout.active_offsets,
        active_scales=runtime_layout.active_scales,
        active_coeff_index_rows=runtime_layout.active_coeff_index_rows,
        active_lengths=runtime_layout.active_lengths,
        c_family_fields=runtime_layout.c_family_fields,
        s_family_fields=runtime_layout.s_family_fields,
        c_family_base_fields=runtime_layout.c_family_base_fields,
        s_family_base_fields=runtime_layout.s_family_base_fields,
        active_u_fields=runtime_layout.active_u_fields,
        c_family_source_slots=runtime_layout.c_family_source_slots,
        s_family_source_slots=runtime_layout.s_family_source_slots,
        geometry_surface_workspace=runtime_layout.geometry_surface_workspace,
        geometry_radial_workspace=runtime_layout.geometry_radial_workspace,
        rho=static_layout.rho,
        theta=static_layout.theta,
        cos_mtheta=static_layout.cos_mtheta,
        sin_mtheta=static_layout.sin_mtheta,
        m_cos_mtheta=static_layout.m_cos_mtheta,
        m_sin_mtheta=static_layout.m_sin_mtheta,
        m2_cos_mtheta=static_layout.m2_cos_mtheta,
        m2_sin_mtheta=static_layout.m2_sin_mtheta,
        h_fields=runtime_layout.h_fields,
        v_fields=runtime_layout.v_fields,
        k_fields=runtime_layout.k_fields,
        F_profile_fields=runtime_layout.F_profile_fields,
        psin_active_slot=int(source_execution.psin_active_slot),
        psin_active_length=int(source_execution.psin_active_length),
        psin_coeff_start=int(source_execution.psin_coeff_start),
        F_active_slot=int(source_execution.F_active_slot),
        F_active_length=int(source_execution.F_active_length),
        F_coeff_start=int(source_execution.F_coeff_start),
        c_active_order=c_active_order,
        s_active_order=s_active_order,
        a=a,
        R0=R0,
        Z0=Z0,
    )


def build_fused_residual_pack_abi(
    *,
    backend_state: "BackendState",
    a: float,
    R0: float,
    B0: float,
) -> FusedResidualPackABI:
    static_layout = backend_state.static_layout
    runtime_layout = backend_state.runtime_layout
    residual_binding_layout = backend_state.residual_binding_layout
    return FusedResidualPackABI(
        packed_residual=runtime_layout.packed_residual,
        residual_surface_workspace=runtime_layout.residual_surface_workspace,
        active_residual_block_codes=residual_binding_layout.active_residual_block_codes,
        active_residual_block_orders=residual_binding_layout.active_residual_block_orders,
        active_residual_block_radial_powers=residual_binding_layout.active_residual_block_radial_powers,
        active_coeff_index_rows=runtime_layout.active_coeff_index_rows,
        active_lengths=runtime_layout.active_lengths,
        sin_mtheta=static_layout.sin_mtheta,
        cos_mtheta=static_layout.cos_mtheta,
        rho_powers=static_layout.rho_powers,
        y=static_layout.y,
        T=static_layout.T,
        quadrature=static_layout.quadrature,
        a=a,
        R0=R0,
        B0=B0,
    )


def build_fused_source_eval_abi(
    *,
    source_plan: "SourcePlan",
    backend_state: "BackendState",
    B0: float,
    fix_rho: float,
) -> FusedSourceEvalABI:
    source_kernel = source_plan.kernel
    static_layout = backend_state.static_layout
    runtime_layout = backend_state.runtime_layout
    source_work_state = backend_state.source_runtime_state.work_state

    if source_plan.coordinate == "psin" and source_plan.nodes == "uniform":
        n_axis_fix = 0
    else:
        n_axis_fix = int(np.searchsorted(static_layout.rho, fix_rho))

    return FusedSourceEvalABI(
        source_kernel=source_kernel,
        scratch_source_kernel=resolve_source_scratch_kernel(source_kernel),
        coordinate_code=int(source_plan.coordinate_code),
        quadrature=static_layout.quadrature,
        differentiator=static_layout.differentiator,
        accumulator=static_layout.accumulator,
        rho=static_layout.rho,
        n_axis_fix=n_axis_fix,
        radial_workspace=runtime_layout.geometry_radial_workspace,
        surface_workspace=runtime_layout.geometry_surface_workspace,
        F_profile_u=runtime_layout.F_profile_u,
        Ip=float(source_plan.Ip),
        beta=float(source_plan.beta),
        source_scratch_1d=source_work_state.scratch_1d,
        source_scratch_2d=source_work_state.scratch_2d,
        B0=B0,
    )


def build_profile_owned_psin_source_abi(
    *,
    source_plan: "SourcePlan",
    source_execution: SourceExecutionABI,
    backend_state: "BackendState",
):
    runtime_layout = backend_state.runtime_layout
    source_runtime_state = backend_state.source_runtime_state
    source_work_state = source_runtime_state.work_state
    source_aux_state = source_runtime_state.aux_state
    return SimpleNamespace(
        source_target_root_fields=source_aux_state.target_root_fields,
        source_psin_query=source_work_state.psin_query,
        source_parameter_query=source_work_state.parameter_query,
        heat_projection_coeff=source_aux_state.heat_projection_coeff,
        current_projection_coeff=source_aux_state.current_projection_coeff,
        endpoint_blend=source_runtime_state.const_state.endpoint_blend,
        materialized_heat_input=source_work_state.materialized_heat_input,
        materialized_current_input=source_work_state.materialized_current_input,
        psin_profile_fields=runtime_layout.psin_profile_fields,
        parameterization_code=int(source_plan.parameterization_code),
        has_projection_policy=bool(source_plan.has_projection_policy),
        projection_domain_code=int(source_plan.projection_domain_code),
        endpoint_policy_code=int(source_plan.endpoint_policy_code),
        heat_input=source_plan.heat_input,
        current_input=source_plan.current_input,
        psin_active_slot=int(source_execution.psin_active_slot),
        psin_active_length=int(source_execution.psin_active_length),
        psin_coeff_start=int(source_execution.psin_coeff_start),
        F_active_slot=int(source_execution.F_active_slot),
        F_active_length=int(source_execution.F_active_length),
        F_coeff_start=int(source_execution.F_coeff_start),
        skip_projection_finalize=bool(source_execution.skip_projection_finalize),
    )


def _build_fixed_point_psin_source_abi(
    *,
    source_plan: "SourcePlan",
    source_execution: SourceExecutionABI,
    backend_state: "BackendState",
    barycentric_order_cap: int,
    allow_query_warmstart: bool,
    finalize_iter: int,
):
    runtime_layout = backend_state.runtime_layout
    source_runtime_state = backend_state.source_runtime_state
    source_work_state = source_runtime_state.work_state
    source_aux_state = source_runtime_state.aux_state
    Ip = float(source_plan.Ip)
    return SimpleNamespace(
        source_psin_query=source_work_state.psin_query,
        materialized_heat_input=source_work_state.materialized_heat_input,
        materialized_current_input=source_work_state.materialized_current_input,
        source_scratch_1d=source_work_state.scratch_1d,
        source_scratch_2d=source_work_state.scratch_2d,
        heat_projection_coeff=source_aux_state.heat_projection_coeff,
        current_projection_coeff=source_aux_state.current_projection_coeff,
        endpoint_blend=source_runtime_state.const_state.endpoint_blend,
        F_profile_u=runtime_layout.F_profile_u,
        psin_profile_u=runtime_layout.psin_profile_u,
        heat_input=source_plan.heat_input,
        current_input=source_plan.current_input,
        coordinate_code=int(source_plan.coordinate_code),
        Ip=Ip,
        beta=float(source_plan.beta),
        has_Ip=bool(np.isfinite(Ip)),
        psin_active_slot=int(source_execution.psin_active_slot),
        psin_active_length=int(source_execution.psin_active_length),
        psin_coeff_start=int(source_execution.psin_coeff_start),
        F_active_slot=int(source_execution.F_active_slot),
        F_active_length=int(source_execution.F_active_length),
        F_coeff_start=int(source_execution.F_coeff_start),
        projection_domain_code=int(source_plan.projection_domain_code),
        endpoint_policy_code=int(source_plan.endpoint_policy_code),
        barycentric_weights=uniform_barycentric_weights(
            min(barycentric_order_cap, int(source_plan.source_sample_count))
        ),
        allow_query_warmstart=allow_query_warmstart,
        finalize_iter=finalize_iter,
    )


def build_pj2_fixed_point_psin_source_abi(
    *,
    source_plan: "SourcePlan",
    source_execution: SourceExecutionABI,
    backend_state: "BackendState",
):
    return _build_fixed_point_psin_source_abi(
        source_plan=source_plan,
        source_execution=source_execution,
        backend_state=backend_state,
        barycentric_order_cap=8,
        allow_query_warmstart=False,
        finalize_iter=8,
    )


def build_pq_fixed_point_psin_source_abi(
    *,
    source_plan: "SourcePlan",
    source_execution: SourceExecutionABI,
    backend_state: "BackendState",
):
    endpoint_policy_code = int(source_plan.endpoint_policy_code)
    return _build_fixed_point_psin_source_abi(
        source_plan=source_plan,
        source_execution=source_execution,
        backend_state=backend_state,
        barycentric_order_cap=4,
        allow_query_warmstart=bool(endpoint_policy_code != 0),
        finalize_iter=16,
    )

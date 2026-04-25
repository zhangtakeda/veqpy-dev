"""
Module: engine.backend_abi

Role:
- 定义 numba fused backend 使用的显式 ABI binding 规范.
- 把 bind-time 数据选择从 numba 实现中剥离到 engine 层 ABI 模块.

Public API:
- FusedHotRuntimeABI
- FusedResidualPackABI
- FusedSourceEvalABI
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

from veqpy.engine.numba_source import _uniform_barycentric_weights, resolve_source_scratch_kernel

if TYPE_CHECKING:
    from veqpy.operator.layouts import BackendState
    from veqpy.orchestration import SourcePlan


@dataclass(frozen=True, slots=True)
class FusedHotRuntimeABI:
    active_profile_slab: np.ndarray
    T_fields: np.ndarray
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
    cos_ktheta: np.ndarray
    sin_ktheta: np.ndarray
    k_cos_ktheta: np.ndarray
    k_sin_ktheta: np.ndarray
    k2_cos_ktheta: np.ndarray
    k2_sin_ktheta: np.ndarray
    h_fields: np.ndarray
    v_fields: np.ndarray
    k_fields: np.ndarray
    F_profile_fields: np.ndarray
    convert_f_squared_to_f: bool
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
    sin_ktheta: np.ndarray
    cos_ktheta: np.ndarray
    rho_powers: np.ndarray
    y: np.ndarray
    T_fields: np.ndarray
    weights: np.ndarray
    a: float
    R0: float
    B0: float


@dataclass(frozen=True, slots=True)
class FusedSourceEvalABI:
    source_kernel: Callable
    scratch_source_kernel: Callable | None
    coordinate_code: int
    weights: np.ndarray
    differentiation_matrix: np.ndarray
    integration_matrix: np.ndarray
    rho: np.ndarray
    radial_workspace: np.ndarray
    surface_workspace: np.ndarray
    F_profile_u: np.ndarray
    Ip: float
    beta: float
    source_scratch_1d: np.ndarray
    B0: float


def build_fused_hot_runtime_abi(
    *,
    backend_state: "BackendState",
    convert_f_squared_to_f: bool,
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
        T_fields=static_layout.T_fields,
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
        cos_ktheta=static_layout.cos_ktheta,
        sin_ktheta=static_layout.sin_ktheta,
        k_cos_ktheta=static_layout.k_cos_ktheta,
        k_sin_ktheta=static_layout.k_sin_ktheta,
        k2_cos_ktheta=static_layout.k2_cos_ktheta,
        k2_sin_ktheta=static_layout.k2_sin_ktheta,
        h_fields=runtime_layout.h_fields,
        v_fields=runtime_layout.v_fields,
        k_fields=runtime_layout.k_fields,
        F_profile_fields=runtime_layout.F_profile_fields,
        convert_f_squared_to_f=convert_f_squared_to_f,
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
        sin_ktheta=static_layout.sin_ktheta,
        cos_ktheta=static_layout.cos_ktheta,
        rho_powers=static_layout.rho_powers,
        y=static_layout.y,
        T_fields=static_layout.T_fields,
        weights=static_layout.weights,
        a=a,
        R0=R0,
        B0=B0,
    )


def build_fused_source_eval_abi(
    *,
    source_plan: "SourcePlan",
    backend_state: "BackendState",
    B0: float,
) -> FusedSourceEvalABI:
    source_kernel = source_plan.kernel
    static_layout = backend_state.static_layout
    runtime_layout = backend_state.runtime_layout
    source_work_state = backend_state.source_runtime_state.work_state
    return FusedSourceEvalABI(
        source_kernel=source_kernel,
        scratch_source_kernel=resolve_source_scratch_kernel(source_kernel),
        coordinate_code=int(source_plan.coordinate_code),
        weights=static_layout.weights,
        differentiation_matrix=static_layout.differentiation_matrix,
        integration_matrix=static_layout.integration_matrix,
        rho=static_layout.rho,
        radial_workspace=runtime_layout.geometry_radial_workspace,
        surface_workspace=runtime_layout.geometry_surface_workspace,
        F_profile_u=runtime_layout.F_profile_u,
        Ip=float(source_plan.Ip),
        beta=float(source_plan.beta),
        source_scratch_1d=source_work_state.scratch_1d,
        B0=B0,
    )


def build_profile_owned_psin_source_abi(
    *,
    source_plan: "SourcePlan",
    backend_state: "BackendState",
    skip_projection_finalize: bool,
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
        skip_projection_finalize=skip_projection_finalize,
    )


def _build_fixed_point_psin_source_abi(
    *,
    source_plan: "SourcePlan",
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
        projection_domain_code=int(source_plan.projection_domain_code),
        endpoint_policy_code=int(source_plan.endpoint_policy_code),
        barycentric_weights=_uniform_barycentric_weights(
            min(barycentric_order_cap, int(source_plan.source_sample_count))
        ),
        allow_query_warmstart=allow_query_warmstart,
        finalize_iter=finalize_iter,
    )


def build_pj2_fixed_point_psin_source_abi(
    *,
    source_plan: "SourcePlan",
    backend_state: "BackendState",
):
    return _build_fixed_point_psin_source_abi(
        source_plan=source_plan,
        backend_state=backend_state,
        barycentric_order_cap=8,
        allow_query_warmstart=False,
        finalize_iter=8,
    )


def build_pq_fixed_point_psin_source_abi(
    *,
    source_plan: "SourcePlan",
    backend_state: "BackendState",
):
    endpoint_policy_code = int(source_plan.endpoint_policy_code)
    return _build_fixed_point_psin_source_abi(
        source_plan=source_plan,
        backend_state=backend_state,
        barycentric_order_cap=4,
        allow_query_warmstart=bool(endpoint_policy_code != 0),
        finalize_iter=16,
    )

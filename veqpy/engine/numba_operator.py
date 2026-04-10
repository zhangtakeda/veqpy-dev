"""
Module: engine.numba_operator

Role:
- 提供 fused x -> residual hot runner.
- 把常见 route 的 stage A/B/C/D 串成单个 engine 绑定入口.

Public API:
- bind_fused_residual_runner
- bind_fused_single_pass_residual_runner
- bind_fused_profile_owned_psin_residual_runner
- bind_fused_fixed_point_psin_residual_runner

Notes:
- 这里只覆盖 common route.
- fixed-point psin 仍由上层保留 staged orchestration.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable

import numpy as np
from numba import njit

from veqpy.engine.numba_geometry import update_geometry_hot
from veqpy.engine.numba_profile import update_profiles_packed_bulk
from veqpy.engine.numba_residual import _run_residual_blocks_packed, update_residual_compact
from veqpy.engine.numba_source import (
    _linear_uniform_interpolate_pair,
    _local_barycentric_interpolate_pair,
    _materialize_profile_owned_psin_source_impl,
    _materialize_projected_source_inputs_impl,
    _uniform_barycentric_weights,
    _update_fixed_point_psin_query_impl,
    _update_fourier_family_fields_impl,
    resolve_source_scratch_kernel,
)

if TYPE_CHECKING:
    from veqpy.operator.layouts import ResidualBindingLayout, RuntimeLayout, StaticLayout
    from veqpy.operator.source_setup import SourcePlan


@njit(cache=True, fastmath=True, nogil=True)
def _apply_f2_linear_fields_impl(fields: np.ndarray, scale: float, eps: float = 1.0e-10) -> None:
    nr = fields.shape[1]
    for i in range(nr):
        H = fields[0, i]
        H_r = fields[1, i]
        H_rr = fields[2, i]
        q = 1.0 + H
        if q < eps:
            q = eps
        sqrt_q = np.sqrt(q)
        inv_sqrt_q = 1.0 / sqrt_q
        inv_q_sqrt_q = inv_sqrt_q / q
        fields[0, i] = scale * sqrt_q
        fields[1, i] = scale * 0.5 * H_r * inv_sqrt_q
        fields[2, i] = scale * (0.5 * H_rr * inv_sqrt_q - 0.25 * H_r * H_r * inv_q_sqrt_q)


def _normalize_psin_query(out: np.ndarray, source: np.ndarray) -> None:
    np.copyto(out, np.asarray(source, dtype=np.float64))
    offset = float(out[0])
    scale = float(out[-1] - offset)
    if abs(scale) < 1.0e-12:
        raise ValueError("psin query does not span a valid normalized flux interval")
    out -= offset
    out /= scale
    out[0] = 0.0
    out[-1] = 1.0


@njit(cache=True, nogil=True)
def _call_source_kernel_with_scratch(
    scratch_source_kernel,
    psin: np.ndarray,
    psin_r: np.ndarray,
    psin_rr: np.ndarray,
    FFn_psin: np.ndarray,
    Pn_psin: np.ndarray,
    materialized_heat_input: np.ndarray,
    materialized_current_input: np.ndarray,
    coordinate_code: int,
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
    R_surface: np.ndarray,
    JdivR: np.ndarray,
    F_profile_u: np.ndarray,
    Ip: float,
    beta: float,
    source_scratch_1d: np.ndarray,
) -> tuple[float, float]:
    return scratch_source_kernel(
        psin,
        psin_r,
        psin_rr,
        FFn_psin,
        Pn_psin,
        materialized_heat_input,
        materialized_current_input,
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
        R_surface,
        JdivR,
        F_profile_u,
        Ip,
        beta,
        source_scratch_1d,
    )


@njit(cache=True, nogil=True)
def _run_fixed_point_barycentric_with_scratch_impl(
    scratch_source_kernel,
    max_iter: int,
    tolerance: float,
    source_psin_query: np.ndarray,
    psin: np.ndarray,
    psin_r: np.ndarray,
    psin_rr: np.ndarray,
    FFn_psin: np.ndarray,
    Pn_psin: np.ndarray,
    materialized_heat_input: np.ndarray,
    materialized_current_input: np.ndarray,
    heat_input: np.ndarray,
    current_input: np.ndarray,
    barycentric_weights: np.ndarray,
    coordinate_code: int,
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
    R_surface: np.ndarray,
    JdivR: np.ndarray,
    F_profile_u: np.ndarray,
    Ip: float,
    beta: float,
    source_scratch_1d: np.ndarray,
) -> tuple[float, float]:
    alpha1 = np.nan
    alpha2 = np.nan
    for _ in range(max_iter):
        _local_barycentric_interpolate_pair(
            materialized_heat_input,
            materialized_current_input,
            heat_input,
            current_input,
            source_psin_query,
            barycentric_weights,
        )
        alpha1, alpha2 = _call_source_kernel_with_scratch(
            scratch_source_kernel,
            psin,
            psin_r,
            psin_rr,
            FFn_psin,
            Pn_psin,
            materialized_heat_input,
            materialized_current_input,
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
            R_surface,
            JdivR,
            F_profile_u,
            Ip,
            beta,
            source_scratch_1d,
        )
        if _update_fixed_point_psin_query_impl(source_psin_query, psin, tolerance):
            break
    return alpha1, alpha2


@njit(cache=True, nogil=True)
def _run_projected_finalize_with_scratch_impl(
    scratch_source_kernel,
    finalize_iter: int,
    tolerance: float,
    source_psin_query: np.ndarray,
    psin: np.ndarray,
    psin_r: np.ndarray,
    psin_rr: np.ndarray,
    FFn_psin: np.ndarray,
    Pn_psin: np.ndarray,
    materialized_heat_input: np.ndarray,
    materialized_current_input: np.ndarray,
    heat_projection_coeff: np.ndarray,
    current_projection_coeff: np.ndarray,
    current_input: np.ndarray,
    projection_domain_code: int,
    endpoint_policy_code: int,
    endpoint_blend: np.ndarray,
    coordinate_code: int,
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
    R_surface: np.ndarray,
    JdivR: np.ndarray,
    F_profile_u: np.ndarray,
    Ip: float,
    beta: float,
    source_scratch_1d: np.ndarray,
) -> tuple[float, float]:
    for i in range(source_psin_query.shape[0]):
        source_psin_query[i] = psin[i]

    alpha1 = np.nan
    alpha2 = np.nan
    for _ in range(finalize_iter):
        _materialize_projected_source_inputs_impl(
            materialized_heat_input,
            materialized_current_input,
            heat_projection_coeff,
            current_projection_coeff,
            current_input,
            source_psin_query,
            projection_domain_code,
            endpoint_policy_code,
            endpoint_blend,
        )
        alpha1, alpha2 = _call_source_kernel_with_scratch(
            scratch_source_kernel,
            psin,
            psin_r,
            psin_rr,
            FFn_psin,
            Pn_psin,
            materialized_heat_input,
            materialized_current_input,
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
            R_surface,
            JdivR,
            F_profile_u,
            Ip,
            beta,
            source_scratch_1d,
        )
        if _update_fixed_point_psin_query_impl(source_psin_query, psin, tolerance):
            break
    return alpha1, alpha2


def bind_fused_residual_runner(
    *,
    source_plan: SourcePlan,
    static_layout: StaticLayout,
    residual_binding_layout: ResidualBindingLayout,
    runtime_layout: RuntimeLayout,
    alpha_state: np.ndarray,
    c_active_order: int,
    s_active_order: int,
    a: float,
    R0: float,
    Z0: float,
    B0: float,
    f_parameterization: str = "direct_F",
) -> Callable[[np.ndarray], np.ndarray]:
    if source_plan.is_single_pass:
        return bind_fused_single_pass_residual_runner(
            source_plan=source_plan,
            static_layout=static_layout,
            residual_binding_layout=residual_binding_layout,
            runtime_layout=runtime_layout,
            alpha_state=alpha_state,
            c_active_order=c_active_order,
            s_active_order=s_active_order,
            a=a,
            R0=R0,
            Z0=Z0,
            B0=B0,
            f_parameterization=f_parameterization,
        )
    if source_plan.is_profile_owned_psin:
        return bind_fused_profile_owned_psin_residual_runner(
            source_plan=source_plan,
            static_layout=static_layout,
            residual_binding_layout=residual_binding_layout,
            runtime_layout=runtime_layout,
            alpha_state=alpha_state,
            c_active_order=c_active_order,
            s_active_order=s_active_order,
            a=a,
            R0=R0,
            Z0=Z0,
            B0=B0,
            f_parameterization=f_parameterization,
        )
    if source_plan.is_fixed_point_psin:
        return bind_fused_fixed_point_psin_residual_runner(
            source_plan=source_plan,
            static_layout=static_layout,
            residual_binding_layout=residual_binding_layout,
            runtime_layout=runtime_layout,
            alpha_state=alpha_state,
            c_active_order=c_active_order,
            s_active_order=s_active_order,
            a=a,
            R0=R0,
            Z0=Z0,
            B0=B0,
            f_parameterization=f_parameterization,
        )
    raise ValueError(f"Unsupported source strategy {source_plan.strategy!r}")


def bind_fused_single_pass_residual_runner(
    *,
    source_plan: SourcePlan,
    static_layout: StaticLayout,
    residual_binding_layout: ResidualBindingLayout,
    runtime_layout: RuntimeLayout,
    alpha_state: np.ndarray,
    c_active_order: int,
    s_active_order: int,
    a: float,
    R0: float,
    Z0: float,
    B0: float,
    f_parameterization: str = "direct_F",
) -> Callable[[np.ndarray], np.ndarray]:
    coeff_index_rows = runtime_layout.active_coeff_index_rows
    lengths = runtime_layout.active_lengths
    active_u_fields = runtime_layout.active_u_fields
    active_rp_fields = runtime_layout.active_rp_fields
    active_env_fields = runtime_layout.active_env_fields
    active_offsets = runtime_layout.active_offsets
    active_scales = runtime_layout.active_scales
    c_family_fields = runtime_layout.c_family_fields
    s_family_fields = runtime_layout.s_family_fields
    c_family_base_fields = runtime_layout.c_family_base_fields
    s_family_base_fields = runtime_layout.s_family_base_fields
    c_source_slots = runtime_layout.c_family_source_slots
    s_source_slots = runtime_layout.s_family_source_slots
    surface_workspace = runtime_layout.geometry_surface_workspace
    radial_workspace = runtime_layout.geometry_radial_workspace
    compact_sin_tb = surface_workspace[0]
    compact_R = surface_workspace[1]
    compact_R_t = surface_workspace[2]
    compact_Z_t = surface_workspace[3]
    compact_J = surface_workspace[4]
    compact_JdivR = surface_workspace[5]
    compact_grtdivJR_t = surface_workspace[6]
    compact_gttdivJR = surface_workspace[7]
    compact_gttdivJR_r = surface_workspace[8]
    S_r = radial_workspace[0]
    V_r = radial_workspace[1]
    Kn = radial_workspace[2]
    Kn_r = radial_workspace[3]
    Ln_r = radial_workspace[4]
    T_fields = static_layout.T_fields
    rho = static_layout.rho
    theta = static_layout.theta
    cos_ktheta = static_layout.cos_ktheta
    sin_ktheta = static_layout.sin_ktheta
    k_cos_ktheta = static_layout.k_cos_ktheta
    k_sin_ktheta = static_layout.k_sin_ktheta
    k2_cos_ktheta = static_layout.k2_cos_ktheta
    k2_sin_ktheta = static_layout.k2_sin_ktheta
    weights = static_layout.weights
    differentiation_matrix = static_layout.differentiation_matrix
    integration_matrix = static_layout.integration_matrix
    rho_powers = static_layout.rho_powers
    y = static_layout.y
    root_fields = runtime_layout.root_fields
    residual_fields = runtime_layout.residual_fields
    packed_residual = runtime_layout.packed_residual
    materialized_heat_input = runtime_layout.materialized_heat_input
    materialized_current_input = runtime_layout.materialized_current_input
    source_scratch_1d = runtime_layout.source_scratch_1d
    profiles_by_name = runtime_layout.profiles_by_name
    h_fields = profiles_by_name["h"].u_fields
    v_fields = profiles_by_name["v"].u_fields
    k_fields = profiles_by_name["k"].u_fields
    F_profile_u = profiles_by_name["F"].u
    F_profile_fields = profiles_by_name["F"].u_fields
    apply_f2_linear = f_parameterization == "F2_linear"
    source_kernel = source_plan.kernel
    source_kernel_name = getattr(source_kernel, "__name__", "")
    enforce_source_psin_consistency = source_kernel_name == "update_PJ2_PSIN"
    coordinate_code = int(source_plan.coordinate_code)
    Ip = float(source_plan.Ip)
    beta = float(source_plan.beta)
    block_codes = residual_binding_layout.active_residual_block_codes
    block_orders = residual_binding_layout.active_residual_block_orders
    scratch_source_kernel = resolve_source_scratch_kernel(source_kernel)
    scratch_holder: list[np.ndarray | None] = [None]
    psin = root_fields[0]
    psin_r = root_fields[1]
    psin_rr = root_fields[2]
    FFn_psin = root_fields[3]
    Pn_psin = root_fields[4]

    def runner(x: np.ndarray) -> np.ndarray:
        update_profiles_packed_bulk(
            active_u_fields,
            T_fields,
            active_rp_fields,
            active_env_fields,
            active_offsets,
            active_scales,
            x,
            coeff_index_rows,
            lengths,
        )
        if apply_f2_linear:
            _apply_f2_linear_fields_impl(F_profile_fields, R0 * B0)
        _update_fourier_family_fields_impl(
            c_family_fields,
            s_family_fields,
            c_family_base_fields,
            s_family_base_fields,
            active_u_fields,
            c_source_slots,
            s_source_slots,
            c_active_order,
            s_active_order,
        )
        update_geometry_hot(
            compact_sin_tb,
            compact_R,
            compact_R_t,
            compact_Z_t,
            compact_J,
            compact_JdivR,
            compact_grtdivJR_t,
            compact_gttdivJR,
            compact_gttdivJR_r,
            S_r,
            V_r,
            Kn,
            Kn_r,
            Ln_r,
            a,
            R0,
            Z0,
            rho,
            theta,
            cos_ktheta,
            sin_ktheta,
            k_cos_ktheta,
            k_sin_ktheta,
            k2_cos_ktheta,
            k2_sin_ktheta,
            h_fields,
            v_fields,
            k_fields,
            c_family_fields,
            s_family_fields,
            c_active_order,
            s_active_order,
        )
        if scratch_source_kernel is None:
            alpha1, alpha2 = source_kernel(
                psin,
                psin_r,
                psin_rr,
                FFn_psin,
                Pn_psin,
                materialized_heat_input,
                materialized_current_input,
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
                compact_R,
                compact_JdivR,
                F_profile_u,
                Ip,
                beta,
            )
        else:
            alpha1, alpha2 = scratch_source_kernel(
                psin,
                psin_r,
                psin_rr,
                FFn_psin,
                Pn_psin,
                materialized_heat_input,
                materialized_current_input,
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
                compact_R,
                compact_JdivR,
                F_profile_u,
                Ip,
                beta,
                source_scratch_1d,
            )
        alpha_state[0] = alpha1
        alpha_state[1] = alpha2
        update_residual_compact(
            residual_fields,
            alpha1,
            alpha2,
            root_fields,
            compact_R,
            compact_R_t,
            compact_Z_t,
            compact_J,
            compact_JdivR,
            compact_grtdivJR_t,
            compact_gttdivJR,
            compact_gttdivJR_r,
        )
        packed_residual.fill(0.0)
        scratch = scratch_holder[0]
        nr = residual_fields.shape[1]
        if scratch is None or scratch.shape[0] != nr:
            scratch = np.empty(nr, dtype=np.float64)
            scratch_holder[0] = scratch
        _run_residual_blocks_packed(
            packed_residual,
            scratch,
            block_codes,
            block_orders,
            coeff_index_rows,
            lengths,
            residual_fields[2],
            residual_fields[0],
            residual_fields[1],
            compact_sin_tb,
            sin_ktheta,
            cos_ktheta,
            rho_powers,
            y,
            T_fields[0],
            weights,
            a,
            R0,
            B0,
        )
        return packed_residual.copy()

    return runner


def bind_fused_profile_owned_psin_residual_runner(
    *,
    source_plan: SourcePlan,
    static_layout: StaticLayout,
    residual_binding_layout: ResidualBindingLayout,
    runtime_layout: RuntimeLayout,
    alpha_state: np.ndarray,
    c_active_order: int,
    s_active_order: int,
    a: float,
    R0: float,
    Z0: float,
    B0: float,
    f_parameterization: str = "direct_F",
) -> Callable[[np.ndarray], np.ndarray]:
    coeff_index_rows = runtime_layout.active_coeff_index_rows
    lengths = runtime_layout.active_lengths
    active_u_fields = runtime_layout.active_u_fields
    active_rp_fields = runtime_layout.active_rp_fields
    active_env_fields = runtime_layout.active_env_fields
    active_offsets = runtime_layout.active_offsets
    active_scales = runtime_layout.active_scales
    c_family_fields = runtime_layout.c_family_fields
    s_family_fields = runtime_layout.s_family_fields
    c_family_base_fields = runtime_layout.c_family_base_fields
    s_family_base_fields = runtime_layout.s_family_base_fields
    c_source_slots = runtime_layout.c_family_source_slots
    s_source_slots = runtime_layout.s_family_source_slots
    surface_workspace = runtime_layout.geometry_surface_workspace
    radial_workspace = runtime_layout.geometry_radial_workspace
    compact_sin_tb = surface_workspace[0]
    compact_R = surface_workspace[1]
    compact_R_t = surface_workspace[2]
    compact_Z_t = surface_workspace[3]
    compact_J = surface_workspace[4]
    compact_JdivR = surface_workspace[5]
    compact_grtdivJR_t = surface_workspace[6]
    compact_gttdivJR = surface_workspace[7]
    compact_gttdivJR_r = surface_workspace[8]
    S_r = radial_workspace[0]
    V_r = radial_workspace[1]
    Kn = radial_workspace[2]
    Kn_r = radial_workspace[3]
    Ln_r = radial_workspace[4]
    T_fields = static_layout.T_fields
    rho = static_layout.rho
    theta = static_layout.theta
    cos_ktheta = static_layout.cos_ktheta
    sin_ktheta = static_layout.sin_ktheta
    k_cos_ktheta = static_layout.k_cos_ktheta
    k_sin_ktheta = static_layout.k_sin_ktheta
    k2_cos_ktheta = static_layout.k2_cos_ktheta
    k2_sin_ktheta = static_layout.k2_sin_ktheta
    weights = static_layout.weights
    differentiation_matrix = static_layout.differentiation_matrix
    integration_matrix = static_layout.integration_matrix
    rho_powers = static_layout.rho_powers
    y = static_layout.y
    root_fields = runtime_layout.root_fields
    source_target_root_fields = runtime_layout.source_target_root_fields
    residual_fields = runtime_layout.residual_fields
    packed_residual = runtime_layout.packed_residual
    source_psin_query = runtime_layout.source_psin_query
    source_parameter_query = runtime_layout.source_parameter_query
    heat_projection_coeff = runtime_layout.source_heat_projection_coeff
    current_projection_coeff = runtime_layout.source_current_projection_coeff
    endpoint_blend = runtime_layout.source_endpoint_blend
    materialized_heat_input = runtime_layout.materialized_heat_input
    materialized_current_input = runtime_layout.materialized_current_input
    source_scratch_1d = runtime_layout.source_scratch_1d
    profiles_by_name = runtime_layout.profiles_by_name
    psin_profile_fields = profiles_by_name["psin"].u_fields
    h_fields = profiles_by_name["h"].u_fields
    v_fields = profiles_by_name["v"].u_fields
    k_fields = profiles_by_name["k"].u_fields
    F_profile_u = profiles_by_name["F"].u
    F_profile_fields = profiles_by_name["F"].u_fields
    apply_f2_linear = f_parameterization == "F2_linear"
    source_kernel = source_plan.kernel
    coordinate_code = int(source_plan.coordinate_code)
    parameterization_code = int(source_plan.parameterization_code)
    has_projection_policy = bool(source_plan.has_projection_policy)
    source_kernel_name = getattr(source_kernel, "__name__", "")
    skip_projection_finalize = source_kernel_name == "update_PI_PSIN"
    projection_domain_code = int(source_plan.projection_domain_code)
    endpoint_policy_code = int(source_plan.endpoint_policy_code)
    heat_input = source_plan.heat_input
    current_input = source_plan.current_input
    Ip = float(source_plan.Ip)
    beta = float(source_plan.beta)
    block_codes = residual_binding_layout.active_residual_block_codes
    block_orders = residual_binding_layout.active_residual_block_orders
    scratch_source_kernel = resolve_source_scratch_kernel(source_kernel)
    scratch_holder: list[np.ndarray | None] = [None]
    psin = root_fields[0]
    psin_r = root_fields[1]
    psin_rr = root_fields[2]
    FFn_psin = root_fields[3]
    Pn_psin = root_fields[4]

    def runner(x: np.ndarray) -> np.ndarray:
        update_profiles_packed_bulk(
            active_u_fields,
            T_fields,
            active_rp_fields,
            active_env_fields,
            active_offsets,
            active_scales,
            x,
            coeff_index_rows,
            lengths,
        )
        if apply_f2_linear:
            _apply_f2_linear_fields_impl(F_profile_fields, R0 * B0)
        _update_fourier_family_fields_impl(
            c_family_fields,
            s_family_fields,
            c_family_base_fields,
            s_family_base_fields,
            active_u_fields,
            c_source_slots,
            s_source_slots,
            c_active_order,
            s_active_order,
        )
        update_geometry_hot(
            compact_sin_tb,
            compact_R,
            compact_R_t,
            compact_Z_t,
            compact_J,
            compact_JdivR,
            compact_grtdivJR_t,
            compact_gttdivJR,
            compact_gttdivJR_r,
            S_r,
            V_r,
            Kn,
            Kn_r,
            Ln_r,
            a,
            R0,
            Z0,
            rho,
            theta,
            cos_ktheta,
            sin_ktheta,
            k_cos_ktheta,
            k_sin_ktheta,
            k2_cos_ktheta,
            k2_sin_ktheta,
            h_fields,
            v_fields,
            k_fields,
            c_family_fields,
            s_family_fields,
            c_active_order,
            s_active_order,
        )
        _materialize_profile_owned_psin_source_impl(
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
        # PI psin-uniform is more accurate with the direct source-owned interpolation
        # than with the extra projected rematerialization used by other routes.
        if has_projection_policy and not skip_projection_finalize:
            _materialize_projected_source_inputs_impl(
                materialized_heat_input,
                materialized_current_input,
                heat_projection_coeff,
                current_projection_coeff,
                current_input,
                source_psin_query,
                projection_domain_code,
                endpoint_policy_code,
                endpoint_blend,
            )
        if scratch_source_kernel is None:
            alpha1, alpha2 = source_kernel(
                source_target_root_fields[0],
                source_target_root_fields[1],
                source_target_root_fields[2],
                FFn_psin,
                Pn_psin,
                materialized_heat_input,
                materialized_current_input,
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
                compact_R,
                compact_JdivR,
                F_profile_u,
                Ip,
                beta,
            )
        else:
            alpha1, alpha2 = scratch_source_kernel(
                source_target_root_fields[0],
                source_target_root_fields[1],
                source_target_root_fields[2],
                FFn_psin,
                Pn_psin,
                materialized_heat_input,
                materialized_current_input,
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
                compact_R,
                compact_JdivR,
                F_profile_u,
                Ip,
                beta,
                source_scratch_1d,
            )
        alpha_state[0] = alpha1
        alpha_state[1] = alpha2
        update_residual_compact(
            residual_fields,
            alpha1,
            alpha2,
            root_fields,
            compact_R,
            compact_R_t,
            compact_Z_t,
            compact_J,
            compact_JdivR,
            compact_grtdivJR_t,
            compact_gttdivJR,
            compact_gttdivJR_r,
        )
        packed_residual.fill(0.0)
        scratch = scratch_holder[0]
        nr = residual_fields.shape[1]
        if scratch is None or scratch.shape[0] != nr:
            scratch = np.empty(nr, dtype=np.float64)
            scratch_holder[0] = scratch
        _run_residual_blocks_packed(
            packed_residual,
            scratch,
            block_codes,
            block_orders,
            coeff_index_rows,
            lengths,
            residual_fields[2],
            residual_fields[0],
            residual_fields[1],
            compact_sin_tb,
            sin_ktheta,
            cos_ktheta,
            rho_powers,
            y,
            T_fields[0],
            weights,
            a,
            R0,
            B0,
        )
        return packed_residual.copy()

    return runner


def bind_fused_fixed_point_psin_residual_runner(
    *,
    source_plan: SourcePlan,
    static_layout: StaticLayout,
    residual_binding_layout: ResidualBindingLayout,
    runtime_layout: RuntimeLayout,
    alpha_state: np.ndarray,
    c_active_order: int,
    s_active_order: int,
    a: float,
    R0: float,
    Z0: float,
    B0: float,
    f_parameterization: str = "direct_F",
    max_iter: int | None = None,
    tolerance: float | None = None,
) -> Callable[[np.ndarray], np.ndarray]:
    coeff_index_rows = runtime_layout.active_coeff_index_rows
    lengths = runtime_layout.active_lengths
    active_u_fields = runtime_layout.active_u_fields
    active_rp_fields = runtime_layout.active_rp_fields
    active_env_fields = runtime_layout.active_env_fields
    active_offsets = runtime_layout.active_offsets
    active_scales = runtime_layout.active_scales
    c_family_fields = runtime_layout.c_family_fields
    s_family_fields = runtime_layout.s_family_fields
    c_family_base_fields = runtime_layout.c_family_base_fields
    s_family_base_fields = runtime_layout.s_family_base_fields
    c_source_slots = runtime_layout.c_family_source_slots
    s_source_slots = runtime_layout.s_family_source_slots
    surface_workspace = runtime_layout.geometry_surface_workspace
    radial_workspace = runtime_layout.geometry_radial_workspace
    compact_sin_tb = surface_workspace[0]
    compact_R = surface_workspace[1]
    compact_R_t = surface_workspace[2]
    compact_Z_t = surface_workspace[3]
    compact_J = surface_workspace[4]
    compact_JdivR = surface_workspace[5]
    compact_grtdivJR_t = surface_workspace[6]
    compact_gttdivJR = surface_workspace[7]
    compact_gttdivJR_r = surface_workspace[8]
    S_r = radial_workspace[0]
    V_r = radial_workspace[1]
    Kn = radial_workspace[2]
    Kn_r = radial_workspace[3]
    Ln_r = radial_workspace[4]
    T_fields = static_layout.T_fields
    rho = static_layout.rho
    theta = static_layout.theta
    cos_ktheta = static_layout.cos_ktheta
    sin_ktheta = static_layout.sin_ktheta
    k_cos_ktheta = static_layout.k_cos_ktheta
    k_sin_ktheta = static_layout.k_sin_ktheta
    k2_cos_ktheta = static_layout.k2_cos_ktheta
    k2_sin_ktheta = static_layout.k2_sin_ktheta
    weights = static_layout.weights
    differentiation_matrix = static_layout.differentiation_matrix
    integration_matrix = static_layout.integration_matrix
    rho_powers = static_layout.rho_powers
    y = static_layout.y
    root_fields = runtime_layout.root_fields
    residual_fields = runtime_layout.residual_fields
    packed_residual = runtime_layout.packed_residual
    source_psin_query = runtime_layout.source_psin_query
    materialized_heat_input = runtime_layout.materialized_heat_input
    materialized_current_input = runtime_layout.materialized_current_input
    source_scratch_1d = runtime_layout.source_scratch_1d
    heat_projection_coeff = runtime_layout.source_heat_projection_coeff
    current_projection_coeff = runtime_layout.source_current_projection_coeff
    endpoint_blend = runtime_layout.source_endpoint_blend
    profiles_by_name = runtime_layout.profiles_by_name
    h_fields = profiles_by_name["h"].u_fields
    v_fields = profiles_by_name["v"].u_fields
    k_fields = profiles_by_name["k"].u_fields
    F_profile_u = profiles_by_name["F"].u
    F_profile_fields = profiles_by_name["F"].u_fields
    apply_f2_linear = f_parameterization == "F2_linear"
    source_kernel = source_plan.kernel
    coordinate_code = int(source_plan.coordinate_code)
    heat_input = source_plan.heat_input
    current_input = source_plan.current_input
    Ip = float(source_plan.Ip)
    beta = float(source_plan.beta)
    max_iter = int(source_plan.fixed_point_max_iter if max_iter is None else max_iter)
    finalize_max_iter = int(source_plan.fixed_point_finalize_max_iter)
    tolerance = float(source_plan.fixed_point_tolerance if tolerance is None else tolerance)
    has_Ip = bool(np.isfinite(Ip))
    source_kernel_name = getattr(source_kernel, "__name__", "")
    use_projected_finalize = bool(source_plan.use_projected_finalize)
    projection_domain_code = int(source_plan.projection_domain_code)
    endpoint_policy_code = int(source_plan.endpoint_policy_code)
    allow_query_warmstart = bool(source_plan.allow_query_warmstart)
    # PQ's q-profile is more stable with a slightly narrower local stencil.
    fixed_point_stencil_size = 4 if source_kernel_name == "update_PQ_PSIN" else 8
    fixed_point_barycentric_weights = _uniform_barycentric_weights(
        min(fixed_point_stencil_size, int(source_plan.n_src))
    )
    block_codes = residual_binding_layout.active_residual_block_codes
    block_orders = residual_binding_layout.active_residual_block_orders
    scratch_source_kernel = resolve_source_scratch_kernel(source_kernel)
    scratch_holder: list[np.ndarray | None] = [None]
    psin = root_fields[0]
    psin_r = root_fields[1]
    psin_rr = root_fields[2]
    FFn_psin = root_fields[3]
    Pn_psin = root_fields[4]
    R_surface = compact_R
    JdivR = compact_JdivR

    if scratch_source_kernel is None:

        def run_main_fixed_point() -> tuple[float, float]:
            alpha1 = np.nan
            alpha2 = np.nan
            for _ in range(max_iter):
                if has_Ip:
                    _local_barycentric_interpolate_pair(
                        materialized_heat_input,
                        materialized_current_input,
                        heat_input,
                        current_input,
                        source_psin_query,
                        fixed_point_barycentric_weights,
                    )
                else:
                    _linear_uniform_interpolate_pair(
                        materialized_heat_input,
                        materialized_current_input,
                        heat_input,
                        current_input,
                        source_psin_query,
                    )
                alpha1, alpha2 = source_kernel(
                    psin,
                    psin_r,
                    psin_rr,
                    FFn_psin,
                    Pn_psin,
                    materialized_heat_input,
                    materialized_current_input,
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
                    R_surface,
                    JdivR,
                    F_profile_u,
                    Ip,
                    beta,
                )
                if _update_fixed_point_psin_query_impl(source_psin_query, psin, tolerance):
                    break
            return alpha1, alpha2

        def run_projected_finalize() -> tuple[float, float]:
            np.copyto(source_psin_query, psin)
            alpha1 = np.nan
            alpha2 = np.nan
            for _ in range(finalize_max_iter):
                _materialize_projected_source_inputs_impl(
                    materialized_heat_input,
                    materialized_current_input,
                    heat_projection_coeff,
                    current_projection_coeff,
                    current_input,
                    source_psin_query,
                    projection_domain_code,
                    endpoint_policy_code,
                    endpoint_blend,
                )
                alpha1, alpha2 = source_kernel(
                    psin,
                    psin_r,
                    psin_rr,
                    FFn_psin,
                    Pn_psin,
                    materialized_heat_input,
                    materialized_current_input,
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
                    R_surface,
                    JdivR,
                    F_profile_u,
                    Ip,
                    beta,
                )
                if _update_fixed_point_psin_query_impl(source_psin_query, psin, tolerance):
                    break
            return alpha1, alpha2

    else:
        if has_Ip:

            def run_main_fixed_point() -> tuple[float, float]:
                return _run_fixed_point_barycentric_with_scratch_impl(
                    scratch_source_kernel,
                    max_iter,
                    tolerance,
                    source_psin_query,
                    psin,
                    psin_r,
                    psin_rr,
                    FFn_psin,
                    Pn_psin,
                    materialized_heat_input,
                    materialized_current_input,
                    heat_input,
                    current_input,
                    fixed_point_barycentric_weights,
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
                    R_surface,
                    JdivR,
                    F_profile_u,
                    Ip,
                    beta,
                    source_scratch_1d,
                )

        else:

            def run_main_fixed_point() -> tuple[float, float]:
                alpha1 = np.nan
                alpha2 = np.nan
                for _ in range(max_iter):
                    _linear_uniform_interpolate_pair(
                        materialized_heat_input,
                        materialized_current_input,
                        heat_input,
                        current_input,
                        source_psin_query,
                    )
                    alpha1, alpha2 = scratch_source_kernel(
                        psin,
                        psin_r,
                        psin_rr,
                        FFn_psin,
                        Pn_psin,
                        materialized_heat_input,
                        materialized_current_input,
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
                        R_surface,
                        JdivR,
                        F_profile_u,
                        Ip,
                        beta,
                        source_scratch_1d,
                    )
                    if _update_fixed_point_psin_query_impl(source_psin_query, psin, tolerance):
                        break
                return alpha1, alpha2

        def run_projected_finalize() -> tuple[float, float]:
            return _run_projected_finalize_with_scratch_impl(
                scratch_source_kernel,
                finalize_max_iter,
                tolerance,
                source_psin_query,
                psin,
                psin_r,
                psin_rr,
                FFn_psin,
                Pn_psin,
                materialized_heat_input,
                materialized_current_input,
                heat_projection_coeff,
                current_projection_coeff,
                current_input,
                projection_domain_code,
                endpoint_policy_code,
                endpoint_blend,
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
                R_surface,
                JdivR,
                F_profile_u,
                Ip,
                beta,
                source_scratch_1d,
            )

    def runner(x: np.ndarray) -> np.ndarray:
        update_profiles_packed_bulk(
            active_u_fields,
            T_fields,
            active_rp_fields,
            active_env_fields,
            active_offsets,
            active_scales,
            x,
            coeff_index_rows,
            lengths,
        )
        if apply_f2_linear:
            _apply_f2_linear_fields_impl(F_profile_fields, R0 * B0)
        _update_fourier_family_fields_impl(
            c_family_fields,
            s_family_fields,
            c_family_base_fields,
            s_family_base_fields,
            active_u_fields,
            c_source_slots,
            s_source_slots,
            c_active_order,
            s_active_order,
        )
        update_geometry_hot(
            compact_sin_tb,
            compact_R,
            compact_R_t,
            compact_Z_t,
            compact_J,
            compact_JdivR,
            compact_grtdivJR_t,
            compact_gttdivJR,
            compact_gttdivJR_r,
            S_r,
            V_r,
            Kn,
            Kn_r,
            Ln_r,
            a,
            R0,
            Z0,
            rho,
            theta,
            cos_ktheta,
            sin_ktheta,
            k_cos_ktheta,
            k_sin_ktheta,
            k2_cos_ktheta,
            k2_sin_ktheta,
            h_fields,
            v_fields,
            k_fields,
            c_family_fields,
            s_family_fields,
            c_active_order,
            s_active_order,
        )

        if (not allow_query_warmstart) or source_psin_query[0] < 0.0:
            _normalize_psin_query(source_psin_query, profiles_by_name["psin"].u)

        alpha1, alpha2 = run_main_fixed_point()

        if use_projected_finalize:
            alpha1, alpha2 = run_projected_finalize()

        alpha_state[0] = alpha1
        alpha_state[1] = alpha2
        update_residual_compact(
            residual_fields,
            alpha1,
            alpha2,
            root_fields,
            compact_R,
            compact_R_t,
            compact_Z_t,
            compact_J,
            compact_JdivR,
            compact_grtdivJR_t,
            compact_gttdivJR,
            compact_gttdivJR_r,
        )
        packed_residual.fill(0.0)
        scratch = scratch_holder[0]
        nr = residual_fields.shape[1]
        if scratch is None or scratch.shape[0] != nr:
            scratch = np.empty(nr, dtype=np.float64)
            scratch_holder[0] = scratch
        _run_residual_blocks_packed(
            packed_residual,
            scratch,
            block_codes,
            block_orders,
            coeff_index_rows,
            lengths,
            residual_fields[2],
            residual_fields[0],
            residual_fields[1],
            compact_sin_tb,
            sin_ktheta,
            cos_ktheta,
            rho_powers,
            y,
            T_fields[0],
            weights,
            a,
            R0,
            B0,
        )
        return packed_residual.copy()

    return runner

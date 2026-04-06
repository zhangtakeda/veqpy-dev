"""
Module: engine.numpy_operator

Role:
- 提供 numpy backend 下的 fused x -> residual hot runner.
- 把常见 route 的 stage A/B/C/D 串成单个 engine 绑定入口.

Public API:
- bind_fused_residual_runner
- bind_fused_single_pass_residual_runner
- bind_fused_profile_owned_psin_residual_runner
- bind_fused_fixed_point_psin_residual_runner

Notes:
- numpy 版本优先保持语义对齐.
- fixed-point psin 仍由上层保留 staged orchestration.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable

import numpy as np

from veqpy.engine.numpy_geometry import update_geometry
from veqpy.engine.numpy_profile import update_profiles_packed_bulk
from veqpy.engine.numpy_residual import bind_residual_stage_runner
from veqpy.engine.numpy_source import (
    materialize_profile_owned_psin_source,
    materialize_projected_source_inputs,
    resolve_source_inputs,
    update_fixed_point_psin_query,
    update_fourier_family_fields,
)

if TYPE_CHECKING:
    from veqpy.operator.layouts import ResidualBindingLayout, RuntimeLayout, SetupLayout, StaticLayout
    from veqpy.operator.plans import ResidualPlan


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


def bind_fused_residual_runner(
    *,
    residual_plan: ResidualPlan,
    static_layout: StaticLayout,
    residual_binding_layout: ResidualBindingLayout,
    setup_layout: SetupLayout,
    runtime_layout: RuntimeLayout,
    alpha_state: np.ndarray,
    c_active_order: int,
    s_active_order: int,
    a: float,
    R0: float,
    Z0: float,
    B0: float,
) -> Callable[[np.ndarray], np.ndarray]:
    if residual_plan.is_single_pass:
        return bind_fused_single_pass_residual_runner(
            residual_plan=residual_plan,
            static_layout=static_layout,
            residual_binding_layout=residual_binding_layout,
            setup_layout=setup_layout,
            runtime_layout=runtime_layout,
            alpha_state=alpha_state,
            c_active_order=c_active_order,
            s_active_order=s_active_order,
            a=a,
            R0=R0,
            Z0=Z0,
            B0=B0,
        )
    if residual_plan.is_profile_owned_psin:
        return bind_fused_profile_owned_psin_residual_runner(
            residual_plan=residual_plan,
            static_layout=static_layout,
            residual_binding_layout=residual_binding_layout,
            setup_layout=setup_layout,
            runtime_layout=runtime_layout,
            alpha_state=alpha_state,
            c_active_order=c_active_order,
            s_active_order=s_active_order,
            a=a,
            R0=R0,
            Z0=Z0,
            B0=B0,
        )
    if residual_plan.is_fixed_point_psin:
        return bind_fused_fixed_point_psin_residual_runner(
            residual_plan=residual_plan,
            static_layout=static_layout,
            residual_binding_layout=residual_binding_layout,
            setup_layout=setup_layout,
            runtime_layout=runtime_layout,
            alpha_state=alpha_state,
            c_active_order=c_active_order,
            s_active_order=s_active_order,
            a=a,
            R0=R0,
            Z0=Z0,
            B0=B0,
        )
    raise ValueError(f"Unsupported residual runner code {residual_plan.runner_code}")


def bind_fused_single_pass_residual_runner(
    *,
    residual_plan: ResidualPlan,
    static_layout: StaticLayout,
    residual_binding_layout: ResidualBindingLayout,
    setup_layout: SetupLayout,
    runtime_layout: RuntimeLayout,
    alpha_state: np.ndarray,
    c_active_order: int,
    s_active_order: int,
    a: float,
    R0: float,
    Z0: float,
    B0: float,
) -> Callable[[np.ndarray], np.ndarray]:
    source_plan = residual_plan.source_plan
    profile_names = residual_binding_layout.active_profile_names
    coeff_index_rows = runtime_layout.active_coeff_index_rows
    lengths = runtime_layout.active_lengths
    residual_stage_runner = bind_residual_stage_runner(profile_names, coeff_index_rows, lengths, setup_layout.x_size)
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
    geometry = runtime_layout.geometry
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
    profiles_by_name = runtime_layout.profiles_by_name
    h_fields = profiles_by_name["h"].u_fields
    v_fields = profiles_by_name["v"].u_fields
    k_fields = profiles_by_name["k"].u_fields
    F_profile_u = profiles_by_name["F"].u
    source_kernel = source_plan.kernel
    coordinate_code = int(source_plan.coordinate_code)
    Ip = float(source_plan.Ip)
    beta = float(source_plan.beta)
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
        update_fourier_family_fields(
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
            weights,
            h_fields,
            v_fields,
            k_fields,
            c_family_fields,
            s_family_fields,
            c_active_order,
            s_active_order,
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
            R_fields[0],
            J_fields[6],
            F_profile_u,
            Ip,
            beta,
        )
        alpha_state[0] = alpha1
        alpha_state[1] = alpha2
        residual_stage_runner(
            packed_residual,
            residual_fields,
            alpha1,
            alpha2,
            root_fields,
            R_fields,
            Z_fields,
            J_fields,
            g_fields,
            tb_fields[7],
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
    residual_plan: ResidualPlan,
    static_layout: StaticLayout,
    residual_binding_layout: ResidualBindingLayout,
    setup_layout: SetupLayout,
    runtime_layout: RuntimeLayout,
    alpha_state: np.ndarray,
    c_active_order: int,
    s_active_order: int,
    a: float,
    R0: float,
    Z0: float,
    B0: float,
) -> Callable[[np.ndarray], np.ndarray]:
    source_plan = residual_plan.source_plan
    profile_names = residual_binding_layout.active_profile_names
    coeff_index_rows = runtime_layout.active_coeff_index_rows
    lengths = runtime_layout.active_lengths
    residual_stage_runner = bind_residual_stage_runner(profile_names, coeff_index_rows, lengths, setup_layout.x_size)
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
    geometry = runtime_layout.geometry
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
    profiles_by_name = runtime_layout.profiles_by_name
    psin_profile_fields = profiles_by_name["psin"].u_fields
    h_fields = profiles_by_name["h"].u_fields
    v_fields = profiles_by_name["v"].u_fields
    k_fields = profiles_by_name["k"].u_fields
    F_profile_u = profiles_by_name["F"].u
    source_kernel = source_plan.kernel
    coordinate_code = int(source_plan.coordinate_code)
    parameterization_code = int(source_plan.parameterization_code)
    has_projection_policy = bool(source_plan.has_projection_policy)
    projection_domain_code = int(source_plan.projection_domain_code)
    endpoint_policy_code = int(source_plan.endpoint_policy_code)
    heat_input = source_plan.heat_input
    current_input = source_plan.current_input
    Ip = float(source_plan.Ip)
    beta = float(source_plan.beta)
    if psin_profile_fields is None:
        raise RuntimeError("psin_profile runtime fields are not initialized")
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
        update_fourier_family_fields(
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
            weights,
            h_fields,
            v_fields,
            k_fields,
            c_family_fields,
            s_family_fields,
            c_active_order,
            s_active_order,
        )
        materialize_profile_owned_psin_source(
            root_fields[0],
            root_fields[1],
            root_fields[2],
            source_psin_query,
            source_parameter_query,
            materialized_heat_input,
            materialized_current_input,
            psin_profile_fields,
            heat_input,
            current_input,
            parameterization_code,
        )
    if has_projection_policy:
        materialize_projected_source_inputs(
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
            R_fields[0],
            J_fields[6],
            F_profile_u,
            Ip,
            beta,
        )
        alpha_state[0] = alpha1
        alpha_state[1] = alpha2
        residual_stage_runner(
            packed_residual,
            residual_fields,
            alpha1,
            alpha2,
            root_fields,
            R_fields,
            Z_fields,
            J_fields,
            g_fields,
            tb_fields[7],
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
    residual_plan: ResidualPlan,
    static_layout: StaticLayout,
    residual_binding_layout: ResidualBindingLayout,
    setup_layout: SetupLayout,
    runtime_layout: RuntimeLayout,
    alpha_state: np.ndarray,
    c_active_order: int,
    s_active_order: int,
    a: float,
    R0: float,
    Z0: float,
    B0: float,
    max_iter: int = 8,
    tolerance: float = 1.0e-10,
) -> Callable[[np.ndarray], np.ndarray]:
    source_plan = residual_plan.source_plan
    profile_names = residual_binding_layout.active_profile_names
    coeff_index_rows = runtime_layout.active_coeff_index_rows
    lengths = runtime_layout.active_lengths
    residual_stage_runner = bind_residual_stage_runner(profile_names, coeff_index_rows, lengths, setup_layout.x_size)
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
    geometry = runtime_layout.geometry
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
    source_barycentric_weights = runtime_layout.source_barycentric_weights
    source_fixed_remap_matrix = runtime_layout.source_fixed_remap_matrix
    heat_projection_coeff = runtime_layout.source_heat_projection_coeff
    current_projection_coeff = runtime_layout.source_current_projection_coeff
    endpoint_blend = runtime_layout.source_endpoint_blend
    profiles_by_name = runtime_layout.profiles_by_name
    h_fields = profiles_by_name["h"].u_fields
    v_fields = profiles_by_name["v"].u_fields
    k_fields = profiles_by_name["k"].u_fields
    F_profile_u = profiles_by_name["F"].u
    source_kernel = source_plan.kernel
    coordinate_code = int(source_plan.coordinate_code)
    heat_input = source_plan.heat_input
    current_input = source_plan.current_input
    source_n_src = int(source_plan.n_src)
    Ip = float(source_plan.Ip)
    beta = float(source_plan.beta)
    use_projected_finalize = bool(source_plan.use_projected_finalize)
    projection_domain_code = int(source_plan.projection_domain_code)
    endpoint_policy_code = int(source_plan.endpoint_policy_code)
    allow_query_warmstart = bool(source_plan.allow_query_warmstart)

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
        update_fourier_family_fields(
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
            weights,
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

        alpha1 = np.nan
        alpha2 = np.nan
        for _ in range(max_iter):
            resolve_source_inputs(
                materialized_heat_input,
                materialized_current_input,
                heat_input,
                current_input,
                coordinate_code,
                source_n_src,
                source_barycentric_weights,
                source_fixed_remap_matrix,
                source_psin_query,
            )
            alpha1, alpha2 = source_kernel(
                root_fields[0],
                root_fields[1],
                root_fields[2],
                root_fields[3],
                root_fields[4],
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
                R_fields[0],
                J_fields[6],
                F_profile_u,
                Ip,
                beta,
            )
            if update_fixed_point_psin_query(source_psin_query, root_fields[0], tolerance):
                break

        if use_projected_finalize:
            np.copyto(source_psin_query, root_fields[0])
            for _ in range(4):
                materialize_projected_source_inputs(
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
                    root_fields[0],
                    root_fields[1],
                    root_fields[2],
                    root_fields[3],
                    root_fields[4],
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
                    R_fields[0],
                    J_fields[6],
                    F_profile_u,
                    Ip,
                    beta,
                )
                if update_fixed_point_psin_query(source_psin_query, root_fields[0], tolerance):
                    break

        alpha_state[0] = alpha1
        alpha_state[1] = alpha2
        residual_stage_runner(
            packed_residual,
            residual_fields,
            alpha1,
            alpha2,
            root_fields,
            R_fields,
            Z_fields,
            J_fields,
            g_fields,
            tb_fields[7],
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

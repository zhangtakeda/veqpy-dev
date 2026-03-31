"""
Module: engine.numpy_operator

Role:
- 提供 numpy backend 下的 fused x -> residual hot runner.
- 把常见 route 的 stage A/B/C/D 串成单个 engine 绑定入口.

Public API:
- bind_fused_single_pass_residual_runner
- bind_fused_profile_owned_psin_residual_runner
- bind_fused_fixed_point_psin_residual_runner

Notes:
- numpy 版本优先保持语义对齐.
- fixed-point psin 仍由上层保留 staged orchestration.
"""

from __future__ import annotations

from typing import Callable

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


def bind_fused_single_pass_residual_runner(
    *,
    source_kernel: Callable,
    coordinate_code: int,
    profile_names: tuple[str, ...],
    coeff_index_rows: np.ndarray,
    lengths: np.ndarray,
    residual_size: int,
    alpha_state: np.ndarray,
    active_u_fields: np.ndarray,
    T_fields: np.ndarray,
    active_rp_fields: np.ndarray,
    active_env_fields: np.ndarray,
    active_offsets: np.ndarray,
    active_scales: np.ndarray,
    c_family_fields: np.ndarray,
    s_family_fields: np.ndarray,
    c_family_base_fields: np.ndarray,
    s_family_base_fields: np.ndarray,
    c_source_slots: np.ndarray,
    s_source_slots: np.ndarray,
    c_active_order: int,
    s_active_order: int,
    h_fields: np.ndarray,
    v_fields: np.ndarray,
    k_fields: np.ndarray,
    tb_fields: np.ndarray,
    R_fields: np.ndarray,
    Z_fields: np.ndarray,
    J_fields: np.ndarray,
    g_fields: np.ndarray,
    S_r: np.ndarray,
    V_r: np.ndarray,
    Kn: np.ndarray,
    Kn_r: np.ndarray,
    Ln_r: np.ndarray,
    a: float,
    R0: float,
    Z0: float,
    B0: float,
    rho: np.ndarray,
    theta: np.ndarray,
    cos_ktheta: np.ndarray,
    sin_ktheta: np.ndarray,
    k_cos_ktheta: np.ndarray,
    k_sin_ktheta: np.ndarray,
    k2_cos_ktheta: np.ndarray,
    k2_sin_ktheta: np.ndarray,
    weights: np.ndarray,
    differentiation_matrix: np.ndarray,
    integration_matrix: np.ndarray,
    rho_powers: np.ndarray,
    y: np.ndarray,
    root_fields: np.ndarray,
    residual_fields: np.ndarray,
    packed_residual: np.ndarray,
    materialized_heat_input: np.ndarray,
    materialized_current_input: np.ndarray,
    F_profile_u: np.ndarray,
    Ip: float,
    beta: float,
) -> Callable[[np.ndarray], np.ndarray]:
    residual_stage_runner = bind_residual_stage_runner(profile_names, coeff_index_rows, lengths, residual_size)
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
    source_kernel: Callable,
    coordinate_code: int,
    parameterization_code: int,
    profile_names: tuple[str, ...],
    coeff_index_rows: np.ndarray,
    lengths: np.ndarray,
    residual_size: int,
    alpha_state: np.ndarray,
    active_u_fields: np.ndarray,
    T_fields: np.ndarray,
    active_rp_fields: np.ndarray,
    active_env_fields: np.ndarray,
    active_offsets: np.ndarray,
    active_scales: np.ndarray,
    c_family_fields: np.ndarray,
    s_family_fields: np.ndarray,
    c_family_base_fields: np.ndarray,
    s_family_base_fields: np.ndarray,
    c_source_slots: np.ndarray,
    s_source_slots: np.ndarray,
    c_active_order: int,
    s_active_order: int,
    h_fields: np.ndarray,
    v_fields: np.ndarray,
    k_fields: np.ndarray,
    tb_fields: np.ndarray,
    R_fields: np.ndarray,
    Z_fields: np.ndarray,
    J_fields: np.ndarray,
    g_fields: np.ndarray,
    S_r: np.ndarray,
    V_r: np.ndarray,
    Kn: np.ndarray,
    Kn_r: np.ndarray,
    Ln_r: np.ndarray,
    a: float,
    R0: float,
    Z0: float,
    B0: float,
    rho: np.ndarray,
    theta: np.ndarray,
    cos_ktheta: np.ndarray,
    sin_ktheta: np.ndarray,
    k_cos_ktheta: np.ndarray,
    k_sin_ktheta: np.ndarray,
    k2_cos_ktheta: np.ndarray,
    k2_sin_ktheta: np.ndarray,
    weights: np.ndarray,
    differentiation_matrix: np.ndarray,
    integration_matrix: np.ndarray,
    rho_powers: np.ndarray,
    y: np.ndarray,
    root_fields: np.ndarray,
    source_target_root_fields: np.ndarray,
    residual_fields: np.ndarray,
    packed_residual: np.ndarray,
    source_psin_query: np.ndarray,
    source_parameter_query: np.ndarray,
    materialized_heat_input: np.ndarray,
    materialized_current_input: np.ndarray,
    psin_profile_fields: np.ndarray,
    heat_input: np.ndarray,
    current_input: np.ndarray,
    F_profile_u: np.ndarray,
    Ip: float,
    beta: float,
) -> Callable[[np.ndarray], np.ndarray]:
    residual_stage_runner = bind_residual_stage_runner(profile_names, coeff_index_rows, lengths, residual_size)
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
    source_kernel: Callable,
    coordinate_code: int,
    profile_names: tuple[str, ...],
    coeff_index_rows: np.ndarray,
    lengths: np.ndarray,
    residual_size: int,
    alpha_state: np.ndarray,
    active_u_fields: np.ndarray,
    T_fields: np.ndarray,
    active_rp_fields: np.ndarray,
    active_env_fields: np.ndarray,
    active_offsets: np.ndarray,
    active_scales: np.ndarray,
    c_family_fields: np.ndarray,
    s_family_fields: np.ndarray,
    c_family_base_fields: np.ndarray,
    s_family_base_fields: np.ndarray,
    c_source_slots: np.ndarray,
    s_source_slots: np.ndarray,
    c_active_order: int,
    s_active_order: int,
    h_fields: np.ndarray,
    v_fields: np.ndarray,
    k_fields: np.ndarray,
    tb_fields: np.ndarray,
    R_fields: np.ndarray,
    Z_fields: np.ndarray,
    J_fields: np.ndarray,
    g_fields: np.ndarray,
    S_r: np.ndarray,
    V_r: np.ndarray,
    Kn: np.ndarray,
    Kn_r: np.ndarray,
    Ln_r: np.ndarray,
    a: float,
    R0: float,
    Z0: float,
    B0: float,
    rho: np.ndarray,
    theta: np.ndarray,
    cos_ktheta: np.ndarray,
    sin_ktheta: np.ndarray,
    k_cos_ktheta: np.ndarray,
    k_sin_ktheta: np.ndarray,
    k2_cos_ktheta: np.ndarray,
    k2_sin_ktheta: np.ndarray,
    weights: np.ndarray,
    differentiation_matrix: np.ndarray,
    integration_matrix: np.ndarray,
    rho_powers: np.ndarray,
    y: np.ndarray,
    root_fields: np.ndarray,
    residual_fields: np.ndarray,
    packed_residual: np.ndarray,
    source_psin_query: np.ndarray,
    psin_seed: np.ndarray,
    materialized_heat_input: np.ndarray,
    materialized_current_input: np.ndarray,
    heat_input: np.ndarray,
    current_input: np.ndarray,
    source_n_src: int,
    source_barycentric_weights: np.ndarray,
    source_fixed_remap_matrix: np.ndarray,
    F_profile_u: np.ndarray,
    Ip: float,
    beta: float,
    use_projected_finalize: bool,
    heat_projection_coeff: np.ndarray,
    current_projection_coeff: np.ndarray,
    endpoint_blend: np.ndarray,
    projection_domain_code: int,
    endpoint_policy_code: int,
    max_iter: int = 8,
    tolerance: float = 1.0e-10,
) -> Callable[[np.ndarray], np.ndarray]:
    residual_stage_runner = bind_residual_stage_runner(profile_names, coeff_index_rows, lengths, residual_size)

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

        _normalize_psin_query(source_psin_query, psin_seed)

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

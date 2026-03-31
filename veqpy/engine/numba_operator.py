"""
Module: engine.numba_operator

Role:
- 提供 numba backend 下的 fused x -> residual hot runner.
- 把常见 route 的 stage A/B/C/D 串成单个 engine 绑定入口.

Public API:
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

from veqpy.engine.numba_geometry import update_geometry
from veqpy.engine.numba_profile import update_profiles_packed_bulk
from veqpy.engine.numba_residual import _decode_residual_block_code, _run_residual_blocks_packed, update_residual
from veqpy.engine.numba_source import (
    _linear_uniform_interpolate_pair,
    _materialize_profile_owned_psin_source_impl,
    _materialize_projected_source_inputs_impl,
    _update_fixed_point_psin_query_impl,
    _update_fourier_family_fields_impl,
    resolve_source_scratch_kernel,
)

if TYPE_CHECKING:
    from veqpy.operator.layouts import RuntimeLayout, SetupLayout, StaticLayout


def _build_residual_block_metadata(profile_names: tuple[str, ...]) -> tuple[np.ndarray, np.ndarray]:
    block_codes = np.empty(len(profile_names), dtype=np.int64)
    block_orders = np.zeros(len(profile_names), dtype=np.int64)
    for i, name in enumerate(profile_names):
        block_codes[i], block_orders[i] = _decode_residual_block_code(name)
    return block_codes, block_orders


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
    static_layout: StaticLayout,
    setup_layout: SetupLayout,
    runtime_layout: RuntimeLayout,
    alpha_state: np.ndarray,
    c_active_order: int,
    s_active_order: int,
    a: float,
    R0: float,
    Z0: float,
    B0: float,
    Ip: float,
    beta: float,
) -> Callable[[np.ndarray], np.ndarray]:
    profile_names = tuple(setup_layout.profile_names[int(p)] for p in setup_layout.active_profile_ids)
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
    source_scratch_1d = runtime_layout.source_scratch_1d
    profiles_by_name = runtime_layout.profiles_by_name
    h_fields = profiles_by_name["h"].u_fields
    v_fields = profiles_by_name["v"].u_fields
    k_fields = profiles_by_name["k"].u_fields
    F_profile_u = profiles_by_name["F"].u
    block_codes, block_orders = _build_residual_block_metadata(profile_names)
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
                R_fields[0],
                J_fields[6],
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
                R_fields[0],
                J_fields[6],
                F_profile_u,
                Ip,
                beta,
                source_scratch_1d,
            )
        alpha_state[0] = alpha1
        alpha_state[1] = alpha2
        update_residual(
            residual_fields,
            alpha1,
            alpha2,
            root_fields,
            R_fields,
            Z_fields,
            J_fields,
            g_fields,
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
    static_layout: StaticLayout,
    setup_layout: SetupLayout,
    runtime_layout: RuntimeLayout,
    alpha_state: np.ndarray,
    c_active_order: int,
    s_active_order: int,
    a: float,
    R0: float,
    Z0: float,
    B0: float,
    heat_input: np.ndarray,
    current_input: np.ndarray,
    Ip: float,
    beta: float,
) -> Callable[[np.ndarray], np.ndarray]:
    profile_names = tuple(setup_layout.profile_names[int(p)] for p in setup_layout.active_profile_ids)
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
    materialized_heat_input = runtime_layout.materialized_heat_input
    materialized_current_input = runtime_layout.materialized_current_input
    source_scratch_1d = runtime_layout.source_scratch_1d
    profiles_by_name = runtime_layout.profiles_by_name
    psin_profile_fields = profiles_by_name["psin"].u_fields
    h_fields = profiles_by_name["h"].u_fields
    v_fields = profiles_by_name["v"].u_fields
    k_fields = profiles_by_name["k"].u_fields
    F_profile_u = profiles_by_name["F"].u
    block_codes, block_orders = _build_residual_block_metadata(profile_names)
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
                R_fields[0],
                J_fields[6],
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
                R_fields[0],
                J_fields[6],
                F_profile_u,
                Ip,
                beta,
                source_scratch_1d,
            )
        alpha_state[0] = alpha1
        alpha_state[1] = alpha2
        update_residual(
            residual_fields,
            alpha1,
            alpha2,
            root_fields,
            R_fields,
            Z_fields,
            J_fields,
            g_fields,
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
    static_layout: StaticLayout,
    setup_layout: SetupLayout,
    runtime_layout: RuntimeLayout,
    alpha_state: np.ndarray,
    c_active_order: int,
    s_active_order: int,
    a: float,
    R0: float,
    Z0: float,
    B0: float,
    heat_input: np.ndarray,
    current_input: np.ndarray,
    Ip: float,
    beta: float,
    max_iter: int = 8,
    tolerance: float = 1.0e-10,
) -> Callable[[np.ndarray], np.ndarray]:
    profile_names = tuple(setup_layout.profile_names[int(p)] for p in setup_layout.active_profile_ids)
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
    source_scratch_1d = runtime_layout.source_scratch_1d
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
    source_n_src = int(setup_layout.source_n_src)
    use_projected_finalize = bool(setup_layout.fixed_point_use_projected_finalize)
    projection_domain_code = int(setup_layout.fixed_point_projection_domain_code)
    endpoint_policy_code = int(setup_layout.fixed_point_endpoint_policy_code)
    allow_query_warmstart = (not use_projected_finalize) or (endpoint_policy_code != 0)
    block_codes, block_orders = _build_residual_block_metadata(profile_names)
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
            _linear_uniform_interpolate_pair(
                materialized_heat_input,
                materialized_current_input,
                heat_input,
                current_input,
                source_psin_query,
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
                    R_fields[0],
                    J_fields[6],
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
                    R_fields[0],
                    J_fields[6],
                    F_profile_u,
                    Ip,
                    beta,
                    source_scratch_1d,
                )
            if _update_fixed_point_psin_query_impl(source_psin_query, psin, tolerance):
                break

        if use_projected_finalize:
            for i in range(source_psin_query.shape[0]):
                source_psin_query[i] = psin[i]
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
                    R_fields[0],
                    J_fields[6],
                    F_profile_u,
                    Ip,
                    beta,
                    source_scratch_1d,
                )

        alpha_state[0] = alpha1
        alpha_state[1] = alpha2
        update_residual(
            residual_fields,
            alpha1,
            alpha2,
            root_fields,
            R_fields,
            Z_fields,
            J_fields,
            g_fields,
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

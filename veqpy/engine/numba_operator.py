"""
Module: engine.numba_operator

Role:
- 提供 fused x -> residual hot runner.
- 把常见 route 的 stage A/B/C/D 串成单个 engine 绑定入口.

Public API:
- bind_fused_residual_runner

Notes:
- 这里只覆盖 common route.
- fixed-point psin 仍由上层保留 staged orchestration.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable

import numpy as np
from numba import njit

from veqpy.engine.numba_geometry import update_geometry_hot
from veqpy.engine.numba_profile import update_profiles_packed_bulk
from veqpy.engine.numba_residual import _run_residual_blocks_packed_precomputed, update_residual_compact
from veqpy.engine.numba_source import (
    _linear_uniform_interpolate_pair,
    _local_barycentric_interpolate_pair,
    _materialize_profile_owned_psin_source_impl,
    _materialize_projected_source_inputs_impl,
    _uniform_barycentric_weights,
    _update_fixed_point_psin_query_and_linear_uniform_inputs_impl,
    _update_fixed_point_psin_query_and_local_barycentric_inputs_impl,
    _update_fixed_point_psin_query_and_projected_inputs_impl,
    _update_fourier_family_fields_impl,
    _update_pj2_from_psin_inputs_with_scratch,
    _update_pq_from_psin_inputs_with_scratch,
    resolve_source_scratch_kernel,
)

if TYPE_CHECKING:
    from veqpy.engine.orchestration import SourcePlan
    from veqpy.operator.layouts import ResidualBindingLayout, RuntimeLayout, StaticLayout


def _source_route_key(source_plan: "SourcePlan") -> tuple[str, str, str]:
    return (source_plan.route, source_plan.coordinate, source_plan.nodes)


@dataclass(frozen=True, slots=True)
class _FusedRouteBindingSpec:
    core_kind: str
    skip_projection_finalize: bool = False
    fixed_point_stencil_size: int = 8


_FUSED_ROUTE_BINDINGS: dict[tuple[str, str, str], _FusedRouteBindingSpec] = {
    ("PF", "rho", "uniform"): _FusedRouteBindingSpec("single_pass"),
    ("PF", "rho", "grid"): _FusedRouteBindingSpec("single_pass"),
    ("PF", "psin", "uniform"): _FusedRouteBindingSpec("profile_owned"),
    ("PF", "psin", "grid"): _FusedRouteBindingSpec("single_pass"),
    ("PP", "rho", "uniform"): _FusedRouteBindingSpec("single_pass"),
    ("PP", "rho", "grid"): _FusedRouteBindingSpec("single_pass"),
    ("PP", "psin", "uniform"): _FusedRouteBindingSpec("profile_owned"),
    ("PP", "psin", "grid"): _FusedRouteBindingSpec("single_pass"),
    ("PI", "rho", "uniform"): _FusedRouteBindingSpec("single_pass"),
    ("PI", "rho", "grid"): _FusedRouteBindingSpec("single_pass"),
    ("PI", "psin", "uniform"): _FusedRouteBindingSpec("profile_owned", skip_projection_finalize=True),
    ("PI", "psin", "grid"): _FusedRouteBindingSpec("single_pass"),
    ("PJ1", "rho", "uniform"): _FusedRouteBindingSpec("single_pass"),
    ("PJ1", "rho", "grid"): _FusedRouteBindingSpec("single_pass"),
    ("PJ1", "psin", "uniform"): _FusedRouteBindingSpec("profile_owned"),
    ("PJ1", "psin", "grid"): _FusedRouteBindingSpec("single_pass"),
    ("PJ2", "rho", "uniform"): _FusedRouteBindingSpec("single_pass"),
    ("PJ2", "rho", "grid"): _FusedRouteBindingSpec("single_pass"),
    ("PJ2", "psin", "uniform"): _FusedRouteBindingSpec("fixed_point", fixed_point_stencil_size=8),
    ("PJ2", "psin", "grid"): _FusedRouteBindingSpec("single_pass"),
    ("PQ", "rho", "uniform"): _FusedRouteBindingSpec("single_pass"),
    ("PQ", "rho", "grid"): _FusedRouteBindingSpec("single_pass"),
    ("PQ", "psin", "uniform"): _FusedRouteBindingSpec("fixed_point", fixed_point_stencil_size=4),
    ("PQ", "psin", "grid"): _FusedRouteBindingSpec("single_pass"),
}


def bind_source_eval_runner(
    *,
    source_plan: "SourcePlan",
    static_layout: "StaticLayout",
    runtime_layout: "RuntimeLayout",
    profiles_by_name: dict[str, object],
    B0: float,
) -> Callable:
    source_kernel = source_plan.kernel
    scratch_source_kernel = resolve_source_scratch_kernel(source_kernel)
    coordinate_code = int(source_plan.coordinate_code)
    weights = static_layout.weights
    differentiation_matrix = static_layout.differentiation_matrix
    integration_matrix = static_layout.integration_matrix
    rho = static_layout.rho
    radial_workspace = runtime_layout.geometry_radial_workspace
    surface_workspace = runtime_layout.geometry_surface_workspace
    source_runtime_state = runtime_layout.source_runtime_state
    F_profile_u = profiles_by_name["F2"].u
    Ip = float(source_plan.Ip)
    beta = float(source_plan.beta)
    source_scratch_1d = source_runtime_state.scratch_1d

    def runner(
        out_root_fields: np.ndarray,
        out_FFn_psin: np.ndarray,
        out_Pn_psin: np.ndarray,
        heat_input: np.ndarray,
        current_input: np.ndarray,
        R0: float,
    ) -> tuple[float, float]:
        if scratch_source_kernel is None:
            return source_kernel(
                out_root_fields,
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
                radial_workspace,
                surface_workspace,
                F_profile_u,
                Ip,
                beta,
            )
        return scratch_source_kernel(
            out_root_fields,
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
            radial_workspace,
            surface_workspace,
            F_profile_u,
            Ip,
            beta,
            source_scratch_1d,
        )

    return runner


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


def _refresh_hot_runtime(
    x: np.ndarray,
    *,
    runtime_layout: "RuntimeLayout",
    static_layout: "StaticLayout",
    c_active_order: int,
    s_active_order: int,
    a: float,
    R0: float,
    Z0: float,
    h_fields: np.ndarray,
    v_fields: np.ndarray,
    k_fields: np.ndarray,
) -> None:
    active_profile_slab = runtime_layout.active_profile_slab
    active_u_fields = runtime_layout.active_u_fields
    T_fields = static_layout.T_fields
    active_offsets = runtime_layout.active_offsets
    active_scales = runtime_layout.active_scales
    coeff_index_rows = runtime_layout.active_coeff_index_rows
    lengths = runtime_layout.active_lengths
    c_family_fields = runtime_layout.c_family_fields
    s_family_fields = runtime_layout.s_family_fields
    c_family_base_fields = runtime_layout.c_family_base_fields
    s_family_base_fields = runtime_layout.s_family_base_fields
    c_source_slots = runtime_layout.c_family_source_slots
    s_source_slots = runtime_layout.s_family_source_slots
    surface_workspace = runtime_layout.geometry_surface_workspace
    radial_workspace = runtime_layout.geometry_radial_workspace
    rho = static_layout.rho
    theta = static_layout.theta
    cos_ktheta = static_layout.cos_ktheta
    sin_ktheta = static_layout.sin_ktheta
    k_cos_ktheta = static_layout.k_cos_ktheta
    k_sin_ktheta = static_layout.k_sin_ktheta
    k2_cos_ktheta = static_layout.k2_cos_ktheta
    k2_sin_ktheta = static_layout.k2_sin_ktheta
    update_profiles_packed_bulk(
        active_profile_slab,
        T_fields,
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
    update_geometry_hot(
        surface_workspace,
        radial_workspace,
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


def _pack_residual_output(
    *,
    runtime_layout: "RuntimeLayout",
    residual_binding_layout: "ResidualBindingLayout",
    static_layout: "StaticLayout",
    scratch_holder: list[np.ndarray | None],
    a: float,
    R0: float,
    B0: float,
) -> np.ndarray:
    packed_residual = runtime_layout.packed_residual
    residual_workspace = runtime_layout.residual_surface_workspace
    block_codes = residual_binding_layout.active_residual_block_codes
    block_orders = residual_binding_layout.active_residual_block_orders
    coeff_index_rows = runtime_layout.active_coeff_index_rows
    lengths = runtime_layout.active_lengths
    sin_ktheta = static_layout.sin_ktheta
    cos_ktheta = static_layout.cos_ktheta
    rho_powers = static_layout.rho_powers
    y = static_layout.y
    T_fields = static_layout.T_fields
    weights = static_layout.weights
    packed_residual.fill(0.0)
    scratch = scratch_holder[0]
    nr = residual_workspace.shape[1]
    if scratch is None or scratch.shape[0] != nr:
        scratch = np.empty(nr, dtype=np.float64)
        scratch_holder[0] = scratch
    _run_residual_blocks_packed_precomputed(
        packed_residual,
        scratch,
        block_codes,
        block_orders,
        coeff_index_rows,
        lengths,
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
    return packed_residual.copy()


@njit(cache=True, nogil=True)
def _call_source_kernel_with_scratch(
    scratch_source_kernel,
    root_fields: np.ndarray,
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
    radial_workspace: np.ndarray,
    surface_workspace: np.ndarray,
    F_profile_u: np.ndarray,
    Ip: float,
    beta: float,
    source_scratch_1d: np.ndarray,
) -> tuple[float, float]:
    return scratch_source_kernel(
        root_fields,
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
        radial_workspace,
        surface_workspace,
        F_profile_u,
        Ip,
        beta,
        source_scratch_1d,
    )


@njit(cache=True, nogil=True)
def _run_fixed_point_linear_with_scratch_impl(
    scratch_source_kernel,
    max_iter: int,
    tolerance: float,
    source_psin_query: np.ndarray,
    psin: np.ndarray,
    root_fields: np.ndarray,
    FFn_psin: np.ndarray,
    Pn_psin: np.ndarray,
    materialized_heat_input: np.ndarray,
    materialized_current_input: np.ndarray,
    heat_input: np.ndarray,
    current_input: np.ndarray,
    coordinate_code: int,
    R0: float,
    B0: float,
    weights: np.ndarray,
    differentiation_matrix: np.ndarray,
    integration_matrix: np.ndarray,
    rho: np.ndarray,
    radial_workspace: np.ndarray,
    surface_workspace: np.ndarray,
    F_profile_u: np.ndarray,
    Ip: float,
    beta: float,
    source_scratch_1d: np.ndarray,
) -> tuple[float, float]:
    _linear_uniform_interpolate_pair(
        materialized_heat_input,
        materialized_current_input,
        heat_input,
        current_input,
        source_psin_query,
    )
    alpha1 = np.nan
    alpha2 = np.nan
    for _ in range(max_iter):
        alpha1, alpha2 = _call_source_kernel_with_scratch(
            scratch_source_kernel,
            root_fields,
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
            radial_workspace,
            surface_workspace,
            F_profile_u,
            Ip,
            beta,
            source_scratch_1d,
        )
        if _update_fixed_point_psin_query_and_linear_uniform_inputs_impl(
            source_psin_query,
            psin,
            tolerance,
            materialized_heat_input,
            materialized_current_input,
            heat_input,
            current_input,
        ):
            break
    return alpha1, alpha2


@njit(cache=True, nogil=True)
def _run_fixed_point_barycentric_with_scratch_impl(
    scratch_source_kernel,
    max_iter: int,
    tolerance: float,
    source_psin_query: np.ndarray,
    psin: np.ndarray,
    root_fields: np.ndarray,
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
    radial_workspace: np.ndarray,
    surface_workspace: np.ndarray,
    F_profile_u: np.ndarray,
    Ip: float,
    beta: float,
    source_scratch_1d: np.ndarray,
) -> tuple[float, float]:
    _local_barycentric_interpolate_pair(
        materialized_heat_input,
        materialized_current_input,
        heat_input,
        current_input,
        source_psin_query,
        barycentric_weights,
    )
    alpha1 = np.nan
    alpha2 = np.nan
    for _ in range(max_iter):
        alpha1, alpha2 = _call_source_kernel_with_scratch(
            scratch_source_kernel,
            root_fields,
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
            radial_workspace,
            surface_workspace,
            F_profile_u,
            Ip,
            beta,
            source_scratch_1d,
        )
        if _update_fixed_point_psin_query_and_local_barycentric_inputs_impl(
            source_psin_query,
            psin,
            tolerance,
            materialized_heat_input,
            materialized_current_input,
            heat_input,
            current_input,
            barycentric_weights,
        ):
            break
    return alpha1, alpha2


@njit(cache=True, nogil=True)
def _run_projected_finalize_with_scratch_impl(
    scratch_source_kernel,
    finalize_iter: int,
    tolerance: float,
    source_psin_query: np.ndarray,
    psin: np.ndarray,
    root_fields: np.ndarray,
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
    radial_workspace: np.ndarray,
    surface_workspace: np.ndarray,
    F_profile_u: np.ndarray,
    Ip: float,
    beta: float,
    source_scratch_1d: np.ndarray,
) -> tuple[float, float]:
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

    alpha1 = np.nan
    alpha2 = np.nan
    for _ in range(finalize_iter):
        alpha1, alpha2 = _call_source_kernel_with_scratch(
            scratch_source_kernel,
            root_fields,
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
            radial_workspace,
            surface_workspace,
            F_profile_u,
            Ip,
            beta,
            source_scratch_1d,
        )
        if _update_fixed_point_psin_query_and_projected_inputs_impl(
            source_psin_query,
            psin,
            tolerance,
            materialized_heat_input,
            materialized_current_input,
            heat_projection_coeff,
            current_projection_coeff,
            current_input,
            projection_domain_code,
            endpoint_policy_code,
            endpoint_blend,
        ):
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
) -> Callable[[np.ndarray], np.ndarray]:
    route_key = _source_route_key(source_plan)
    try:
        binding = _FUSED_ROUTE_BINDINGS[route_key]
    except KeyError as exc:
        raise ValueError(f"Unsupported source route key {route_key!r}") from exc

    if binding.core_kind == "single_pass":
        return _bind_single_pass_residual_runner_core(
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
        )
    if binding.core_kind == "profile_owned":
        return _bind_profile_owned_psin_residual_runner_core(
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
            skip_projection_finalize=binding.skip_projection_finalize,
        )
    if binding.core_kind == "fixed_point":
        if route_key == ("PJ2", "psin", "uniform"):
            return _bind_pj2_psin_fixed_point_residual_runner_core(
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
                fixed_point_stencil_size=binding.fixed_point_stencil_size,
            )
        if route_key == ("PQ", "psin", "uniform"):
            return _bind_pq_psin_fixed_point_residual_runner_core(
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
                fixed_point_stencil_size=binding.fixed_point_stencil_size,
            )
        raise ValueError(f"Unsupported fixed-point fused route key {route_key!r}")
    raise ValueError(f"Unsupported fused route binding {binding!r} for key {route_key!r}")


def _bind_single_pass_residual_runner_core(
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
) -> Callable[[np.ndarray], np.ndarray]:
    surface_workspace = runtime_layout.geometry_surface_workspace
    residual_workspace = runtime_layout.residual_surface_workspace
    root_fields = runtime_layout.root_fields
    source_runtime_state = runtime_layout.source_runtime_state
    materialized_heat_input = source_runtime_state.materialized_heat_input
    materialized_current_input = source_runtime_state.materialized_current_input
    profiles_by_name = runtime_layout.profiles_by_name
    h_fields = profiles_by_name["h"].u_fields
    v_fields = profiles_by_name["v"].u_fields
    k_fields = profiles_by_name["k"].u_fields
    source_eval_runner = bind_source_eval_runner(
        source_plan=source_plan,
        static_layout=static_layout,
        runtime_layout=runtime_layout,
        profiles_by_name=profiles_by_name,
        B0=B0,
    )
    scratch_holder: list[np.ndarray | None] = [None]
    FFn_psin = root_fields[3]
    Pn_psin = root_fields[4]

    def runner(x: np.ndarray) -> np.ndarray:
        _refresh_hot_runtime(
            x,
            runtime_layout=runtime_layout,
            static_layout=static_layout,
            c_active_order=c_active_order,
            s_active_order=s_active_order,
            a=a,
            R0=R0,
            Z0=Z0,
            h_fields=h_fields,
            v_fields=v_fields,
            k_fields=k_fields,
        )
        alpha1, alpha2 = source_eval_runner(
            root_fields,
            FFn_psin,
            Pn_psin,
            materialized_heat_input,
            materialized_current_input,
            R0,
        )
        alpha_state[0] = alpha1
        alpha_state[1] = alpha2
        update_residual_compact(
            residual_workspace,
            alpha1,
            alpha2,
            root_fields,
            surface_workspace,
        )
        return _pack_residual_output(
            runtime_layout=runtime_layout,
            residual_binding_layout=residual_binding_layout,
            static_layout=static_layout,
            scratch_holder=scratch_holder,
            a=a,
            R0=R0,
            B0=B0,
        )

    return runner


def _bind_profile_owned_psin_residual_runner_core(
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
    skip_projection_finalize: bool,
) -> Callable[[np.ndarray], np.ndarray]:
    surface_workspace = runtime_layout.geometry_surface_workspace
    residual_workspace = runtime_layout.residual_surface_workspace
    root_fields = runtime_layout.root_fields
    source_runtime_state = runtime_layout.source_runtime_state
    source_target_root_fields = source_runtime_state.target_root_fields
    source_psin_query = source_runtime_state.psin_query
    source_parameter_query = source_runtime_state.parameter_query
    heat_projection_coeff = source_runtime_state.heat_projection_coeff
    current_projection_coeff = source_runtime_state.current_projection_coeff
    endpoint_blend = source_runtime_state.endpoint_blend
    materialized_heat_input = source_runtime_state.materialized_heat_input
    materialized_current_input = source_runtime_state.materialized_current_input
    profiles_by_name = runtime_layout.profiles_by_name
    psin_profile_fields = profiles_by_name["psin"].u_fields
    h_fields = profiles_by_name["h"].u_fields
    v_fields = profiles_by_name["v"].u_fields
    k_fields = profiles_by_name["k"].u_fields
    parameterization_code = int(source_plan.parameterization_code)
    has_projection_policy = bool(source_plan.has_projection_policy)
    projection_domain_code = int(source_plan.projection_domain_code)
    endpoint_policy_code = int(source_plan.endpoint_policy_code)
    heat_input = source_plan.heat_input
    current_input = source_plan.current_input
    source_eval_runner = bind_source_eval_runner(
        source_plan=source_plan,
        static_layout=static_layout,
        runtime_layout=runtime_layout,
        profiles_by_name=profiles_by_name,
        B0=B0,
    )
    scratch_holder: list[np.ndarray | None] = [None]
    psin = root_fields[0]
    psin_r = root_fields[1]
    psin_rr = root_fields[2]
    FFn_psin = root_fields[3]
    Pn_psin = root_fields[4]

    def runner(x: np.ndarray) -> np.ndarray:
        _refresh_hot_runtime(
            x,
            runtime_layout=runtime_layout,
            static_layout=static_layout,
            c_active_order=c_active_order,
            s_active_order=s_active_order,
            a=a,
            R0=R0,
            Z0=Z0,
            h_fields=h_fields,
            v_fields=v_fields,
            k_fields=k_fields,
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
        alpha1, alpha2 = source_eval_runner(
            source_target_root_fields,
            FFn_psin,
            Pn_psin,
            materialized_heat_input,
            materialized_current_input,
            R0,
        )
        alpha_state[0] = alpha1
        alpha_state[1] = alpha2
        update_residual_compact(
            residual_workspace,
            alpha1,
            alpha2,
            root_fields,
            surface_workspace,
        )
        return _pack_residual_output(
            runtime_layout=runtime_layout,
            residual_binding_layout=residual_binding_layout,
            static_layout=static_layout,
            scratch_holder=scratch_holder,
            a=a,
            R0=R0,
            B0=B0,
        )

    return runner


def _bind_pj2_psin_fixed_point_residual_runner_core(
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
    fixed_point_stencil_size: int,
    max_iter: int | None = None,
    tolerance: float | None = None,
) -> Callable[[np.ndarray], np.ndarray]:
    return _bind_fixed_point_psin_residual_runner_core(
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
        fixed_point_stencil_size=fixed_point_stencil_size,
        scratch_source_kernel=_update_pj2_from_psin_inputs_with_scratch,
        max_iter=max_iter,
        tolerance=tolerance,
    )


def _bind_pq_psin_fixed_point_residual_runner_core(
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
    fixed_point_stencil_size: int,
    max_iter: int | None = None,
    tolerance: float | None = None,
) -> Callable[[np.ndarray], np.ndarray]:
    return _bind_fixed_point_psin_residual_runner_core(
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
        fixed_point_stencil_size=fixed_point_stencil_size,
        scratch_source_kernel=_update_pq_from_psin_inputs_with_scratch,
        max_iter=max_iter,
        tolerance=tolerance,
    )


def _bind_fixed_point_psin_residual_runner_core(
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
    fixed_point_stencil_size: int,
    scratch_source_kernel,
    max_iter: int | None = None,
    tolerance: float | None = None,
) -> Callable[[np.ndarray], np.ndarray]:
    surface_workspace = runtime_layout.geometry_surface_workspace
    radial_workspace = runtime_layout.geometry_radial_workspace
    residual_workspace = runtime_layout.residual_surface_workspace
    rho = static_layout.rho
    weights = static_layout.weights
    differentiation_matrix = static_layout.differentiation_matrix
    integration_matrix = static_layout.integration_matrix
    root_fields = runtime_layout.root_fields
    source_runtime_state = runtime_layout.source_runtime_state
    source_psin_query = source_runtime_state.psin_query
    materialized_heat_input = source_runtime_state.materialized_heat_input
    materialized_current_input = source_runtime_state.materialized_current_input
    source_scratch_1d = source_runtime_state.scratch_1d
    heat_projection_coeff = source_runtime_state.heat_projection_coeff
    current_projection_coeff = source_runtime_state.current_projection_coeff
    endpoint_blend = source_runtime_state.endpoint_blend
    profiles_by_name = runtime_layout.profiles_by_name
    h_fields = profiles_by_name["h"].u_fields
    v_fields = profiles_by_name["v"].u_fields
    k_fields = profiles_by_name["k"].u_fields
    F_profile_u = profiles_by_name["F2"].u
    heat_input = source_plan.heat_input
    current_input = source_plan.current_input
    coordinate_code = int(source_plan.coordinate_code)
    Ip = float(source_plan.Ip)
    beta = float(source_plan.beta)
    max_iter = int(source_plan.fixed_point_max_iter if max_iter is None else max_iter)
    finalize_max_iter = int(source_plan.fixed_point_finalize_max_iter)
    tolerance = float(source_plan.fixed_point_tolerance if tolerance is None else tolerance)
    has_Ip = bool(np.isfinite(Ip))
    use_projected_finalize = bool(source_plan.use_projected_finalize)
    projection_domain_code = int(source_plan.projection_domain_code)
    endpoint_policy_code = int(source_plan.endpoint_policy_code)
    allow_query_warmstart = bool(source_plan.allow_query_warmstart)
    fixed_point_barycentric_weights = _uniform_barycentric_weights(
        min(fixed_point_stencil_size, int(source_plan.n_src))
    )
    scratch_holder: list[np.ndarray | None] = [None]
    psin = root_fields[0]
    FFn_psin = root_fields[3]
    Pn_psin = root_fields[4]

    def runner(x: np.ndarray) -> np.ndarray:
        _refresh_hot_runtime(
            x,
            runtime_layout=runtime_layout,
            static_layout=static_layout,
            c_active_order=c_active_order,
            s_active_order=s_active_order,
            a=a,
            R0=R0,
            Z0=Z0,
            h_fields=h_fields,
            v_fields=v_fields,
            k_fields=k_fields,
        )

        if (not allow_query_warmstart) or source_psin_query[0] < 0.0:
            _normalize_psin_query(source_psin_query, profiles_by_name["psin"].u)

        if has_Ip:
            alpha1, alpha2 = _run_fixed_point_barycentric_with_scratch_impl(
                scratch_source_kernel,
                max_iter,
                tolerance,
                source_psin_query,
                psin,
                root_fields,
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
                radial_workspace,
                surface_workspace,
                F_profile_u,
                Ip,
                beta,
                source_scratch_1d,
            )
        else:
            alpha1, alpha2 = _run_fixed_point_linear_with_scratch_impl(
                scratch_source_kernel,
                max_iter,
                tolerance,
                source_psin_query,
                psin,
                root_fields,
                FFn_psin,
                Pn_psin,
                materialized_heat_input,
                materialized_current_input,
                heat_input,
                current_input,
                coordinate_code,
                R0,
                B0,
                weights,
                differentiation_matrix,
                integration_matrix,
                rho,
                radial_workspace,
                surface_workspace,
                F_profile_u,
                Ip,
                beta,
                source_scratch_1d,
            )

        if use_projected_finalize:
            alpha1, alpha2 = _run_projected_finalize_with_scratch_impl(
                scratch_source_kernel,
                finalize_max_iter,
                tolerance,
                source_psin_query,
                psin,
                root_fields,
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
                radial_workspace,
                surface_workspace,
                F_profile_u,
                Ip,
                beta,
                source_scratch_1d,
            )

        alpha_state[0] = alpha1
        alpha_state[1] = alpha2
        update_residual_compact(
            residual_workspace,
            alpha1,
            alpha2,
            root_fields,
            surface_workspace,
        )
        return _pack_residual_output(
            runtime_layout=runtime_layout,
            residual_binding_layout=residual_binding_layout,
            static_layout=static_layout,
            scratch_holder=scratch_holder,
            a=a,
            R0=R0,
            B0=B0,
        )

    return runner

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

import veqpy.engine.backend_abi as backend_abi
from veqpy.engine.numba_geometry import update_geometry_hot
from veqpy.engine.numba_profile import update_profiles_packed_bulk
from veqpy.engine.numba_residual import _run_residual_blocks_packed_precomputed, update_residual_compact
from veqpy.engine.numba_source import (
    _linear_uniform_interpolate_pair,
    _local_barycentric_interpolate_pair,
    _materialize_profile_owned_psin_source_impl,
    _materialize_projected_source_inputs_impl,
    _update_fixed_point_psin_query_and_linear_uniform_inputs_impl,
    _update_fixed_point_psin_query_and_local_barycentric_inputs_impl,
    _update_fixed_point_psin_query_and_projected_inputs_impl,
    _update_fourier_family_fields_impl,
    _update_pj2_from_psin_inputs_with_scratch,
    _update_pq_from_psin_inputs_with_scratch,
)

if TYPE_CHECKING:
    from veqpy.engine.orchestration import SourcePlan
    from veqpy.operator.layouts import BackendState


def _source_route_key(source_plan: "SourcePlan") -> tuple[str, str, str]:
    return (source_plan.route, source_plan.coordinate, source_plan.nodes)


@dataclass(frozen=True, slots=True)
class _FusedRouteBindingSpec:
    core_kind: str
    skip_projection_finalize: bool = False


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
    ("PJ2", "psin", "grid"): _FusedRouteBindingSpec("single_pass"),
    ("PQ", "rho", "uniform"): _FusedRouteBindingSpec("single_pass"),
    ("PQ", "rho", "grid"): _FusedRouteBindingSpec("single_pass"),
    ("PQ", "psin", "grid"): _FusedRouteBindingSpec("single_pass"),
}


def bind_source_eval_runner(
    *,
    source_plan: "SourcePlan",
    backend_state: "BackendState",
    B0: float,
) -> Callable:
    return _bind_source_eval_runner_for_fused_backend(
        source_eval_binding=backend_abi.build_fused_source_eval_abi(
            source_plan=source_plan,
            backend_state=backend_state,
            B0=B0,
        )
    )


@njit(cache=True, fastmath=True, nogil=True)
def _apply_f2_profile_fields_impl(fields: np.ndarray, eps: float = 1.0e-10) -> None:
    nr = fields.shape[1]
    for i in range(nr):
        F2 = fields[0, i]
        F2_r = fields[1, i]
        F2_rr = fields[2, i]
        if F2 < eps:
            F2 = eps
        F = np.sqrt(F2)
        inv_F = 1.0 / F
        inv_F3 = inv_F / F2
        fields[0, i] = F
        fields[1, i] = 0.5 * F2_r * inv_F
        fields[2, i] = 0.5 * F2_rr * inv_F - 0.25 * F2_r * F2_r * inv_F3


def apply_f2_profile_fields(fields: np.ndarray, eps: float = 1.0e-10) -> None:
    _apply_f2_profile_fields_impl(fields, eps=eps)


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
    hot_runtime_binding: backend_abi.FusedHotRuntimeABI,
) -> None:
    update_profiles_packed_bulk(
        hot_runtime_binding.active_profile_slab,
        hot_runtime_binding.T_fields,
        hot_runtime_binding.active_offsets,
        hot_runtime_binding.active_scales,
        x,
        hot_runtime_binding.active_coeff_index_rows,
        hot_runtime_binding.active_lengths,
    )
    if hot_runtime_binding.apply_f2_transform:
        _apply_f2_profile_fields_impl(hot_runtime_binding.F_profile_fields)
    _update_fourier_family_fields_impl(
        hot_runtime_binding.c_family_fields,
        hot_runtime_binding.s_family_fields,
        hot_runtime_binding.c_family_base_fields,
        hot_runtime_binding.s_family_base_fields,
        hot_runtime_binding.active_u_fields,
        hot_runtime_binding.c_family_source_slots,
        hot_runtime_binding.s_family_source_slots,
        hot_runtime_binding.c_active_order,
        hot_runtime_binding.s_active_order,
    )
    update_geometry_hot(
        hot_runtime_binding.geometry_surface_workspace,
        hot_runtime_binding.geometry_radial_workspace,
        hot_runtime_binding.a,
        hot_runtime_binding.R0,
        hot_runtime_binding.Z0,
        hot_runtime_binding.rho,
        hot_runtime_binding.theta,
        hot_runtime_binding.cos_ktheta,
        hot_runtime_binding.sin_ktheta,
        hot_runtime_binding.k_cos_ktheta,
        hot_runtime_binding.k_sin_ktheta,
        hot_runtime_binding.k2_cos_ktheta,
        hot_runtime_binding.k2_sin_ktheta,
        hot_runtime_binding.h_fields,
        hot_runtime_binding.v_fields,
        hot_runtime_binding.k_fields,
        hot_runtime_binding.c_family_fields,
        hot_runtime_binding.s_family_fields,
        hot_runtime_binding.c_active_order,
        hot_runtime_binding.s_active_order,
    )


def _pack_residual_output(
    *,
    residual_pack_binding: backend_abi.FusedResidualPackABI,
    scratch_holder: list[np.ndarray | None],
) -> np.ndarray:
    packed_residual = residual_pack_binding.packed_residual
    packed_residual.fill(0.0)
    scratch = scratch_holder[0]
    nr = residual_pack_binding.residual_surface_workspace.shape[1]
    if scratch is None or scratch.shape[0] != nr:
        scratch = np.empty(nr, dtype=np.float64)
        scratch_holder[0] = scratch
    _run_residual_blocks_packed_precomputed(
        packed_residual,
        scratch,
        residual_pack_binding.active_residual_block_codes,
        residual_pack_binding.active_residual_block_orders,
        residual_pack_binding.active_residual_block_radial_powers,
        residual_pack_binding.active_coeff_index_rows,
        residual_pack_binding.active_lengths,
        residual_pack_binding.residual_surface_workspace,
        residual_pack_binding.sin_ktheta,
        residual_pack_binding.cos_ktheta,
        residual_pack_binding.rho_powers,
        residual_pack_binding.y,
        residual_pack_binding.T_fields,
        residual_pack_binding.weights,
        residual_pack_binding.a,
        residual_pack_binding.R0,
        residual_pack_binding.B0,
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
    backend_state: "BackendState",
    alpha_state: np.ndarray,
    c_active_order: int,
    s_active_order: int,
    a: float,
    R0: float,
    Z0: float,
    B0: float,
) -> Callable[[np.ndarray], np.ndarray]:
    apply_f2_transform = "F" in backend_state.residual_binding_layout.active_profile_names
    route_key = _source_route_key(source_plan)
    if route_key == ("PJ2", "psin", "uniform"):
        return _bind_pj2_psin_fixed_point_residual_runner_core(
            source_plan=source_plan,
            backend_state=backend_state,
            apply_f2_transform=apply_f2_transform,
            alpha_state=alpha_state,
            c_active_order=c_active_order,
            s_active_order=s_active_order,
            a=a,
            R0=R0,
            Z0=Z0,
            B0=B0,
        )
    if route_key == ("PQ", "psin", "uniform"):
        return _bind_pq_psin_fixed_point_residual_runner_core(
            source_plan=source_plan,
            backend_state=backend_state,
            apply_f2_transform=apply_f2_transform,
            alpha_state=alpha_state,
            c_active_order=c_active_order,
            s_active_order=s_active_order,
            a=a,
            R0=R0,
            Z0=Z0,
            B0=B0,
        )
    try:
        binding = _FUSED_ROUTE_BINDINGS[route_key]
    except KeyError as exc:
        raise ValueError(f"Unsupported source route key {route_key!r}") from exc

    if binding.core_kind == "single_pass":
        return _bind_single_pass_residual_runner_core(
            source_plan=source_plan,
            backend_state=backend_state,
            apply_f2_transform=apply_f2_transform,
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
            backend_state=backend_state,
            apply_f2_transform=apply_f2_transform,
            alpha_state=alpha_state,
            c_active_order=c_active_order,
            s_active_order=s_active_order,
            a=a,
            R0=R0,
            Z0=Z0,
            B0=B0,
            skip_projection_finalize=binding.skip_projection_finalize,
        )
    raise ValueError(f"Unsupported fused route binding {binding!r} for key {route_key!r}")


def _bind_single_pass_residual_runner_core(
    *,
    source_plan: SourcePlan,
    backend_state: "BackendState",
    apply_f2_transform: bool,
    alpha_state: np.ndarray,
    c_active_order: int,
    s_active_order: int,
    a: float,
    R0: float,
    Z0: float,
    B0: float,
) -> Callable[[np.ndarray], np.ndarray]:
    runtime_layout = backend_state.runtime_layout
    surface_workspace = runtime_layout.geometry_surface_workspace
    residual_workspace = runtime_layout.residual_surface_workspace
    root_fields = runtime_layout.root_fields
    source_work_state = backend_state.source_runtime_state.work_state
    materialized_heat_input = source_work_state.materialized_heat_input
    materialized_current_input = source_work_state.materialized_current_input
    hot_runtime_binding = backend_abi.build_fused_hot_runtime_abi(
        backend_state=backend_state,
        apply_f2_transform=apply_f2_transform,
        c_active_order=c_active_order,
        s_active_order=s_active_order,
        a=a,
        R0=R0,
        Z0=Z0,
    )
    source_eval_runner = _bind_source_eval_runner_for_fused_backend(
        source_eval_binding=backend_abi.build_fused_source_eval_abi(
            source_plan=source_plan,
            backend_state=backend_state,
            B0=B0,
        )
    )
    residual_pack_binding = backend_abi.build_fused_residual_pack_abi(
        backend_state=backend_state,
        a=a,
        R0=R0,
        B0=B0,
    )
    scratch_holder: list[np.ndarray | None] = [None]
    FFn_psin = root_fields[3]
    Pn_psin = root_fields[4]

    def runner(x: np.ndarray) -> np.ndarray:
        _refresh_hot_runtime(x, hot_runtime_binding=hot_runtime_binding)
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
        return _pack_residual_output(residual_pack_binding=residual_pack_binding, scratch_holder=scratch_holder)

    return runner


def _bind_profile_owned_psin_residual_runner_core(
    *,
    source_plan: SourcePlan,
    backend_state: "BackendState",
    apply_f2_transform: bool,
    alpha_state: np.ndarray,
    c_active_order: int,
    s_active_order: int,
    a: float,
    R0: float,
    Z0: float,
    B0: float,
    skip_projection_finalize: bool,
) -> Callable[[np.ndarray], np.ndarray]:
    runtime_layout = backend_state.runtime_layout
    surface_workspace = runtime_layout.geometry_surface_workspace
    residual_workspace = runtime_layout.residual_surface_workspace
    root_fields = runtime_layout.root_fields
    hot_runtime_binding = backend_abi.build_fused_hot_runtime_abi(
        backend_state=backend_state,
        apply_f2_transform=apply_f2_transform,
        c_active_order=c_active_order,
        s_active_order=s_active_order,
        a=a,
        R0=R0,
        Z0=Z0,
    )
    source_eval_runner = _bind_source_eval_runner_for_fused_backend(
        source_eval_binding=backend_abi.build_fused_source_eval_abi(
            source_plan=source_plan,
            backend_state=backend_state,
            B0=B0,
        )
    )
    residual_pack_binding = backend_abi.build_fused_residual_pack_abi(
        backend_state=backend_state,
        a=a,
        R0=R0,
        B0=B0,
    )
    profile_owned_psin_binding = backend_abi.build_profile_owned_psin_source_abi(
        source_plan=source_plan,
        backend_state=backend_state,
        skip_projection_finalize=skip_projection_finalize,
    )
    scratch_holder: list[np.ndarray | None] = [None]
    psin = root_fields[0]
    psin_r = root_fields[1]
    psin_rr = root_fields[2]
    FFn_psin = root_fields[3]
    Pn_psin = root_fields[4]

    def runner(x: np.ndarray) -> np.ndarray:
        _refresh_hot_runtime(x, hot_runtime_binding=hot_runtime_binding)
        _materialize_profile_owned_psin_source_impl(
            psin,
            psin_r,
            psin_rr,
            profile_owned_psin_binding.source_psin_query,
            profile_owned_psin_binding.source_parameter_query,
            profile_owned_psin_binding.materialized_heat_input,
            profile_owned_psin_binding.materialized_current_input,
            profile_owned_psin_binding.psin_profile_fields,
            profile_owned_psin_binding.heat_input,
            profile_owned_psin_binding.current_input,
            profile_owned_psin_binding.parameterization_code,
        )
        # PI psin-uniform is more accurate with the direct source-owned interpolation
        # than with the extra projected rematerialization used by other routes.
        if (
            profile_owned_psin_binding.has_projection_policy
            and not profile_owned_psin_binding.skip_projection_finalize
        ):
            _materialize_projected_source_inputs_impl(
                profile_owned_psin_binding.materialized_heat_input,
                profile_owned_psin_binding.materialized_current_input,
                profile_owned_psin_binding.heat_projection_coeff,
                profile_owned_psin_binding.current_projection_coeff,
                profile_owned_psin_binding.current_input,
                profile_owned_psin_binding.source_psin_query,
                profile_owned_psin_binding.projection_domain_code,
                profile_owned_psin_binding.endpoint_policy_code,
                profile_owned_psin_binding.endpoint_blend,
            )
        alpha1, alpha2 = source_eval_runner(
            profile_owned_psin_binding.source_target_root_fields,
            FFn_psin,
            Pn_psin,
            profile_owned_psin_binding.materialized_heat_input,
            profile_owned_psin_binding.materialized_current_input,
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
        return _pack_residual_output(residual_pack_binding=residual_pack_binding, scratch_holder=scratch_holder)

    return runner


def _bind_pj2_psin_fixed_point_residual_runner_core(
    *,
    source_plan: SourcePlan,
    backend_state: "BackendState",
    apply_f2_transform: bool,
    alpha_state: np.ndarray,
    c_active_order: int,
    s_active_order: int,
    a: float,
    R0: float,
    Z0: float,
    B0: float,
) -> Callable[[np.ndarray], np.ndarray]:
    static_layout = backend_state.static_layout
    runtime_layout = backend_state.runtime_layout
    surface_workspace = runtime_layout.geometry_surface_workspace
    radial_workspace = runtime_layout.geometry_radial_workspace
    residual_workspace = runtime_layout.residual_surface_workspace
    rho = static_layout.rho
    weights = static_layout.weights
    differentiation_matrix = static_layout.differentiation_matrix
    integration_matrix = static_layout.integration_matrix
    root_fields = runtime_layout.root_fields
    hot_runtime_binding = backend_abi.build_fused_hot_runtime_abi(
        backend_state=backend_state,
        apply_f2_transform=apply_f2_transform,
        c_active_order=c_active_order,
        s_active_order=s_active_order,
        a=a,
        R0=R0,
        Z0=Z0,
    )
    residual_pack_binding = backend_abi.build_fused_residual_pack_abi(
        backend_state=backend_state,
        a=a,
        R0=R0,
        B0=B0,
    )
    fixed_point_psin_binding = backend_abi.build_pj2_fixed_point_psin_source_abi(
        source_plan=source_plan,
        backend_state=backend_state,
    )
    scratch_holder: list[np.ndarray | None] = [None]
    psin = root_fields[0]
    FFn_psin = root_fields[3]
    Pn_psin = root_fields[4]

    def runner(x: np.ndarray) -> np.ndarray:
        _refresh_hot_runtime(x, hot_runtime_binding=hot_runtime_binding)
        if fixed_point_psin_binding.source_psin_query[0] < 0.0:
            _normalize_psin_query(
                fixed_point_psin_binding.source_psin_query,
                fixed_point_psin_binding.psin_profile_u,
            )
        if fixed_point_psin_binding.has_Ip:
            alpha1, alpha2 = _run_fixed_point_barycentric_with_scratch_impl(
                _update_pj2_from_psin_inputs_with_scratch,
                16,
                1.0e-10,
                fixed_point_psin_binding.source_psin_query,
                psin,
                root_fields,
                FFn_psin,
                Pn_psin,
                fixed_point_psin_binding.materialized_heat_input,
                fixed_point_psin_binding.materialized_current_input,
                fixed_point_psin_binding.heat_input,
                fixed_point_psin_binding.current_input,
                fixed_point_psin_binding.barycentric_weights,
                fixed_point_psin_binding.coordinate_code,
                R0,
                B0,
                weights,
                differentiation_matrix,
                integration_matrix,
                rho,
                radial_workspace,
                surface_workspace,
                fixed_point_psin_binding.F_profile_u,
                fixed_point_psin_binding.Ip,
                fixed_point_psin_binding.beta,
                fixed_point_psin_binding.source_scratch_1d,
            )
        else:
            alpha1, alpha2 = _run_fixed_point_linear_with_scratch_impl(
                _update_pj2_from_psin_inputs_with_scratch,
                16,
                1.0e-10,
                fixed_point_psin_binding.source_psin_query,
                psin,
                root_fields,
                FFn_psin,
                Pn_psin,
                fixed_point_psin_binding.materialized_heat_input,
                fixed_point_psin_binding.materialized_current_input,
                fixed_point_psin_binding.heat_input,
                fixed_point_psin_binding.current_input,
                fixed_point_psin_binding.coordinate_code,
                R0,
                B0,
                weights,
                differentiation_matrix,
                integration_matrix,
                rho,
                radial_workspace,
                surface_workspace,
                fixed_point_psin_binding.F_profile_u,
                fixed_point_psin_binding.Ip,
                fixed_point_psin_binding.beta,
                fixed_point_psin_binding.source_scratch_1d,
            )
        alpha1, alpha2 = _run_projected_finalize_with_scratch_impl(
            _update_pj2_from_psin_inputs_with_scratch,
            fixed_point_psin_binding.finalize_iter,
            1.0e-10,
            fixed_point_psin_binding.source_psin_query,
            psin,
            root_fields,
            FFn_psin,
            Pn_psin,
            fixed_point_psin_binding.materialized_heat_input,
            fixed_point_psin_binding.materialized_current_input,
            fixed_point_psin_binding.heat_projection_coeff,
            fixed_point_psin_binding.current_projection_coeff,
            fixed_point_psin_binding.current_input,
            fixed_point_psin_binding.projection_domain_code,
            fixed_point_psin_binding.endpoint_policy_code,
            fixed_point_psin_binding.endpoint_blend,
            fixed_point_psin_binding.coordinate_code,
            R0,
            B0,
            weights,
            differentiation_matrix,
            integration_matrix,
            rho,
            radial_workspace,
            surface_workspace,
            fixed_point_psin_binding.F_profile_u,
            fixed_point_psin_binding.Ip,
            fixed_point_psin_binding.beta,
            fixed_point_psin_binding.source_scratch_1d,
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
        return _pack_residual_output(residual_pack_binding=residual_pack_binding, scratch_holder=scratch_holder)

    return runner


def _bind_source_eval_runner_for_fused_backend(
    *,
    source_eval_binding: backend_abi.FusedSourceEvalABI,
) -> Callable:
    def runner(
        out_root_fields: np.ndarray,
        out_FFn_psin: np.ndarray,
        out_Pn_psin: np.ndarray,
        heat_input: np.ndarray,
        current_input: np.ndarray,
        R0: float,
    ) -> tuple[float, float]:
        if source_eval_binding.scratch_source_kernel is None:
            return source_eval_binding.source_kernel(
                out_root_fields,
                out_FFn_psin,
                out_Pn_psin,
                heat_input,
                current_input,
                source_eval_binding.coordinate_code,
                R0,
                source_eval_binding.B0,
                source_eval_binding.weights,
                source_eval_binding.differentiation_matrix,
                source_eval_binding.integration_matrix,
                source_eval_binding.rho,
                source_eval_binding.radial_workspace,
                source_eval_binding.surface_workspace,
                source_eval_binding.F_profile_u,
                source_eval_binding.Ip,
                source_eval_binding.beta,
            )
        return source_eval_binding.scratch_source_kernel(
            out_root_fields,
            out_FFn_psin,
            out_Pn_psin,
            heat_input,
            current_input,
            source_eval_binding.coordinate_code,
            R0,
            source_eval_binding.B0,
            source_eval_binding.weights,
            source_eval_binding.differentiation_matrix,
            source_eval_binding.integration_matrix,
            source_eval_binding.rho,
            source_eval_binding.radial_workspace,
            source_eval_binding.surface_workspace,
            source_eval_binding.F_profile_u,
            source_eval_binding.Ip,
            source_eval_binding.beta,
            source_eval_binding.source_scratch_1d,
        )

    return runner


def _bind_pq_psin_fixed_point_residual_runner_core(
    *,
    source_plan: SourcePlan,
    backend_state: "BackendState",
    apply_f2_transform: bool,
    alpha_state: np.ndarray,
    c_active_order: int,
    s_active_order: int,
    a: float,
    R0: float,
    Z0: float,
    B0: float,
) -> Callable[[np.ndarray], np.ndarray]:
    static_layout = backend_state.static_layout
    runtime_layout = backend_state.runtime_layout
    surface_workspace = runtime_layout.geometry_surface_workspace
    radial_workspace = runtime_layout.geometry_radial_workspace
    residual_workspace = runtime_layout.residual_surface_workspace
    rho = static_layout.rho
    weights = static_layout.weights
    differentiation_matrix = static_layout.differentiation_matrix
    integration_matrix = static_layout.integration_matrix
    root_fields = runtime_layout.root_fields
    hot_runtime_binding = backend_abi.build_fused_hot_runtime_abi(
        backend_state=backend_state,
        apply_f2_transform=apply_f2_transform,
        c_active_order=c_active_order,
        s_active_order=s_active_order,
        a=a,
        R0=R0,
        Z0=Z0,
    )
    residual_pack_binding = backend_abi.build_fused_residual_pack_abi(
        backend_state=backend_state,
        a=a,
        R0=R0,
        B0=B0,
    )
    fixed_point_psin_binding = backend_abi.build_pq_fixed_point_psin_source_abi(
        source_plan=source_plan,
        backend_state=backend_state,
    )
    scratch_holder: list[np.ndarray | None] = [None]
    psin = root_fields[0]
    FFn_psin = root_fields[3]
    Pn_psin = root_fields[4]

    def runner(x: np.ndarray) -> np.ndarray:
        _refresh_hot_runtime(x, hot_runtime_binding=hot_runtime_binding)
        if (not fixed_point_psin_binding.allow_query_warmstart) or fixed_point_psin_binding.source_psin_query[0] < 0.0:
            _normalize_psin_query(
                fixed_point_psin_binding.source_psin_query,
                fixed_point_psin_binding.psin_profile_u,
            )
        if fixed_point_psin_binding.has_Ip:
            alpha1, alpha2 = _run_fixed_point_barycentric_with_scratch_impl(
                _update_pq_from_psin_inputs_with_scratch,
                16,
                1.0e-10,
                fixed_point_psin_binding.source_psin_query,
                psin,
                root_fields,
                FFn_psin,
                Pn_psin,
                fixed_point_psin_binding.materialized_heat_input,
                fixed_point_psin_binding.materialized_current_input,
                fixed_point_psin_binding.heat_input,
                fixed_point_psin_binding.current_input,
                fixed_point_psin_binding.barycentric_weights,
                fixed_point_psin_binding.coordinate_code,
                R0,
                B0,
                weights,
                differentiation_matrix,
                integration_matrix,
                rho,
                radial_workspace,
                surface_workspace,
                fixed_point_psin_binding.F_profile_u,
                fixed_point_psin_binding.Ip,
                fixed_point_psin_binding.beta,
                fixed_point_psin_binding.source_scratch_1d,
            )
        else:
            alpha1, alpha2 = _run_fixed_point_linear_with_scratch_impl(
                _update_pq_from_psin_inputs_with_scratch,
                16,
                1.0e-10,
                fixed_point_psin_binding.source_psin_query,
                psin,
                root_fields,
                FFn_psin,
                Pn_psin,
                fixed_point_psin_binding.materialized_heat_input,
                fixed_point_psin_binding.materialized_current_input,
                fixed_point_psin_binding.heat_input,
                fixed_point_psin_binding.current_input,
                fixed_point_psin_binding.coordinate_code,
                R0,
                B0,
                weights,
                differentiation_matrix,
                integration_matrix,
                rho,
                radial_workspace,
                surface_workspace,
                fixed_point_psin_binding.F_profile_u,
                fixed_point_psin_binding.Ip,
                fixed_point_psin_binding.beta,
                fixed_point_psin_binding.source_scratch_1d,
            )
        alpha1, alpha2 = _run_projected_finalize_with_scratch_impl(
            _update_pq_from_psin_inputs_with_scratch,
            fixed_point_psin_binding.finalize_iter,
            1.0e-10,
            fixed_point_psin_binding.source_psin_query,
            psin,
            root_fields,
            FFn_psin,
            Pn_psin,
            fixed_point_psin_binding.materialized_heat_input,
            fixed_point_psin_binding.materialized_current_input,
            fixed_point_psin_binding.heat_projection_coeff,
            fixed_point_psin_binding.current_projection_coeff,
            fixed_point_psin_binding.current_input,
            fixed_point_psin_binding.projection_domain_code,
            fixed_point_psin_binding.endpoint_policy_code,
            fixed_point_psin_binding.endpoint_blend,
            fixed_point_psin_binding.coordinate_code,
            R0,
            B0,
            weights,
            differentiation_matrix,
            integration_matrix,
            rho,
            radial_workspace,
            surface_workspace,
            fixed_point_psin_binding.F_profile_u,
            fixed_point_psin_binding.Ip,
            fixed_point_psin_binding.beta,
            fixed_point_psin_binding.source_scratch_1d,
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
        return _pack_residual_output(residual_pack_binding=residual_pack_binding, scratch_holder=scratch_holder)

    return runner

"""
Module: layout.stage_binding

Role:
- Bind source and geometry stage runners from already-built layout/workspace state.
- Keep Python closure wiring separate from source planning and runtime memory refresh.

Notes:
- This module binds preallocated arrays and engine callables; it does not allocate memory.
- Numerical kernels remain in ``veqpy.engine``.
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np

from veqpy.engine.numba_geometry import update_geometry_hot
from veqpy.engine.numba_source import (
    PJ2_PSIN_UNIFORM_FIXED_POINT_FINALIZE_ITER,
    PJ2_PSIN_UNIFORM_FIXED_POINT_MAX_ITER,
    PJ2_PSIN_UNIFORM_FIXED_POINT_MAX_RESIDUAL,
    materialize_profile_owned_psin_source,
    materialize_projected_source_inputs,
    resolve_source_inputs,
    update_fixed_point_psin_query,
    update_fourier_family_fields,
)


def build_bound_source_stage_runner(
    operator_core: Any, *, source_eval_runner: Callable | None = None
) -> Callable:
    route_key = tuple(operator_core.plan.source_execution.route_key)
    if route_key == ("PJ2", "psin", "uniform"):
        return _build_pj2_psin_uniform_source_stage_runner(
            operator_core, source_eval_runner=source_eval_runner
        )
    return _build_source_stage_runner_shared(operator_core, source_eval_runner=source_eval_runner)


def _build_source_stage_runner_shared(
    operator_core: Any, *, source_eval_runner: Callable | None = None
) -> Callable:
    source_plan = operator_core.plan.source_plan
    source_execution = operator_core.plan.source_execution
    if source_eval_runner is None:
        source_eval_runner = operator_core.layout.source.run_eval
    workspace = operator_core.workspace
    source_runtime_state = workspace.source.runtime_state
    source_work_state = source_runtime_state.work_state
    source_aux_state = source_runtime_state.aux_state
    root_fields = workspace.residual.root_fields
    psin = root_fields[0]
    psin_r = root_fields[1]
    psin_rr = root_fields[2]
    FFn_psin = root_fields[3]
    Pn_psin = root_fields[4]
    materialized_heat_input = source_work_state.materialized_heat_input
    materialized_current_input = source_work_state.materialized_current_input
    source_target_root_fields = source_aux_state.target_root_fields
    case_R0 = float(operator_core.case.R0)

    if source_execution.requires_optimized_psin_profile:
        if source_plan.is_psin_coordinate and not source_plan.is_grid_nodes:
            source_psin_query = source_work_state.psin_query
            source_parameter_query = source_work_state.parameter_query
            psin_profile_fields = workspace.psin_profile_fields
            heat_input = source_plan.heat_input
            current_input = source_plan.current_input
            parameterization_code = source_plan.parameterization_code
            static_layout = operator_core.plan.static_layout
            n_axis_fix = int(np.searchsorted(static_layout.rho, operator_core.fix_rho))

            def runner() -> tuple[float, float]:
                if psin_profile_fields.size == 0:
                    raise RuntimeError("psin_profile runtime fields are not initialized")
                materialize_profile_owned_psin_source(
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
                    source_aux_state.heat_spline_coeff,
                    source_aux_state.current_spline_coeff,
                    parameterization_code,
                    static_layout.rho,
                    static_layout.differentiator,
                    static_layout.accumulator,
                    n_axis_fix,
                    source_runtime_state.const_state.barycentric_weights,
                    source_plan.uses_barycentric_interpolation,
                )
                if source_plan.has_projection_policy:
                    materialize_projected_source_inputs(
                        source_work_state.materialized_heat_input,
                        source_work_state.materialized_current_input,
                        source_aux_state.heat_projection_coeff,
                        source_aux_state.current_projection_coeff,
                        source_plan.current_input,
                        source_psin_query,
                        source_plan.projection_domain_code,
                        source_plan.endpoint_policy_code,
                        source_runtime_state.const_state.endpoint_blend,
                    )
                return source_eval_runner(
                    source_target_root_fields,
                    FFn_psin,
                    Pn_psin,
                    materialized_heat_input,
                    materialized_current_input,
                    case_R0,
                )

            return runner

        source_psin_query = source_work_state.psin_query
        psin_profile_u = workspace.psin_profile_u
        psin_profile_fields = workspace.psin_profile_fields

        def runner() -> tuple[float, float]:
            if psin_profile_fields.size == 0:
                raise RuntimeError("psin_profile runtime fields are not initialized")
            np.copyto(psin, psin_profile_u)
            np.copyto(psin_r, psin_profile_fields[1])
            np.copyto(psin_rr, psin_profile_fields[2])
            np.copyto(source_psin_query, psin)
            if source_plan.has_projection_policy:
                materialize_projected_source_inputs(
                    source_work_state.materialized_heat_input,
                    source_work_state.materialized_current_input,
                    source_aux_state.heat_projection_coeff,
                    source_aux_state.current_projection_coeff,
                    source_plan.current_input,
                    source_psin_query,
                    source_plan.projection_domain_code,
                    source_plan.endpoint_policy_code,
                    source_runtime_state.const_state.endpoint_blend,
                )
            else:
                if source_plan.parameterization == "identity":
                    np.copyto(source_work_state.parameter_query, source_psin_query)
                elif source_plan.parameterization == "sqrt_psin":
                    np.copyto(source_work_state.parameter_query, source_psin_query)
                    np.maximum(
                        source_work_state.parameter_query,
                        0.0,
                        out=source_work_state.parameter_query,
                    )
                    np.sqrt(
                        source_work_state.parameter_query, out=source_work_state.parameter_query
                    )
                else:
                    raise ValueError(
                        f"Unsupported source parameterization {source_plan.parameterization!r}"
                    )
                resolve_source_inputs(
                    source_work_state.materialized_heat_input,
                    source_work_state.materialized_current_input,
                    source_plan.heat_input,
                    source_plan.current_input,
                    source_plan.coordinate_code,
                    source_plan.source_sample_count,
                    source_runtime_state.const_state.barycentric_weights,
                    source_runtime_state.const_state.fixed_remap_matrix,
                    source_aux_state.heat_spline_coeff,
                    source_aux_state.current_spline_coeff,
                    source_work_state.parameter_query,
                    source_plan.uses_barycentric_interpolation,
                )
            return source_eval_runner(
                source_target_root_fields,
                FFn_psin,
                Pn_psin,
                materialized_heat_input,
                materialized_current_input,
                case_R0,
            )

        return runner

    def runner() -> tuple[float, float]:
        return source_eval_runner(
            root_fields,
            FFn_psin,
            Pn_psin,
            materialized_heat_input,
            materialized_current_input,
            case_R0,
        )

    return runner


def _build_pj2_psin_uniform_source_stage_runner(
    operator_core: Any,
    *,
    source_eval_runner: Callable | None = None,
) -> Callable[[], tuple[float, float]]:
    source_plan = operator_core.plan.source_plan
    if source_eval_runner is None:
        source_eval_runner = operator_core.layout.source.run_eval
    workspace = operator_core.workspace
    source_runtime_state = workspace.source.runtime_state
    source_work_state = source_runtime_state.work_state
    source_aux_state = source_runtime_state.aux_state
    target_root_fields = source_aux_state.target_root_fields
    psin_profile_u = workspace.psin_profile_u
    root_fields = workspace.residual.root_fields
    psin = root_fields[0]
    psin_r = root_fields[1]
    psin_rr = root_fields[2]
    FFn_psin = root_fields[3]
    Pn_psin = root_fields[4]
    case_R0 = float(operator_core.case.R0)

    def runner() -> tuple[float, float]:
        if source_work_state.psin_query[0] < 0.0:
            if psin_profile_u is None:
                raise RuntimeError("psin_profile.u is not initialized")
            np.copyto(source_work_state.psin_query, np.asarray(psin_profile_u, dtype=np.float64))
            if source_work_state.psin_query.ndim != 1 or source_work_state.psin_query.size < 2:
                raise ValueError(
                    f"Expected psin query to be 1D with at least two points, "
                    f"got {source_work_state.psin_query.shape}"
                )
            offset = float(source_work_state.psin_query[0])
            scale = float(source_work_state.psin_query[-1] - offset)
            if abs(scale) < 1e-12:
                raise ValueError("psin query does not span a valid normalized flux interval")
            source_work_state.psin_query -= offset
            source_work_state.psin_query /= scale
            source_work_state.psin_query[0] = 0.0
            source_work_state.psin_query[-1] = 1.0
        alpha1 = float("nan")
        alpha2 = float("nan")
        for _ in range(PJ2_PSIN_UNIFORM_FIXED_POINT_MAX_ITER):
            if source_plan.parameterization == "identity":
                np.copyto(source_work_state.parameter_query, source_work_state.psin_query)
            elif source_plan.parameterization == "sqrt_psin":
                np.copyto(source_work_state.parameter_query, source_work_state.psin_query)
                np.maximum(
                    source_work_state.parameter_query, 0.0, out=source_work_state.parameter_query
                )
                np.sqrt(source_work_state.parameter_query, out=source_work_state.parameter_query)
            else:
                raise ValueError(
                    f"Unsupported source parameterization {source_plan.parameterization!r}"
                )
            resolve_source_inputs(
                source_work_state.materialized_heat_input,
                source_work_state.materialized_current_input,
                source_plan.heat_input,
                source_plan.current_input,
                source_plan.coordinate_code,
                source_plan.source_sample_count,
                source_runtime_state.const_state.barycentric_weights,
                source_runtime_state.const_state.fixed_remap_matrix,
                source_aux_state.heat_spline_coeff,
                source_aux_state.current_spline_coeff,
                source_work_state.parameter_query,
                source_plan.uses_barycentric_interpolation,
            )
            alpha1, alpha2 = source_eval_runner(
                target_root_fields,
                FFn_psin,
                Pn_psin,
                source_work_state.materialized_heat_input,
                source_work_state.materialized_current_input,
                case_R0,
            )
            if update_fixed_point_psin_query(
                source_work_state.psin_query,
                target_root_fields[0],
                PJ2_PSIN_UNIFORM_FIXED_POINT_MAX_RESIDUAL,
            ):
                break
        np.copyto(source_work_state.psin_query, target_root_fields[0])
        if source_plan.has_projection_policy:
            for _ in range(PJ2_PSIN_UNIFORM_FIXED_POINT_FINALIZE_ITER):
                materialize_projected_source_inputs(
                    source_work_state.materialized_heat_input,
                    source_work_state.materialized_current_input,
                    source_aux_state.heat_projection_coeff,
                    source_aux_state.current_projection_coeff,
                    source_plan.current_input,
                    source_work_state.psin_query,
                    source_plan.projection_domain_code,
                    source_plan.endpoint_policy_code,
                    source_runtime_state.const_state.endpoint_blend,
                )
                alpha1, alpha2 = source_eval_runner(
                    target_root_fields,
                    FFn_psin,
                    Pn_psin,
                    source_work_state.materialized_heat_input,
                    source_work_state.materialized_current_input,
                    case_R0,
                )
                if update_fixed_point_psin_query(
                    source_work_state.psin_query,
                    target_root_fields[0],
                    PJ2_PSIN_UNIFORM_FIXED_POINT_MAX_RESIDUAL,
                ):
                    break
        np.copyto(psin, target_root_fields[0])
        np.copyto(psin_r, target_root_fields[1])
        np.copyto(psin_rr, target_root_fields[2])
        return alpha1, alpha2

    return runner


def build_geometry_stage_runner(
    *,
    c_family_fields: np.ndarray,
    s_family_fields: np.ndarray,
    c_family_base_fields: np.ndarray,
    s_family_base_fields: np.ndarray,
    active_u_fields: np.ndarray,
    c_family_source_slots: np.ndarray,
    s_family_source_slots: np.ndarray,
    c_effective_order: int,
    s_effective_order: int,
    h_fields: np.ndarray,
    v_fields: np.ndarray,
    k_fields: np.ndarray,
    a: float,
    R0: float,
    Z0: float,
    surface_workspace: np.ndarray,
    radial_workspace: np.ndarray,
    rho: np.ndarray,
    theta: np.ndarray,
    cos_mtheta: np.ndarray,
    sin_mtheta: np.ndarray,
    m_cos_mtheta: np.ndarray,
    m_sin_mtheta: np.ndarray,
    m2_cos_mtheta: np.ndarray,
    m2_sin_mtheta: np.ndarray,
) -> Callable[[], None]:
    c_effective_order = int(c_effective_order)
    s_effective_order = int(s_effective_order)
    a = float(a)
    R0 = float(R0)
    Z0 = float(Z0)

    def runner() -> None:
        update_fourier_family_fields(
            c_family_fields,
            s_family_fields,
            c_family_base_fields,
            s_family_base_fields,
            active_u_fields,
            c_family_source_slots,
            s_family_source_slots,
            c_effective_order,
            s_effective_order,
        )
        update_geometry_hot(
            surface_workspace,
            radial_workspace,
            a,
            R0,
            Z0,
            rho,
            theta,
            cos_mtheta,
            sin_mtheta,
            m_cos_mtheta,
            m_sin_mtheta,
            m2_cos_mtheta,
            m2_sin_mtheta,
            h_fields,
            v_fields,
            k_fields,
            c_family_fields,
            s_family_fields,
            c_effective_order,
            s_effective_order,
        )

    return runner

__all__ = ["build_bound_source_stage_runner", "build_geometry_stage_runner"]

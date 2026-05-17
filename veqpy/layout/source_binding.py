"""
Module: layout.source_binding

Role:
- Bind source stage runners from already-built layout/workspace state.
- Keep Python closure wiring separate from source planning and runtime memory refresh.

Notes:
- This module binds preallocated arrays and engine callables; it does not allocate memory.
- Numerical kernels remain in ``veqpy.engine``.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from veqpy.engine.numba_source import (
    PJ2_PSIN_UNIFORM_FIXED_POINT_MAX_ITER,
    PJ2_PSIN_UNIFORM_FIXED_POINT_MAX_RESIDUAL,
    materialize_profile_owned_psin_source,
    resolve_source_inputs,
    update_fixed_point_psin_query,
)


def build_bound_source_stage_runner(
    *,
    plan,
    case,
    workspace,
    fix_rho: float,
    source_eval_runner: Callable,
) -> Callable:
    route_key = tuple(plan.source_execution.route_key)
    if route_key == ("PJ2", "psin", "uniform"):
        return _build_pj2_psin_uniform_source_stage_runner(
            plan=plan,
            case=case,
            workspace=workspace,
            source_eval_runner=source_eval_runner,
        )
    return _build_source_stage_runner_shared(
        plan=plan,
        case=case,
        workspace=workspace,
        fix_rho=fix_rho,
        source_eval_runner=source_eval_runner,
    )


def _build_source_stage_runner_shared(
    *,
    plan,
    case,
    workspace,
    fix_rho: float,
    source_eval_runner: Callable,
) -> Callable:
    source_plan = plan.source_plan
    source_execution = plan.source_execution
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
    case_R0 = float(case.R0)

    if source_execution.requires_optimized_psin_profile:
        if source_plan.is_psin_coordinate and not source_plan.is_grid_nodes:
            source_psin_query = source_work_state.psin_query
            source_parameter_query = source_work_state.parameter_query
            psin_profile_fields = workspace.psin_profile_fields
            heat_input = source_plan.heat_input
            current_input = source_plan.current_input
            parameterization_code = source_plan.parameterization_code
            static_layout = plan.static_layout
            n_axis_fix = int(np.searchsorted(static_layout.rho, fix_rho))

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
            if source_plan.parameterization == "identity":
                np.copyto(source_work_state.parameter_query, source_psin_query)
            elif source_plan.parameterization == "sqrt_psin":
                np.copyto(source_work_state.parameter_query, source_psin_query)
                np.maximum(
                    source_work_state.parameter_query,
                    0.0,
                    out=source_work_state.parameter_query,
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


__all__ = ["build_bound_source_stage_runner"]
def _build_pj2_psin_uniform_source_stage_runner(
    *,
    plan,
    case,
    workspace,
    source_eval_runner: Callable,
) -> Callable[[], tuple[float, float]]:
    source_plan = plan.source_plan
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
    case_R0 = float(case.R0)

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
        np.copyto(psin, target_root_fields[0])
        np.copyto(psin_r, target_root_fields[1])
        np.copyto(psin_rr, target_root_fields[2])
        return alpha1, alpha2

    return runner


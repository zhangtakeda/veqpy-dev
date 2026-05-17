"""
Module: operator.source_runtime

Role:
- Refresh source runtime memory from a bound ``SourcePlan`` and current psin state.
- Own source input materialization cache updates outside engine kernels.

Notes:
- This module mutates preallocated source runtime arrays in place.
- It does not choose routes, allocate runtime state, or bind source stage runners.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from veqpy.engine.numba_source import (
    build_source_remap_cache,
    resolve_source_inputs,
)
from veqpy.math.interpolate import build_uniform_source_interpolation_coefficients
from veqpy.operator.source_plan import SourcePlan, validate_source_inputs

if TYPE_CHECKING:
    from veqpy.operator.operator_case import OperatorCase
    from veqpy.workspace import SourceRuntimeState


def refresh_source_runtime(
    *,
    case: "OperatorCase",
    grid_rho: np.ndarray,
    source_plan: SourcePlan,
    source_execution: object,
    source_runtime_state: "SourceRuntimeState",
    psin: np.ndarray,
) -> None:
    validate_source_inputs(case, int(grid_rho.shape[0]))
    source_const_state = source_runtime_state.const_state
    source_aux_state = source_runtime_state.aux_state
    source_work_state = source_runtime_state.work_state
    case_key = (
        source_plan.coordinate,
        source_plan.nodes,
        source_plan.source_sample_count,
        source_plan.interpolation_kind if not source_plan.is_grid_nodes else "",
    )
    if source_runtime_state.cache_key != case_key:
        if source_plan.is_grid_nodes:
            source_const_state.barycentric_weights = np.empty(0, dtype=np.float64)
            source_const_state.fixed_remap_matrix = np.empty((0, 0), dtype=np.float64)
            source_aux_state.heat_spline_coeff = np.empty((0, 4), dtype=np.float64)
            source_aux_state.current_spline_coeff = np.empty((0, 4), dtype=np.float64)
        else:
            (
                _,
                source_const_state.barycentric_weights,
                source_const_state.fixed_remap_matrix,
            ) = build_source_remap_cache(
                source_plan.coordinate,
                source_plan.source_sample_count,
                rho=grid_rho,
                interpolation_kind=source_plan.interpolation_kind,
            )
        source_runtime_state.cache_key = case_key
    if source_plan.is_grid_nodes:
        source_aux_state.heat_spline_coeff = np.empty((0, 4), dtype=np.float64)
        source_aux_state.current_spline_coeff = np.empty((0, 4), dtype=np.float64)
    else:
        source_aux_state.heat_spline_coeff = build_uniform_source_interpolation_coefficients(
            source_plan.heat_input,
            kind=source_plan.interpolation_kind,
        )
        source_aux_state.current_spline_coeff = build_uniform_source_interpolation_coefficients(
            source_plan.current_input,
            kind=source_plan.interpolation_kind,
        )
    if source_plan.is_grid_nodes or not source_plan.is_psin_coordinate:
        if source_plan.is_grid_nodes:
            np.copyto(source_work_state.materialized_heat_input, source_plan.heat_input)
            np.copyto(source_work_state.materialized_current_input, source_plan.current_input)
        else:
            resolve_source_inputs(
                source_work_state.materialized_heat_input,
                source_work_state.materialized_current_input,
                source_plan.heat_input,
                source_plan.current_input,
                source_plan.coordinate_code,
                source_plan.source_sample_count,
                source_const_state.barycentric_weights,
                source_const_state.fixed_remap_matrix,
                source_aux_state.heat_spline_coeff,
                source_aux_state.current_spline_coeff,
                psin,
                source_plan.uses_barycentric_interpolation,
            )
    elif tuple(source_execution.route_key) == ("PJ2", "psin", "uniform"):
        source_work_state.psin_query.fill(-1.0)


__all__ = ["refresh_source_runtime"]

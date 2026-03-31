"""
Module: operator.source_orchestration

Role:
- 承载 Operator 的 source/fixed-point Python 协调逻辑.
- 保持热核接口稳定, 仅整理 orchestration 层职责.
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np

from veqpy.engine import materialize_profile_owned_psin_source, update_fixed_point_psin_query
from veqpy.operator.source_runtime import normalize_psin_query_inplace


def build_bound_source_stage_runner(operator_core: Any) -> Callable:
    source_plan = operator_core.source_plan
    psin = operator_core.psin
    psin_r = operator_core.psin_r
    psin_rr = operator_core.psin_rr
    FFn_psin = operator_core.FFn_psin
    Pn_psin = operator_core.Pn_psin
    materialized_heat_input = operator_core.materialized_heat_input
    materialized_current_input = operator_core.materialized_current_input
    source_scratch_1d = operator_core.source_scratch_1d
    source_target_root_fields = operator_core.source_target_root_fields
    grid = operator_core.grid
    geometry = operator_core.geometry
    F_profile_u = operator_core.F_profile.u
    case_R0 = float(operator_core.case.R0)
    case_B0 = float(operator_core.case.B0)
    case_Ip = source_plan.Ip
    case_beta = source_plan.beta

    if source_plan.strategy == "profile_owned_psin":
        if source_plan.is_psin_coordinate and not source_plan.is_grid_nodes:
            source_psin_query = operator_core.source_psin_query
            source_parameter_query = operator_core.source_parameter_query
            psin_profile_fields = operator_core.psin_profile.u_fields
            heat_input = source_plan.heat_input
            current_input = source_plan.current_input
            parameterization_code = source_plan.parameterization_code

            def runner() -> tuple[float, float]:
                if psin_profile_fields is None:
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
                    parameterization_code,
                )
                return _run_source_kernel(
                    operator_core,
                    source_target_root_fields[0],
                    source_target_root_fields[1],
                    source_target_root_fields[2],
                    FFn_psin,
                    Pn_psin,
                    materialized_heat_input,
                    materialized_current_input,
                    case_R0,
                    case_B0,
                    grid.weights,
                    grid.differentiation_matrix,
                    grid.integration_matrix,
                    grid.rho,
                    geometry.V_r,
                    geometry.Kn,
                    geometry.Kn_r,
                    geometry.Ln_r,
                    geometry.S_r,
                    geometry.R,
                    geometry.JdivR,
                    F_profile_u,
                    case_Ip,
                    case_beta,
                    source_scratch_1d,
                )

            return runner

        source_psin_query = operator_core.source_psin_query

        def runner() -> tuple[float, float]:
            copy_psin_profile_to_root_fields(operator_core)
            np.copyto(source_psin_query, psin)
            operator_core._materialize_source_inputs(source_psin_query)
            return _run_source_kernel(
                operator_core,
                source_target_root_fields[0],
                source_target_root_fields[1],
                source_target_root_fields[2],
                FFn_psin,
                Pn_psin,
                materialized_heat_input,
                materialized_current_input,
                case_R0,
                case_B0,
                grid.weights,
                grid.differentiation_matrix,
                grid.integration_matrix,
                grid.rho,
                geometry.V_r,
                geometry.Kn,
                geometry.Kn_r,
                geometry.Ln_r,
                geometry.S_r,
                geometry.R,
                geometry.JdivR,
                F_profile_u,
                case_Ip,
                case_beta,
                source_scratch_1d,
            )

        return runner

    if source_plan.strategy == "fixed_point_psin":

        def runner() -> tuple[float, float]:
            return run_psin_source_fixed_point(operator_core)

        return runner

    def runner() -> tuple[float, float]:
        return _run_source_kernel(
            operator_core,
            psin,
            psin_r,
            psin_rr,
            FFn_psin,
            Pn_psin,
            materialized_heat_input,
            materialized_current_input,
            case_R0,
            case_B0,
            grid.weights,
            grid.differentiation_matrix,
            grid.integration_matrix,
            grid.rho,
            geometry.V_r,
            geometry.Kn,
            geometry.Kn_r,
            geometry.Ln_r,
            geometry.S_r,
            geometry.R,
            geometry.JdivR,
            F_profile_u,
            case_Ip,
            case_beta,
            source_scratch_1d,
        )

    return runner


def copy_psin_profile_to_root_fields(operator_core: Any) -> None:
    if operator_core.psin_profile.u_fields is None:
        raise RuntimeError("psin_profile runtime fields are not initialized")
    np.copyto(operator_core.psin, operator_core.psin_profile.u)
    np.copyto(operator_core.psin_r, operator_core.psin_profile.u_r)
    np.copyto(operator_core.psin_rr, operator_core.psin_profile.u_rr)


def run_profile_owned_psin_source(operator_core: Any) -> tuple[float, float]:
    source_plan = operator_core.source_plan
    if source_plan.is_psin_coordinate and not source_plan.is_grid_nodes:
        if operator_core.psin_profile.u_fields is None:
            raise RuntimeError("psin_profile runtime fields are not initialized")
        materialize_profile_owned_psin_source(
            operator_core.psin,
            operator_core.psin_r,
            operator_core.psin_rr,
            operator_core.source_psin_query,
            operator_core.source_parameter_query,
            operator_core.materialized_heat_input,
            operator_core.materialized_current_input,
            operator_core.psin_profile.u_fields,
            source_plan.heat_input,
            source_plan.current_input,
            source_plan.parameterization_code,
        )
    else:
        copy_psin_profile_to_root_fields(operator_core)
        np.copyto(operator_core.source_psin_query, operator_core.psin)
        operator_core._materialize_source_inputs(operator_core.source_psin_query)
    return _run_source_kernel_from_operator(operator_core)


def run_psin_source_fixed_point(operator_core: Any) -> tuple[float, float]:
    source_plan = operator_core.source_plan
    if (not source_plan.allow_query_warmstart) or operator_core.source_psin_query[0] < 0.0:
        normalize_psin_query_inplace(operator_core.source_psin_query, operator_core.psin_profile.u)
    alpha1 = float("nan")
    alpha2 = float("nan")
    for _ in range(8):
        operator_core._materialize_source_inputs(
            operator_core.source_psin_query,
            enable_projection=not source_plan.use_projected_finalize,
        )
        alpha1, alpha2 = _run_source_kernel_from_operator(operator_core)
        if update_fixed_point_psin_query(operator_core.source_psin_query, operator_core.psin, 1e-10):
            break
    if source_plan.use_projected_finalize:
        np.copyto(operator_core.source_psin_query, operator_core.psin)
        operator_core._materialize_source_inputs(operator_core.source_psin_query, enable_projection=True)
        alpha1, alpha2 = _run_source_kernel_from_operator(operator_core)
    return alpha1, alpha2


def invalidate_source_state(operator_core: Any) -> None:
    if operator_core.source_strategy == "fixed_point_psin":
        operator_core.source_psin_query.fill(-1.0)


def _run_source_kernel_from_operator(operator_core: Any) -> tuple[float, float]:
    return _run_source_kernel(
        operator_core,
        operator_core.source_target_root_fields[0],
        operator_core.source_target_root_fields[1],
        operator_core.source_target_root_fields[2],
        operator_core.FFn_psin,
        operator_core.Pn_psin,
        operator_core.materialized_heat_input,
        operator_core.materialized_current_input,
        float(operator_core.case.R0),
        float(operator_core.case.B0),
        operator_core.grid.weights,
        operator_core.grid.differentiation_matrix,
        operator_core.grid.integration_matrix,
        operator_core.grid.rho,
        operator_core.geometry.V_r,
        operator_core.geometry.Kn,
        operator_core.geometry.Kn_r,
        operator_core.geometry.Ln_r,
        operator_core.geometry.S_r,
        operator_core.geometry.R,
        operator_core.geometry.JdivR,
        operator_core.F_profile.u,
        float(operator_core.case.Ip),
        float(operator_core.case.beta),
        operator_core.source_scratch_1d,
    )


def _run_source_kernel(
    operator_core: Any,
    out_psin: np.ndarray,
    out_psin_r: np.ndarray,
    out_psin_rr: np.ndarray,
    out_FFn_psin: np.ndarray,
    out_Pn_psin: np.ndarray,
    heat_input: np.ndarray,
    current_input: np.ndarray,
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
    R: np.ndarray,
    JdivR: np.ndarray,
    F: np.ndarray,
    Ip: float,
    beta: float,
    source_scratch_1d: np.ndarray,
) -> tuple[float, float]:
    return operator_core._source_runner(
        out_psin,
        out_psin_r,
        out_psin_rr,
        out_FFn_psin,
        out_Pn_psin,
        heat_input,
        current_input,
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
        R,
        JdivR,
        F,
        Ip,
        beta,
        source_scratch_1d,
    )

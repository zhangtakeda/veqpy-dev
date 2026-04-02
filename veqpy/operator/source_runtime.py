"""
Module: operator.source_runtime

Role:
- 承载 source runtime 的共享静态策略与 helper.
- 避免 operator.py 混入过多 projection/materialization 细节.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable

import numpy as np
from numpy.polynomial.chebyshev import chebvander

from veqpy.engine import (
    build_source_remap_cache,
    materialize_projected_source_inputs,
    resolve_source_inputs,
    resolve_source_scratch_kernel,
)

if TYPE_CHECKING:
    from veqpy.operator.layouts import SourceRuntimeState
    from veqpy.operator.operator_case import OperatorCase
    from veqpy.operator.plans import SourcePlan


@dataclass(frozen=True, slots=True)
class SourceProjectionPolicy:
    domain: str
    heat_degree: int
    current_degree: int
    current_ip_endpoint_policy: str = "none"
    current_other_endpoint_policy: str = "none"


SOURCE_PROJECTION_POLICIES: dict[tuple[str, str, str], SourceProjectionPolicy] = {
    ("PI", "psin", "uniform"): SourceProjectionPolicy(
        domain="psin",
        heat_degree=7,
        current_degree=8,
        current_ip_endpoint_policy="both",
        current_other_endpoint_policy="none",
    ),
    ("PJ1", "psin", "uniform"): SourceProjectionPolicy(
        domain="psin",
        heat_degree=7,
        current_degree=8,
        current_ip_endpoint_policy="both",
        current_other_endpoint_policy="none",
    ),
    ("PQ", "psin", "uniform"): SourceProjectionPolicy(
        domain="psin",
        heat_degree=8,
        current_degree=8,
        current_ip_endpoint_policy="affine_both",
        current_other_endpoint_policy="none",
    )
}

PROJECTION_DOMAIN_CODES = {
    "psin": 0,
    "sqrt_psin": 1,
}

ENDPOINT_POLICY_CODES = {
    "none": 0,
    "right": 1,
    "both": 2,
    "affine_both": 3,
}

SOURCE_PARAMETERIZATION_CODES = {
    "identity": 0,
    "sqrt_psin": 1,
}


def normalize_psin_query_inplace(out: np.ndarray, source: np.ndarray | None) -> np.ndarray:
    if source is None:
        raise RuntimeError("psin_profile.u is not initialized")

    np.copyto(out, np.asarray(source, dtype=np.float64))
    if out.ndim != 1 or out.size < 2:
        raise ValueError(f"Expected psin query to be 1D with at least two points, got {out.shape}")

    offset = float(out[0])
    scale = float(out[-1] - offset)
    if abs(scale) < 1e-12:
        raise ValueError("psin query does not span a valid normalized flux interval")

    out -= offset
    out /= scale
    out[0] = 0.0
    out[-1] = 1.0
    return out


def parameterize_psin_query_inplace(out: np.ndarray, source: np.ndarray, parameterization: str) -> np.ndarray:
    np.copyto(out, np.asarray(source, dtype=np.float64))
    if parameterization == "identity":
        return out
    if parameterization == "sqrt_psin":
        np.maximum(out, 0.0, out=out)
        np.sqrt(out, out=out)
        return out
    raise ValueError(f"Unsupported source parameterization {parameterization!r}")


def _parameterize_projection_query_inplace(out: np.ndarray, source: np.ndarray, domain: str) -> np.ndarray:
    np.copyto(out, np.asarray(source, dtype=np.float64))
    np.clip(out, 0.0, 1.0, out=out)
    if domain == "psin":
        out *= 2.0
        out -= 1.0
        return out
    if domain == "sqrt_psin":
        np.sqrt(out, out=out)
        out *= 2.0
        out -= 1.0
        return out
    raise ValueError(f"Unsupported source projection domain {domain!r}")


def build_source_projection_fit_matrix(n_src: int, *, degree: int, domain: str) -> np.ndarray:
    if degree < 0:
        raise ValueError(f"Projection degree must be non-negative, got {degree}")
    source_axis = np.linspace(0.0, 1.0, int(n_src), dtype=np.float64)
    source_query = np.empty_like(source_axis)
    _parameterize_projection_query_inplace(source_query, source_axis, domain)
    vandermonde = chebvander(source_query, degree)
    return np.linalg.pinv(vandermonde)


def validate_source_inputs(case: "OperatorCase", nr: int) -> None:
    if case.heat_input.shape != case.current_input.shape:
        raise ValueError(
            "Expected heat_input/current_input to share a shape, "
            f"got {case.heat_input.shape} and {case.current_input.shape}"
        )
    if case.nodes == "grid" and case.heat_input.shape[0] != nr:
        raise ValueError(f"Expected grid inputs to have shape ({nr},), got {case.heat_input.shape}")
    if case.heat_input.shape[0] < 1:
        raise ValueError(f"Expected {case.coordinate}-coordinate inputs to contain at least one sample")


def materialize_source_inputs(
    *,
    source_plan: "SourcePlan",
    source_runtime_state: "SourceRuntimeState",
    psin_query: np.ndarray,
    enable_projection: bool = True,
) -> None:
    if source_plan.is_grid_nodes:
        np.copyto(source_runtime_state.materialized_heat_input, source_plan.heat_input)
        np.copyto(source_runtime_state.materialized_current_input, source_plan.current_input)
        return
    if enable_projection and source_plan.is_psin_coordinate and source_plan.has_projection_policy:
        materialize_projected_psin_source_inputs(
            source_plan=source_plan,
            source_runtime_state=source_runtime_state,
            psin_query=psin_query,
        )
        return
    query = psin_query
    if source_plan.is_psin_coordinate:
        parameterize_psin_query_inplace(
            source_runtime_state.parameter_query,
            psin_query,
            source_plan.parameterization,
        )
        query = source_runtime_state.parameter_query
    resolve_source_inputs(
        source_runtime_state.materialized_heat_input,
        source_runtime_state.materialized_current_input,
        source_plan.heat_input,
        source_plan.current_input,
        source_plan.coordinate_code,
        source_plan.n_src,
        source_runtime_state.barycentric_weights,
        source_runtime_state.fixed_remap_matrix,
        query,
    )


def materialize_projected_psin_source_inputs(
    *,
    source_plan: "SourcePlan",
    source_runtime_state: "SourceRuntimeState",
    psin_query: np.ndarray,
) -> None:
    if not source_plan.has_projection_policy:
        raise RuntimeError("Projected psin source materialization requested without a projection policy")
    materialize_projected_source_inputs(
        source_runtime_state.materialized_heat_input,
        source_runtime_state.materialized_current_input,
        source_runtime_state.heat_projection_coeff,
        source_runtime_state.current_projection_coeff,
        source_plan.current_input,
        psin_query,
        source_plan.projection_domain_code,
        source_plan.endpoint_policy_code,
        source_runtime_state.endpoint_blend,
    )


def refresh_source_runtime(
    *,
    case: "OperatorCase",
    grid_rho: np.ndarray,
    source_plan: "SourcePlan",
    source_runtime_state: "SourceRuntimeState",
    psin: np.ndarray,
) -> None:
    validate_source_inputs(case, int(grid_rho.shape[0]))
    case_key = (source_plan.coordinate, source_plan.nodes, source_plan.n_src)
    if source_runtime_state.cache_key != case_key:
        if source_plan.is_grid_nodes:
            source_runtime_state.barycentric_weights = np.empty(0, dtype=np.float64)
            source_runtime_state.fixed_remap_matrix = np.empty((0, 0), dtype=np.float64)
            source_runtime_state.heat_projection_fit_matrix = np.empty((0, 0), dtype=np.float64)
            source_runtime_state.current_projection_fit_matrix = np.empty((0, 0), dtype=np.float64)
            source_runtime_state.heat_projection_coeff = np.empty(0, dtype=np.float64)
            source_runtime_state.current_projection_coeff = np.empty(0, dtype=np.float64)
        else:
            (
                _,
                source_runtime_state.barycentric_weights,
                source_runtime_state.fixed_remap_matrix,
            ) = build_source_remap_cache(
                source_plan.coordinate,
                source_plan.n_src,
                rho=grid_rho,
            )
            if not source_plan.has_projection_policy:
                source_runtime_state.heat_projection_fit_matrix = np.empty((0, 0), dtype=np.float64)
                source_runtime_state.current_projection_fit_matrix = np.empty((0, 0), dtype=np.float64)
            else:
                source_runtime_state.heat_projection_fit_matrix = build_source_projection_fit_matrix(
                    source_plan.n_src,
                    degree=source_plan.heat_projection_degree,
                    domain=source_plan.projection_domain,
                )
                source_runtime_state.current_projection_fit_matrix = build_source_projection_fit_matrix(
                    source_plan.n_src,
                    degree=source_plan.current_projection_degree,
                    domain=source_plan.projection_domain,
                )
        source_runtime_state.cache_key = case_key
    if source_plan.is_grid_nodes or not source_plan.has_projection_policy:
        source_runtime_state.heat_projection_coeff = np.empty(0, dtype=np.float64)
        source_runtime_state.current_projection_coeff = np.empty(0, dtype=np.float64)
    else:
        source_runtime_state.heat_projection_coeff = source_runtime_state.heat_projection_fit_matrix @ np.asarray(
            source_plan.heat_input,
            dtype=np.float64,
        )
        source_runtime_state.current_projection_coeff = (
            source_runtime_state.current_projection_fit_matrix @ np.asarray(source_plan.current_input, dtype=np.float64)
        )
    if source_plan.is_grid_nodes or not source_plan.is_psin_coordinate:
        materialize_source_inputs(
            source_plan=source_plan,
            source_runtime_state=source_runtime_state,
            psin_query=psin,
        )
    elif source_plan.strategy == "fixed_point_psin":
        source_runtime_state.psin_query.fill(-1.0)


def build_source_stage_runner(route_spec) -> Callable:
    coordinate_code = int(route_spec.coordinate_code)
    operator_kernel = route_spec.implementation
    scratch_kernel = resolve_source_scratch_kernel(operator_kernel)

    def runner(
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
        if scratch_kernel is not None:
            return scratch_kernel(
                out_psin,
                out_psin_r,
                out_psin_rr,
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
        return operator_kernel(
            out_psin,
            out_psin_r,
            out_psin_rr,
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
        )

    return runner

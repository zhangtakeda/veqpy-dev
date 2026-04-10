"""
Module: operator.source_setup

Role:
- 收敛 source plan/runtime/orchestration 的共享 Python 规则.
- 避免 operator.py 混入过多 source/fixed-point/projection 细节.

Public API:
- SourcePlan
- SourceProjectionPolicy
- resolve_source_projection_policy
- validate_source_inputs
- refresh_source_runtime
- build_bound_source_stage_runner
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable

import numpy as np
from numpy.polynomial.chebyshev import chebvander

from veqpy.engine import (
    build_source_remap_cache,
    materialize_profile_owned_psin_source,
    materialize_projected_source_inputs,
    resolve_source_inputs,
    update_fixed_point_psin_query,
)

if TYPE_CHECKING:
    from veqpy.operator.layouts import SourceRuntimeState
    from veqpy.operator.operator_case import OperatorCase


@dataclass(frozen=True, slots=True)
class SourcePlan:
    """描述 source 语义与 runner 绑定所需的只读计划."""

    route: str
    kernel: Callable
    coordinate: str
    nodes: str
    coordinate_code: int
    strategy: str
    parameterization: str
    parameterization_code: int
    n_src: int
    heat_input: np.ndarray
    current_input: np.ndarray
    Ip: float
    beta: float
    has_projection_policy: bool
    projection_domain: str
    use_projected_finalize: bool
    heat_projection_degree: int
    current_projection_degree: int
    projection_domain_code: int
    endpoint_policy_code: int
    allow_query_warmstart: bool
    fixed_point_max_iter: int
    fixed_point_finalize_max_iter: int
    fixed_point_tolerance: float

    @property
    def is_grid_nodes(self) -> bool:
        return self.nodes == "grid"

    @property
    def is_psin_coordinate(self) -> bool:
        return self.coordinate == "psin"

    @property
    def is_single_pass(self) -> bool:
        return self.strategy == "single_pass"

    @property
    def is_profile_owned_psin(self) -> bool:
        return self.strategy == "profile_owned_psin"

    @property
    def is_fixed_point_psin(self) -> bool:
        return self.strategy == "fixed_point_psin"

    @property
    def supports_fused_residual(self) -> bool:
        return self.strategy in {"single_pass", "profile_owned_psin", "fixed_point_psin"}

    @property
    def requires_psin_profile_fields(self) -> bool:
        return self.is_profile_owned_psin


def _source_route_key(source_plan: SourcePlan) -> tuple[str, str, str]:
    return (
        str(source_plan.route).upper(),
        str(source_plan.coordinate).lower(),
        str(source_plan.nodes).lower(),
    )


@dataclass(frozen=True, slots=True)
class SourceProjectionPolicy:
    domain: str
    heat_degree: int
    current_degree: int
    ip_current_degree: int | None = None
    current_ip_endpoint_policy: str = "none"
    current_other_endpoint_policy: str = "none"


@dataclass(frozen=True, slots=True)
class FixedPointPolicy:
    max_iter: int
    finalize_max_iter: int
    tolerance: float


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
    ("PJ2", "psin", "uniform"): SourceProjectionPolicy(
        domain="psin",
        heat_degree=5,
        current_degree=6,
        current_ip_endpoint_policy="none",
        current_other_endpoint_policy="none",
    ),
    ("PQ", "psin", "uniform"): SourceProjectionPolicy(
        domain="sqrt_psin",
        heat_degree=8,
        current_degree=10,
        ip_current_degree=12,
        current_ip_endpoint_policy="affine_both",
        current_other_endpoint_policy="none",
    ),
}


FIXED_POINT_POLICIES: dict[tuple[str, str, str], FixedPointPolicy] = {
    ("PJ2", "psin", "uniform"): FixedPointPolicy(max_iter=16, finalize_max_iter=8, tolerance=1.0e-10),
    ("PQ", "psin", "uniform"): FixedPointPolicy(max_iter=16, finalize_max_iter=16, tolerance=1.0e-10),
}


def resolve_source_projection_policy(
    route: str,
    coordinate: str,
    nodes: str,
    *,
    has_ip_constraint: bool,
    has_beta_constraint: bool,
) -> SourceProjectionPolicy | None:
    policy = SOURCE_PROJECTION_POLICIES.get((route, coordinate, nodes))
    if policy is None:
        return None
    if route != "PQ" or coordinate != "psin" or nodes != "uniform":
        return policy
    if has_ip_constraint and has_beta_constraint:
        return SourceProjectionPolicy(
            domain="sqrt_psin",
            heat_degree=7,
            current_degree=7,
            ip_current_degree=7,
            current_ip_endpoint_policy="affine_both",
            current_other_endpoint_policy="none",
        )
    if has_ip_constraint:
        return SourceProjectionPolicy(
            domain="sqrt_psin",
            heat_degree=7,
            current_degree=9,
            ip_current_degree=7,
            current_ip_endpoint_policy="affine_both",
            current_other_endpoint_policy="none",
        )
    if has_beta_constraint:
        return SourceProjectionPolicy(
            domain="sqrt_psin",
            heat_degree=7,
            current_degree=9,
            ip_current_degree=7,
            current_ip_endpoint_policy="affine_both",
            current_other_endpoint_policy="both",
        )
    return SourceProjectionPolicy(
        domain="sqrt_psin",
        heat_degree=7,
        current_degree=10,
        ip_current_degree=7,
        current_ip_endpoint_policy="affine_both",
        current_other_endpoint_policy="both",
    )


def resolve_fixed_point_policy(route: str, coordinate: str, nodes: str) -> FixedPointPolicy:
    return FIXED_POINT_POLICIES.get(
        (str(route).upper(), str(coordinate).lower(), str(nodes).lower()),
        FixedPointPolicy(max_iter=8, finalize_max_iter=4, tolerance=1.0e-10),
    )


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
            f"Expected heat_input/current_input to share a shape, got {case.heat_input.shape} and {case.current_input.shape}"
        )
    if case.nodes == "grid" and case.heat_input.shape[0] != nr:
        raise ValueError(f"Expected grid inputs to have shape ({nr},), got {case.heat_input.shape}")
    if case.heat_input.shape[0] < 1:
        raise ValueError(f"Expected {case.coordinate}-coordinate inputs to contain at least one sample")


def materialize_source_inputs(
    *,
    source_plan: SourcePlan,
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
    source_plan: SourcePlan,
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
    source_plan: SourcePlan,
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


def build_bound_source_stage_runner(operator_core: Any) -> Callable:
    route_key = _source_route_key(operator_core.source_plan)
    valid_route_keys = {
        ("PF", "rho", "uniform"),
        ("PF", "rho", "grid"),
        ("PF", "psin", "uniform"),
        ("PF", "psin", "grid"),
        ("PP", "rho", "uniform"),
        ("PP", "rho", "grid"),
        ("PP", "psin", "uniform"),
        ("PP", "psin", "grid"),
        ("PI", "rho", "uniform"),
        ("PI", "rho", "grid"),
        ("PI", "psin", "uniform"),
        ("PI", "psin", "grid"),
        ("PJ1", "rho", "uniform"),
        ("PJ1", "rho", "grid"),
        ("PJ1", "psin", "uniform"),
        ("PJ1", "psin", "grid"),
        ("PJ2", "rho", "uniform"),
        ("PJ2", "rho", "grid"),
        ("PJ2", "psin", "uniform"),
        ("PJ2", "psin", "grid"),
        ("PQ", "rho", "uniform"),
        ("PQ", "rho", "grid"),
        ("PQ", "psin", "uniform"),
        ("PQ", "psin", "grid"),
    }
    if route_key not in valid_route_keys:
        raise ValueError(f"Unsupported source route key {route_key!r}")
    return _build_source_stage_runner_shared(operator_core)


def _build_source_stage_runner_shared(operator_core: Any) -> Callable:
    source_plan = operator_core.source_plan
    source_eval_runner = operator_core.execution_state.source_eval_runner
    source_runtime_state = operator_core.source_runtime_state
    psin = operator_core.psin
    psin_r = operator_core.psin_r
    psin_rr = operator_core.psin_rr
    FFn_psin = operator_core.FFn_psin
    Pn_psin = operator_core.Pn_psin
    materialized_heat_input = source_runtime_state.materialized_heat_input
    materialized_current_input = source_runtime_state.materialized_current_input
    source_target_root_fields = source_runtime_state.target_root_fields
    case_R0 = float(operator_core.case.R0)

    if source_plan.strategy == "profile_owned_psin":
        if source_plan.is_psin_coordinate and not source_plan.is_grid_nodes:
            source_psin_query = source_runtime_state.psin_query
            source_parameter_query = source_runtime_state.parameter_query
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
                if source_plan.has_projection_policy:
                    materialize_source_inputs(
                        source_plan=source_plan,
                        source_runtime_state=source_runtime_state,
                        psin_query=source_psin_query,
                        enable_projection=True,
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

        source_psin_query = source_runtime_state.psin_query

        def runner() -> tuple[float, float]:
            copy_psin_profile_to_root_fields(operator_core)
            np.copyto(source_psin_query, psin)
            materialize_source_inputs(
                source_plan=source_plan,
                source_runtime_state=source_runtime_state,
                psin_query=source_psin_query,
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

    if source_plan.strategy == "fixed_point_psin":

        def runner() -> tuple[float, float]:
            return run_psin_source_fixed_point(operator_core)

        return runner

    def runner() -> tuple[float, float]:
        return source_eval_runner(
            operator_core.root_fields,
            FFn_psin,
            Pn_psin,
            materialized_heat_input,
            materialized_current_input,
            case_R0,
        )

    return runner


def copy_psin_profile_to_root_fields(operator_core: Any) -> None:
    if operator_core.psin_profile.u_fields is None:
        raise RuntimeError("psin_profile runtime fields are not initialized")
    np.copyto(operator_core.psin, operator_core.psin_profile.u)
    np.copyto(operator_core.psin_r, operator_core.psin_profile.u_r)
    np.copyto(operator_core.psin_rr, operator_core.psin_profile.u_rr)


def run_psin_source_fixed_point(operator_core: Any) -> tuple[float, float]:
    source_plan = operator_core.source_plan
    source_eval_runner = operator_core.execution_state.source_eval_runner
    source_runtime_state = operator_core.source_runtime_state
    target_root_fields = source_runtime_state.target_root_fields
    tolerance = float(source_plan.fixed_point_tolerance)
    if (not source_plan.allow_query_warmstart) or source_runtime_state.psin_query[0] < 0.0:
        normalize_psin_query_inplace(source_runtime_state.psin_query, operator_core.psin_profile.u)
    alpha1 = float("nan")
    alpha2 = float("nan")
    for _ in range(int(source_plan.fixed_point_max_iter)):
        materialize_source_inputs(
            source_plan=source_plan,
            source_runtime_state=source_runtime_state,
            psin_query=source_runtime_state.psin_query,
            enable_projection=not source_plan.use_projected_finalize,
        )
        alpha1, alpha2 = source_eval_runner(
            target_root_fields,
            operator_core.FFn_psin,
            operator_core.Pn_psin,
            source_runtime_state.materialized_heat_input,
            source_runtime_state.materialized_current_input,
            float(operator_core.case.R0),
        )
        if update_fixed_point_psin_query(source_runtime_state.psin_query, target_root_fields[0], tolerance):
            break
    if source_plan.use_projected_finalize:
        np.copyto(source_runtime_state.psin_query, target_root_fields[0])
        for _ in range(int(source_plan.fixed_point_finalize_max_iter)):
            materialize_source_inputs(
                source_plan=source_plan,
                source_runtime_state=source_runtime_state,
                psin_query=source_runtime_state.psin_query,
                enable_projection=True,
            )
            alpha1, alpha2 = source_eval_runner(
                target_root_fields,
                operator_core.FFn_psin,
                operator_core.Pn_psin,
                source_runtime_state.materialized_heat_input,
                source_runtime_state.materialized_current_input,
                float(operator_core.case.R0),
            )
            if update_fixed_point_psin_query(source_runtime_state.psin_query, target_root_fields[0], tolerance):
                break
    np.copyto(operator_core.psin, target_root_fields[0])
    np.copyto(operator_core.psin_r, target_root_fields[1])
    np.copyto(operator_core.psin_rr, target_root_fields[2])
    return alpha1, alpha2

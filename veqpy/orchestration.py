"""
Module: veqpy.orchestration

Role:
- Centralize Python-level stage orchestration shared by operator/runtime code.
- Keep route policy, source materialization, residual metadata, and K_max
  decisions out of numeric kernels.

Notes:
- This module assembles and refreshes runtime stages; it does not implement
  numba kernels.
- ABI dataclasses remain in ``veqpy.engine.backend_abi`` because they describe
  backend binding shapes rather than orchestration policy.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable

import numpy as np
from numpy.polynomial.chebyshev import chebvander

from veqpy.engine.numba_geometry import update_geometry_hot
from veqpy.engine.numba_source import (
    COORDINATE_CODES,
    build_source_remap_cache,
    materialize_profile_owned_psin_source,
    materialize_projected_source_inputs,
    resolve_source_inputs,
    update_fixed_point_psin_query,
    update_fourier_family_fields,
)

if TYPE_CHECKING:
    from veqpy.operator.operator_case import OperatorCase
    from veqpy.operator.runtime_layout import SourceRuntimeState


RESIDUAL_BLOCK_CODE_BY_NAME = {
    "h": 0,
    "v": 1,
    "k": 2,
    "c0": 3,
    "c_family": 4,
    "s_family": 5,
    "psin": 6,
    "F": 7,
}

PACKED_PROFILE_FAMILY_ORDER = ("h", "v", "k", "c0", "c", "s", "psin", "F")
PREFIX_PROFILE_FAMILIES = ("psin", "F")
SHAPE_PROFILE_FAMILIES = ("h", "v", "k", "c0", "c", "s")
ALL_PROFILE_FAMILIES = SHAPE_PROFILE_FAMILIES + PREFIX_PROFILE_FAMILIES

PROFILE_STATIC_KWARGS: dict[str, dict[str, int]] = {
    "psin": {"power": 2},
    "F": {"envelope_power": 2},
}
PROFILE_OFFSET_SPECS: dict[str, float | str] = {
    "h": 0.0,
    "v": 0.0,
    "k": "ka",
    "psin": 1.0,
    "F": 1.0,
}


def validate_profile_family_order(
    family_order: tuple[str, ...] = PACKED_PROFILE_FAMILY_ORDER,
) -> tuple[str, ...]:
    family_order = tuple(family_order)
    if len(family_order) != len(ALL_PROFILE_FAMILIES) or set(family_order) != set(ALL_PROFILE_FAMILIES):
        raise ValueError(
            "PACKED_PROFILE_FAMILY_ORDER must contain each family exactly once: "
            f"{ALL_PROFILE_FAMILIES!r}, got {family_order!r}"
        )
    return family_order


def get_prefix_profile_names(
    family_order: tuple[str, ...] = PACKED_PROFILE_FAMILY_ORDER,
) -> tuple[str, ...]:
    return tuple(family for family in validate_profile_family_order(family_order) if family in PREFIX_PROFILE_FAMILIES)


def expand_profile_family(family: str, M_max: int) -> tuple[str, ...]:
    if family == "psin":
        return ("psin",)
    if family == "F":
        return ("F",)
    if family == "h":
        return ("h",)
    if family == "v":
        return ("v",)
    if family == "k":
        return ("k",)
    if family == "c0":
        return ("c0",)
    if family == "c":
        return tuple(f"c{k}" for k in range(1, M_max + 1))
    if family == "s":
        return tuple(f"s{k}" for k in range(1, M_max + 1))
    raise KeyError(f"Unknown profile family {family!r}")


def build_fourier_profile_names(
    M_max: int,
    family_order: tuple[str, ...] = PACKED_PROFILE_FAMILY_ORDER,
) -> tuple[str, ...]:
    M_max = int(M_max)
    if M_max < 0:
        raise ValueError(f"M_max must be non-negative, got {M_max}")

    fourier_names: list[str] = []
    for family in validate_profile_family_order(family_order):
        if family not in ("c0", "c", "s"):
            continue
        fourier_names.extend(expand_profile_family(family, M_max))
    return tuple(fourier_names)


def build_shape_profile_names(
    M_max: int,
    family_order: tuple[str, ...] = PACKED_PROFILE_FAMILY_ORDER,
) -> tuple[str, ...]:
    shape_profile_names: list[str] = []
    for family in validate_profile_family_order(family_order):
        if family in PREFIX_PROFILE_FAMILIES:
            continue
        shape_profile_names.extend(expand_profile_family(family, int(M_max)))
    return tuple(shape_profile_names)


def build_profile_names(
    M_max: int,
    family_order: tuple[str, ...] = PACKED_PROFILE_FAMILY_ORDER,
) -> tuple[str, ...]:
    M_max = int(M_max)
    profile_names: list[str] = []
    for family in validate_profile_family_order(family_order):
        profile_names.extend(expand_profile_family(family, M_max))
    return tuple(profile_names)


def normalize_fourier_power_K_max(value: int | None) -> int | None:
    """Normalize and validate a Fourier radial-power cap."""
    if value is None:
        return None
    K_max = int(value)
    if K_max < 1:
        raise ValueError(f"K_max must be None or an integer >= 1, got {value!r}")
    return K_max


def resolve_fourier_power(order: int, K_max: int | None = None) -> int:
    """Resolve the radial prefactor power for a Fourier shape profile order."""
    order = int(order)
    if order <= 0:
        return 0
    normalized_K_max = normalize_fourier_power_K_max(K_max)
    if normalized_K_max is None:
        return order
    return min(order, normalized_K_max)


def _decode_residual_block_code(name: str) -> tuple[int, int]:
    if name.startswith("c") and name[1:].isdigit():
        order = int(name[1:])
        if order == 0:
            return (RESIDUAL_BLOCK_CODE_BY_NAME["c0"], 0)
        return (RESIDUAL_BLOCK_CODE_BY_NAME["c_family"], order)
    if name.startswith("s") and name[1:].isdigit():
        order = int(name[1:])
        if order == 0:
            raise KeyError("s0 is not a valid residual block")
        return (RESIDUAL_BLOCK_CODE_BY_NAME["s_family"], order)
    try:
        return (RESIDUAL_BLOCK_CODE_BY_NAME[name], 0)
    except KeyError as exc:
        supported = ", ".join(RESIDUAL_BLOCK_CODE_BY_NAME)
        raise KeyError(f"Unknown residual block {name!r}. Supported blocks: {supported}") from exc


def build_residual_block_metadata(profile_names: tuple[str, ...]) -> tuple[np.ndarray, np.ndarray]:
    block_codes = np.empty(len(profile_names), dtype=np.int64)
    block_orders = np.zeros(len(profile_names), dtype=np.int64)
    for i, name in enumerate(profile_names):
        block_codes[i], block_orders[i] = _decode_residual_block_code(name)
    return block_codes, block_orders


def build_residual_block_radial_powers(profile_names: tuple[str, ...], *, K_max: int | None) -> np.ndarray:
    radial_powers = np.zeros(len(profile_names), dtype=np.int64)
    for i, name in enumerate(profile_names):
        if name.startswith(("c", "s")) and name[1:].isdigit():
            radial_powers[i] = resolve_fourier_power(int(name[1:]), K_max)
    return radial_powers


@dataclass(frozen=True, slots=True)
class SourcePlan:
    """描述 source 语义与 runner 绑定所需的只读计划."""

    route: str
    kernel: Callable
    coordinate: str
    nodes: str
    strategy: str
    parameterization: str
    source_sample_count: int
    heat_input: np.ndarray
    current_input: np.ndarray
    Ip: float
    beta: float
    has_projection_policy: bool
    projection_domain: str
    endpoint_policy: str
    heat_projection_degree: int
    current_projection_degree: int

    @property
    def is_grid_nodes(self) -> bool:
        return self.nodes == "grid"

    @property
    def is_psin_coordinate(self) -> bool:
        return self.coordinate == "psin"

    @property
    def is_profile_owned_psin(self) -> bool:
        return self.strategy == "profile_owned_psin"

    @property
    def is_fixed_point_psin_route(self) -> bool:
        return (self.route, self.coordinate, self.nodes) in {
            ("PJ2", "psin", "uniform"),
            ("PQ", "psin", "uniform"),
        }

    @property
    def supports_fused_residual(self) -> bool:
        return self.strategy in {"single_pass", "profile_owned_psin"}

    @property
    def requires_psin_profile_fields(self) -> bool:
        return self.is_profile_owned_psin

    @property
    def requires_psin_query_workspace(self) -> bool:
        return self.is_profile_owned_psin or self.is_fixed_point_psin_route

    @property
    def requires_source_parameter_query(self) -> bool:
        return self.is_psin_coordinate and self.parameterization != "identity"

    @property
    def requires_target_root_fields(self) -> bool:
        return self.is_profile_owned_psin or self.is_fixed_point_psin_route

    @property
    def coordinate_code(self) -> int:
        return int(COORDINATE_CODES[self.coordinate])

    @property
    def parameterization_code(self) -> int:
        return int(SOURCE_PARAMETERIZATION_CODES[self.parameterization])

    @property
    def projection_domain_code(self) -> int:
        return int(PROJECTION_DOMAIN_CODES[self.projection_domain])

    @property
    def endpoint_policy_code(self) -> int:
        return int(ENDPOINT_POLICY_CODES[self.endpoint_policy])


def _source_route_key(source_plan: SourcePlan) -> tuple[str, str, str]:
    return (source_plan.route, source_plan.coordinate, source_plan.nodes)


@dataclass(frozen=True, slots=True)
class SourceProjectionPolicy:
    domain: str
    heat_degree: int
    current_degree: int
    ip_current_degree: int | None = None
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


def _resolve_source_projection_policy(
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


def build_source_plan(
    *,
    case: "OperatorCase",
    source_route_spec: object,
) -> SourcePlan:
    policy = _resolve_source_projection_policy(
        case.route,
        case.coordinate,
        case.nodes,
        has_ip_constraint=bool(np.isfinite(case.Ip)),
        has_beta_constraint=bool(np.isfinite(case.beta)),
    )
    has_projection_policy = policy is not None
    has_ip_constraint = bool(np.isfinite(case.Ip))
    projection_domain = "psin"
    heat_projection_degree = 0
    current_projection_degree = 0
    endpoint_policy = "none"

    if policy is not None:
        endpoint_policy = (
            policy.current_ip_endpoint_policy if has_ip_constraint else policy.current_other_endpoint_policy
        )
        projection_domain = policy.domain
        heat_projection_degree = int(policy.heat_degree)
        current_projection_degree = int(
            policy.ip_current_degree
            if has_ip_constraint and policy.ip_current_degree is not None
            else policy.current_degree
        )

    return SourcePlan(
        route=str(case.route).upper(),
        kernel=source_route_spec.implementation,
        coordinate=str(case.coordinate).lower(),
        nodes=str(case.nodes).lower(),
        strategy=str(source_route_spec.source_strategy).lower(),
        parameterization=str(source_route_spec.source_parameterization).lower(),
        source_sample_count=int(case.heat_input.shape[0]),
        heat_input=case.heat_input,
        current_input=case.current_input,
        Ip=float(case.Ip),
        beta=float(case.beta),
        has_projection_policy=has_projection_policy,
        projection_domain=str(projection_domain).lower(),
        endpoint_policy=str(endpoint_policy).lower(),
        heat_projection_degree=heat_projection_degree,
        current_projection_degree=current_projection_degree,
    )


def validate_source_plan_profile_support(
    *,
    source_plan: SourcePlan,
    profile_L: np.ndarray,
    profile_index: dict[str, int],
    case: "OperatorCase",
) -> None:
    has_active_psin = int(profile_L[profile_index["psin"]]) >= 0
    if source_plan.strategy == "profile_owned_psin" and not has_active_psin:
        raise ValueError(f"{case.route} requires an active psin profile because psin is optimized externally")
    if case.coordinate == "psin" and source_plan.strategy != "profile_owned_psin" and has_active_psin:
        raise ValueError(f"{case.route} does not accept an active psin profile because psin is source-owned")


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


def refresh_source_runtime(
    *,
    case: "OperatorCase",
    grid_rho: np.ndarray,
    source_plan: SourcePlan,
    source_runtime_state: "SourceRuntimeState",
    psin: np.ndarray,
) -> None:
    validate_source_inputs(case, int(grid_rho.shape[0]))
    source_const_state = source_runtime_state.const_state
    source_aux_state = source_runtime_state.aux_state
    source_work_state = source_runtime_state.work_state
    case_key = (source_plan.coordinate, source_plan.nodes, source_plan.source_sample_count)
    if source_runtime_state.cache_key != case_key:
        if source_plan.is_grid_nodes:
            source_const_state.barycentric_weights = np.empty(0, dtype=np.float64)
            source_const_state.fixed_remap_matrix = np.empty((0, 0), dtype=np.float64)
            source_const_state.heat_projection_fit_matrix = np.empty((0, 0), dtype=np.float64)
            source_const_state.current_projection_fit_matrix = np.empty((0, 0), dtype=np.float64)
            source_aux_state.heat_projection_coeff = np.empty(0, dtype=np.float64)
            source_aux_state.current_projection_coeff = np.empty(0, dtype=np.float64)
        else:
            (
                _,
                source_const_state.barycentric_weights,
                source_const_state.fixed_remap_matrix,
            ) = build_source_remap_cache(
                source_plan.coordinate,
                source_plan.source_sample_count,
                rho=grid_rho,
            )
            if not source_plan.has_projection_policy:
                source_const_state.heat_projection_fit_matrix = np.empty((0, 0), dtype=np.float64)
                source_const_state.current_projection_fit_matrix = np.empty((0, 0), dtype=np.float64)
            else:
                source_axis = np.linspace(0.0, 1.0, int(source_plan.source_sample_count), dtype=np.float64)
                source_query = np.empty_like(source_axis)
                np.copyto(source_query, source_axis)
                np.clip(source_query, 0.0, 1.0, out=source_query)
                if source_plan.projection_domain == "psin":
                    source_query *= 2.0
                    source_query -= 1.0
                elif source_plan.projection_domain == "sqrt_psin":
                    np.sqrt(source_query, out=source_query)
                    source_query *= 2.0
                    source_query -= 1.0
                else:
                    raise ValueError(f"Unsupported source projection domain {source_plan.projection_domain!r}")
                if source_plan.heat_projection_degree < 0:
                    raise ValueError(
                        f"Projection degree must be non-negative, got {source_plan.heat_projection_degree}"
                    )
                if source_plan.current_projection_degree < 0:
                    raise ValueError(
                        f"Projection degree must be non-negative, got {source_plan.current_projection_degree}"
                    )
                source_const_state.heat_projection_fit_matrix = np.linalg.pinv(
                    chebvander(source_query, source_plan.heat_projection_degree)
                )
                source_const_state.current_projection_fit_matrix = np.linalg.pinv(
                    chebvander(source_query, source_plan.current_projection_degree)
                )
        source_runtime_state.cache_key = case_key
    if source_plan.is_grid_nodes or not source_plan.has_projection_policy:
        source_aux_state.heat_projection_coeff = np.empty(0, dtype=np.float64)
        source_aux_state.current_projection_coeff = np.empty(0, dtype=np.float64)
    else:
        source_aux_state.heat_projection_coeff = source_const_state.heat_projection_fit_matrix @ np.asarray(
            source_plan.heat_input,
            dtype=np.float64,
        )
        source_aux_state.current_projection_coeff = source_const_state.current_projection_fit_matrix @ np.asarray(
            source_plan.current_input, dtype=np.float64
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
                psin,
            )
    elif (source_plan.route, source_plan.coordinate, source_plan.nodes) in {
        ("PJ2", "psin", "uniform"),
        ("PQ", "psin", "uniform"),
    }:
        source_work_state.psin_query.fill(-1.0)


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
    if route_key == ("PJ2", "psin", "uniform"):
        return _build_pj2_psin_uniform_source_stage_runner(operator_core)
    if route_key == ("PQ", "psin", "uniform"):
        return _build_pq_psin_uniform_source_stage_runner(operator_core)
    return _build_source_stage_runner_shared(operator_core)


def _build_source_stage_runner_shared(operator_core: Any) -> Callable:
    source_plan = operator_core.source_plan
    source_eval_runner = operator_core.execution_state.source_eval_runner
    runtime_layout = operator_core.runtime_layout
    source_runtime_state = operator_core.source_runtime_state
    source_work_state = source_runtime_state.work_state
    source_aux_state = source_runtime_state.aux_state
    psin = operator_core.psin
    psin_r = operator_core.psin_r
    psin_rr = operator_core.psin_rr
    FFn_psin = operator_core.FFn_psin
    Pn_psin = operator_core.Pn_psin
    materialized_heat_input = source_work_state.materialized_heat_input
    materialized_current_input = source_work_state.materialized_current_input
    source_target_root_fields = source_aux_state.target_root_fields
    case_R0 = float(operator_core.case.R0)

    if source_plan.strategy == "profile_owned_psin":
        if source_plan.is_psin_coordinate and not source_plan.is_grid_nodes:
            source_psin_query = source_work_state.psin_query
            source_parameter_query = source_work_state.parameter_query
            psin_profile_fields = runtime_layout.psin_profile_fields
            heat_input = source_plan.heat_input
            current_input = source_plan.current_input
            parameterization_code = source_plan.parameterization_code

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
                    parameterization_code,
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
        psin_profile_u = runtime_layout.psin_profile_u
        psin_profile_fields = runtime_layout.psin_profile_fields

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
                    np.maximum(source_work_state.parameter_query, 0.0, out=source_work_state.parameter_query)
                    np.sqrt(source_work_state.parameter_query, out=source_work_state.parameter_query)
                else:
                    raise ValueError(f"Unsupported source parameterization {source_plan.parameterization!r}")
                resolve_source_inputs(
                    source_work_state.materialized_heat_input,
                    source_work_state.materialized_current_input,
                    source_plan.heat_input,
                    source_plan.current_input,
                    source_plan.coordinate_code,
                    source_plan.source_sample_count,
                    source_runtime_state.const_state.barycentric_weights,
                    source_runtime_state.const_state.fixed_remap_matrix,
                    source_work_state.parameter_query,
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
            operator_core.root_fields,
            FFn_psin,
            Pn_psin,
            materialized_heat_input,
            materialized_current_input,
            case_R0,
        )

    return runner


def _build_pj2_psin_uniform_source_stage_runner(operator_core: Any) -> Callable[[], tuple[float, float]]:
    source_plan = operator_core.source_plan
    source_eval_runner = operator_core.execution_state.source_eval_runner
    runtime_layout = operator_core.runtime_layout
    source_runtime_state = operator_core.source_runtime_state
    source_work_state = source_runtime_state.work_state
    source_aux_state = source_runtime_state.aux_state
    target_root_fields = source_aux_state.target_root_fields
    psin_profile_u = runtime_layout.psin_profile_u
    max_residual = 1.0e-10

    def runner() -> tuple[float, float]:
        if source_work_state.psin_query[0] < 0.0:
            if psin_profile_u is None:
                raise RuntimeError("psin_profile.u is not initialized")
            np.copyto(source_work_state.psin_query, np.asarray(psin_profile_u, dtype=np.float64))
            if source_work_state.psin_query.ndim != 1 or source_work_state.psin_query.size < 2:
                raise ValueError(
                    f"Expected psin query to be 1D with at least two points, got {source_work_state.psin_query.shape}"
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
        for _ in range(16):
            if source_plan.parameterization == "identity":
                np.copyto(source_work_state.parameter_query, source_work_state.psin_query)
            elif source_plan.parameterization == "sqrt_psin":
                np.copyto(source_work_state.parameter_query, source_work_state.psin_query)
                np.maximum(source_work_state.parameter_query, 0.0, out=source_work_state.parameter_query)
                np.sqrt(source_work_state.parameter_query, out=source_work_state.parameter_query)
            else:
                raise ValueError(f"Unsupported source parameterization {source_plan.parameterization!r}")
            resolve_source_inputs(
                source_work_state.materialized_heat_input,
                source_work_state.materialized_current_input,
                source_plan.heat_input,
                source_plan.current_input,
                source_plan.coordinate_code,
                source_plan.source_sample_count,
                source_runtime_state.const_state.barycentric_weights,
                source_runtime_state.const_state.fixed_remap_matrix,
                source_work_state.parameter_query,
            )
            alpha1, alpha2 = source_eval_runner(
                target_root_fields,
                operator_core.FFn_psin,
                operator_core.Pn_psin,
                source_work_state.materialized_heat_input,
                source_work_state.materialized_current_input,
                float(operator_core.case.R0),
            )
            if update_fixed_point_psin_query(source_work_state.psin_query, target_root_fields[0], max_residual):
                break
        np.copyto(source_work_state.psin_query, target_root_fields[0])
        for _ in range(8):
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
                operator_core.FFn_psin,
                operator_core.Pn_psin,
                source_work_state.materialized_heat_input,
                source_work_state.materialized_current_input,
                float(operator_core.case.R0),
            )
            if update_fixed_point_psin_query(source_work_state.psin_query, target_root_fields[0], max_residual):
                break
        np.copyto(operator_core.psin, target_root_fields[0])
        np.copyto(operator_core.psin_r, target_root_fields[1])
        np.copyto(operator_core.psin_rr, target_root_fields[2])
        return alpha1, alpha2

    return runner


def _build_pq_psin_uniform_source_stage_runner(operator_core: Any) -> Callable[[], tuple[float, float]]:
    source_plan = operator_core.source_plan
    source_eval_runner = operator_core.execution_state.source_eval_runner
    runtime_layout = operator_core.runtime_layout
    source_runtime_state = operator_core.source_runtime_state
    source_work_state = source_runtime_state.work_state
    source_aux_state = source_runtime_state.aux_state
    target_root_fields = source_aux_state.target_root_fields
    psin_profile_u = runtime_layout.psin_profile_u
    max_residual = 1.0e-10
    allow_query_warmstart = bool(source_plan.endpoint_policy_code != ENDPOINT_POLICY_CODES["none"])

    def runner() -> tuple[float, float]:
        if (not allow_query_warmstart) or source_work_state.psin_query[0] < 0.0:
            if psin_profile_u is None:
                raise RuntimeError("psin_profile.u is not initialized")
            np.copyto(source_work_state.psin_query, np.asarray(psin_profile_u, dtype=np.float64))
            if source_work_state.psin_query.ndim != 1 or source_work_state.psin_query.size < 2:
                raise ValueError(
                    f"Expected psin query to be 1D with at least two points, got {source_work_state.psin_query.shape}"
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
        for _ in range(16):
            if source_plan.parameterization == "identity":
                np.copyto(source_work_state.parameter_query, source_work_state.psin_query)
            elif source_plan.parameterization == "sqrt_psin":
                np.copyto(source_work_state.parameter_query, source_work_state.psin_query)
                np.maximum(source_work_state.parameter_query, 0.0, out=source_work_state.parameter_query)
                np.sqrt(source_work_state.parameter_query, out=source_work_state.parameter_query)
            else:
                raise ValueError(f"Unsupported source parameterization {source_plan.parameterization!r}")
            resolve_source_inputs(
                source_work_state.materialized_heat_input,
                source_work_state.materialized_current_input,
                source_plan.heat_input,
                source_plan.current_input,
                source_plan.coordinate_code,
                source_plan.source_sample_count,
                source_runtime_state.const_state.barycentric_weights,
                source_runtime_state.const_state.fixed_remap_matrix,
                source_work_state.parameter_query,
            )
            alpha1, alpha2 = source_eval_runner(
                target_root_fields,
                operator_core.FFn_psin,
                operator_core.Pn_psin,
                source_work_state.materialized_heat_input,
                source_work_state.materialized_current_input,
                float(operator_core.case.R0),
            )
            if update_fixed_point_psin_query(source_work_state.psin_query, target_root_fields[0], max_residual):
                break
        np.copyto(source_work_state.psin_query, target_root_fields[0])
        for _ in range(16):
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
                operator_core.FFn_psin,
                operator_core.Pn_psin,
                source_work_state.materialized_heat_input,
                source_work_state.materialized_current_input,
                float(operator_core.case.R0),
            )
            if update_fixed_point_psin_query(source_work_state.psin_query, target_root_fields[0], max_residual):
                break
        np.copyto(operator_core.psin, target_root_fields[0])
        np.copyto(operator_core.psin_r, target_root_fields[1])
        np.copyto(operator_core.psin_rr, target_root_fields[2])
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
    cos_ktheta: np.ndarray,
    sin_ktheta: np.ndarray,
    k_cos_ktheta: np.ndarray,
    k_sin_ktheta: np.ndarray,
    k2_cos_ktheta: np.ndarray,
    k2_sin_ktheta: np.ndarray,
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
            c_effective_order,
            s_effective_order,
        )

    return runner


__all__ = [
    "SourcePlan",
    "build_bound_source_stage_runner",
    "build_fourier_profile_names",
    "build_geometry_stage_runner",
    "build_profile_names",
    "build_residual_block_metadata",
    "build_residual_block_radial_powers",
    "build_shape_profile_names",
    "build_source_plan",
    "expand_profile_family",
    "get_prefix_profile_names",
    "normalize_fourier_power_K_max",
    "refresh_source_runtime",
    "resolve_fourier_power",
    "validate_profile_family_order",
    "validate_source_inputs",
    "validate_source_plan_profile_support",
]

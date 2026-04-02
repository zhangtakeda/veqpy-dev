"""
Module: engine.numpy_source

Role:
- 负责在 numpy backend 下注册 source operators.
- 负责校验 operator/coordinate/nodes 组合并执行 source kernels.

Public API:
- register_operator
- validate_route
- build_source_remap_cache
- resolve_source_inputs

Notes:
- 这个文件同时作为 Stage-C source routing 与 field update 的 vectorized reference.
"""

import warnings
from dataclasses import dataclass
from typing import Callable

import numpy as np

MU0 = 4.0 * np.pi * 1e-7
DEFAULT_LOCAL_BARYCENTRIC_STENCIL = 12

RHO_AXIS = 0
THETA_AXIS = 1

RHO_COORDINATE = 0
PSIN_COORDINATE = 1

COORDINATE_NAMES = {
    RHO_COORDINATE: "rho",
    PSIN_COORDINATE: "psin",
}

COORDINATE_CODES = {
    "rho": RHO_COORDINATE,
    "psin": PSIN_COORDINATE,
}


@dataclass(frozen=True, slots=True)
class _SourceSpec:
    supported_coordinates: tuple[int, ...]
    implementation: Callable


OPERATOR_REGISTRY: dict[str, _SourceSpec] = {}

UNIFORM_NODES = "uniform"
GRID_NODES = "grid"
NODE_NAMES = (UNIFORM_NODES, GRID_NODES)

SOURCE_STRATEGY_SINGLE_PASS = "single_pass"
SOURCE_STRATEGY_PROFILE_OWNED_PSIN = "profile_owned_psin"
SOURCE_STRATEGY_FIXED_POINT_PSIN = "fixed_point_psin"
SOURCE_PARAMETERIZATION_IDENTITY = "identity"
SOURCE_PARAMETERIZATION_SQRT_PSIN = "sqrt_psin"
SOURCE_PARAMETERIZATION_CODE_IDENTITY = 0
SOURCE_PARAMETERIZATION_CODE_SQRT_PSIN = 1
PROJECTION_DOMAIN_PSIN = 0
PROJECTION_DOMAIN_SQRT_PSIN = 1
ENDPOINT_POLICY_NONE = 0
ENDPOINT_POLICY_RIGHT = 1
ENDPOINT_POLICY_BOTH = 2
ENDPOINT_POLICY_AFFINE_BOTH = 3


@dataclass(frozen=True, slots=True)
class _SourceRouteSpec:
    coordinate_code: int
    nodes: str
    implementation: Callable
    source_strategy: str
    source_parameterization: str


ROUTE_REGISTRY: dict[tuple[str, int, str], _SourceRouteSpec] = {}


def register_operator(
    name: str,
    *,
    supported_coordinates: tuple[int, ...] = (RHO_COORDINATE, PSIN_COORDINATE),
) -> Callable:
    """注册一个 source operator kernel."""

    def decorator(func: Callable) -> Callable:
        existing = OPERATOR_REGISTRY.get(name)
        if existing is not None:
            raise ValueError(f"_SourceSpec {name!r} is already registered")

        OPERATOR_REGISTRY[name] = _SourceSpec(
            supported_coordinates=supported_coordinates,
            implementation=func,
        )
        return func

    return decorator


def register_route(
    route: str,
    coordinate: str,
    nodes: str,
    *,
    implementation_name: str,
    source_strategy: str,
    source_parameterization: str = SOURCE_PARAMETERIZATION_IDENTITY,
) -> None:
    coordinate_code = _normalize_coordinate(coordinate)
    normalized_nodes = _normalize_nodes(nodes)
    try:
        implementation_spec = OPERATOR_REGISTRY[implementation_name]
    except KeyError as exc:
        raise KeyError(
            f"Unknown implementation {implementation_name!r} for route {(route, coordinate, nodes)!r}"
        ) from exc

    if coordinate_code not in implementation_spec.supported_coordinates:
        raise ValueError(f"Implementation {implementation_name!r} does not support coordinate={coordinate!r}")

    key = (str(route).upper(), coordinate_code, normalized_nodes)
    if key in ROUTE_REGISTRY:
        raise ValueError(f"Source route {key!r} is already registered")

    ROUTE_REGISTRY[key] = _SourceRouteSpec(
        coordinate_code=coordinate_code,
        nodes=normalized_nodes,
        implementation=implementation_spec.implementation,
        source_strategy=source_strategy,
        source_parameterization=source_parameterization,
    )


def validate_route(
    route: str,
    coordinate: str,
    nodes: str = UNIFORM_NODES,
) -> _SourceRouteSpec:
    """校验 operator/coordinate/nodes 组合并返回 route 规格."""

    coordinate_code = _normalize_coordinate(coordinate)
    normalized_nodes = _normalize_nodes(nodes)
    key = (str(route).upper(), coordinate_code, normalized_nodes)
    try:
        return ROUTE_REGISTRY[key]
    except KeyError as exc:
        supported_names = sorted({route_name for route_name, _, _ in ROUTE_REGISTRY})
        supported = ", ".join(supported_names)
        raise KeyError(
            f"Unknown source route route={route!r}, coordinate={coordinate!r}, nodes={nodes!r}. "
            f"Supported routes: {supported}"
        ) from exc


def build_source_remap_cache(
    coordinate: str,
    n_src: int,
    *,
    rho: np.ndarray | None = None,
    stencil_size: int = DEFAULT_LOCAL_BARYCENTRIC_STENCIL,
) -> tuple[int, np.ndarray, np.ndarray]:
    coord = str(coordinate).lower()
    if coord not in ("rho", "psin"):
        raise ValueError(f"Unsupported coordinate {coordinate!r}")

    count = int(n_src)
    if count < 1:
        raise ValueError(f"n_src must be positive, got {n_src!r}")

    coord_code = PSIN_COORDINATE if coord == "psin" else RHO_COORDINATE
    local_size = min(count, int(stencil_size))
    if local_size < 1:
        raise ValueError(f"stencil_size must be positive, got {stencil_size!r}")
    weights = _uniform_barycentric_weights(local_size)
    fixed_remap_matrix = np.empty((0, 0), dtype=np.float64)
    if coord_code == RHO_COORDINATE:
        if rho is None:
            raise ValueError("rho is required when coordinate='rho'")
        query = np.clip(np.asarray(rho, dtype=np.float64), 0.0, 1.0)
        fixed_remap_matrix = _build_uniform_barycentric_matrix(query, count, local_size, weights)

    return local_size, weights, fixed_remap_matrix


def _normalize_coordinate(value: str) -> int:
    try:
        return COORDINATE_CODES[value]
    except KeyError as exc:
        raise ValueError(f"Unsupported coordinate {value!r}") from exc


def _normalize_nodes(value: str) -> str:
    nodes = str(value).lower()
    if nodes not in NODE_NAMES:
        raise ValueError(f"Unsupported nodes {value!r}")
    return nodes


@register_operator("PF_RHO", supported_coordinates=(RHO_COORDINATE,))
def update_PF_rho(
    out_psin: np.ndarray,
    out_psin_r: np.ndarray,
    out_psin_rr: np.ndarray,
    out_FFn_psin: np.ndarray,
    out_Pn_psin: np.ndarray,
    heat_input: np.ndarray,
    current_input: np.ndarray,
    coordinate_code: int,
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
) -> tuple[float, float]:
    del coordinate_code
    has_Ip = not np.isnan(Ip)
    has_beta = not np.isnan(beta)
    pressure_scale = MU0 if not has_Ip and not has_beta else 1.0
    return _update_pf_from_rho_inputs(
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
        pressure_scale,
    )


@register_operator("PF_PSIN", supported_coordinates=(PSIN_COORDINATE,))
def update_PF_psin(
    out_psin: np.ndarray,
    out_psin_r: np.ndarray,
    out_psin_rr: np.ndarray,
    out_FFn_psin: np.ndarray,
    out_Pn_psin: np.ndarray,
    heat_input: np.ndarray,
    current_input: np.ndarray,
    coordinate_code: int,
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
) -> tuple[float, float]:
    del coordinate_code
    has_Ip = not np.isnan(Ip)
    has_beta = not np.isnan(beta)
    pressure_scale = MU0 if not has_Ip and not has_beta else 1.0
    return _update_pf_from_psin_inputs(
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
        pressure_scale,
    )


def _update_pf_from_rho_inputs(
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
    pressure_scale: float,
) -> tuple[float, float]:
    has_Ip = not np.isnan(Ip)
    has_beta = not np.isnan(beta)
    if not has_Ip and not has_beta and np.max(np.abs(heat_input)) <= 1e-14 and np.max(np.abs(current_input)) <= 1e-14:
        out_psin[:] = rho
        out_psin_r.fill(1.0)
        out_psin_rr.fill(0.0)
        out_FFn_psin.fill(0.0)
        out_Pn_psin.fill(0.0)
        return 0.0, 0.0
    integrand = Kn * (current_input * Ln_r + V_r * (pressure_scale * heat_input) / (4.0 * np.pi**2))
    corrected_integration(
        out_psin_r,
        integrand,
        integration_matrix,
        p=1,
        rho=rho,
        differentiation_matrix=differentiation_matrix,
    )
    out_psin_r *= -2.0
    np.maximum(out_psin_r, 0.0, out=out_psin_r)
    np.sqrt(out_psin_r, out=out_psin_r)
    out_psin_r /= Kn
    _enforce_axis_linear_psin_r(out_psin_r, rho)

    prof = out_psin_r
    integral_prof = quadrature(prof, weights)
    out_psin_r /= integral_prof
    full_differentiation(out_psin_rr, out_psin_r, differentiation_matrix)
    _update_psin_coordinate(out_psin, out_psin_r, integration_matrix, rho, differentiation_matrix)

    psin_r_safe = np.maximum(out_psin_r, 1e-10)
    if not has_Ip and not has_beta:
        alpha2 = integral_prof
        alpha1 = -MU0 / alpha2 * quadrature(heat_input, weights)
        out_Pn_psin[:] = MU0 * heat_input / (alpha1 * alpha2 * psin_r_safe)
        out_FFn_psin[:] = current_input / (alpha1 * alpha2 * psin_r_safe)
        return alpha1, alpha2

    c2 = integral_prof**2
    if has_Ip and not has_beta:
        G1n_integral = quadrature(
            JdivR * (current_input[:, None] + R * R * heat_input[:, None]) / psin_r_safe[:, None],
            weights,
        )
        alpha1 = -MU0 * Ip / G1n_integral
    elif has_beta and not has_Ip:
        Pn = _compute_Pn(heat_input, integration_matrix, weights)
        c1 = 0.5 * beta * B0**2 * quadrature(V_r, weights) / quadrature(Pn * V_r, weights)
        alpha1 = np.sqrt(c1 / c2)
    else:
        raise ValueError("PF does not support applying Ip and beta constraints simultaneously")

    alpha2 = c2 * alpha1
    np.divide(heat_input, psin_r_safe, out=out_Pn_psin)
    np.divide(current_input, psin_r_safe, out=out_FFn_psin)
    return alpha1, alpha2


def _update_pf_from_psin_inputs(
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
    pressure_scale: float,
) -> tuple[float, float]:
    del R0, Kn_r, S_r, F
    has_Ip = not np.isnan(Ip)
    has_beta = not np.isnan(beta)
    if not has_Ip and not has_beta and np.max(np.abs(heat_input)) <= 1e-14 and np.max(np.abs(current_input)) <= 1e-14:
        out_psin[:] = rho
        out_psin_r.fill(1.0)
        out_psin_rr.fill(0.0)
        out_FFn_psin.fill(0.0)
        out_Pn_psin.fill(0.0)
        return 0.0, 0.0
    integrand = current_input * Ln_r + V_r * (pressure_scale * heat_input) / (4.0 * np.pi**2)
    corrected_integration(
        out_psin_r,
        integrand,
        integration_matrix,
        p=1,
        rho=rho,
        differentiation_matrix=differentiation_matrix,
    )
    out_psin_r *= -1.0
    out_psin_r /= Kn

    prof = out_psin_r
    integral_prof = quadrature(prof, weights)
    out_psin_r /= integral_prof
    full_differentiation(out_psin_rr, out_psin_r, differentiation_matrix)
    _update_psin_coordinate(out_psin, out_psin_r, integration_matrix, rho, differentiation_matrix)

    if not has_Ip and not has_beta:
        alpha2 = integral_prof
        alpha1 = -MU0 * quadrature(heat_input * prof, weights)
        out_Pn_psin[:] = MU0 * heat_input / alpha1
        out_FFn_psin[:] = current_input / alpha1
        return alpha1, alpha2

    c2 = integral_prof
    out_Pn_psin[:] = heat_input
    out_FFn_psin[:] = current_input

    if has_Ip and not has_beta:
        G1n_integral = quadrature(
            JdivR * (out_FFn_psin[:, None] + R * R * out_Pn_psin[:, None]),
            weights,
        )
        alpha1 = -MU0 * Ip / G1n_integral
    elif has_beta and not has_Ip:
        Pn_r = out_Pn_psin * out_psin_r
        Pn = _compute_Pn(Pn_r, integration_matrix, weights)
        c1 = 0.5 * beta * B0**2 * quadrature(V_r, weights) / quadrature(Pn * V_r, weights)
        alpha1 = np.sqrt(c1 / c2)
    else:
        raise ValueError("PF does not support applying Ip and beta constraints simultaneously")

    alpha2 = c2 * alpha1
    return alpha1, alpha2


def _update_pp_from_rho_inputs(
    out_psin: np.ndarray,
    out_psin_r: np.ndarray,
    out_psin_rr: np.ndarray,
    out_FFn_psin: np.ndarray,
    out_Pn_psin: np.ndarray,
    heat_input: np.ndarray,
    current_input: np.ndarray,
    coordinate_code: int,
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
) -> tuple[float, float]:
    has_Ip = not np.isnan(Ip)
    has_beta = not np.isnan(beta)

    if has_Ip:
        out_psin_r[:] = current_input
        alpha2 = MU0 * Ip / (2.0 * np.pi * Kn[-1] * out_psin_r[-1])
    else:
        alpha2 = quadrature(current_input, weights)
        out_psin_r[:] = current_input / alpha2

    full_differentiation(out_psin_rr, out_psin_r, differentiation_matrix)
    _update_psin_coordinate(out_psin, out_psin_r, integration_matrix, rho, differentiation_matrix)
    psin_r_safe = np.maximum(out_psin_r, 1e-10)

    if has_beta:
        np.divide(heat_input, psin_r_safe, out=out_Pn_psin)
        Pn = _compute_Pn(out_Pn_psin * out_psin_r, integration_matrix, weights)
        alpha1 = 0.5 * beta * B0**2 / alpha2 * quadrature(V_r, weights) / quadrature(Pn * V_r, weights)
    else:
        P_r = heat_input
        alpha1 = -MU0 / alpha2 * quadrature(P_r, weights)
        out_Pn_psin[:] = MU0 * P_r / (alpha1 * alpha2 * psin_r_safe)

    out_FFn_psin[:] = (
        -((alpha2 / alpha1) * (Kn_r * out_psin_r + Kn * out_psin_rr) + V_r * out_Pn_psin / (4.0 * np.pi**2)) / Ln_r
    )
    return alpha1, alpha2


def _update_pp_from_psin_inputs(
    out_psin: np.ndarray,
    out_psin_r: np.ndarray,
    out_psin_rr: np.ndarray,
    out_FFn_psin: np.ndarray,
    out_Pn_psin: np.ndarray,
    heat_input: np.ndarray,
    current_input: np.ndarray,
    coordinate_code: int,
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
) -> tuple[float, float]:
    has_Ip = not np.isnan(Ip)
    has_beta = not np.isnan(beta)

    if has_Ip:
        out_psin_r[:] = current_input
        alpha2 = MU0 * Ip / (2.0 * np.pi * Kn[-1] * out_psin_r[-1])
    else:
        alpha2 = quadrature(current_input, weights)
        out_psin_r[:] = current_input / alpha2

    full_differentiation(out_psin_rr, out_psin_r, differentiation_matrix)
    _update_psin_coordinate(out_psin, out_psin_r, integration_matrix, rho, differentiation_matrix)
    psin_r_safe = np.maximum(out_psin_r, 1e-10)

    if has_beta:
        out_Pn_psin[:] = heat_input
        Pn = _compute_Pn(out_Pn_psin * out_psin_r, integration_matrix, weights)
        alpha1 = 0.5 * beta * B0**2 / alpha2 * quadrature(V_r, weights) / quadrature(Pn * V_r, weights)
    else:
        P_r = heat_input * out_psin_r * alpha2
        alpha1 = -MU0 / alpha2 * quadrature(P_r, weights)
        out_Pn_psin[:] = MU0 * P_r / (alpha1 * alpha2 * psin_r_safe)

    out_FFn_psin[:] = (
        -((alpha2 / alpha1) * (Kn_r * out_psin_r + Kn * out_psin_rr) + V_r * out_Pn_psin / (4.0 * np.pi**2)) / Ln_r
    )
    return alpha1, alpha2


@register_operator("PP_RHO", supported_coordinates=(RHO_COORDINATE,))
def update_PP_RHO(
    out_psin: np.ndarray,
    out_psin_r: np.ndarray,
    out_psin_rr: np.ndarray,
    out_FFn_psin: np.ndarray,
    out_Pn_psin: np.ndarray,
    heat_input: np.ndarray,
    current_input: np.ndarray,
    coordinate_code: int,
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
) -> tuple[float, float]:
    return _update_pp_from_rho_inputs(
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


@register_operator("PP_PSIN", supported_coordinates=(PSIN_COORDINATE,))
def update_PP_PSIN(
    out_psin: np.ndarray,
    out_psin_r: np.ndarray,
    out_psin_rr: np.ndarray,
    out_FFn_psin: np.ndarray,
    out_Pn_psin: np.ndarray,
    heat_input: np.ndarray,
    current_input: np.ndarray,
    coordinate_code: int,
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
) -> tuple[float, float]:
    return _update_pp_from_psin_inputs(
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


def _update_pi_from_rho_inputs(
    out_psin: np.ndarray,
    out_psin_r: np.ndarray,
    out_psin_rr: np.ndarray,
    out_FFn_psin: np.ndarray,
    out_Pn_psin: np.ndarray,
    heat_input: np.ndarray,
    current_input: np.ndarray,
    coordinate_code: int,
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
) -> tuple[float, float]:
    has_Ip = not np.isnan(Ip)
    has_beta = not np.isnan(beta)

    if has_Ip:
        Itor = Ip / current_input[-1] * current_input
    else:
        Itor = current_input
    # PI treats current_input as enclosed toroidal current; interpolation undershoot can
    # create tiny non-physical negatives near the magnetic axis for rho-route samples.
    # Keep a tiny positive floor so axis diagnostics that divide by psin_r stay finite.
    itor_floor = max(float(Itor[-1]), 1.0) * 1e-12
    Itor = np.maximum(Itor, itor_floor)

    alpha2 = quadrature(MU0 * Itor / (2.0 * np.pi * Kn), weights)

    out_psin_r[:] = MU0 * Itor / (2.0 * np.pi * alpha2 * Kn)
    full_differentiation(out_psin_rr, out_psin_r, differentiation_matrix)
    _update_psin_coordinate(out_psin, out_psin_r, integration_matrix, rho, differentiation_matrix)
    psin_r_safe = np.maximum(out_psin_r, 1e-10)
    Itor_r = np.empty_like(Itor)
    full_differentiation(Itor_r, Itor, differentiation_matrix)

    if has_beta:
        np.divide(heat_input, psin_r_safe, out=out_Pn_psin)
        Pn = _compute_Pn(out_Pn_psin * out_psin_r, integration_matrix, weights)
        alpha1 = 0.5 * beta * B0**2 / alpha2 * quadrature(V_r, weights) / quadrature(Pn * V_r, weights)
    else:
        P_r = heat_input
        alpha1 = -MU0 / alpha2 * quadrature(P_r, weights)
        out_Pn_psin[:] = MU0 * P_r / (alpha1 * alpha2 * psin_r_safe)

    out_FFn_psin[:] = -((MU0 / (2.0 * np.pi * alpha1)) * Itor_r + V_r * out_Pn_psin / (4.0 * np.pi**2)) / Ln_r
    return alpha1, alpha2


def _update_pi_from_psin_inputs(
    out_psin: np.ndarray,
    out_psin_r: np.ndarray,
    out_psin_rr: np.ndarray,
    out_FFn_psin: np.ndarray,
    out_Pn_psin: np.ndarray,
    heat_input: np.ndarray,
    current_input: np.ndarray,
    coordinate_code: int,
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
) -> tuple[float, float]:
    has_Ip = not np.isnan(Ip)
    has_beta = not np.isnan(beta)

    if has_Ip:
        Itor = Ip / current_input[-1] * current_input
    else:
        Itor = current_input
    itor_floor = max(float(Itor[-1]), 1.0) * 1e-12
    Itor = np.maximum(Itor, itor_floor)

    alpha2 = quadrature(MU0 * Itor / (2.0 * np.pi * Kn), weights)
    out_psin_r[:] = MU0 * Itor / (2.0 * np.pi * alpha2 * Kn)
    full_differentiation(out_psin_rr, out_psin_r, differentiation_matrix)
    _update_psin_coordinate(out_psin, out_psin_r, integration_matrix, rho, differentiation_matrix)
    psin_r_safe = np.maximum(out_psin_r, 1e-10)
    Itor_r = np.empty_like(Itor)
    full_differentiation(Itor_r, Itor, differentiation_matrix)

    if has_beta:
        out_Pn_psin[:] = heat_input
        Pn = _compute_Pn(out_Pn_psin * out_psin_r, integration_matrix, weights)
        alpha1 = 0.5 * beta * B0**2 / alpha2 * quadrature(V_r, weights) / quadrature(Pn * V_r, weights)
    else:
        P_r = heat_input * out_psin_r * alpha2
        alpha1 = -MU0 / alpha2 * quadrature(P_r, weights)
        out_Pn_psin[:] = MU0 * P_r / (alpha1 * alpha2 * psin_r_safe)

    out_FFn_psin[:] = -((MU0 / (2.0 * np.pi * alpha1)) * Itor_r + V_r * out_Pn_psin / (4.0 * np.pi**2)) / Ln_r
    return alpha1, alpha2


@register_operator("PI_RHO", supported_coordinates=(RHO_COORDINATE,))
def update_PI_RHO(
    out_psin: np.ndarray,
    out_psin_r: np.ndarray,
    out_psin_rr: np.ndarray,
    out_FFn_psin: np.ndarray,
    out_Pn_psin: np.ndarray,
    heat_input: np.ndarray,
    current_input: np.ndarray,
    coordinate_code: int,
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
) -> tuple[float, float]:
    return _update_pi_from_rho_inputs(
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


@register_operator("PI_PSIN", supported_coordinates=(PSIN_COORDINATE,))
def update_PI_PSIN(
    out_psin: np.ndarray,
    out_psin_r: np.ndarray,
    out_psin_rr: np.ndarray,
    out_FFn_psin: np.ndarray,
    out_Pn_psin: np.ndarray,
    heat_input: np.ndarray,
    current_input: np.ndarray,
    coordinate_code: int,
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
) -> tuple[float, float]:
    return _update_pi_from_psin_inputs(
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


def _update_pj1_from_rho_inputs(
    out_psin: np.ndarray,
    out_psin_r: np.ndarray,
    out_psin_rr: np.ndarray,
    out_FFn_psin: np.ndarray,
    out_Pn_psin: np.ndarray,
    heat_input: np.ndarray,
    current_input: np.ndarray,
    coordinate_code: int,
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
) -> tuple[float, float]:
    has_Ip = not np.isnan(Ip)
    has_beta = not np.isnan(beta)

    integrand_j = current_input * S_r
    corrected_integration(
        out_psin_r,
        integrand_j,
        integration_matrix,
        p=2,
        rho=rho,
        differentiation_matrix=differentiation_matrix,
    )
    I_tor_prof = out_psin_r

    if has_Ip:
        I_tor = Ip * (I_tor_prof / I_tor_prof[-1])
        jtor = current_input * (Ip / I_tor_prof[-1])
    else:
        I_tor = I_tor_prof
        jtor = current_input

    alpha2 = quadrature(MU0 * I_tor / (2.0 * np.pi * Kn), weights)
    out_psin_r[:] = MU0 * I_tor / (2.0 * np.pi * alpha2 * Kn)
    full_differentiation(out_psin_rr, out_psin_r, differentiation_matrix)
    _update_psin_coordinate(out_psin, out_psin_r, integration_matrix, rho, differentiation_matrix)
    psin_r_safe = np.maximum(out_psin_r, 1e-10)

    if has_beta:
        np.divide(heat_input, psin_r_safe, out=out_Pn_psin)
        Pn = _compute_Pn(out_Pn_psin * out_psin_r, integration_matrix, weights)
        alpha1 = 0.5 * beta * B0**2 / alpha2 * quadrature(V_r, weights) / quadrature(Pn * V_r, weights)
    else:
        P_r = heat_input
        alpha1 = -MU0 / alpha2 * quadrature(P_r, weights)
        out_Pn_psin[:] = MU0 * P_r / (alpha1 * alpha2 * psin_r_safe)

    out_FFn_psin[:] = -((MU0 / (2.0 * np.pi * alpha1)) * jtor * S_r + V_r * out_Pn_psin / (4.0 * np.pi**2)) / Ln_r
    return alpha1, alpha2


def _update_pj1_from_psin_inputs(
    out_psin: np.ndarray,
    out_psin_r: np.ndarray,
    out_psin_rr: np.ndarray,
    out_FFn_psin: np.ndarray,
    out_Pn_psin: np.ndarray,
    heat_input: np.ndarray,
    current_input: np.ndarray,
    coordinate_code: int,
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
) -> tuple[float, float]:
    has_Ip = not np.isnan(Ip)
    has_beta = not np.isnan(beta)

    integrand_j = current_input * S_r
    corrected_integration(
        out_psin_r, integrand_j, integration_matrix, p=2, rho=rho, differentiation_matrix=differentiation_matrix
    )
    I_tor_prof = out_psin_r

    if has_Ip:
        I_tor = Ip * (I_tor_prof / I_tor_prof[-1])
        jtor = current_input * (Ip / I_tor_prof[-1])
    else:
        I_tor = I_tor_prof
        jtor = current_input

    alpha2 = quadrature(MU0 * I_tor / (2.0 * np.pi * Kn), weights)
    out_psin_r[:] = MU0 * I_tor / (2.0 * np.pi * alpha2 * Kn)
    full_differentiation(out_psin_rr, out_psin_r, differentiation_matrix)
    _update_psin_coordinate(out_psin, out_psin_r, integration_matrix, rho, differentiation_matrix)
    psin_r_safe = np.maximum(out_psin_r, 1e-10)

    if has_beta:
        out_Pn_psin[:] = heat_input
        Pn = _compute_Pn(out_Pn_psin * out_psin_r, integration_matrix, weights)
        alpha1 = 0.5 * beta * B0**2 / alpha2 * quadrature(V_r, weights) / quadrature(Pn * V_r, weights)
    else:
        P_r = heat_input * out_psin_r * alpha2
        alpha1 = -MU0 / alpha2 * quadrature(P_r, weights)
        out_Pn_psin[:] = MU0 * P_r / (alpha1 * alpha2 * psin_r_safe)

    out_FFn_psin[:] = -((MU0 / (2.0 * np.pi * alpha1)) * jtor * S_r + V_r * out_Pn_psin / (4.0 * np.pi**2)) / Ln_r
    return alpha1, alpha2


@register_operator("PJ1_RHO", supported_coordinates=(RHO_COORDINATE,))
def update_PJ1_RHO(
    out_psin: np.ndarray,
    out_psin_r: np.ndarray,
    out_psin_rr: np.ndarray,
    out_FFn_psin: np.ndarray,
    out_Pn_psin: np.ndarray,
    heat_input: np.ndarray,
    current_input: np.ndarray,
    coordinate_code: int,
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
) -> tuple[float, float]:
    return _update_pj1_from_rho_inputs(
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


@register_operator("PJ1_PSIN", supported_coordinates=(PSIN_COORDINATE,))
def update_PJ1_PSIN(
    out_psin: np.ndarray,
    out_psin_r: np.ndarray,
    out_psin_rr: np.ndarray,
    out_FFn_psin: np.ndarray,
    out_Pn_psin: np.ndarray,
    heat_input: np.ndarray,
    current_input: np.ndarray,
    coordinate_code: int,
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
) -> tuple[float, float]:
    return _update_pj1_from_psin_inputs(
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


def _update_pj2_from_rho_inputs(
    out_psin: np.ndarray,
    out_psin_r: np.ndarray,
    out_psin_rr: np.ndarray,
    out_FFn_psin: np.ndarray,
    out_Pn_psin: np.ndarray,
    heat_input: np.ndarray,
    current_input: np.ndarray,
    coordinate_code: int,
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
) -> tuple[float, float]:
    has_Ip = not np.isnan(Ip)
    has_beta = not np.isnan(beta)

    integrand = (Ln_r * current_input) / F
    corrected_integration(
        out_psin_r,
        integrand,
        integration_matrix,
        p=1,
        rho=rho,
        differentiation_matrix=differentiation_matrix,
    )
    integral_val = out_psin_r

    if has_Ip:
        I_tor = Ip * (F * integral_val) / (R0 * B0 * integral_val[-1])
    else:
        I_tor = 2.0 * np.pi * F * integral_val
    alpha2 = quadrature(MU0 * I_tor / (2.0 * np.pi * Kn), weights)
    out_psin_r[:] = MU0 * I_tor / (2.0 * np.pi * alpha2 * Kn)
    full_differentiation(out_psin_rr, out_psin_r, differentiation_matrix)
    _update_psin_coordinate(out_psin, out_psin_r, integration_matrix, rho, differentiation_matrix)
    psin_r_safe = np.maximum(out_psin_r, 1e-10)

    if has_beta:
        np.divide(heat_input, psin_r_safe, out=out_Pn_psin)
        Pn = _compute_Pn(out_Pn_psin * out_psin_r, integration_matrix, weights)
        alpha1 = 0.5 * beta * B0**2 / alpha2 * quadrature(V_r, weights) / quadrature(Pn * V_r, weights)
    else:
        P_r = heat_input
        alpha1 = -MU0 / alpha2 * quadrature(P_r, weights)
        out_Pn_psin[:] = MU0 * P_r / (alpha1 * alpha2 * psin_r_safe)

    F_r = np.empty_like(F)
    full_differentiation(F_r, F, differentiation_matrix)
    FF_r = F * F_r
    out_FFn_psin[:] = FF_r / (alpha1 * alpha2 * psin_r_safe)
    return alpha1, alpha2


def _update_pj2_from_psin_inputs(
    out_psin: np.ndarray,
    out_psin_r: np.ndarray,
    out_psin_rr: np.ndarray,
    out_FFn_psin: np.ndarray,
    out_Pn_psin: np.ndarray,
    heat_input: np.ndarray,
    current_input: np.ndarray,
    coordinate_code: int,
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
) -> tuple[float, float]:
    has_Ip = not np.isnan(Ip)
    has_beta = not np.isnan(beta)

    integrand = (Ln_r * current_input) / F
    corrected_integration(
        out_psin_r, integrand, integration_matrix, p=1, rho=rho, differentiation_matrix=differentiation_matrix
    )
    integral_val = out_psin_r

    if has_Ip:
        I_tor = Ip * (F * integral_val) / (R0 * B0 * integral_val[-1])
    else:
        I_tor = 2.0 * np.pi * F * integral_val
    alpha2 = quadrature(MU0 * I_tor / (2.0 * np.pi * Kn), weights)
    out_psin_r[:] = MU0 * I_tor / (2.0 * np.pi * alpha2 * Kn)
    full_differentiation(out_psin_rr, out_psin_r, differentiation_matrix)
    _update_psin_coordinate(out_psin, out_psin_r, integration_matrix, rho, differentiation_matrix)
    psin_r_safe = np.maximum(out_psin_r, 1e-10)

    if has_beta:
        out_Pn_psin[:] = heat_input
        Pn = _compute_Pn(out_Pn_psin * out_psin_r, integration_matrix, weights)
        alpha1 = 0.5 * beta * B0**2 / alpha2 * quadrature(V_r, weights) / quadrature(Pn * V_r, weights)
    else:
        P_r = heat_input * out_psin_r * alpha2
        alpha1 = -MU0 / alpha2 * quadrature(P_r, weights)
        out_Pn_psin[:] = MU0 * P_r / (alpha1 * alpha2 * psin_r_safe)

    F_r = np.empty_like(F)
    full_differentiation(F_r, F, differentiation_matrix)
    FF_r = F * F_r
    out_FFn_psin[:] = FF_r / (alpha1 * alpha2 * psin_r_safe)
    return alpha1, alpha2


@register_operator("PJ2_RHO", supported_coordinates=(RHO_COORDINATE,))
def update_PJ2_RHO(
    out_psin: np.ndarray,
    out_psin_r: np.ndarray,
    out_psin_rr: np.ndarray,
    out_FFn_psin: np.ndarray,
    out_Pn_psin: np.ndarray,
    heat_input: np.ndarray,
    current_input: np.ndarray,
    coordinate_code: int,
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
) -> tuple[float, float]:
    return _update_pj2_from_rho_inputs(
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


@register_operator("PJ2_PSIN", supported_coordinates=(PSIN_COORDINATE,))
def update_PJ2_PSIN(
    out_psin: np.ndarray,
    out_psin_r: np.ndarray,
    out_psin_rr: np.ndarray,
    out_FFn_psin: np.ndarray,
    out_Pn_psin: np.ndarray,
    heat_input: np.ndarray,
    current_input: np.ndarray,
    coordinate_code: int,
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
) -> tuple[float, float]:
    return _update_pj2_from_psin_inputs(
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


def _update_pq_from_rho_inputs(
    out_psin: np.ndarray,
    out_psin_r: np.ndarray,
    out_psin_rr: np.ndarray,
    out_FFn_psin: np.ndarray,
    out_Pn_psin: np.ndarray,
    heat_input: np.ndarray,
    current_input: np.ndarray,
    coordinate_code: int,
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
) -> tuple[float, float]:
    has_Ip = not np.isnan(Ip)
    has_beta = not np.isnan(beta)

    if has_Ip:
        q_scale = (2.0 * np.pi * R0 * B0) / (MU0 * Ip)
        q_prof = current_input * q_scale * (Kn[-1] * Ln_r[-1] / current_input[-1])
    else:
        q_prof = current_input

    integrand_alpha2 = (F * Ln_r) / q_prof
    alpha2 = quadrature(integrand_alpha2, weights)

    out_psin_r[:] = (F * Ln_r) / (alpha2 * q_prof)
    full_differentiation(out_psin_rr, out_psin_r, differentiation_matrix)
    _update_psin_coordinate(out_psin, out_psin_r, integration_matrix, rho, differentiation_matrix)
    psin_r_safe = np.maximum(out_psin_r, 1e-10)

    if has_beta:
        np.divide(heat_input, psin_r_safe, out=out_Pn_psin)
        Pn = _compute_Pn(out_Pn_psin * out_psin_r, integration_matrix, weights)
        alpha1 = 0.5 * beta * B0**2 / alpha2 * quadrature(V_r, weights) / quadrature(Pn * V_r, weights)
    else:
        P_r = heat_input
        alpha1 = -MU0 / alpha2 * quadrature(P_r, weights)
        out_Pn_psin[:] = MU0 * P_r / (alpha1 * alpha2 * psin_r_safe)

    F_r = np.empty_like(F)
    full_differentiation(F_r, F, differentiation_matrix)
    FF_r = F * F_r
    out_FFn_psin[:] = FF_r / (alpha1 * alpha2 * psin_r_safe)
    return alpha1, alpha2


def _update_pq_from_psin_inputs(
    out_psin: np.ndarray,
    out_psin_r: np.ndarray,
    out_psin_rr: np.ndarray,
    out_FFn_psin: np.ndarray,
    out_Pn_psin: np.ndarray,
    heat_input: np.ndarray,
    current_input: np.ndarray,
    coordinate_code: int,
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
) -> tuple[float, float]:
    has_Ip = not np.isnan(Ip)
    has_beta = not np.isnan(beta)

    if has_Ip:
        q_scale = (2.0 * np.pi * R0 * B0) / (MU0 * Ip)
        q_prof = current_input * q_scale * (Kn[-1] * Ln_r[-1] / current_input[-1])
    else:
        q_prof = current_input

    integrand_alpha2 = (F * Ln_r) / q_prof
    alpha2 = quadrature(integrand_alpha2, weights)

    out_psin_r[:] = (F * Ln_r) / (alpha2 * q_prof)
    full_differentiation(out_psin_rr, out_psin_r, differentiation_matrix)
    _update_psin_coordinate(out_psin, out_psin_r, integration_matrix, rho, differentiation_matrix)
    psin_r_safe = np.maximum(out_psin_r, 1e-10)

    if has_beta:
        out_Pn_psin[:] = heat_input
        Pn = _compute_Pn(out_Pn_psin * out_psin_r, integration_matrix, weights)
        alpha1 = 0.5 * beta * B0**2 / alpha2 * quadrature(V_r, weights) / quadrature(Pn * V_r, weights)
    else:
        P_r = heat_input * out_psin_r * alpha2
        alpha1 = -MU0 / alpha2 * quadrature(P_r, weights)
        out_Pn_psin[:] = MU0 * P_r / (alpha1 * alpha2 * psin_r_safe)

    F_r = np.empty_like(F)
    full_differentiation(F_r, F, differentiation_matrix)
    FF_r = F * F_r
    out_FFn_psin[:] = FF_r / (alpha1 * alpha2 * psin_r_safe)
    return alpha1, alpha2


@register_operator("PQ_RHO", supported_coordinates=(RHO_COORDINATE,))
def update_PQ_RHO(
    out_psin: np.ndarray,
    out_psin_r: np.ndarray,
    out_psin_rr: np.ndarray,
    out_FFn_psin: np.ndarray,
    out_Pn_psin: np.ndarray,
    heat_input: np.ndarray,
    current_input: np.ndarray,
    coordinate_code: int,
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
) -> tuple[float, float]:
    return _update_pq_from_rho_inputs(
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


@register_operator("PQ_PSIN", supported_coordinates=(PSIN_COORDINATE,))
def update_PQ_PSIN(
    out_psin: np.ndarray,
    out_psin_r: np.ndarray,
    out_psin_rr: np.ndarray,
    out_FFn_psin: np.ndarray,
    out_Pn_psin: np.ndarray,
    heat_input: np.ndarray,
    current_input: np.ndarray,
    coordinate_code: int,
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
) -> tuple[float, float]:
    return _update_pq_from_psin_inputs(
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


def _register_standard_routes(
    base_name: str,
    *,
    rho_implementation: str,
    psin_implementation: str,
    psin_uniform_strategy: str,
    psin_uniform_parameterization: str = SOURCE_PARAMETERIZATION_IDENTITY,
) -> None:
    register_route(
        base_name,
        "rho",
        UNIFORM_NODES,
        implementation_name=rho_implementation,
        source_strategy=SOURCE_STRATEGY_SINGLE_PASS,
    )
    register_route(
        base_name,
        "rho",
        GRID_NODES,
        implementation_name=rho_implementation,
        source_strategy=SOURCE_STRATEGY_SINGLE_PASS,
    )
    register_route(
        base_name,
        "psin",
        UNIFORM_NODES,
        implementation_name=psin_implementation,
        source_strategy=psin_uniform_strategy,
        source_parameterization=psin_uniform_parameterization,
    )
    register_route(
        base_name,
        "psin",
        GRID_NODES,
        implementation_name=psin_implementation,
        source_strategy=SOURCE_STRATEGY_SINGLE_PASS,
    )


def _register_default_source_routes() -> None:
    _register_standard_routes(
        "PF",
        rho_implementation="PF_RHO",
        psin_implementation="PF_PSIN",
        psin_uniform_strategy=SOURCE_STRATEGY_PROFILE_OWNED_PSIN,
    )
    _register_standard_routes(
        "PP",
        rho_implementation="PP_RHO",
        psin_implementation="PP_PSIN",
        psin_uniform_strategy=SOURCE_STRATEGY_PROFILE_OWNED_PSIN,
        psin_uniform_parameterization=SOURCE_PARAMETERIZATION_SQRT_PSIN,
    )
    _register_standard_routes(
        "PI",
        rho_implementation="PI_RHO",
        psin_implementation="PI_PSIN",
        psin_uniform_strategy=SOURCE_STRATEGY_PROFILE_OWNED_PSIN,
    )
    _register_standard_routes(
        "PJ1",
        rho_implementation="PJ1_RHO",
        psin_implementation="PJ1_PSIN",
        psin_uniform_strategy=SOURCE_STRATEGY_PROFILE_OWNED_PSIN,
    )
    _register_standard_routes(
        "PJ2",
        rho_implementation="PJ2_RHO",
        psin_implementation="PJ2_PSIN",
        psin_uniform_strategy=SOURCE_STRATEGY_FIXED_POINT_PSIN,
    )
    _register_standard_routes(
        "PQ",
        rho_implementation="PQ_RHO",
        psin_implementation="PQ_PSIN",
        psin_uniform_strategy=SOURCE_STRATEGY_FIXED_POINT_PSIN,
    )


_register_default_source_routes()


def full_differentiation(
    out: np.ndarray,
    arr: np.ndarray,
    differentiation_matrix: np.ndarray,
) -> np.ndarray:
    """执行全径向微分."""

    _validate_vector_matrix_kernel(
        out=out,
        arr=arr,
        matrix=differentiation_matrix,
        matrix_route="differentiation_matrix",
    )
    np.matmul(differentiation_matrix, arr, out=out)
    return out


def theta_reduction(
    out: np.ndarray,
    arr: np.ndarray,
    weights: np.ndarray,
    axis: int,
) -> np.ndarray:
    """沿指定轴执行求积约化."""

    if arr.ndim not in (1, 2):
        raise ValueError(f"Expected 1D or 2D array, got {arr.shape}")
    if weights.ndim != 1 or weights.shape[0] != arr.shape[0]:
        raise ValueError(f"Expected weights to have shape ({arr.shape[0]},), got {weights.shape}")

    if axis == RHO_AXIS:
        if arr.ndim != 2 or out.ndim != 1 or out.shape[0] != arr.shape[1]:
            raise ValueError(f"Expected 2D arr and out shape ({arr.shape[1]},), got arr {arr.shape}, out {out.shape}")
        out[:] = weights @ arr
        return out

    if axis == THETA_AXIS:
        if arr.ndim != 2 or out.ndim != 1 or out.shape[0] != arr.shape[0]:
            raise ValueError(f"Expected 2D arr and out shape ({arr.shape[0]},), got arr {arr.shape}, out {out.shape}")
        np.sum(arr, axis=1, out=out)
        out *= 2.0 * np.pi / arr.shape[1]
        return out

    raise ValueError(f"Unsupported quadrature axis {axis}")


def quadrature(
    arr: np.ndarray,
    weights: np.ndarray,
) -> float:
    """返回全域标量求积值."""

    if arr.ndim not in (1, 2):
        raise ValueError(f"Expected 1D or 2D array, got {arr.shape}")
    if weights.ndim != 1 or weights.shape[0] != arr.shape[0]:
        raise ValueError(f"Expected weights to have shape ({arr.shape[0]},), got {weights.shape}")

    radial_sum = np.dot(arr, weights) if arr.ndim == 1 else weights @ arr
    if arr.ndim == 1:
        return float(radial_sum)
    return float((2.0 * np.pi / arr.shape[1]) * np.sum(radial_sum))


def full_integration(
    out: np.ndarray,
    arr: np.ndarray,
    integration_matrix: np.ndarray,
) -> np.ndarray:
    """执行全径向积分."""

    _validate_vector_matrix_kernel(
        out=out,
        arr=arr,
        matrix=integration_matrix,
        matrix_route="integration_matrix",
    )
    np.matmul(integration_matrix, arr, out=out)
    return out


def corrected_integration(
    out: np.ndarray,
    arr: np.ndarray,
    integration_matrix: np.ndarray,
    *,
    p: int,
    rho: np.ndarray,
    differentiation_matrix: np.ndarray,
) -> np.ndarray:
    """执行原点修正后的径向积分."""

    _validate_vector_matrix_kernel(
        out=out,
        arr=arr,
        matrix=integration_matrix,
        matrix_route="integration_matrix",
    )

    if p < 0:
        raise ValueError("p must be non-negative")
    if rho.ndim != 1 or rho.shape != arr.shape:
        raise ValueError(f"Expected rho to have shape {arr.shape}, got {rho.shape}")
    if differentiation_matrix.ndim != 2 or differentiation_matrix.shape != (arr.shape[0], arr.shape[0]):
        raise ValueError(
            f"Expected differentiation_matrix to have shape ({arr.shape[0]}, {arr.shape[0]}), "
            f"got {differentiation_matrix.shape}"
        )

    rho_safe = np.where(rho > 1e-10, rho, 1e-10)
    q_int = arr / rho_safe**p
    system = float(p + 1) * np.eye(rho.shape[0]) + rho[:, None] * differentiation_matrix

    try:
        q_solution = np.linalg.solve(system, q_int)
        out[:] = q_solution
        out *= rho ** (p + 1)
        if not np.all(np.isfinite(out)):
            raise FloatingPointError
        return out
    except (np.linalg.LinAlgError, FloatingPointError):
        warnings.warn("Corrected spectral integration failed; falling back to full integration")
        return full_integration(out, arr, integration_matrix)


def corrected_linear_derivative(
    out: np.ndarray,
    arr: np.ndarray,
    differentiation_matrix: np.ndarray,
    *,
    rho: np.ndarray,
) -> np.ndarray:
    """对轴心奇函数/线性起始量执行修正微分."""

    _validate_vector_matrix_kernel(
        out=out,
        arr=arr,
        matrix=differentiation_matrix,
        matrix_route="differentiation_matrix",
    )
    if rho.ndim != 1 or rho.shape != arr.shape:
        raise ValueError(f"Expected rho to have shape {arr.shape}, got {rho.shape}")

    reduced = np.empty_like(arr)
    if arr.size == 0:
        return out
    if arr.size == 1:
        out[0] = 0.0
        return out

    rho_safe = np.where(rho > 1e-10, rho, 1.0)
    np.divide(arr, rho_safe, out=reduced)
    reduced[0] = reduced[1]
    _enforce_axis_even_profile(reduced, rho)

    reduced_r = np.empty_like(arr)
    full_differentiation(reduced_r, reduced, differentiation_matrix)
    _enforce_axis_linear_psin_r(reduced_r, rho)

    out[:] = reduced + rho * reduced_r
    out[0] = reduced[0]
    return out


def corrected_even_derivative(
    out: np.ndarray,
    arr: np.ndarray,
    differentiation_matrix: np.ndarray,
    *,
    rho: np.ndarray,
) -> np.ndarray:
    """对轴心偶函数量执行修正微分."""

    _validate_vector_matrix_kernel(
        out=out,
        arr=arr,
        matrix=differentiation_matrix,
        matrix_route="differentiation_matrix",
    )
    if rho.ndim != 1 or rho.shape != arr.shape:
        raise ValueError(f"Expected rho to have shape {arr.shape}, got {rho.shape}")

    if arr.size == 0:
        return out
    if arr.size == 1:
        out[0] = 0.0
        return out

    smooth = np.array(arr, copy=True)
    _enforce_axis_even_profile(smooth, rho)
    base = float(smooth[0])

    reduced = np.empty_like(arr)
    rho2_safe = np.where(rho > 1e-10, rho * rho, 1.0)
    reduced[:] = (smooth - base) / rho2_safe
    reduced[0] = reduced[1]
    _enforce_axis_even_profile(reduced, rho)

    reduced_r = np.empty_like(arr)
    full_differentiation(reduced_r, reduced, differentiation_matrix)
    _enforce_axis_linear_psin_r(reduced_r, rho)

    out[:] = 2.0 * rho * reduced + (rho * rho) * reduced_r
    out[0] = 0.0
    _enforce_axis_linear_psin_r(out, rho)
    return out


def _update_psin_coordinate(
    out_psin: np.ndarray,
    psin_r: np.ndarray,
    integration_matrix: np.ndarray,
    rho: np.ndarray,
    differentiation_matrix: np.ndarray,
) -> np.ndarray:
    corrected_integration(
        out_psin,
        psin_r,
        integration_matrix,
        p=2,
        rho=rho,
        differentiation_matrix=differentiation_matrix,
    )
    return _normalize_psin_coordinate_inplace(out_psin)


def _normalize_psin_coordinate_inplace(psin: np.ndarray) -> np.ndarray:
    if psin.ndim != 1 or psin.size < 2:
        raise ValueError(f"Expected psin to be 1D with at least two points, got {psin.shape}")

    offset = float(psin[0])
    scale = float(psin[-1] - offset)
    if abs(scale) < 1e-12:
        raise ValueError("psin does not span a valid normalized flux interval")

    psin -= offset
    psin /= scale
    psin[0] = 0.0
    psin[-1] = 1.0
    return psin


def _enforce_axis_linear_psin_r(psin_r: np.ndarray, rho: np.ndarray) -> np.ndarray:
    if psin_r.ndim != 1 or rho.ndim != 1 or psin_r.shape != rho.shape:
        raise ValueError(f"Expected psin_r/rho to share a 1D shape, got {psin_r.shape} and {rho.shape}")
    if psin_r.size < 2 or abs(rho[1]) < 1e-14:
        return psin_r
    if psin_r.size >= 3 and abs(rho[2]) >= 1e-14:
        slope = psin_r[2] / rho[2]
        psin_r[0] = slope * rho[0]
        psin_r[1] = slope * rho[1]
        return psin_r
    psin_r[0] = psin_r[1] * rho[0] / rho[1]
    return psin_r


def _enforce_axis_quadratic_itor(itor: np.ndarray, rho: np.ndarray) -> np.ndarray:
    if itor.ndim != 1 or rho.ndim != 1 or itor.shape != rho.shape:
        raise ValueError(f"Expected itor/rho to share a 1D shape, got {itor.shape} and {rho.shape}")
    if itor.size < 2 or abs(rho[1]) < 1e-14:
        return itor
    if itor.size >= 3 and abs(rho[2]) >= 1e-14:
        scale = itor[2] / (rho[2] * rho[2])
        itor[0] = scale * rho[0] * rho[0]
        itor[1] = scale * rho[1] * rho[1]
        return itor
    scale = itor[1] / (rho[1] * rho[1])
    itor[0] = scale * rho[0] * rho[0]
    return itor


def _enforce_axis_even_profile(profile: np.ndarray, rho: np.ndarray) -> np.ndarray:
    if profile.ndim != 1 or rho.ndim != 1 or profile.shape != rho.shape:
        raise ValueError(f"Expected profile/rho to share a 1D shape, got {profile.shape} and {rho.shape}")
    if profile.size < 3:
        return profile
    x1 = rho[1] * rho[1]
    x2 = rho[2] * rho[2]
    if abs(x2 - x1) < 1e-14:
        return profile
    slope = (profile[2] - profile[1]) / (x2 - x1)
    intercept = profile[1] - slope * x1
    profile[0] = intercept + slope * rho[0] * rho[0]
    profile[1] = intercept + slope * x1
    return profile


def _smooth_even_profile_on_rho2(profile: np.ndarray, rho: np.ndarray, *, degree: int = 5) -> np.ndarray:
    if profile.ndim != 1 or rho.ndim != 1 or profile.shape != rho.shape:
        raise ValueError(f"Expected profile/rho to share a 1D shape, got {profile.shape} and {rho.shape}")
    fit_degree = min(int(degree), profile.shape[0] - 1)
    if fit_degree <= 0:
        return profile

    x = rho * rho
    vandermonde = np.empty((profile.shape[0], fit_degree + 1), dtype=np.float64)
    vandermonde[:, 0] = 1.0
    for order in range(1, fit_degree + 1):
        vandermonde[:, order] = vandermonde[:, order - 1] * x

    gram = vandermonde.T @ vandermonde
    rhs = vandermonde.T @ profile
    coeff = np.linalg.solve(gram, rhs)
    profile[:] = vandermonde @ coeff
    return profile


def _stabilize_odd_profile_head_on_rho(
    profile: np.ndarray,
    rho: np.ndarray,
    *,
    fit_start: int = 6,
    fit_count: int = 12,
    replace_count: int = 12,
    degree: int = 2,
) -> np.ndarray:
    if profile.ndim != 1 or rho.ndim != 1 or profile.shape != rho.shape:
        raise ValueError(f"Expected profile/rho to share a 1D shape, got {profile.shape} and {rho.shape}")
    n = profile.shape[0]
    start = max(int(fit_start), 1)
    stop = min(start + int(fit_count), n)
    replace_stop = min(int(replace_count), stop)
    fit_degree = min(int(degree), stop - start - 1)
    if replace_stop <= 0 or stop - start < 2 or fit_degree <= 0:
        return profile

    x_fit = rho[start:stop] * rho[start:stop]
    y_fit = profile[start:stop] / np.maximum(rho[start:stop], 1e-12)
    vandermonde = np.empty((x_fit.shape[0], fit_degree + 1), dtype=np.float64)
    vandermonde[:, 0] = 1.0
    for order in range(1, fit_degree + 1):
        vandermonde[:, order] = vandermonde[:, order - 1] * x_fit

    gram = vandermonde.T @ vandermonde
    rhs = vandermonde.T @ y_fit
    coeff = np.linalg.solve(gram, rhs)

    x_replace = rho[:replace_stop] * rho[:replace_stop]
    replace_vandermonde = np.empty((replace_stop, fit_degree + 1), dtype=np.float64)
    replace_vandermonde[:, 0] = 1.0
    for order in range(1, fit_degree + 1):
        replace_vandermonde[:, order] = replace_vandermonde[:, order - 1] * x_replace

    profile[:replace_stop] = rho[:replace_stop] * (replace_vandermonde @ coeff)
    profile[0] = 0.0
    return profile


def _smooth_profile_head_three_point(
    profile: np.ndarray,
    *,
    replace_count: int = 8,
    passes: int = 1,
) -> np.ndarray:
    if profile.ndim != 1:
        raise ValueError(f"Expected profile to be 1D, got {profile.shape}")
    stop = min(max(int(replace_count), 0), profile.shape[0] - 1)
    if stop <= 1 or passes <= 0:
        return profile

    for _ in range(int(passes)):
        prev = profile.copy()
        for i in range(1, stop):
            profile[i] = 0.25 * prev[i - 1] + 0.5 * prev[i] + 0.25 * prev[i + 1]
    return profile
def resolve_source_inputs(
    out_heat_input: np.ndarray,
    out_current_input: np.ndarray,
    heat_input: np.ndarray,
    current_input: np.ndarray,
    coordinate_code: int,
    n_src: int,
    barycentric_weights: np.ndarray,
    fixed_remap_matrix: np.ndarray,
    psin_query: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """按 uniform source + coordinate 语义把输入解析到 operator rho 节点."""

    heat = np.asarray(heat_input, dtype=np.float64)
    current = np.asarray(current_input, dtype=np.float64)
    if heat.ndim != 1 or current.ndim != 1:
        raise ValueError(f"Expected 1D heat/current inputs, got {heat.shape} and {current.shape}")
    if heat.shape != current.shape:
        raise ValueError(f"Expected heat/current inputs to share a shape, got {heat.shape} and {current.shape}")
    if heat.shape[0] != n_src:
        raise ValueError(f"Expected heat/current inputs to have length {n_src}, got {heat.shape[0]}")
    if out_heat_input.ndim != 1 or out_current_input.ndim != 1 or out_heat_input.shape != out_current_input.shape:
        raise ValueError(
            "Expected out_heat_input/out_current_input to be 1D arrays with matching shapes, "
            f"got {out_heat_input.shape} and {out_current_input.shape}"
        )
    if psin_query.ndim != 1:
        raise ValueError(f"Expected psin_query to be 1D, got {psin_query.shape}")

    if coordinate_code == RHO_COORDINATE:
        np.matmul(fixed_remap_matrix, heat, out=out_heat_input)
        np.matmul(fixed_remap_matrix, current, out=out_current_input)
        return out_heat_input, out_current_input

    if psin_query.shape != out_heat_input.shape:
        raise ValueError(f"Expected psin_query to have shape {out_heat_input.shape}, got {psin_query.shape}")

    _linear_uniform_interpolate_pair(
        out_heat_input,
        out_current_input,
        heat,
        current,
        psin_query,
    )
    return out_heat_input, out_current_input


def resolve_source_scratch_kernel(operator_kernel: Callable) -> Callable | None:
    """numpy backend 当前不提供 scratch-aware source kernel."""
    return None


def materialize_profile_owned_psin_source(
    out_psin: np.ndarray,
    out_psin_r: np.ndarray,
    out_psin_rr: np.ndarray,
    out_source_psin_query: np.ndarray,
    out_parameter_query: np.ndarray,
    out_heat_input: np.ndarray,
    out_current_input: np.ndarray,
    psin_fields: np.ndarray,
    heat_input: np.ndarray,
    current_input: np.ndarray,
    parameterization_code: int,
) -> tuple[np.ndarray, np.ndarray]:
    if psin_fields.ndim != 2 or psin_fields.shape[0] != 3:
        raise ValueError(f"Expected psin_fields to have shape (3, Nr), got {psin_fields.shape}")
    nr = psin_fields.shape[1]
    expected = (nr,)
    arrays = {
        "out_psin": out_psin,
        "out_psin_r": out_psin_r,
        "out_psin_rr": out_psin_rr,
        "out_source_psin_query": out_source_psin_query,
        "out_parameter_query": out_parameter_query,
        "out_heat_input": out_heat_input,
        "out_current_input": out_current_input,
    }
    for name, arr in arrays.items():
        if arr.ndim != 1 or arr.shape != expected:
            raise ValueError(f"Expected {name} to have shape {expected}, got {arr.shape}")

    heat = np.asarray(heat_input, dtype=np.float64)
    current = np.asarray(current_input, dtype=np.float64)
    if heat.ndim != 1 or current.ndim != 1 or heat.shape != current.shape:
        raise ValueError(f"Expected 1D heat/current inputs with matching shapes, got {heat.shape} and {current.shape}")

    _materialize_profile_owned_psin_source_impl(
        out_psin,
        out_psin_r,
        out_psin_rr,
        out_source_psin_query,
        out_parameter_query,
        out_heat_input,
        out_current_input,
        np.asarray(psin_fields, dtype=np.float64),
        heat,
        current,
        int(parameterization_code),
    )
    return out_heat_input, out_current_input


def update_fourier_family_fields(
    out_c_fields: np.ndarray,
    out_s_fields: np.ndarray,
    base_c_fields: np.ndarray,
    base_s_fields: np.ndarray,
    active_u_fields: np.ndarray,
    c_source_slots: np.ndarray,
    s_source_slots: np.ndarray,
    c_active_order: int,
    s_active_order: int,
) -> tuple[np.ndarray, np.ndarray]:
    if out_c_fields.ndim != 3 or out_s_fields.ndim != 3:
        raise ValueError(
            f"Expected out_c_fields/out_s_fields to be 3D, got {out_c_fields.shape} and {out_s_fields.shape}"
        )
    if base_c_fields.shape != out_c_fields.shape or base_s_fields.shape != out_s_fields.shape:
        raise ValueError(
            "Expected base_c_fields/base_s_fields to match output shapes, "
            f"got {base_c_fields.shape} and {base_s_fields.shape}"
        )
    if active_u_fields.ndim != 3:
        raise ValueError(f"Expected active_u_fields to be 3D, got {active_u_fields.shape}")
    if c_source_slots.ndim != 1 or s_source_slots.ndim != 1:
        raise ValueError(
            f"Expected c_source_slots/s_source_slots to be 1D, got {c_source_slots.shape} and {s_source_slots.shape}"
        )

    _update_fourier_family_fields_impl(
        out_c_fields,
        out_s_fields,
        np.asarray(base_c_fields, dtype=np.float64),
        np.asarray(base_s_fields, dtype=np.float64),
        np.asarray(active_u_fields, dtype=np.float64),
        np.asarray(c_source_slots, dtype=np.int64),
        np.asarray(s_source_slots, dtype=np.int64),
        int(c_active_order),
        int(s_active_order),
    )
    return out_c_fields, out_s_fields


def materialize_projected_source_inputs(
    out_heat_input: np.ndarray,
    out_current_input: np.ndarray,
    heat_coeff: np.ndarray,
    current_coeff: np.ndarray,
    current_source_values: np.ndarray,
    psin_query: np.ndarray,
    projection_domain_code: int,
    endpoint_policy_code: int,
    blend: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    if out_heat_input.ndim != 1 or out_current_input.ndim != 1 or out_heat_input.shape != out_current_input.shape:
        raise ValueError(
            "Expected out_heat_input/out_current_input to be 1D arrays with matching shapes, "
            f"got {out_heat_input.shape} and {out_current_input.shape}"
        )
    if psin_query.ndim != 1 or psin_query.shape != out_heat_input.shape:
        raise ValueError(f"Expected psin_query to have shape {out_heat_input.shape}, got {psin_query.shape}")
    if blend.ndim != 1 or blend.shape != out_heat_input.shape:
        raise ValueError(f"Expected blend to have shape {out_heat_input.shape}, got {blend.shape}")

    _materialize_projected_source_inputs_impl(
        out_heat_input,
        out_current_input,
        np.asarray(heat_coeff, dtype=np.float64),
        np.asarray(current_coeff, dtype=np.float64),
        np.asarray(current_source_values, dtype=np.float64),
        np.asarray(psin_query, dtype=np.float64),
        int(projection_domain_code),
        int(endpoint_policy_code),
        np.asarray(blend, dtype=np.float64),
    )
    return out_heat_input, out_current_input


def update_fixed_point_psin_query(
    query: np.ndarray,
    psin: np.ndarray,
    tolerance: float,
) -> bool:
    if query.ndim != 1 or psin.ndim != 1 or query.shape != psin.shape:
        raise ValueError(f"Expected query/psin to share a 1D shape, got {query.shape} and {psin.shape}")
    return _update_fixed_point_psin_query_impl(
        np.asarray(query, dtype=np.float64),
        np.asarray(psin, dtype=np.float64),
        float(tolerance),
    )


def _materialize_projected_source_inputs_impl(
    out_heat_input: np.ndarray,
    out_current_input: np.ndarray,
    heat_coeff: np.ndarray,
    current_coeff: np.ndarray,
    current_source_values: np.ndarray,
    psin_query: np.ndarray,
    projection_domain_code: int,
    endpoint_policy_code: int,
    blend: np.ndarray,
) -> None:
    for i in range(out_heat_input.shape[0]):
        q = float(psin_query[i])
        if q < 0.0:
            q = 0.0
        elif q > 1.0:
            q = 1.0
        if projection_domain_code == PROJECTION_DOMAIN_SQRT_PSIN:
            q = float(np.sqrt(q))
        elif projection_domain_code != PROJECTION_DOMAIN_PSIN:
            raise ValueError(f"Unsupported projection domain code {projection_domain_code!r}")
        x = 2.0 * q - 1.0
        out_heat_input[i] = _evaluate_chebyshev_scalar(heat_coeff, x)
        out_current_input[i] = _evaluate_chebyshev_scalar(current_coeff, x)

    if endpoint_policy_code == ENDPOINT_POLICY_NONE:
        return
    if endpoint_policy_code == ENDPOINT_POLICY_RIGHT:
        out_current_input[-1] = float(current_source_values[-1])
        return
    if endpoint_policy_code == ENDPOINT_POLICY_BOTH:
        out_current_input[0] = float(current_source_values[0])
        out_current_input[-1] = float(current_source_values[-1])
        return
    if endpoint_policy_code == ENDPOINT_POLICY_AFFINE_BOTH:
        delta_left = float(current_source_values[0]) - float(out_current_input[0])
        delta_right = float(current_source_values[-1]) - float(out_current_input[-1])
        for i in range(out_current_input.shape[0]):
            out_current_input[i] += (1.0 - blend[i]) * delta_left + blend[i] * delta_right
        return
    raise ValueError(f"Unsupported endpoint policy code {endpoint_policy_code!r}")


def _materialize_profile_owned_psin_source_impl(
    out_psin: np.ndarray,
    out_psin_r: np.ndarray,
    out_psin_rr: np.ndarray,
    out_source_psin_query: np.ndarray,
    out_parameter_query: np.ndarray,
    out_heat_input: np.ndarray,
    out_current_input: np.ndarray,
    psin_fields: np.ndarray,
    heat_input: np.ndarray,
    current_input: np.ndarray,
    parameterization_code: int,
) -> None:
    psin = psin_fields[0]
    np.copyto(out_psin, psin)
    np.copyto(out_psin_r, psin_fields[1])
    np.copyto(out_psin_rr, psin_fields[2])
    np.copyto(out_source_psin_query, psin)
    np.copyto(out_parameter_query, psin)

    if parameterization_code == SOURCE_PARAMETERIZATION_CODE_SQRT_PSIN:
        np.maximum(out_parameter_query, 0.0, out=out_parameter_query)
        np.sqrt(out_parameter_query, out=out_parameter_query)
    elif parameterization_code != SOURCE_PARAMETERIZATION_CODE_IDENTITY:
        raise ValueError(f"Unsupported source parameterization code {parameterization_code!r}")

    _linear_uniform_interpolate_pair(
        out_heat_input,
        out_current_input,
        heat_input,
        current_input,
        out_parameter_query,
    )


def _update_fourier_family_fields_impl(
    out_c_fields: np.ndarray,
    out_s_fields: np.ndarray,
    base_c_fields: np.ndarray,
    base_s_fields: np.ndarray,
    active_u_fields: np.ndarray,
    c_source_slots: np.ndarray,
    s_source_slots: np.ndarray,
    c_active_order: int,
    s_active_order: int,
) -> None:
    for order in range(out_c_fields.shape[0]):
        target = out_c_fields[order]
        if order <= c_active_order:
            slot = int(c_source_slots[order])
            if slot >= 0:
                np.copyto(target, active_u_fields[slot])
            else:
                np.copyto(target, base_c_fields[order])
        else:
            target.fill(0.0)

    np.copyto(out_s_fields[0], base_s_fields[0])
    for order in range(1, out_s_fields.shape[0]):
        target = out_s_fields[order]
        if order <= s_active_order:
            slot = int(s_source_slots[order])
            if slot >= 0:
                np.copyto(target, active_u_fields[slot])
            else:
                np.copyto(target, base_s_fields[order])
        else:
            target.fill(0.0)


def _update_fixed_point_psin_query_impl(
    query: np.ndarray,
    psin: np.ndarray,
    tolerance: float,
) -> bool:
    max_abs_diff = 0.0
    for i in range(query.shape[0]):
        diff = abs(float(psin[i]) - float(query[i]))
        if diff > max_abs_diff:
            max_abs_diff = diff
        query[i] = psin[i]
    return bool(max_abs_diff <= tolerance)


def _evaluate_chebyshev_scalar(coeff: np.ndarray, x: float) -> float:
    if coeff.size == 0:
        return 0.0
    if coeff.size == 1:
        return float(coeff[0])
    b_kplus1 = 0.0
    b_kplus2 = 0.0
    for idx in range(coeff.size - 1, 0, -1):
        b_k = 2.0 * x * b_kplus1 - b_kplus2 + float(coeff[idx])
        b_kplus2 = b_kplus1
        b_kplus1 = b_k
    return x * b_kplus1 - b_kplus2 + float(coeff[0])


def _linear_uniform_interpolate_pair(
    out0: np.ndarray,
    out1: np.ndarray,
    values0: np.ndarray,
    values1: np.ndarray,
    query: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    n_src = values0.shape[0]
    if n_src == 1:
        out0.fill(float(values0[0]))
        out1.fill(float(values1[0]))
        return out0, out1

    step = 1.0 / float(n_src - 1)
    for i, q_raw in enumerate(query):
        q = min(max(float(q_raw), 0.0), 1.0)
        if q >= 1.0:
            out0[i] = values0[-1]
            out1[i] = values1[-1]
            continue

        position = q / step
        left = int(position)
        right = left + 1
        frac = position - float(left)
        out0[i] = (1.0 - frac) * values0[left] + frac * values0[right]
        out1[i] = (1.0 - frac) * values1[left] + frac * values1[right]
    return out0, out1


def _uniform_barycentric_weights(n_src: int) -> np.ndarray:
    weights = np.empty(n_src, dtype=np.float64)
    weights[0] = 1.0
    for j in range(1, n_src):
        weights[j] = -weights[j - 1] * float(n_src - j) / float(j)
    return weights


def _build_uniform_barycentric_matrix(
    query: np.ndarray,
    n_src: int,
    stencil_size: int,
    weights: np.ndarray,
) -> np.ndarray:
    matrix = np.empty((query.shape[0], n_src), dtype=np.float64)
    if n_src == 1:
        matrix[:, 0] = 1.0
        return matrix

    for i, q in enumerate(query):
        matrix[i].fill(0.0)
        start = _local_uniform_stencil_start(q, n_src, stencil_size)
        hit = False
        for local_j in range(stencil_size):
            j = start + local_j
            diff = q - j / (n_src - 1.0)
            if abs(diff) <= 1e-14:
                matrix[i, j] = 1.0
                hit = True
                break
        if hit:
            continue

        denominator = 0.0
        for local_j in range(stencil_size):
            j = start + local_j
            denominator += weights[local_j] / (q - j / (n_src - 1.0))
        for local_j in range(stencil_size):
            j = start + local_j
            matrix[i, j] = (weights[local_j] / (q - j / (n_src - 1.0))) / denominator
    return matrix


def _local_uniform_stencil_start(q: float, n_src: int, stencil_size: int) -> int:
    if stencil_size >= n_src:
        return 0
    pos = q * (n_src - 1.0)
    center = int(np.searchsorted(np.arange(n_src, dtype=np.float64), pos, side="left"))
    start = center - stencil_size // 2
    if start < 0:
        return 0
    max_start = n_src - stencil_size
    if start > max_start:
        return max_start
    return start


def _validate_vector_matrix_kernel(
    *,
    out: np.ndarray,
    arr: np.ndarray,
    matrix: np.ndarray,
    matrix_name: str | None = None,
    matrix_route: str | None = None,
) -> None:
    label = matrix_name if matrix_name is not None else matrix_route
    if label is None:
        label = "matrix"
    if arr.ndim != 1:
        raise ValueError(f"Expected a 1D array, got {arr.shape}")
    if out.ndim != 1 or out.shape != arr.shape:
        raise ValueError(f"Expected out to have shape {arr.shape}, got {out.shape}")
    if matrix.ndim != 2 or matrix.shape != (arr.shape[0], arr.shape[0]):
        raise ValueError(f"Expected {label} to have shape ({arr.shape[0]}, {arr.shape[0]}), got {matrix.shape}")


def _compute_Pn(
    Pn_r: np.ndarray,
    integration_matrix: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    Pn = np.empty_like(Pn_r)
    full_integration(Pn, Pn_r, integration_matrix)
    Pn -= quadrature(Pn_r, weights)
    return Pn

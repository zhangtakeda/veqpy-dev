"""
Module: engine.numpy_source

Role:
- 负责在 numpy backend 下注册 source operators.
- 负责校验 operator/coordinate/nodes 组合并执行 source kernels.

Public API:
- register_operator
- validate_operator
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


@dataclass(frozen=True, slots=True)
class _SourceRouteSpec:
    coordinate_code: int
    nodes: str
    implementation: Callable
    source_strategy: str


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
    name: str,
    coordinate: str,
    nodes: str,
    *,
    implementation_name: str,
    source_strategy: str,
) -> None:
    coordinate_code = _normalize_coordinate(coordinate)
    normalized_nodes = _normalize_nodes(nodes)
    try:
        implementation_spec = OPERATOR_REGISTRY[implementation_name]
    except KeyError as exc:
        raise KeyError(f"Unknown implementation {implementation_name!r} for route {(name, coordinate, nodes)!r}") from exc

    if coordinate_code not in implementation_spec.supported_coordinates:
        raise ValueError(
            f"Implementation {implementation_name!r} does not support coordinate={coordinate!r}"
        )

    key = (str(name).upper(), coordinate_code, normalized_nodes)
    if key in ROUTE_REGISTRY:
        raise ValueError(f"Source route {key!r} is already registered")

    ROUTE_REGISTRY[key] = _SourceRouteSpec(
        coordinate_code=coordinate_code,
        nodes=normalized_nodes,
        implementation=implementation_spec.implementation,
        source_strategy=source_strategy,
    )


def validate_operator(
    name: str,
    coordinate: str,
    nodes: str = UNIFORM_NODES,
) -> _SourceRouteSpec:
    """校验 operator/coordinate/nodes 组合并返回 route 规格."""

    coordinate_code = _normalize_coordinate(coordinate)
    normalized_nodes = _normalize_nodes(nodes)
    key = (str(name).upper(), coordinate_code, normalized_nodes)
    try:
        return ROUTE_REGISTRY[key]
    except KeyError as exc:
        supported_names = sorted({route_name for route_name, _, _ in ROUTE_REGISTRY})
        supported = ", ".join(supported_names)
        raise KeyError(
            f"Unknown source route name={name!r}, coordinate={coordinate!r}, nodes={nodes!r}. "
            f"Supported operator names: {supported}"
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
    if (
        not has_Ip
        and not has_beta
        and np.max(np.abs(heat_input)) <= 1e-14
        and np.max(np.abs(current_input)) <= 1e-14
    ):
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
    if (
        not has_Ip
        and not has_beta
        and np.max(np.abs(heat_input)) <= 1e-14
        and np.max(np.abs(current_input)) <= 1e-14
    ):
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

    out_FFn_psin[:] = -(
        (alpha2 / alpha1) * (Kn_r * out_psin_r + Kn * out_psin_rr) + V_r * out_Pn_psin / (4.0 * np.pi**2)
    ) / Ln_r
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

    out_FFn_psin[:] = -(
        (alpha2 / alpha1) * (Kn_r * out_psin_r + Kn * out_psin_rr) + V_r * out_Pn_psin / (4.0 * np.pi**2)
    ) / Ln_r
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
        out_psin, out_psin_r, out_psin_rr, out_FFn_psin, out_Pn_psin, heat_input, current_input, coordinate_code,
        R0, B0, weights, differentiation_matrix, integration_matrix, rho, V_r, Kn, Kn_r, Ln_r, S_r, R, JdivR, F, Ip, beta,
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
        out_psin, out_psin_r, out_psin_rr, out_FFn_psin, out_Pn_psin, heat_input, current_input, coordinate_code,
        R0, B0, weights, differentiation_matrix, integration_matrix, rho, V_r, Kn, Kn_r, Ln_r, S_r, R, JdivR, F, Ip, beta,
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
        out_psin, out_psin_r, out_psin_rr, out_FFn_psin, out_Pn_psin, heat_input, current_input, coordinate_code,
        R0, B0, weights, differentiation_matrix, integration_matrix, rho, V_r, Kn, Kn_r, Ln_r, S_r, R, JdivR, F, Ip, beta,
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
        out_psin, out_psin_r, out_psin_rr, out_FFn_psin, out_Pn_psin, heat_input, current_input, coordinate_code,
        R0, B0, weights, differentiation_matrix, integration_matrix, rho, V_r, Kn, Kn_r, Ln_r, S_r, R, JdivR, F, Ip, beta,
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
    corrected_integration(out_psin_r, integrand_j, integration_matrix, p=2, rho=rho, differentiation_matrix=differentiation_matrix)
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
        out_psin, out_psin_r, out_psin_rr, out_FFn_psin, out_Pn_psin, heat_input, current_input, coordinate_code,
        R0, B0, weights, differentiation_matrix, integration_matrix, rho, V_r, Kn, Kn_r, Ln_r, S_r, R, JdivR, F, Ip, beta,
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
        out_psin, out_psin_r, out_psin_rr, out_FFn_psin, out_Pn_psin, heat_input, current_input, coordinate_code,
        R0, B0, weights, differentiation_matrix, integration_matrix, rho, V_r, Kn, Kn_r, Ln_r, S_r, R, JdivR, F, Ip, beta,
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
    corrected_integration(out_psin_r, integrand, integration_matrix, p=1, rho=rho, differentiation_matrix=differentiation_matrix)
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
        out_psin, out_psin_r, out_psin_rr, out_FFn_psin, out_Pn_psin, heat_input, current_input, coordinate_code,
        R0, B0, weights, differentiation_matrix, integration_matrix, rho, V_r, Kn, Kn_r, Ln_r, S_r, R, JdivR, F, Ip, beta,
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
        out_psin, out_psin_r, out_psin_rr, out_FFn_psin, out_Pn_psin, heat_input, current_input, coordinate_code,
        R0, B0, weights, differentiation_matrix, integration_matrix, rho, V_r, Kn, Kn_r, Ln_r, S_r, R, JdivR, F, Ip, beta,
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
        out_psin, out_psin_r, out_psin_rr, out_FFn_psin, out_Pn_psin, heat_input, current_input, coordinate_code,
        R0, B0, weights, differentiation_matrix, integration_matrix, rho, V_r, Kn, Kn_r, Ln_r, S_r, R, JdivR, F, Ip, beta,
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
        out_psin, out_psin_r, out_psin_rr, out_FFn_psin, out_Pn_psin, heat_input, current_input, coordinate_code,
        R0, B0, weights, differentiation_matrix, integration_matrix, rho, V_r, Kn, Kn_r, Ln_r, S_r, R, JdivR, F, Ip, beta,
    )


def _register_standard_routes(
    base_name: str,
    *,
    rho_implementation: str,
    psin_implementation: str,
    psin_uniform_strategy: str,
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
        matrix_name="differentiation_matrix",
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
        matrix_name="integration_matrix",
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
        matrix_name="integration_matrix",
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

    _barycentric_uniform_interpolate_pair(
        out_heat_input,
        out_current_input,
        heat,
        current,
        psin_query,
        barycentric_weights,
    )
    return out_heat_input, out_current_input


def _barycentric_uniform_interpolate_pair(
    out0: np.ndarray,
    out1: np.ndarray,
    values0: np.ndarray,
    values1: np.ndarray,
    query: np.ndarray,
    weights: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    n_src = values0.shape[0]
    stencil_size = weights.shape[0]
    if n_src == 1:
        out0.fill(float(values0[0]))
        out1.fill(float(values1[0]))
        return out0, out1

    for i, q in enumerate(query):
        start = _local_uniform_stencil_start(q, n_src, stencil_size)
        numerator0 = 0.0
        numerator1 = 0.0
        denominator = 0.0
        hit = False
        hit_value0 = 0.0
        hit_value1 = 0.0
        for local_j in range(stencil_size):
            j = start + local_j
            node = j / (n_src - 1.0)
            diff = q - node
            if abs(diff) <= 1e-14:
                hit = True
                hit_value0 = values0[j]
                hit_value1 = values1[j]
                break
            term = weights[local_j] / diff
            numerator0 += term * values0[j]
            numerator1 += term * values1[j]
            denominator += term
        if hit:
            out0[i] = hit_value0
            out1[i] = hit_value1
        else:
            inv_denominator = 1.0 / denominator
            out0[i] = numerator0 * inv_denominator
            out1[i] = numerator1 * inv_denominator
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
    matrix_name: str,
) -> None:
    if arr.ndim != 1:
        raise ValueError(f"Expected a 1D array, got {arr.shape}")
    if out.ndim != 1 or out.shape != arr.shape:
        raise ValueError(f"Expected out to have shape {arr.shape}, got {out.shape}")
    if matrix.ndim != 2 or matrix.shape != (arr.shape[0], arr.shape[0]):
        raise ValueError(f"Expected {matrix_name} to have shape ({arr.shape[0]}, {arr.shape[0]}), got {matrix.shape}")


def _compute_Pn(
    Pn_r: np.ndarray,
    integration_matrix: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    Pn = np.empty_like(Pn_r)
    full_integration(Pn, Pn_r, integration_matrix)
    Pn -= quadrature(Pn_r, weights)
    return Pn

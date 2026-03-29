"""
Module: engine.numba_source

Role:
- 负责在 numba backend 下注册 source operators.
- 负责校验 operator/coordinate/nodes 组合并执行 source kernels.

Public API:
- register_operator
- validate_operator
- build_source_remap_cache
- resolve_source_inputs

Notes:
- operator routing 保留在这里.
- operator 层只 bind 一个 source runner, 并把它作为 Stage-C 执行入口.
"""

from dataclasses import dataclass
from typing import Callable

import numpy as np
from numba import njit

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
        raise KeyError(
            f"Unknown implementation {implementation_name!r} for route {(name, coordinate, nodes)!r}"
        ) from exc

    if coordinate_code not in implementation_spec.supported_coordinates:
        raise ValueError(f"Implementation {implementation_name!r} does not support coordinate={coordinate!r}")

    key = (str(name).upper(), coordinate_code, normalized_nodes)
    if key in ROUTE_REGISTRY:
        raise ValueError(f"Source route {key!r} is already registered")

    ROUTE_REGISTRY[key] = _SourceRouteSpec(
        coordinate_code=coordinate_code,
        nodes=normalized_nodes,
        implementation=implementation_spec.implementation,
        source_strategy=source_strategy,
    )


def validate_operator(name: str, coordinate: str, nodes: str = UNIFORM_NODES) -> _SourceRouteSpec:
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


@register_operator("PF_RHO", supported_coordinates=(RHO_COORDINATE,))
@njit(cache=True, nogil=True)
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
@njit(cache=True, nogil=True)
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


@njit(cache=True, nogil=True)
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
    if (not has_Ip) and (not has_beta):
        zero_heat = True
        zero_current = True
        for i in range(heat_input.shape[0]):
            if abs(heat_input[i]) > 1e-14:
                zero_heat = False
                break
        for i in range(current_input.shape[0]):
            if abs(current_input[i]) > 1e-14:
                zero_current = False
                break
        if zero_heat and zero_current:
            for i in range(rho.shape[0]):
                out_psin[i] = rho[i]
                out_psin_r[i] = 1.0
                out_psin_rr[i] = 0.0
                out_FFn_psin[i] = 0.0
                out_Pn_psin[i] = 0.0
            return 0.0, 0.0
    integrand = np.empty_like(out_psin_r)
    _fill_pf_rho_integrand(integrand, Kn, current_input, Ln_r, V_r, heat_input, pressure_scale)
    corrected_integration(out_psin_r, integrand, integration_matrix, 1, rho, differentiation_matrix)
    out_psin_r *= -2.0
    for i in range(out_psin_r.shape[0]):
        if out_psin_r[i] < 0.0:
            out_psin_r[i] = 0.0
    out_psin_r[:] = np.sqrt(out_psin_r)
    out_psin_r /= Kn
    _enforce_axis_linear_psin_r(out_psin_r, rho)

    prof = out_psin_r
    integral_prof = quadrature(prof, weights)
    out_psin_r /= integral_prof
    full_differentiation(out_psin_rr, out_psin_r, differentiation_matrix)
    _update_psin_coordinate(out_psin, out_psin_r, integration_matrix, rho, differentiation_matrix)

    psin_r_safe = _maximum_floor(out_psin_r, 1e-10)
    if (not has_Ip) and (not has_beta):
        alpha2 = integral_prof
        alpha1 = -MU0 / alpha2 * quadrature(heat_input, weights)
        _fill_scaled_ratio(out_Pn_psin, heat_input, psin_r_safe, MU0 / (alpha1 * alpha2))
        _fill_scaled_ratio(out_FFn_psin, current_input, psin_r_safe, 1.0 / (alpha1 * alpha2))
        return alpha1, alpha2

    c2 = integral_prof * integral_prof
    if has_Ip and (not has_beta):
        g1n_integrand = np.empty_like(R)
        _fill_g1n_rho_integrand(g1n_integrand, JdivR, current_input, R, heat_input, psin_r_safe)
        G1n_integral = quadrature(g1n_integrand, weights)
        alpha1 = -MU0 * Ip / G1n_integral
    elif has_beta and (not has_Ip):
        Pn = _compute_Pn(heat_input, integration_matrix, weights)
        c1 = 0.5 * beta * B0**2 * quadrature(V_r, weights) / quadrature(Pn * V_r, weights)
        alpha1 = np.sqrt(c1 / c2)
    else:
        raise ValueError("PF does not support applying Ip and beta constraints simultaneously")

    alpha2 = c2 * alpha1
    _fill_scaled_ratio(out_Pn_psin, heat_input, psin_r_safe, 1.0)
    _fill_scaled_ratio(out_FFn_psin, current_input, psin_r_safe, 1.0)
    return alpha1, alpha2


@njit(cache=True, nogil=True)
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
    has_Ip = not np.isnan(Ip)
    has_beta = not np.isnan(beta)
    if (not has_Ip) and (not has_beta):
        zero_heat = True
        zero_current = True
        for i in range(heat_input.shape[0]):
            if abs(heat_input[i]) > 1e-14:
                zero_heat = False
                break
        for i in range(current_input.shape[0]):
            if abs(current_input[i]) > 1e-14:
                zero_current = False
                break
        if zero_heat and zero_current:
            for i in range(rho.shape[0]):
                out_psin[i] = rho[i]
                out_psin_r[i] = 1.0
                out_psin_rr[i] = 0.0
                out_FFn_psin[i] = 0.0
                out_Pn_psin[i] = 0.0
            return 0.0, 0.0
    integrand = np.empty_like(out_psin_r)
    _fill_pf_psin_integrand(integrand, current_input, Ln_r, V_r, heat_input, pressure_scale)
    corrected_integration(out_psin_r, integrand, integration_matrix, 1, rho, differentiation_matrix)
    out_psin_r *= -1.0
    out_psin_r /= Kn

    prof = out_psin_r
    integral_prof = quadrature(prof, weights)
    out_psin_r /= integral_prof
    full_differentiation(out_psin_rr, out_psin_r, differentiation_matrix)
    _update_psin_coordinate(out_psin, out_psin_r, integration_matrix, rho, differentiation_matrix)

    if (not has_Ip) and (not has_beta):
        alpha2 = integral_prof
        pressure_profile = np.empty_like(out_psin_r)
        _fill_pointwise_product(pressure_profile, heat_input, prof)
        alpha1 = -MU0 * quadrature(pressure_profile, weights)
        _fill_scaled_vector(out_Pn_psin, heat_input, MU0 / alpha1)
        _fill_scaled_vector(out_FFn_psin, current_input, 1.0 / alpha1)
        return alpha1, alpha2

    c2 = integral_prof
    _copy_vector(out_Pn_psin, heat_input)
    _copy_vector(out_FFn_psin, current_input)

    if has_Ip and (not has_beta):
        g1n_integrand = np.empty_like(R)
        _fill_g1n_psin_integrand(g1n_integrand, JdivR, out_FFn_psin, R, out_Pn_psin)
        G1n_integral = quadrature(g1n_integrand, weights)
        alpha1 = -MU0 * Ip / G1n_integral
    elif has_beta and (not has_Ip):
        Pn_r = np.empty_like(out_psin_r)
        _fill_pointwise_product(Pn_r, out_Pn_psin, out_psin_r)
        Pn = _compute_Pn(Pn_r, integration_matrix, weights)
        c1 = 0.5 * beta * B0**2 * quadrature(V_r, weights) / quadrature(Pn * V_r, weights)
        alpha1 = np.sqrt(c1 / c2)
    else:
        raise ValueError("PF does not support applying Ip and beta constraints simultaneously")

    alpha2 = c2 * alpha1
    return alpha1, alpha2


@njit(cache=True, nogil=True)
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
        _copy_vector(out_psin_r, current_input)
        alpha2 = MU0 * Ip / (2.0 * np.pi * Kn[-1] * out_psin_r[-1])
    else:
        alpha2 = quadrature(current_input, weights)
        _fill_scaled_vector(out_psin_r, current_input, 1.0 / alpha2)

    full_differentiation(out_psin_rr, out_psin_r, differentiation_matrix)
    _update_psin_coordinate(out_psin, out_psin_r, integration_matrix, rho, differentiation_matrix)
    psin_r_safe = _maximum_floor(out_psin_r, 1e-10)

    if has_beta:
        _fill_scaled_ratio(out_Pn_psin, heat_input, psin_r_safe, 1.0)
        Pn_r = np.empty_like(out_psin_r)
        _fill_pointwise_product(Pn_r, out_Pn_psin, out_psin_r)
        Pn = _compute_Pn(Pn_r, integration_matrix, weights)
        alpha1 = 0.5 * beta * B0**2 / alpha2 * quadrature(V_r, weights) / quadrature(Pn * V_r, weights)
    else:
        P_r = np.empty_like(out_psin_r)
        _copy_vector(P_r, heat_input)
        alpha1 = -MU0 / alpha2 * quadrature(P_r, weights)
        _fill_scaled_ratio(out_Pn_psin, P_r, psin_r_safe, MU0 / (alpha1 * alpha2))

    _fill_pp_ffn_psin(out_FFn_psin, out_psin_r, Kn_r, Kn, out_psin_rr, V_r, out_Pn_psin, Ln_r, alpha2 / alpha1)
    return alpha1, alpha2


@njit(cache=True, nogil=True)
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
        _copy_vector(out_psin_r, current_input)
        alpha2 = MU0 * Ip / (2.0 * np.pi * Kn[-1] * out_psin_r[-1])
    else:
        alpha2 = quadrature(current_input, weights)
        _fill_scaled_vector(out_psin_r, current_input, 1.0 / alpha2)

    full_differentiation(out_psin_rr, out_psin_r, differentiation_matrix)
    _update_psin_coordinate(out_psin, out_psin_r, integration_matrix, rho, differentiation_matrix)
    psin_r_safe = _maximum_floor(out_psin_r, 1e-10)

    if has_beta:
        _copy_vector(out_Pn_psin, heat_input)
        Pn_r = np.empty_like(out_psin_r)
        _fill_pointwise_product(Pn_r, out_Pn_psin, out_psin_r)
        Pn = _compute_Pn(Pn_r, integration_matrix, weights)
        alpha1 = 0.5 * beta * B0**2 / alpha2 * quadrature(V_r, weights) / quadrature(Pn * V_r, weights)
    else:
        P_r = np.empty_like(out_psin_r)
        _fill_scaled_product(P_r, heat_input, out_psin_r, alpha2)
        alpha1 = -MU0 / alpha2 * quadrature(P_r, weights)
        _fill_scaled_ratio(out_Pn_psin, P_r, psin_r_safe, MU0 / (alpha1 * alpha2))

    _fill_pp_ffn_psin(out_FFn_psin, out_psin_r, Kn_r, Kn, out_psin_rr, V_r, out_Pn_psin, Ln_r, alpha2 / alpha1)
    return alpha1, alpha2


@register_operator("PP_RHO", supported_coordinates=(RHO_COORDINATE,))
@njit(cache=True, nogil=True)
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
@njit(cache=True, nogil=True)
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


@njit(cache=True, nogil=True)
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
    Itor = np.empty_like(current_input)

    if has_Ip:
        _fill_scaled_vector(Itor, current_input, Ip / current_input[-1])
    else:
        _copy_vector(Itor, current_input)
    itor_floor = max(Itor[-1], 1.0) * 1e-12
    Itor[:] = _maximum_floor(Itor, itor_floor)

    itor_over_kn = np.empty_like(current_input)
    _fill_scaled_ratio(itor_over_kn, Itor, Kn, MU0 / (2.0 * np.pi))
    alpha2 = quadrature(itor_over_kn, weights)

    _fill_scaled_ratio(out_psin_r, Itor, Kn, MU0 / (2.0 * np.pi * alpha2))
    full_differentiation(out_psin_rr, out_psin_r, differentiation_matrix)
    _update_psin_coordinate(out_psin, out_psin_r, integration_matrix, rho, differentiation_matrix)
    psin_r_safe = _maximum_floor(out_psin_r, 1e-10)
    Itor_r = np.empty_like(Itor)
    full_differentiation(Itor_r, Itor, differentiation_matrix)

    if has_beta:
        _fill_scaled_ratio(out_Pn_psin, heat_input, psin_r_safe, 1.0)
        Pn_r = np.empty_like(out_psin_r)
        _fill_pointwise_product(Pn_r, out_Pn_psin, out_psin_r)
        Pn = _compute_Pn(Pn_r, integration_matrix, weights)
        alpha1 = 0.5 * beta * B0**2 / alpha2 * quadrature(V_r, weights) / quadrature(Pn * V_r, weights)
    else:
        P_r = np.empty_like(out_psin_r)
        _copy_vector(P_r, heat_input)
        alpha1 = -MU0 / alpha2 * quadrature(P_r, weights)
        _fill_scaled_ratio(out_Pn_psin, P_r, psin_r_safe, MU0 / (alpha1 * alpha2))

    _fill_pi_ffn_psin(out_FFn_psin, Itor_r, V_r, out_Pn_psin, Ln_r, MU0 / (2.0 * np.pi * alpha1))
    return alpha1, alpha2


@njit(cache=True, nogil=True)
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
    Itor = np.empty_like(current_input)

    if has_Ip:
        _fill_scaled_vector(Itor, current_input, Ip / current_input[-1])
    else:
        _copy_vector(Itor, current_input)
    itor_floor = max(Itor[-1], 1.0) * 1e-12
    Itor[:] = _maximum_floor(Itor, itor_floor)

    itor_over_kn = np.empty_like(current_input)
    _fill_scaled_ratio(itor_over_kn, Itor, Kn, MU0 / (2.0 * np.pi))
    alpha2 = quadrature(itor_over_kn, weights)

    _fill_scaled_ratio(out_psin_r, Itor, Kn, MU0 / (2.0 * np.pi * alpha2))
    full_differentiation(out_psin_rr, out_psin_r, differentiation_matrix)
    _update_psin_coordinate(out_psin, out_psin_r, integration_matrix, rho, differentiation_matrix)
    psin_r_safe = _maximum_floor(out_psin_r, 1e-10)
    Itor_r = np.empty_like(Itor)
    full_differentiation(Itor_r, Itor, differentiation_matrix)

    if has_beta:
        _copy_vector(out_Pn_psin, heat_input)
        Pn_r = np.empty_like(out_psin_r)
        _fill_pointwise_product(Pn_r, out_Pn_psin, out_psin_r)
        Pn = _compute_Pn(Pn_r, integration_matrix, weights)
        alpha1 = 0.5 * beta * B0**2 / alpha2 * quadrature(V_r, weights) / quadrature(Pn * V_r, weights)
    else:
        P_r = np.empty_like(out_psin_r)
        _fill_scaled_product(P_r, heat_input, out_psin_r, alpha2)
        alpha1 = -MU0 / alpha2 * quadrature(P_r, weights)
        _fill_scaled_ratio(out_Pn_psin, P_r, psin_r_safe, MU0 / (alpha1 * alpha2))

    _fill_pi_ffn_psin(out_FFn_psin, Itor_r, V_r, out_Pn_psin, Ln_r, MU0 / (2.0 * np.pi * alpha1))
    return alpha1, alpha2


@register_operator("PI_RHO", supported_coordinates=(RHO_COORDINATE,))
@njit(cache=True, nogil=True)
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
@njit(cache=True, nogil=True)
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


@njit(cache=True, nogil=True)
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

    integrand_j = np.empty_like(current_input)
    _fill_pointwise_product(integrand_j, current_input, S_r)
    corrected_integration(out_psin_r, integrand_j, integration_matrix, 2, rho, differentiation_matrix)
    I_tor_prof = np.empty_like(out_psin_r)
    _copy_vector(I_tor_prof, out_psin_r)
    I_tor = np.empty_like(current_input)
    jtor = np.empty_like(current_input)

    if has_Ip:
        _fill_scaled_vector(I_tor, I_tor_prof, Ip / I_tor_prof[-1])
        _fill_scaled_vector(jtor, current_input, Ip / I_tor_prof[-1])
    else:
        _copy_vector(I_tor, I_tor_prof)
        _copy_vector(jtor, current_input)

    itor_over_kn = np.empty_like(current_input)
    _fill_scaled_ratio(itor_over_kn, I_tor, Kn, MU0 / (2.0 * np.pi))
    alpha2 = quadrature(itor_over_kn, weights)
    _fill_scaled_ratio(out_psin_r, I_tor, Kn, MU0 / (2.0 * np.pi * alpha2))
    full_differentiation(out_psin_rr, out_psin_r, differentiation_matrix)
    _update_psin_coordinate(out_psin, out_psin_r, integration_matrix, rho, differentiation_matrix)
    psin_r_safe = _maximum_floor(out_psin_r, 1e-10)

    if has_beta:
        _fill_scaled_ratio(out_Pn_psin, heat_input, psin_r_safe, 1.0)
        Pn_r = np.empty_like(out_psin_r)
        _fill_pointwise_product(Pn_r, out_Pn_psin, out_psin_r)
        Pn = _compute_Pn(Pn_r, integration_matrix, weights)
        alpha1 = 0.5 * beta * B0**2 / alpha2 * quadrature(V_r, weights) / quadrature(Pn * V_r, weights)
    else:
        P_r = np.empty_like(out_psin_r)
        _copy_vector(P_r, heat_input)
        alpha1 = -MU0 / alpha2 * quadrature(P_r, weights)
        _fill_scaled_ratio(out_Pn_psin, P_r, psin_r_safe, MU0 / (alpha1 * alpha2))

    _fill_pj_ffn_psin(out_FFn_psin, jtor, S_r, V_r, out_Pn_psin, Ln_r, MU0 / (2.0 * np.pi * alpha1))
    return alpha1, alpha2


@njit(cache=True, nogil=True)
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

    integrand_j = np.empty_like(current_input)
    _fill_pointwise_product(integrand_j, current_input, S_r)
    corrected_integration(out_psin_r, integrand_j, integration_matrix, 2, rho, differentiation_matrix)
    I_tor_prof = np.empty_like(out_psin_r)
    _copy_vector(I_tor_prof, out_psin_r)
    I_tor = np.empty_like(current_input)
    jtor = np.empty_like(current_input)

    if has_Ip:
        _fill_scaled_vector(I_tor, I_tor_prof, Ip / I_tor_prof[-1])
        _fill_scaled_vector(jtor, current_input, Ip / I_tor_prof[-1])
    else:
        _copy_vector(I_tor, I_tor_prof)
        _copy_vector(jtor, current_input)

    itor_over_kn = np.empty_like(current_input)
    _fill_scaled_ratio(itor_over_kn, I_tor, Kn, MU0 / (2.0 * np.pi))
    alpha2 = quadrature(itor_over_kn, weights)
    _fill_scaled_ratio(out_psin_r, I_tor, Kn, MU0 / (2.0 * np.pi * alpha2))
    full_differentiation(out_psin_rr, out_psin_r, differentiation_matrix)
    _update_psin_coordinate(out_psin, out_psin_r, integration_matrix, rho, differentiation_matrix)
    psin_r_safe = _maximum_floor(out_psin_r, 1e-10)

    if has_beta:
        _copy_vector(out_Pn_psin, heat_input)
        Pn_r = np.empty_like(out_psin_r)
        _fill_pointwise_product(Pn_r, out_Pn_psin, out_psin_r)
        Pn = _compute_Pn(Pn_r, integration_matrix, weights)
        alpha1 = 0.5 * beta * B0**2 / alpha2 * quadrature(V_r, weights) / quadrature(Pn * V_r, weights)
    else:
        P_r = np.empty_like(out_psin_r)
        _fill_scaled_product(P_r, heat_input, out_psin_r, alpha2)
        alpha1 = -MU0 / alpha2 * quadrature(P_r, weights)
        _fill_scaled_ratio(out_Pn_psin, P_r, psin_r_safe, MU0 / (alpha1 * alpha2))

    _fill_pj_ffn_psin(out_FFn_psin, jtor, S_r, V_r, out_Pn_psin, Ln_r, MU0 / (2.0 * np.pi * alpha1))
    return alpha1, alpha2


@register_operator("PJ1_RHO", supported_coordinates=(RHO_COORDINATE,))
@njit(cache=True, nogil=True)
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
@njit(cache=True, nogil=True)
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


@njit(cache=True, nogil=True)
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

    integrand = np.empty_like(out_psin_r)
    _fill_product_ratio(integrand, Ln_r, current_input, F, 1.0)
    corrected_integration(out_psin_r, integrand, integration_matrix, 1, rho, differentiation_matrix)
    integral_val = np.empty_like(out_psin_r)
    _copy_vector(integral_val, out_psin_r)
    I_tor = np.empty_like(current_input)

    if has_Ip:
        _fill_scaled_product(I_tor, F, integral_val, Ip / (R0 * B0 * integral_val[-1]))
    else:
        _fill_scaled_product(I_tor, F, integral_val, 2.0 * np.pi)

    itor_over_kn = np.empty_like(current_input)
    _fill_scaled_ratio(itor_over_kn, I_tor, Kn, MU0 / (2.0 * np.pi))
    alpha2 = quadrature(itor_over_kn, weights)
    _fill_scaled_ratio(out_psin_r, I_tor, Kn, MU0 / (2.0 * np.pi * alpha2))
    full_differentiation(out_psin_rr, out_psin_r, differentiation_matrix)
    _update_psin_coordinate(out_psin, out_psin_r, integration_matrix, rho, differentiation_matrix)
    psin_r_safe = _maximum_floor(out_psin_r, 1e-10)

    if has_beta:
        _fill_scaled_ratio(out_Pn_psin, heat_input, psin_r_safe, 1.0)
        Pn_r = np.empty_like(out_psin_r)
        _fill_pointwise_product(Pn_r, out_Pn_psin, out_psin_r)
        Pn = _compute_Pn(Pn_r, integration_matrix, weights)
        alpha1 = 0.5 * beta * B0**2 / alpha2 * quadrature(V_r, weights) / quadrature(Pn * V_r, weights)
    else:
        P_r = np.empty_like(out_psin_r)
        _copy_vector(P_r, heat_input)
        alpha1 = -MU0 / alpha2 * quadrature(P_r, weights)
        _fill_scaled_ratio(out_Pn_psin, P_r, psin_r_safe, MU0 / (alpha1 * alpha2))

    F_r = np.empty_like(F)
    full_differentiation(F_r, F, differentiation_matrix)
    _fill_scaled_product(out_FFn_psin, F, F_r, 1.0 / (alpha1 * alpha2))
    _fill_scaled_ratio(out_FFn_psin, out_FFn_psin, psin_r_safe, 1.0)
    return alpha1, alpha2


@njit(cache=True, nogil=True)
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

    integrand = np.empty_like(out_psin_r)
    _fill_product_ratio(integrand, Ln_r, current_input, F, 1.0)
    corrected_integration(out_psin_r, integrand, integration_matrix, 1, rho, differentiation_matrix)
    integral_val = np.empty_like(out_psin_r)
    _copy_vector(integral_val, out_psin_r)
    I_tor = np.empty_like(current_input)

    if has_Ip:
        _fill_scaled_product(I_tor, F, integral_val, Ip / (R0 * B0 * integral_val[-1]))
    else:
        _fill_scaled_product(I_tor, F, integral_val, 2.0 * np.pi)

    itor_over_kn = np.empty_like(current_input)
    _fill_scaled_ratio(itor_over_kn, I_tor, Kn, MU0 / (2.0 * np.pi))
    alpha2 = quadrature(itor_over_kn, weights)
    _fill_scaled_ratio(out_psin_r, I_tor, Kn, MU0 / (2.0 * np.pi * alpha2))
    full_differentiation(out_psin_rr, out_psin_r, differentiation_matrix)
    _update_psin_coordinate(out_psin, out_psin_r, integration_matrix, rho, differentiation_matrix)
    psin_r_safe = _maximum_floor(out_psin_r, 1e-10)

    if has_beta:
        _copy_vector(out_Pn_psin, heat_input)
        Pn_r = np.empty_like(out_psin_r)
        _fill_pointwise_product(Pn_r, out_Pn_psin, out_psin_r)
        Pn = _compute_Pn(Pn_r, integration_matrix, weights)
        alpha1 = 0.5 * beta * B0**2 / alpha2 * quadrature(V_r, weights) / quadrature(Pn * V_r, weights)
    else:
        P_r = np.empty_like(out_psin_r)
        _fill_scaled_product(P_r, heat_input, out_psin_r, alpha2)
        alpha1 = -MU0 / alpha2 * quadrature(P_r, weights)
        _fill_scaled_ratio(out_Pn_psin, P_r, psin_r_safe, MU0 / (alpha1 * alpha2))

    F_r = np.empty_like(F)
    full_differentiation(F_r, F, differentiation_matrix)
    _fill_scaled_product(out_FFn_psin, F, F_r, 1.0 / (alpha1 * alpha2))
    _fill_scaled_ratio(out_FFn_psin, out_FFn_psin, psin_r_safe, 1.0)
    return alpha1, alpha2


@register_operator("PJ2_RHO", supported_coordinates=(RHO_COORDINATE,))
@njit(cache=True, nogil=True)
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
@njit(cache=True, nogil=True)
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


@njit(cache=True, nogil=True)
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
    q_prof = np.empty_like(current_input)

    if has_Ip:
        q_scale = (2.0 * np.pi * R0 * B0) / (MU0 * Ip)
        _fill_scaled_vector(q_prof, current_input, q_scale * (Kn[-1] * Ln_r[-1] / current_input[-1]))
    else:
        _copy_vector(q_prof, current_input)

    integrand_alpha2 = np.empty_like(out_psin_r)
    _fill_product_ratio(integrand_alpha2, F, Ln_r, q_prof, 1.0)
    alpha2 = quadrature(integrand_alpha2, weights)

    _fill_product_ratio(out_psin_r, F, Ln_r, q_prof, 1.0 / alpha2)
    full_differentiation(out_psin_rr, out_psin_r, differentiation_matrix)
    _update_psin_coordinate(out_psin, out_psin_r, integration_matrix, rho, differentiation_matrix)
    psin_r_safe = _maximum_floor(out_psin_r, 1e-10)

    if has_beta:
        _fill_scaled_ratio(out_Pn_psin, heat_input, psin_r_safe, 1.0)
        Pn_r = np.empty_like(out_psin_r)
        _fill_pointwise_product(Pn_r, out_Pn_psin, out_psin_r)
        Pn = _compute_Pn(Pn_r, integration_matrix, weights)
        alpha1 = 0.5 * beta * B0**2 / alpha2 * quadrature(V_r, weights) / quadrature(Pn * V_r, weights)
    else:
        P_r = np.empty_like(out_psin_r)
        _copy_vector(P_r, heat_input)
        alpha1 = -MU0 / alpha2 * quadrature(P_r, weights)
        _fill_scaled_ratio(out_Pn_psin, P_r, psin_r_safe, MU0 / (alpha1 * alpha2))

    F_r = np.empty_like(F)
    full_differentiation(F_r, F, differentiation_matrix)
    _fill_scaled_product(out_FFn_psin, F, F_r, 1.0 / (alpha1 * alpha2))
    _fill_scaled_ratio(out_FFn_psin, out_FFn_psin, psin_r_safe, 1.0)
    return alpha1, alpha2


@njit(cache=True, nogil=True)
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
    q_prof = np.empty_like(current_input)

    if has_Ip:
        q_scale = (2.0 * np.pi * R0 * B0) / (MU0 * Ip)
        _fill_scaled_vector(q_prof, current_input, q_scale * (Kn[-1] * Ln_r[-1] / current_input[-1]))
    else:
        _copy_vector(q_prof, current_input)

    integrand_alpha2 = np.empty_like(out_psin_r)
    _fill_product_ratio(integrand_alpha2, F, Ln_r, q_prof, 1.0)
    alpha2 = quadrature(integrand_alpha2, weights)

    _fill_product_ratio(out_psin_r, F, Ln_r, q_prof, 1.0 / alpha2)
    full_differentiation(out_psin_rr, out_psin_r, differentiation_matrix)
    _update_psin_coordinate(out_psin, out_psin_r, integration_matrix, rho, differentiation_matrix)
    psin_r_safe = _maximum_floor(out_psin_r, 1e-10)

    if has_beta:
        _copy_vector(out_Pn_psin, heat_input)
        Pn_r = np.empty_like(out_psin_r)
        _fill_pointwise_product(Pn_r, out_Pn_psin, out_psin_r)
        Pn = _compute_Pn(Pn_r, integration_matrix, weights)
        alpha1 = 0.5 * beta * B0**2 / alpha2 * quadrature(V_r, weights) / quadrature(Pn * V_r, weights)
    else:
        P_r = np.empty_like(out_psin_r)
        _fill_scaled_product(P_r, heat_input, out_psin_r, alpha2)
        alpha1 = -MU0 / alpha2 * quadrature(P_r, weights)
        _fill_scaled_ratio(out_Pn_psin, P_r, psin_r_safe, MU0 / (alpha1 * alpha2))

    F_r = np.empty_like(F)
    full_differentiation(F_r, F, differentiation_matrix)
    _fill_scaled_product(out_FFn_psin, F, F_r, 1.0 / (alpha1 * alpha2))
    _fill_scaled_ratio(out_FFn_psin, out_FFn_psin, psin_r_safe, 1.0)
    return alpha1, alpha2


@register_operator("PQ_RHO", supported_coordinates=(RHO_COORDINATE,))
@njit(cache=True, nogil=True)
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
@njit(cache=True, nogil=True)
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


@njit(cache=True, nogil=True)
def full_differentiation(out: np.ndarray, arr: np.ndarray, differentiation_matrix: np.ndarray) -> np.ndarray:
    """执行全径向微分."""
    rows = differentiation_matrix.shape[0]
    cols = differentiation_matrix.shape[1]
    for i in range(rows):
        total = 0.0
        for j in range(cols):
            total += differentiation_matrix[i, j] * arr[j]
        out[i] = total
    return out


@njit(cache=True, nogil=True)
def theta_reduction(out: np.ndarray, arr: np.ndarray, weights: np.ndarray, axis: int) -> np.ndarray:
    """沿指定轴执行求积约化."""
    if axis == RHO_AXIS:
        for j in range(arr.shape[1]):
            total = 0.0
            for i in range(arr.shape[0]):
                total += weights[i] * arr[i, j]
            out[j] = total
        return out

    if axis == THETA_AXIS:
        scale = 2.0 * np.pi / arr.shape[1]
        for i in range(arr.shape[0]):
            total = 0.0
            for j in range(arr.shape[1]):
                total += arr[i, j]
            out[i] = total * scale
        return out

    raise ValueError(f"Unsupported quadrature axis {axis}")


@njit(cache=True, nogil=True)
def quadrature(arr: np.ndarray, weights: np.ndarray) -> float:
    """返回全域标量求积值."""
    if arr.ndim == 1:
        total = 0.0
        for i in range(arr.shape[0]):
            total += arr[i] * weights[i]
        return total

    radial_sum = np.empty(arr.shape[1], dtype=arr.dtype)
    for j in range(arr.shape[1]):
        total = 0.0
        for i in range(arr.shape[0]):
            total += weights[i] * arr[i, j]
        radial_sum[j] = total

    total = 0.0
    for j in range(radial_sum.shape[0]):
        total += radial_sum[j]
    return (2.0 * np.pi / arr.shape[1]) * total


@njit(cache=True, nogil=True)
def full_integration(out: np.ndarray, arr: np.ndarray, integration_matrix: np.ndarray) -> np.ndarray:
    """执行全径向积分."""
    rows = integration_matrix.shape[0]
    cols = integration_matrix.shape[1]
    for i in range(rows):
        total = 0.0
        for j in range(cols):
            total += integration_matrix[i, j] * arr[j]
        out[i] = total
    return out


@njit(cache=True, nogil=True)
def corrected_integration(
    out: np.ndarray,
    arr: np.ndarray,
    integration_matrix: np.ndarray,
    p: int,
    rho: np.ndarray,
    differentiation_matrix: np.ndarray,
) -> np.ndarray:
    """执行原点修正后的径向积分."""
    n = arr.shape[0]
    rho_safe = np.empty_like(rho)
    for i in range(n):
        rho_safe[i] = rho[i] if rho[i] > 1e-10 else 1e-10

    q_int = arr / (rho_safe**p)
    system = rho[:, None] * differentiation_matrix
    for i in range(n):
        system[i, i] += float(p + 1)

    q_solution = np.linalg.solve(system, q_int)
    out[:] = q_solution * (rho ** (p + 1))
    return out


@njit(cache=True, nogil=True)
def _update_psin_coordinate(
    out_psin: np.ndarray,
    psin_r: np.ndarray,
    integration_matrix: np.ndarray,
    rho: np.ndarray,
    differentiation_matrix: np.ndarray,
) -> np.ndarray:
    corrected_integration(out_psin, psin_r, integration_matrix, 2, rho, differentiation_matrix)
    return _normalize_psin_coordinate_inplace(out_psin)


@njit(cache=True, nogil=True)
def _normalize_psin_coordinate_inplace(psin: np.ndarray) -> np.ndarray:
    offset = psin[0]
    scale = psin[-1] - offset
    if abs(scale) < 1e-12:
        raise ValueError("psin does not span a valid normalized flux interval")

    for i in range(psin.shape[0]):
        psin[i] = (psin[i] - offset) / scale
    psin[0] = 0.0
    psin[-1] = 1.0
    return psin


@njit(cache=True, fastmath=True, nogil=True)
def _enforce_axis_linear_psin_r(psin_r: np.ndarray, rho: np.ndarray) -> np.ndarray:
    if psin_r.shape[0] < 2:
        return psin_r
    if abs(rho[1]) < 1e-14:
        return psin_r
    if psin_r.shape[0] >= 3 and abs(rho[2]) >= 1e-14:
        slope = psin_r[2] / rho[2]
        psin_r[0] = slope * rho[0]
        psin_r[1] = slope * rho[1]
        return psin_r
    psin_r[0] = psin_r[1] * rho[0] / rho[1]
    return psin_r


@njit(cache=True, nogil=True)
def _compute_Pn(Pn_r: np.ndarray, integration_matrix: np.ndarray, weights: np.ndarray) -> np.ndarray:
    Pn = np.empty_like(Pn_r)
    full_integration(Pn, Pn_r, integration_matrix)
    Pn -= quadrature(Pn_r, weights)
    return Pn


@njit(cache=True, fastmath=True, nogil=True)
def _copy_vector(out: np.ndarray, src: np.ndarray) -> np.ndarray:
    for i in range(out.shape[0]):
        out[i] = src[i]
    return out


@njit(cache=True, fastmath=True, nogil=True)
def _fill_scaled_vector(out: np.ndarray, src: np.ndarray, scale: float) -> np.ndarray:
    for i in range(out.shape[0]):
        out[i] = scale * src[i]
    return out


@njit(cache=True, fastmath=True, nogil=True)
def _fill_pointwise_product(out: np.ndarray, lhs: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    for i in range(out.shape[0]):
        out[i] = lhs[i] * rhs[i]
    return out


@njit(cache=True, fastmath=True, nogil=True)
def _fill_scaled_product(out: np.ndarray, lhs: np.ndarray, rhs: np.ndarray, scale: float) -> np.ndarray:
    for i in range(out.shape[0]):
        out[i] = scale * lhs[i] * rhs[i]
    return out


@njit(cache=True, fastmath=True, nogil=True)
def _fill_scaled_ratio(out: np.ndarray, num: np.ndarray, den: np.ndarray, scale: float) -> np.ndarray:
    for i in range(out.shape[0]):
        out[i] = scale * num[i] / den[i]
    return out


@njit(cache=True, fastmath=True, nogil=True)
def _fill_product_ratio(
    out: np.ndarray,
    lhs: np.ndarray,
    rhs: np.ndarray,
    den: np.ndarray,
    scale: float,
) -> np.ndarray:
    for i in range(out.shape[0]):
        out[i] = scale * lhs[i] * rhs[i] / den[i]
    return out


@njit(cache=True, fastmath=True, nogil=True)
def _mul_vector_inplace(out: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    for i in range(out.shape[0]):
        out[i] *= rhs[i]
    return out


@njit(cache=True, fastmath=True, nogil=True)
def _fill_pf_rho_integrand(
    out: np.ndarray,
    Kn: np.ndarray,
    current_input: np.ndarray,
    Ln_r: np.ndarray,
    V_r: np.ndarray,
    heat_input: np.ndarray,
    pressure_scale: float,
) -> np.ndarray:
    pressure_factor = pressure_scale / (4.0 * np.pi**2)
    for i in range(out.shape[0]):
        out[i] = Kn[i] * (current_input[i] * Ln_r[i] + V_r[i] * heat_input[i] * pressure_factor)
    return out


@njit(cache=True, fastmath=True, nogil=True)
def _fill_pf_psin_integrand(
    out: np.ndarray,
    current_input: np.ndarray,
    Ln_r: np.ndarray,
    V_r: np.ndarray,
    heat_input: np.ndarray,
    pressure_scale: float,
) -> np.ndarray:
    pressure_factor = pressure_scale / (4.0 * np.pi**2)
    for i in range(out.shape[0]):
        out[i] = current_input[i] * Ln_r[i] + V_r[i] * heat_input[i] * pressure_factor
    return out


@njit(cache=True, fastmath=True, nogil=True)
def _fill_g1n_psin_integrand(
    out: np.ndarray,
    JdivR: np.ndarray,
    FFn_psin: np.ndarray,
    R: np.ndarray,
    Pn_psin: np.ndarray,
) -> np.ndarray:
    nr, nt = out.shape
    for i in range(nr):
        ffn_i = FFn_psin[i]
        pn_i = Pn_psin[i]
        for j in range(nt):
            out[i, j] = JdivR[i, j] * (ffn_i + R[i, j] * R[i, j] * pn_i)
    return out


@njit(cache=True, fastmath=True, nogil=True)
def _fill_g1n_rho_integrand(
    out: np.ndarray,
    JdivR: np.ndarray,
    FFn_r: np.ndarray,
    R: np.ndarray,
    Pn_r: np.ndarray,
    psin_r_safe: np.ndarray,
) -> np.ndarray:
    nr, nt = out.shape
    for i in range(nr):
        ffn_i = FFn_r[i]
        pn_i = Pn_r[i]
        psin_r_i = psin_r_safe[i]
        for j in range(nt):
            out[i, j] = JdivR[i, j] * (ffn_i + R[i, j] * R[i, j] * pn_i) / psin_r_i
    return out


@njit(cache=True, fastmath=True, nogil=True)
def _fill_pp_ffn_psin(
    out: np.ndarray,
    psin_r: np.ndarray,
    Kn_r: np.ndarray,
    Kn: np.ndarray,
    psin_rr: np.ndarray,
    V_r: np.ndarray,
    Pn_psin: np.ndarray,
    Ln_r: np.ndarray,
    alpha_ratio: float,
) -> np.ndarray:
    pressure_factor = 1.0 / (4.0 * np.pi**2)
    for i in range(out.shape[0]):
        term0 = alpha_ratio * (Kn_r[i] * psin_r[i] + Kn[i] * psin_rr[i])
        term1 = V_r[i] * Pn_psin[i] * pressure_factor
        out[i] = -(term0 + term1) / Ln_r[i]
    return out


@njit(cache=True, fastmath=True, nogil=True)
def _fill_pi_ffn_psin(
    out: np.ndarray,
    Itor_r: np.ndarray,
    V_r: np.ndarray,
    Pn_psin: np.ndarray,
    Ln_r: np.ndarray,
    current_scale: float,
) -> np.ndarray:
    pressure_factor = 1.0 / (4.0 * np.pi**2)
    for i in range(out.shape[0]):
        term0 = current_scale * Itor_r[i]
        term1 = V_r[i] * Pn_psin[i] * pressure_factor
        out[i] = -(term0 + term1) / Ln_r[i]
    return out


@njit(cache=True, fastmath=True, nogil=True)
def _fill_pj_ffn_psin(
    out: np.ndarray,
    jtor: np.ndarray,
    S_r: np.ndarray,
    V_r: np.ndarray,
    Pn_psin: np.ndarray,
    Ln_r: np.ndarray,
    current_scale: float,
) -> np.ndarray:
    pressure_factor = 1.0 / (4.0 * np.pi**2)
    for i in range(out.shape[0]):
        term0 = current_scale * jtor[i] * S_r[i]
        term1 = V_r[i] * Pn_psin[i] * pressure_factor
        out[i] = -(term0 + term1) / Ln_r[i]
    return out


@njit(cache=True, nogil=True)
def _maximum_floor(arr: np.ndarray, floor: float) -> np.ndarray:
    out = np.empty_like(arr)
    for i in range(arr.shape[0]):
        value = arr[i]
        out[i] = value if value > floor else floor
    return out


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

    _barycentric_uniform_interpolate(out_heat_input, heat, psin_query, barycentric_weights)
    _barycentric_uniform_interpolate(out_current_input, current, psin_query, barycentric_weights)
    return out_heat_input, out_current_input


@njit(cache=True, fastmath=True, nogil=True)
def _barycentric_uniform_interpolate(
    out: np.ndarray,
    values: np.ndarray,
    query: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    n_src = values.shape[0]
    stencil_size = weights.shape[0]
    if n_src == 1:
        value = values[0]
        for i in range(out.shape[0]):
            out[i] = value
        return out

    for i in range(out.shape[0]):
        q = query[i]
        start = _local_uniform_stencil_start(q, n_src, stencil_size)
        numerator = 0.0
        denominator = 0.0
        hit = False
        hit_value = 0.0
        for local_j in range(stencil_size):
            j = start + local_j
            node = j / (n_src - 1.0)
            diff = q - node
            if abs(diff) <= 1e-14:
                hit = True
                hit_value = values[j]
                break
            term = weights[local_j] / diff
            numerator += term * values[j]
            denominator += term
        out[i] = hit_value if hit else numerator / denominator
    return out


@njit(cache=True, fastmath=True, nogil=True)
def _uniform_barycentric_weights(n_src: int) -> np.ndarray:
    weights = np.empty(n_src, dtype=np.float64)
    weights[0] = 1.0
    for j in range(1, n_src):
        weights[j] = -weights[j - 1] * (n_src - j) / j
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
        for j in range(n_src):
            matrix[i, j] = 0.0
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


@njit(cache=True, fastmath=True, nogil=True)
def _local_uniform_stencil_start(q: float, n_src: int, stencil_size: int) -> int:
    if stencil_size >= n_src:
        return 0
    pos = q * (n_src - 1.0)
    center = int(pos)
    if pos > center:
        center += 1
    start = center - stencil_size // 2
    if start < 0:
        return 0
    max_start = n_src - stencil_size
    if start > max_start:
        return max_start
    return start


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


_register_default_source_routes()

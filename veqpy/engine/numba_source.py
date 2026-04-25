"""
Module: engine.numba_source

Role:
- 负责注册 source operators.
- 负责校验 operator/coordinate/nodes 组合并执行 source kernels.

Public API:
- register_operator
- validate_route
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

DEFAULT_LOCAL_BARYCENTRIC_STENCIL = 8

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


def validate_route(route: str, coordinate: str, nodes: str = UNIFORM_NODES) -> _SourceRouteSpec:
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


@njit(cache=True, nogil=True)
def _source_output_root_views(out_root_fields: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return out_root_fields[0], out_root_fields[1], out_root_fields[2]


@njit(cache=True, nogil=True)
def _source_geometry_workspace_views(
    radial_workspace: np.ndarray,
    surface_workspace: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    return (
        radial_workspace[1],
        radial_workspace[2],
        radial_workspace[3],
        radial_workspace[4],
        radial_workspace[0],
        surface_workspace[1],
        surface_workspace[5],
    )


@register_operator("PF_RHO", supported_coordinates=(RHO_COORDINATE,))
@njit(cache=True, nogil=True)
def update_PF_rho(
    out_root_fields: np.ndarray,
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
    radial_workspace: np.ndarray,
    surface_workspace: np.ndarray,
    F: np.ndarray,
    Ip: float,
    beta: float,
) -> tuple[float, float]:
    out_psin, out_psin_r, out_psin_rr = _source_output_root_views(out_root_fields)
    V_r, Kn, _, Ln_r, _, R, JdivR = _source_geometry_workspace_views(radial_workspace, surface_workspace)
    return _update_pf_from_rho_inputs(
        out_psin,
        out_psin_r,
        out_psin_rr,
        out_FFn_psin,
        out_Pn_psin,
        heat_input,
        current_input,
        B0,
        weights,
        differentiation_matrix,
        integration_matrix,
        rho,
        V_r,
        Kn,
        Ln_r,
        R,
        JdivR,
        Ip,
        beta,
    )


@register_operator("PF_PSIN", supported_coordinates=(PSIN_COORDINATE,))
@njit(cache=True, nogil=True)
def update_PF_psin(
    out_root_fields: np.ndarray,
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
    radial_workspace: np.ndarray,
    surface_workspace: np.ndarray,
    F: np.ndarray,
    Ip: float,
    beta: float,
) -> tuple[float, float]:
    out_psin, out_psin_r, out_psin_rr = _source_output_root_views(out_root_fields)
    V_r, Kn, _, Ln_r, _, R, JdivR = _source_geometry_workspace_views(radial_workspace, surface_workspace)
    return _update_pf_from_psin_inputs(
        out_psin,
        out_psin_r,
        out_psin_rr,
        out_FFn_psin,
        out_Pn_psin,
        heat_input,
        current_input,
        B0,
        weights,
        differentiation_matrix,
        integration_matrix,
        rho,
        V_r,
        Kn,
        Ln_r,
        R,
        JdivR,
        Ip,
        beta,
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
    B0: float,
    weights: np.ndarray,
    differentiation_matrix: np.ndarray,
    integration_matrix: np.ndarray,
    rho: np.ndarray,
    V_r: np.ndarray,
    Kn: np.ndarray,
    Ln_r: np.ndarray,
    R: np.ndarray,
    JdivR: np.ndarray,
    Ip: float,
    beta: float,
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
    _fill_pf_rho_integrand(integrand, Kn, current_input, Ln_r, V_r, heat_input)
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
    corrected_linear_derivative(out_psin_rr, out_psin_r, differentiation_matrix, rho=rho)
    _update_psin_coordinate(out_psin, out_psin_r, integration_matrix, rho, differentiation_matrix)

    psin_r_safe = _maximum_floor(out_psin_r, 1e-10)
    if (not has_Ip) and (not has_beta):
        alpha2 = integral_prof
        alpha1 = -quadrature(heat_input, weights) / alpha2
        _fill_scaled_ratio(out_Pn_psin, heat_input, psin_r_safe, 1.0 / (alpha1 * alpha2))
        _fill_scaled_ratio(out_FFn_psin, current_input, psin_r_safe, 1.0 / (alpha1 * alpha2))
        return alpha1, alpha2

    c2 = integral_prof * integral_prof
    if has_Ip and (not has_beta):
        g1n_integrand = np.empty_like(R)
        _fill_g1n_rho_integrand(g1n_integrand, JdivR, current_input, R, heat_input, psin_r_safe)
        G1n_integral = quadrature(g1n_integrand, weights)
        alpha1 = -Ip / G1n_integral
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
    B0: float,
    weights: np.ndarray,
    differentiation_matrix: np.ndarray,
    integration_matrix: np.ndarray,
    rho: np.ndarray,
    V_r: np.ndarray,
    Kn: np.ndarray,
    Ln_r: np.ndarray,
    R: np.ndarray,
    JdivR: np.ndarray,
    Ip: float,
    beta: float,
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
    _fill_pf_psin_integrand(integrand, current_input, Ln_r, V_r, heat_input)
    corrected_integration(out_psin_r, integrand, integration_matrix, 1, rho, differentiation_matrix)
    out_psin_r *= -1.0
    out_psin_r /= Kn

    prof = out_psin_r
    integral_prof = quadrature(prof, weights)
    out_psin_r /= integral_prof
    corrected_linear_derivative(out_psin_rr, out_psin_r, differentiation_matrix, rho=rho)
    _update_psin_coordinate(out_psin, out_psin_r, integration_matrix, rho, differentiation_matrix)

    if (not has_Ip) and (not has_beta):
        alpha2 = integral_prof
        pressure_profile = np.empty_like(out_psin_r)
        _fill_pointwise_product(pressure_profile, heat_input, prof)
        alpha1 = -quadrature(pressure_profile, weights)
        _fill_scaled_vector(out_Pn_psin, heat_input, 1.0 / alpha1)
        _fill_scaled_vector(out_FFn_psin, current_input, 1.0 / alpha1)
        return alpha1, alpha2

    c2 = integral_prof
    _copy_vector(out_Pn_psin, heat_input)
    _copy_vector(out_FFn_psin, current_input)

    if has_Ip and (not has_beta):
        g1n_integrand = np.empty_like(R)
        _fill_g1n_psin_integrand(g1n_integrand, JdivR, out_FFn_psin, R, out_Pn_psin)
        G1n_integral = quadrature(g1n_integrand, weights)
        alpha1 = -Ip / G1n_integral
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
        alpha2 = Ip / (2.0 * np.pi * Kn[-1] * out_psin_r[-1])
    else:
        alpha2 = quadrature(current_input, weights)
        _fill_scaled_vector(out_psin_r, current_input, 1.0 / alpha2)

    _enforce_axis_linear_psin_r(out_psin_r, rho)
    corrected_linear_derivative(out_psin_rr, out_psin_r, differentiation_matrix, rho=rho)
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
        alpha1 = -quadrature(P_r, weights) / alpha2
        _fill_scaled_ratio(out_Pn_psin, P_r, psin_r_safe, 1.0 / (alpha1 * alpha2))

    _fill_pp_ffn_psin(
        out_FFn_psin,
        out_psin_r,
        psin_r_safe,
        Kn_r,
        Kn,
        out_psin_rr,
        V_r,
        out_Pn_psin,
        Ln_r,
        alpha2 / alpha1,
    )
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
        alpha2 = Ip / (2.0 * np.pi * Kn[-1] * out_psin_r[-1])
    else:
        alpha2 = quadrature(current_input, weights)
        _fill_scaled_vector(out_psin_r, current_input, 1.0 / alpha2)

    _enforce_axis_linear_psin_r(out_psin_r, rho)
    corrected_linear_derivative(out_psin_rr, out_psin_r, differentiation_matrix, rho=rho)
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
        alpha1 = -quadrature(P_r, weights) / alpha2
        _fill_scaled_ratio(out_Pn_psin, P_r, psin_r_safe, 1.0 / (alpha1 * alpha2))

    _fill_pp_ffn_psin(
        out_FFn_psin,
        out_psin_r,
        psin_r_safe,
        Kn_r,
        Kn,
        out_psin_rr,
        V_r,
        out_Pn_psin,
        Ln_r,
        alpha2 / alpha1,
    )
    return alpha1, alpha2


@register_operator("PP_RHO", supported_coordinates=(RHO_COORDINATE,))
@njit(cache=True, nogil=True)
def update_PP_RHO(
    out_root_fields: np.ndarray,
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
    radial_workspace: np.ndarray,
    surface_workspace: np.ndarray,
    F: np.ndarray,
    Ip: float,
    beta: float,
) -> tuple[float, float]:
    out_psin, out_psin_r, out_psin_rr = _source_output_root_views(out_root_fields)
    V_r, Kn, Kn_r, Ln_r, S_r, R, JdivR = _source_geometry_workspace_views(radial_workspace, surface_workspace)
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
    out_root_fields: np.ndarray,
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
    radial_workspace: np.ndarray,
    surface_workspace: np.ndarray,
    F: np.ndarray,
    Ip: float,
    beta: float,
) -> tuple[float, float]:
    out_psin, out_psin_r, out_psin_rr = _source_output_root_views(out_root_fields)
    V_r, Kn, Kn_r, Ln_r, S_r, R, JdivR = _source_geometry_workspace_views(radial_workspace, surface_workspace)
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
    _enforce_axis_quadratic_itor(Itor, rho)
    itor_floor = max(Itor[-1], 1.0) * 1e-12
    Itor[:] = _maximum_floor(Itor, itor_floor)

    itor_over_kn = np.empty_like(current_input)
    _fill_scaled_ratio(itor_over_kn, Itor, Kn, 1.0 / (2.0 * np.pi))
    alpha2 = quadrature(itor_over_kn, weights)

    _fill_scaled_ratio(out_psin_r, Itor, Kn, 1.0 / (2.0 * np.pi * alpha2))
    _enforce_axis_linear_psin_r(out_psin_r, rho)
    corrected_linear_derivative(out_psin_rr, out_psin_r, differentiation_matrix, rho=rho)
    _update_psin_coordinate(out_psin, out_psin_r, integration_matrix, rho, differentiation_matrix)
    psin_r_safe = _maximum_floor(out_psin_r, 1e-10)
    Itor_r = np.empty_like(Itor)
    corrected_even_derivative(Itor_r, Itor, differentiation_matrix, rho=rho)

    if has_beta:
        _fill_scaled_ratio(out_Pn_psin, heat_input, psin_r_safe, 1.0)
        Pn_r = np.empty_like(out_psin_r)
        _fill_pointwise_product(Pn_r, out_Pn_psin, out_psin_r)
        Pn = _compute_Pn(Pn_r, integration_matrix, weights)
        alpha1 = 0.5 * beta * B0**2 / alpha2 * quadrature(V_r, weights) / quadrature(Pn * V_r, weights)
    else:
        P_r = np.empty_like(out_psin_r)
        _copy_vector(P_r, heat_input)
        alpha1 = -quadrature(P_r, weights) / alpha2
        _fill_scaled_ratio(out_Pn_psin, P_r, psin_r_safe, 1.0 / (alpha1 * alpha2))

    _fill_pi_ffn_psin(out_FFn_psin, Itor_r, V_r, out_Pn_psin, Ln_r, 1.0 / (2.0 * np.pi * alpha1))
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
    _enforce_axis_quadratic_itor(Itor, rho)
    itor_floor = max(Itor[-1], 1.0) * 1e-12
    Itor[:] = _maximum_floor(Itor, itor_floor)

    itor_over_kn = np.empty_like(current_input)
    _fill_scaled_ratio(itor_over_kn, Itor, Kn, 1.0 / (2.0 * np.pi))
    alpha2 = quadrature(itor_over_kn, weights)

    _fill_scaled_ratio(out_psin_r, Itor, Kn, 1.0 / (2.0 * np.pi * alpha2))
    corrected_linear_derivative(out_psin_rr, out_psin_r, differentiation_matrix, rho=rho)
    _update_psin_coordinate(out_psin, out_psin_r, integration_matrix, rho, differentiation_matrix)
    psin_r_safe = _maximum_floor(out_psin_r, 1e-10)
    Itor_r = np.empty_like(Itor)
    corrected_even_derivative(Itor_r, Itor, differentiation_matrix, rho=rho)

    if has_beta:
        _copy_vector(out_Pn_psin, heat_input)
        Pn_r = np.empty_like(out_psin_r)
        _fill_pointwise_product(Pn_r, out_Pn_psin, out_psin_r)
        Pn = _compute_Pn(Pn_r, integration_matrix, weights)
        alpha1 = 0.5 * beta * B0**2 / alpha2 * quadrature(V_r, weights) / quadrature(Pn * V_r, weights)
    else:
        P_r = np.empty_like(out_psin_r)
        _fill_scaled_product(P_r, heat_input, out_psin_r, alpha2)
        alpha1 = -quadrature(P_r, weights) / alpha2
        _fill_scaled_ratio(out_Pn_psin, P_r, psin_r_safe, 1.0 / (alpha1 * alpha2))

    _fill_pi_ffn_psin(out_FFn_psin, Itor_r, V_r, out_Pn_psin, Ln_r, 1.0 / (2.0 * np.pi * alpha1))
    return alpha1, alpha2


@register_operator("PI_RHO", supported_coordinates=(RHO_COORDINATE,))
@njit(cache=True, nogil=True)
def update_PI_RHO(
    out_root_fields: np.ndarray,
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
    radial_workspace: np.ndarray,
    surface_workspace: np.ndarray,
    F: np.ndarray,
    Ip: float,
    beta: float,
) -> tuple[float, float]:
    out_psin, out_psin_r, out_psin_rr = _source_output_root_views(out_root_fields)
    V_r, Kn, Kn_r, Ln_r, S_r, R, JdivR = _source_geometry_workspace_views(radial_workspace, surface_workspace)
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
    out_root_fields: np.ndarray,
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
    radial_workspace: np.ndarray,
    surface_workspace: np.ndarray,
    F: np.ndarray,
    Ip: float,
    beta: float,
) -> tuple[float, float]:
    out_psin, out_psin_r, out_psin_rr = _source_output_root_views(out_root_fields)
    V_r, Kn, Kn_r, Ln_r, S_r, R, JdivR = _source_geometry_workspace_views(radial_workspace, surface_workspace)
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
    corrected_integration(out_psin_r, integrand_j, integration_matrix, 1, rho, differentiation_matrix)
    I_tor_prof = np.empty_like(out_psin_r)
    _copy_vector(I_tor_prof, out_psin_r)
    _enforce_axis_quadratic_itor(I_tor_prof, rho)
    I_tor = np.empty_like(current_input)
    jtor = np.empty_like(current_input)

    if has_Ip:
        _fill_scaled_vector(I_tor, I_tor_prof, Ip / I_tor_prof[-1])
        _fill_scaled_vector(jtor, current_input, Ip / I_tor_prof[-1])
    else:
        _copy_vector(I_tor, I_tor_prof)
        _copy_vector(jtor, current_input)
    _enforce_axis_even_profile(jtor, rho)

    itor_floor = max(I_tor[-1], 1.0) * 1e-12
    I_tor[:] = _maximum_floor(I_tor, itor_floor)

    itor_over_kn = np.empty_like(current_input)
    _fill_scaled_ratio(itor_over_kn, I_tor, Kn, 1.0 / (2.0 * np.pi))
    alpha2 = quadrature(itor_over_kn, weights)
    _fill_scaled_ratio(out_psin_r, I_tor, Kn, 1.0 / (2.0 * np.pi * alpha2))
    corrected_linear_derivative(out_psin_rr, out_psin_r, differentiation_matrix, rho=rho)
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
        alpha1 = -quadrature(P_r, weights) / alpha2
        _fill_scaled_ratio(out_Pn_psin, P_r, psin_r_safe, 1.0 / (alpha1 * alpha2))

    _fill_pj_ffn_psin(
        out_FFn_psin,
        jtor,
        S_r,
        V_r,
        out_Pn_psin,
        out_psin_r,
        psin_r_safe,
        Ln_r,
        1.0 / (2.0 * np.pi * alpha1),
    )
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
    corrected_integration(out_psin_r, integrand_j, integration_matrix, 1, rho, differentiation_matrix)
    I_tor_prof = np.empty_like(out_psin_r)
    _copy_vector(I_tor_prof, out_psin_r)
    _enforce_axis_quadratic_itor(I_tor_prof, rho)
    I_tor = np.empty_like(current_input)
    jtor = np.empty_like(current_input)

    if has_Ip:
        _fill_scaled_vector(I_tor, I_tor_prof, Ip / I_tor_prof[-1])
        _fill_scaled_vector(jtor, current_input, Ip / I_tor_prof[-1])
    else:
        _copy_vector(I_tor, I_tor_prof)
        _copy_vector(jtor, current_input)
    _enforce_axis_even_profile(jtor, rho)

    itor_floor = max(I_tor[-1], 1.0) * 1e-12
    I_tor[:] = _maximum_floor(I_tor, itor_floor)

    itor_over_kn = np.empty_like(current_input)
    _fill_scaled_ratio(itor_over_kn, I_tor, Kn, 1.0 / (2.0 * np.pi))
    alpha2 = quadrature(itor_over_kn, weights)
    _fill_scaled_ratio(out_psin_r, I_tor, Kn, 1.0 / (2.0 * np.pi * alpha2))
    corrected_linear_derivative(out_psin_rr, out_psin_r, differentiation_matrix, rho=rho)
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
        alpha1 = -quadrature(P_r, weights) / alpha2
        _fill_scaled_ratio(out_Pn_psin, P_r, psin_r_safe, 1.0 / (alpha1 * alpha2))

    _fill_pj_ffn_psin(
        out_FFn_psin,
        jtor,
        S_r,
        V_r,
        out_Pn_psin,
        out_psin_r,
        psin_r_safe,
        Ln_r,
        1.0 / (2.0 * np.pi * alpha1),
    )
    return alpha1, alpha2


@register_operator("PJ1_RHO", supported_coordinates=(RHO_COORDINATE,))
@njit(cache=True, nogil=True)
def update_PJ1_RHO(
    out_root_fields: np.ndarray,
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
    radial_workspace: np.ndarray,
    surface_workspace: np.ndarray,
    F: np.ndarray,
    Ip: float,
    beta: float,
) -> tuple[float, float]:
    out_psin, out_psin_r, out_psin_rr = _source_output_root_views(out_root_fields)
    V_r, Kn, Kn_r, Ln_r, S_r, R, JdivR = _source_geometry_workspace_views(radial_workspace, surface_workspace)
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
    out_root_fields: np.ndarray,
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
    radial_workspace: np.ndarray,
    surface_workspace: np.ndarray,
    F: np.ndarray,
    Ip: float,
    beta: float,
) -> tuple[float, float]:
    out_psin, out_psin_r, out_psin_rr = _source_output_root_views(out_root_fields)
    V_r, Kn, Kn_r, Ln_r, S_r, R, JdivR = _source_geometry_workspace_views(radial_workspace, surface_workspace)
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
        _fill_scaled_product(I_tor, F, integral_val, Ip / (F[-1] * integral_val[-1]))
    else:
        _fill_scaled_product(I_tor, F, integral_val, 2.0 * np.pi)
    itor_over_kn = np.empty_like(current_input)
    _fill_scaled_ratio(itor_over_kn, I_tor, Kn, 1.0 / (2.0 * np.pi))
    alpha2 = quadrature(itor_over_kn, weights)
    _fill_scaled_ratio(out_psin_r, I_tor, Kn, 1.0 / (2.0 * np.pi * alpha2))
    corrected_linear_derivative(out_psin_rr, out_psin_r, differentiation_matrix, rho=rho)
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
        alpha1 = -quadrature(P_r, weights) / alpha2
        _fill_scaled_ratio(out_Pn_psin, P_r, psin_r_safe, 1.0 / (alpha1 * alpha2))

    F_r = np.empty_like(F)
    corrected_even_derivative(F_r, F, differentiation_matrix, rho=rho)
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
        _fill_scaled_product(I_tor, F, integral_val, Ip / (F[-1] * integral_val[-1]))
    else:
        _fill_scaled_product(I_tor, F, integral_val, 2.0 * np.pi)
    itor_over_kn = np.empty_like(current_input)
    _fill_scaled_ratio(itor_over_kn, I_tor, Kn, 1.0 / (2.0 * np.pi))
    alpha2 = quadrature(itor_over_kn, weights)
    _fill_scaled_ratio(out_psin_r, I_tor, Kn, 1.0 / (2.0 * np.pi * alpha2))
    corrected_linear_derivative(out_psin_rr, out_psin_r, differentiation_matrix, rho=rho)
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
        alpha1 = -quadrature(P_r, weights) / alpha2
        _fill_scaled_ratio(out_Pn_psin, P_r, psin_r_safe, 1.0 / (alpha1 * alpha2))

    F_r = np.empty_like(F)
    corrected_even_derivative(F_r, F, differentiation_matrix, rho=rho)
    _fill_scaled_product(out_FFn_psin, F, F_r, 1.0 / (alpha1 * alpha2))
    _fill_scaled_ratio(out_FFn_psin, out_FFn_psin, psin_r_safe, 1.0)
    return alpha1, alpha2


@njit(cache=True, nogil=True)
def _update_pj2_from_psin_inputs_with_scratch(
    out_root_fields: np.ndarray,
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
    radial_workspace: np.ndarray,
    surface_workspace: np.ndarray,
    F: np.ndarray,
    Ip: float,
    beta: float,
    source_scratch_1d: np.ndarray,
) -> tuple[float, float]:
    out_psin, out_psin_r, out_psin_rr = _source_output_root_views(out_root_fields)
    V_r, Kn, Kn_r, Ln_r, S_r, R, JdivR = _source_geometry_workspace_views(radial_workspace, surface_workspace)
    has_Ip = not np.isnan(Ip)
    has_beta = not np.isnan(beta)
    integrand = source_scratch_1d[0]
    integral_val = source_scratch_1d[1]
    I_tor = source_scratch_1d[2]
    scratch_Pn_r = source_scratch_1d[3]
    psin_r_safe = source_scratch_1d[4]
    scratch_aux = source_scratch_1d[5]

    _fill_product_ratio(integrand, Ln_r, current_input, F, 1.0)
    corrected_integration(out_psin_r, integrand, integration_matrix, 1, rho, differentiation_matrix)
    _copy_vector(integral_val, out_psin_r)

    if has_Ip:
        _fill_scaled_product(I_tor, F, integral_val, Ip / (R0 * B0 * integral_val[-1]))
    else:
        _fill_scaled_product(I_tor, F, integral_val, 2.0 * np.pi)
    _fill_scaled_ratio(integrand, I_tor, Kn, 1.0 / (2.0 * np.pi))
    alpha2 = quadrature(integrand, weights)
    _fill_scaled_vector(out_psin_r, integrand, 1.0 / alpha2)
    full_differentiation(out_psin_rr, out_psin_r, differentiation_matrix)
    _update_psin_coordinate(out_psin, out_psin_r, integration_matrix, rho, differentiation_matrix)
    _maximum_floor_out(psin_r_safe, out_psin_r, 1e-10)

    if has_beta:
        _fill_pointwise_product(scratch_Pn_r, heat_input, out_psin_r)
        _compute_Pn_out(scratch_aux, scratch_Pn_r, integration_matrix, weights)
        alpha1 = (
            0.5
            * beta
            * B0**2
            / alpha2
            * quadrature(V_r, weights)
            / _quadrature_product(
                scratch_aux,
                V_r,
                weights,
            )
        )
        _copy_vector(out_Pn_psin, heat_input)
    else:
        alpha1 = -_quadrature_product(heat_input, out_psin_r, weights)
        _fill_product_ratio(out_Pn_psin, heat_input, out_psin_r, psin_r_safe, 1.0 / alpha1)

    full_differentiation(scratch_aux, F, differentiation_matrix)
    _fill_pointwise_product(out_FFn_psin, F, scratch_aux)
    _fill_scaled_ratio(out_FFn_psin, out_FFn_psin, psin_r_safe, 1.0 / (alpha1 * alpha2))
    return alpha1, alpha2


@register_operator("PJ2_RHO", supported_coordinates=(RHO_COORDINATE,))
@njit(cache=True, nogil=True)
def update_PJ2_RHO(
    out_root_fields: np.ndarray,
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
    radial_workspace: np.ndarray,
    surface_workspace: np.ndarray,
    F: np.ndarray,
    Ip: float,
    beta: float,
) -> tuple[float, float]:
    out_psin, out_psin_r, out_psin_rr = _source_output_root_views(out_root_fields)
    V_r, Kn, Kn_r, Ln_r, S_r, R, JdivR = _source_geometry_workspace_views(radial_workspace, surface_workspace)
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
    out_root_fields: np.ndarray,
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
    radial_workspace: np.ndarray,
    surface_workspace: np.ndarray,
    F: np.ndarray,
    Ip: float,
    beta: float,
) -> tuple[float, float]:
    out_psin, out_psin_r, out_psin_rr = _source_output_root_views(out_root_fields)
    V_r, Kn, Kn_r, Ln_r, S_r, R, JdivR = _source_geometry_workspace_views(radial_workspace, surface_workspace)
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
        q_scale = (2.0 * np.pi * F[-1]) / Ip
        _fill_scaled_vector(q_prof, current_input, q_scale * (Kn[-1] * Ln_r[-1] / current_input[-1]))
    else:
        _copy_vector(q_prof, current_input)

    integrand_alpha2 = np.empty_like(out_psin_r)
    _fill_product_ratio(integrand_alpha2, F, Ln_r, q_prof, 1.0)
    alpha2 = quadrature(integrand_alpha2, weights)

    _fill_product_ratio(out_psin_r, F, Ln_r, q_prof, 1.0 / alpha2)
    corrected_linear_derivative(out_psin_rr, out_psin_r, differentiation_matrix, rho=rho)
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
        alpha1 = -quadrature(P_r, weights) / alpha2
        _fill_scaled_ratio(out_Pn_psin, P_r, psin_r_safe, 1.0 / (alpha1 * alpha2))

    F_r = np.empty_like(F)
    corrected_even_derivative(F_r, F, differentiation_matrix, rho=rho)
    _fill_pointwise_product(out_FFn_psin, F, F_r)
    _fill_scaled_ratio(out_FFn_psin, out_FFn_psin, psin_r_safe, 1.0 / (alpha1 * alpha2))
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
        q_scale = (2.0 * np.pi * F[-1]) / Ip
        _fill_scaled_vector(q_prof, current_input, q_scale * (Kn[-1] * Ln_r[-1] / current_input[-1]))
    else:
        _copy_vector(q_prof, current_input)

    integrand_alpha2 = np.empty_like(out_psin_r)
    _fill_product_ratio(integrand_alpha2, F, Ln_r, q_prof, 1.0)
    alpha2 = quadrature(integrand_alpha2, weights)

    _fill_product_ratio(out_psin_r, F, Ln_r, q_prof, 1.0 / alpha2)
    corrected_linear_derivative(out_psin_rr, out_psin_r, differentiation_matrix, rho=rho)
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
        alpha1 = -quadrature(P_r, weights) / alpha2
        _fill_scaled_ratio(out_Pn_psin, P_r, psin_r_safe, 1.0 / (alpha1 * alpha2))

    F_r = np.empty_like(F)
    corrected_even_derivative(F_r, F, differentiation_matrix, rho=rho)
    _fill_pointwise_product(out_FFn_psin, F, F_r)
    _fill_scaled_ratio(out_FFn_psin, out_FFn_psin, psin_r_safe, 1.0 / (alpha1 * alpha2))
    return alpha1, alpha2


@njit(cache=True, nogil=True)
def _update_pq_from_psin_inputs_with_scratch(
    out_root_fields: np.ndarray,
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
    radial_workspace: np.ndarray,
    surface_workspace: np.ndarray,
    F: np.ndarray,
    Ip: float,
    beta: float,
    source_scratch_1d: np.ndarray,
) -> tuple[float, float]:
    out_psin, out_psin_r, out_psin_rr = _source_output_root_views(out_root_fields)
    V_r, Kn, Kn_r, Ln_r, S_r, R, JdivR = _source_geometry_workspace_views(radial_workspace, surface_workspace)
    has_Ip = not np.isnan(Ip)
    has_beta = not np.isnan(beta)
    q_prof = source_scratch_1d[0]
    scratch_Pn_r = source_scratch_1d[3]
    psin_r_safe = source_scratch_1d[4]
    scratch_aux = source_scratch_1d[5]

    if has_Ip:
        q_scale = (2.0 * np.pi * F[-1]) / Ip
        _fill_scaled_vector(q_prof, current_input, q_scale * (Kn[-1] * Ln_r[-1] / current_input[-1]))
        alpha2 = _quadrature_product_ratio(F, Ln_r, q_prof, weights)
        _fill_product_ratio(out_psin_r, F, Ln_r, q_prof, 1.0 / alpha2)
    else:
        alpha2 = _quadrature_product_ratio(F, Ln_r, current_input, weights)
        _fill_product_ratio(out_psin_r, F, Ln_r, current_input, 1.0 / alpha2)

    corrected_linear_derivative(out_psin_rr, out_psin_r, differentiation_matrix, rho=rho)
    _update_psin_coordinate(out_psin, out_psin_r, integration_matrix, rho, differentiation_matrix)
    _maximum_floor_out(psin_r_safe, out_psin_r, 1e-10)

    if has_beta:
        _fill_pointwise_product(scratch_Pn_r, heat_input, out_psin_r)
        _compute_Pn_out(scratch_aux, scratch_Pn_r, integration_matrix, weights)
        alpha1 = (
            0.5
            * beta
            * B0**2
            / alpha2
            * quadrature(V_r, weights)
            / _quadrature_product(
                scratch_aux,
                V_r,
                weights,
            )
        )
        _copy_vector(out_Pn_psin, heat_input)
    else:
        alpha1 = -_quadrature_product(heat_input, out_psin_r, weights)
        _fill_product_ratio(out_Pn_psin, heat_input, out_psin_r, psin_r_safe, 1.0 / alpha1)

    corrected_even_derivative(scratch_aux, F, differentiation_matrix, rho=rho)
    _fill_pointwise_product(out_FFn_psin, F, scratch_aux)
    _fill_scaled_ratio(out_FFn_psin, out_FFn_psin, psin_r_safe, 1.0 / (alpha1 * alpha2))
    return alpha1, alpha2


@register_operator("PQ_RHO", supported_coordinates=(RHO_COORDINATE,))
@njit(cache=True, nogil=True)
def update_PQ_RHO(
    out_root_fields: np.ndarray,
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
    radial_workspace: np.ndarray,
    surface_workspace: np.ndarray,
    F: np.ndarray,
    Ip: float,
    beta: float,
) -> tuple[float, float]:
    out_psin, out_psin_r, out_psin_rr = _source_output_root_views(out_root_fields)
    V_r, Kn, Kn_r, Ln_r, S_r, R, JdivR = _source_geometry_workspace_views(radial_workspace, surface_workspace)
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
    out_root_fields: np.ndarray,
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
    radial_workspace: np.ndarray,
    surface_workspace: np.ndarray,
    F: np.ndarray,
    Ip: float,
    beta: float,
) -> tuple[float, float]:
    out_psin, out_psin_r, out_psin_rr = _source_output_root_views(out_root_fields)
    V_r, Kn, Kn_r, Ln_r, S_r, R, JdivR = _source_geometry_workspace_views(radial_workspace, surface_workspace)
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
        psin_uniform_strategy=SOURCE_STRATEGY_SINGLE_PASS,
    )
    _register_standard_routes(
        "PQ",
        rho_implementation="PQ_RHO",
        psin_implementation="PQ_PSIN",
        psin_uniform_strategy=SOURCE_STRATEGY_SINGLE_PASS,
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
def _quadrature_product(lhs: np.ndarray, rhs: np.ndarray, weights: np.ndarray) -> float:
    total = 0.0
    for i in range(lhs.shape[0]):
        total += weights[i] * lhs[i] * rhs[i]
    return total


@njit(cache=True, nogil=True)
def _quadrature_product_ratio(lhs: np.ndarray, rhs: np.ndarray, den: np.ndarray, weights: np.ndarray) -> float:
    total = 0.0
    for i in range(lhs.shape[0]):
        total += weights[i] * lhs[i] * rhs[i] / den[i]
    return total


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
def corrected_linear_derivative(
    out: np.ndarray,
    arr: np.ndarray,
    differentiation_matrix: np.ndarray,
    rho: np.ndarray,
) -> np.ndarray:
    """对轴心奇函数/线性起始量执行修正微分."""
    n = arr.shape[0]
    if n == 0:
        return out
    if n == 1:
        out[0] = 0.0
        return out

    reduced = np.empty_like(arr)
    for i in range(n):
        if rho[i] > 1e-10:
            reduced[i] = arr[i] / rho[i]
        else:
            reduced[i] = 0.0
    reduced[0] = reduced[1]
    _enforce_axis_even_profile(reduced, rho)

    reduced_r = np.empty_like(arr)
    full_differentiation(reduced_r, reduced, differentiation_matrix)
    _enforce_axis_linear_psin_r(reduced_r, rho)

    for i in range(n):
        out[i] = reduced[i] + rho[i] * reduced_r[i]
    out[0] = reduced[0]
    return out


@njit(cache=True, nogil=True)
def corrected_even_derivative(
    out: np.ndarray,
    arr: np.ndarray,
    differentiation_matrix: np.ndarray,
    rho: np.ndarray,
) -> np.ndarray:
    """对轴心偶函数量执行修正微分."""
    n = arr.shape[0]
    if n == 0:
        return out
    if n == 1:
        out[0] = 0.0
        return out

    smooth = np.empty_like(arr)
    for i in range(n):
        smooth[i] = arr[i]
    _enforce_axis_even_profile(smooth, rho)
    base = smooth[0]

    reduced = np.empty_like(arr)
    for i in range(n):
        rho2 = rho[i] * rho[i]
        if rho2 > 1e-10:
            reduced[i] = (smooth[i] - base) / rho2
        else:
            reduced[i] = 0.0
    reduced[0] = reduced[1]
    _enforce_axis_even_profile(reduced, rho)

    reduced_r = np.empty_like(arr)
    full_differentiation(reduced_r, reduced, differentiation_matrix)
    _enforce_axis_linear_psin_r(reduced_r, rho)

    for i in range(n):
        rho2 = rho[i] * rho[i]
        out[i] = 2.0 * rho[i] * reduced[i] + rho2 * reduced_r[i]
    out[0] = 0.0
    _enforce_axis_linear_psin_r(out, rho)
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


@njit(cache=True, fastmath=True, nogil=True)
def _enforce_axis_quadratic_itor(itor: np.ndarray, rho: np.ndarray) -> np.ndarray:
    if itor.shape[0] < 2:
        return itor
    if abs(rho[1]) < 1e-14:
        return itor
    if itor.shape[0] >= 3 and abs(rho[2]) >= 1e-14:
        scale = itor[2] / (rho[2] * rho[2])
        itor[0] = scale * rho[0] * rho[0]
        itor[1] = scale * rho[1] * rho[1]
        return itor
    scale = itor[1] / (rho[1] * rho[1])
    itor[0] = scale * rho[0] * rho[0]
    return itor


@njit(cache=True, fastmath=True, nogil=True)
def _enforce_axis_even_profile(profile: np.ndarray, rho: np.ndarray) -> np.ndarray:
    if profile.shape[0] < 3:
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


@njit(cache=True, fastmath=True, nogil=True)
def _smooth_even_profile_on_rho2(profile: np.ndarray, rho: np.ndarray, degree: int = 5) -> np.ndarray:
    n = profile.shape[0]
    fit_degree = degree
    if fit_degree > n - 1:
        fit_degree = n - 1
    if fit_degree <= 0:
        return profile

    x = np.empty(n, dtype=np.float64)
    for i in range(n):
        x[i] = rho[i] * rho[i]

    vandermonde = np.empty((n, fit_degree + 1), dtype=np.float64)
    for i in range(n):
        vandermonde[i, 0] = 1.0
    for order in range(1, fit_degree + 1):
        for i in range(n):
            vandermonde[i, order] = vandermonde[i, order - 1] * x[i]

    gram = np.empty((fit_degree + 1, fit_degree + 1), dtype=np.float64)
    rhs = np.empty(fit_degree + 1, dtype=np.float64)
    for row in range(fit_degree + 1):
        total_rhs = 0.0
        for i in range(n):
            total_rhs += vandermonde[i, row] * profile[i]
        rhs[row] = total_rhs
        for col in range(fit_degree + 1):
            total = 0.0
            for i in range(n):
                total += vandermonde[i, row] * vandermonde[i, col]
            gram[row, col] = total

    coeff = np.linalg.solve(gram, rhs)
    for i in range(n):
        total = 0.0
        for order in range(fit_degree + 1):
            total += vandermonde[i, order] * coeff[order]
        profile[i] = total
    return profile


@njit(cache=True, fastmath=True, nogil=True)
def _stabilize_odd_profile_head_on_rho(
    profile: np.ndarray,
    rho: np.ndarray,
    fit_start: int = 6,
    fit_count: int = 12,
    replace_count: int = 12,
    degree: int = 2,
) -> np.ndarray:
    n = profile.shape[0]
    start = fit_start if fit_start > 1 else 1
    stop = start + fit_count
    if stop > n:
        stop = n
    replace_stop = replace_count
    if replace_stop > stop:
        replace_stop = stop
    fit_degree = degree
    available = stop - start - 1
    if fit_degree > available:
        fit_degree = available
    if replace_stop <= 0 or stop - start < 2 or fit_degree <= 0:
        return profile

    m = stop - start
    x_fit = np.empty(m, dtype=np.float64)
    y_fit = np.empty(m, dtype=np.float64)
    for i in range(m):
        idx = start + i
        x_fit[i] = rho[idx] * rho[idx]
        denom = rho[idx] if rho[idx] > 1e-12 else 1e-12
        y_fit[i] = profile[idx] / denom

    vandermonde = np.empty((m, fit_degree + 1), dtype=np.float64)
    for i in range(m):
        vandermonde[i, 0] = 1.0
    for order in range(1, fit_degree + 1):
        for i in range(m):
            vandermonde[i, order] = vandermonde[i, order - 1] * x_fit[i]

    gram = np.empty((fit_degree + 1, fit_degree + 1), dtype=np.float64)
    rhs = np.empty(fit_degree + 1, dtype=np.float64)
    for row in range(fit_degree + 1):
        total_rhs = 0.0
        for i in range(m):
            total_rhs += vandermonde[i, row] * y_fit[i]
        rhs[row] = total_rhs
        for col in range(fit_degree + 1):
            total = 0.0
            for i in range(m):
                total += vandermonde[i, row] * vandermonde[i, col]
            gram[row, col] = total

    coeff = np.linalg.solve(gram, rhs)

    for i in range(replace_stop):
        x_val = rho[i] * rho[i]
        poly = 1.0
        total = coeff[0]
        for order in range(1, fit_degree + 1):
            poly *= x_val
            total += coeff[order] * poly
        profile[i] = rho[i] * total
    profile[0] = 0.0
    return profile


@njit(cache=True, fastmath=True, nogil=True)
def _smooth_profile_head_three_point(
    profile: np.ndarray,
    replace_count: int = 8,
    passes: int = 1,
) -> np.ndarray:
    n = profile.shape[0]
    stop = replace_count
    if stop > n - 1:
        stop = n - 1
    if stop <= 1 or passes <= 0:
        return profile

    scratch = np.empty_like(profile)
    for _ in range(passes):
        _copy_vector(scratch, profile)
        for i in range(1, stop):
            profile[i] = 0.25 * scratch[i - 1] + 0.5 * scratch[i] + 0.25 * scratch[i + 1]
    return profile


@njit(cache=True, nogil=True)
def _compute_Pn(Pn_r: np.ndarray, integration_matrix: np.ndarray, weights: np.ndarray) -> np.ndarray:
    Pn = np.empty_like(Pn_r)
    full_integration(Pn, Pn_r, integration_matrix)
    Pn -= quadrature(Pn_r, weights)
    return Pn


@njit(cache=True, nogil=True)
def _compute_Pn_out(
    out_Pn: np.ndarray,
    Pn_r: np.ndarray,
    integration_matrix: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    full_integration(out_Pn, Pn_r, integration_matrix)
    out_Pn -= quadrature(Pn_r, weights)
    return out_Pn


@njit(cache=True, fastmath=True, nogil=True)
def _copy_vector(out: np.ndarray, src: np.ndarray) -> np.ndarray:
    for i in range(out.shape[0]):
        out[i] = src[i]
    return out


@njit(cache=True, fastmath=True, nogil=True)
def _fill_half_square(out: np.ndarray, src: np.ndarray) -> np.ndarray:
    for i in range(out.shape[0]):
        out[i] = 0.5 * src[i] * src[i]
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
def _fill_pf_rho_integrand(
    out: np.ndarray,
    Kn: np.ndarray,
    current_input: np.ndarray,
    Ln_r: np.ndarray,
    V_r: np.ndarray,
    heat_input: np.ndarray,
) -> np.ndarray:
    pressure_factor = 1.0 / (4.0 * np.pi**2)
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
) -> np.ndarray:
    pressure_factor = 1.0 / (4.0 * np.pi**2)
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
    psin_r_safe: np.ndarray,
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
        ffn_r = -(term0 + term1) * (psin_r[i] / Ln_r[i])
        out[i] = ffn_r / psin_r_safe[i]
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
    psin_r: np.ndarray,
    psin_r_safe: np.ndarray,
    Ln_r: np.ndarray,
    current_scale: float,
) -> np.ndarray:
    pressure_factor = 1.0 / (4.0 * np.pi**2)
    for i in range(out.shape[0]):
        term0 = current_scale * jtor[i] * S_r[i]
        term1 = V_r[i] * Pn_psin[i] * pressure_factor
        ffn_r = -(term0 + term1) * (psin_r[i] / Ln_r[i])
        out[i] = ffn_r / psin_r_safe[i]
    return out


@njit(cache=True, nogil=True)
def _maximum_floor(arr: np.ndarray, floor: float) -> np.ndarray:
    out = np.empty_like(arr)
    for i in range(arr.shape[0]):
        value = arr[i]
        out[i] = value if value > floor else floor
    return out


@njit(cache=True, nogil=True)
def _maximum_floor_out(out: np.ndarray, arr: np.ndarray, floor: float) -> np.ndarray:
    for i in range(arr.shape[0]):
        value = arr[i]
        out[i] = value if value > floor else floor
    return out


def build_source_remap_cache(
    coordinate: str,
    source_sample_count: int,
    *,
    rho: np.ndarray | None = None,
    stencil_size: int = DEFAULT_LOCAL_BARYCENTRIC_STENCIL,
) -> tuple[int, np.ndarray, np.ndarray]:
    coord = str(coordinate).lower()
    if coord not in ("rho", "psin"):
        raise ValueError(f"Unsupported coordinate {coordinate!r}")

    count = int(source_sample_count)
    if count < 1:
        raise ValueError(f"source_sample_count must be positive, got {source_sample_count!r}")

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
    source_sample_count: int,
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
    if heat.shape[0] != source_sample_count:
        raise ValueError(f"Expected heat/current inputs to have length {source_sample_count}, got {heat.shape[0]}")
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


_SOURCE_SCRATCH_KERNELS: dict[str, Callable] = {
    "update_PJ2_PSIN": _update_pj2_from_psin_inputs_with_scratch,
    "update_PQ_PSIN": _update_pq_from_psin_inputs_with_scratch,
}


def resolve_source_scratch_kernel(operator_kernel: Callable) -> Callable | None:
    """返回支持显式 scratch 的 source kernel 实现."""
    if getattr(operator_kernel, "__module__", "") != __name__:
        return None
    return _SOURCE_SCRATCH_KERNELS.get(getattr(operator_kernel, "__name__", ""))


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
        np.asarray(out_psin, dtype=np.float64),
        np.asarray(out_psin_r, dtype=np.float64),
        np.asarray(out_psin_rr, dtype=np.float64),
        np.asarray(out_source_psin_query, dtype=np.float64),
        np.asarray(out_parameter_query, dtype=np.float64),
        np.asarray(out_heat_input, dtype=np.float64),
        np.asarray(out_current_input, dtype=np.float64),
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
        np.asarray(out_c_fields, dtype=np.float64),
        np.asarray(out_s_fields, dtype=np.float64),
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
    max_residual: float,
) -> bool:
    if query.ndim != 1 or psin.ndim != 1 or query.shape != psin.shape:
        raise ValueError(f"Expected query/psin to share a 1D shape, got {query.shape} and {psin.shape}")
    return bool(
        _update_fixed_point_psin_query_impl(
            np.asarray(query, dtype=np.float64),
            np.asarray(psin, dtype=np.float64),
            float(max_residual),
        )
    )


@njit(cache=True, fastmath=True, nogil=True)
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
    n = out_heat_input.shape[0]
    if endpoint_policy_code == ENDPOINT_POLICY_AFFINE_BOTH:
        _, left_current = _evaluate_chebyshev_pair(
            heat_coeff,
            current_coeff,
            _project_psin_query_to_chebyshev_x(psin_query[0], projection_domain_code),
        )
        _, right_current = _evaluate_chebyshev_pair(
            heat_coeff,
            current_coeff,
            _project_psin_query_to_chebyshev_x(psin_query[n - 1], projection_domain_code),
        )
        delta_left = current_source_values[0] - left_current
        delta_right = current_source_values[-1] - right_current
        for i in range(n):
            heat_val, current_val = _evaluate_chebyshev_pair(
                heat_coeff,
                current_coeff,
                _project_psin_query_to_chebyshev_x(psin_query[i], projection_domain_code),
            )
            out_heat_input[i] = heat_val
            out_current_input[i] = current_val + (1.0 - blend[i]) * delta_left + blend[i] * delta_right
        return

    for i in range(n):
        heat_val, current_val = _evaluate_chebyshev_pair(
            heat_coeff,
            current_coeff,
            _project_psin_query_to_chebyshev_x(psin_query[i], projection_domain_code),
        )
        out_heat_input[i] = heat_val
        out_current_input[i] = current_val

    if endpoint_policy_code == ENDPOINT_POLICY_NONE:
        return
    if endpoint_policy_code == ENDPOINT_POLICY_RIGHT:
        out_current_input[-1] = current_source_values[-1]
        return
    if endpoint_policy_code == ENDPOINT_POLICY_BOTH:
        out_current_input[0] = current_source_values[0]
        out_current_input[-1] = current_source_values[-1]
        return
    raise ValueError("Unsupported endpoint policy code")


@njit(cache=True, fastmath=True, nogil=True)
def _update_fixed_point_psin_query_impl(
    query: np.ndarray,
    psin: np.ndarray,
    max_residual: float,
) -> bool:
    max_abs_diff = 0.0
    for i in range(query.shape[0]):
        diff = abs(psin[i] - query[i])
        if diff > max_abs_diff:
            max_abs_diff = diff
        query[i] = psin[i]
    return max_abs_diff <= max_residual


@njit(cache=True, fastmath=True, nogil=True)
def _update_fixed_point_psin_query_and_linear_uniform_inputs_impl(
    query: np.ndarray,
    psin: np.ndarray,
    max_residual: float,
    out_heat_input: np.ndarray,
    out_current_input: np.ndarray,
    heat_input: np.ndarray,
    current_input: np.ndarray,
) -> bool:
    max_abs_diff = 0.0
    source_sample_count = heat_input.shape[0]
    if source_sample_count == 1:
        heat0 = heat_input[0]
        current0 = current_input[0]
        for i in range(query.shape[0]):
            q = psin[i]
            diff = abs(q - query[i])
            if diff > max_abs_diff:
                max_abs_diff = diff
            query[i] = q
            out_heat_input[i] = heat0
            out_current_input[i] = current0
        return max_abs_diff <= max_residual

    step = 1.0 / (source_sample_count - 1.0)
    for i in range(query.shape[0]):
        q = psin[i]
        diff = abs(q - query[i])
        if diff > max_abs_diff:
            max_abs_diff = diff
        query[i] = q

        if q < 0.0:
            q = 0.0
        elif q > 1.0:
            q = 1.0

        if q >= 1.0:
            out_heat_input[i] = heat_input[-1]
            out_current_input[i] = current_input[-1]
            continue

        position = q / step
        left = int(position)
        right = left + 1
        frac = position - left
        out_heat_input[i] = (1.0 - frac) * heat_input[left] + frac * heat_input[right]
        out_current_input[i] = (1.0 - frac) * current_input[left] + frac * current_input[right]
    return max_abs_diff <= max_residual


@njit(cache=True, fastmath=True, nogil=True)
def _update_fixed_point_psin_query_and_local_barycentric_inputs_impl(
    query: np.ndarray,
    psin: np.ndarray,
    max_residual: float,
    out_heat_input: np.ndarray,
    out_current_input: np.ndarray,
    heat_input: np.ndarray,
    current_input: np.ndarray,
    weights: np.ndarray,
) -> bool:
    max_abs_diff = 0.0
    source_sample_count = heat_input.shape[0]
    if source_sample_count == 1:
        heat0 = heat_input[0]
        current0 = current_input[0]
        for i in range(query.shape[0]):
            q = psin[i]
            diff = abs(q - query[i])
            if diff > max_abs_diff:
                max_abs_diff = diff
            query[i] = q
            out_heat_input[i] = heat0
            out_current_input[i] = current0
        return max_abs_diff <= max_residual

    local_size = weights.shape[0]
    denom_scale = source_sample_count - 1.0
    for i in range(query.shape[0]):
        q = psin[i]
        diff = abs(q - query[i])
        if diff > max_abs_diff:
            max_abs_diff = diff
        query[i] = q

        if q < 0.0:
            q = 0.0
        elif q > 1.0:
            q = 1.0

        start = _local_uniform_stencil_start(q, source_sample_count, local_size)
        hit = -1
        for local_j in range(local_size):
            j = start + local_j
            xj = j / denom_scale
            if abs(q - xj) <= 1e-14:
                hit = j
                break
        if hit >= 0:
            out_heat_input[i] = heat_input[hit]
            out_current_input[i] = current_input[hit]
            continue

        denominator = 0.0
        numerator_heat = 0.0
        numerator_current = 0.0
        for local_j in range(local_size):
            j = start + local_j
            term = weights[local_j] / (q - j / denom_scale)
            denominator += term
            numerator_heat += term * heat_input[j]
            numerator_current += term * current_input[j]
        out_heat_input[i] = numerator_heat / denominator
        out_current_input[i] = numerator_current / denominator
    return max_abs_diff <= max_residual


@njit(cache=True, fastmath=True, nogil=True)
def _update_fixed_point_psin_query_and_projected_inputs_impl(
    query: np.ndarray,
    psin: np.ndarray,
    max_residual: float,
    out_heat_input: np.ndarray,
    out_current_input: np.ndarray,
    heat_coeff: np.ndarray,
    current_coeff: np.ndarray,
    current_source_values: np.ndarray,
    projection_domain_code: int,
    endpoint_policy_code: int,
    blend: np.ndarray,
) -> bool:
    max_abs_diff = 0.0
    n = out_heat_input.shape[0]
    if endpoint_policy_code == ENDPOINT_POLICY_AFFINE_BOTH:
        _, left_current = _evaluate_chebyshev_pair(
            heat_coeff,
            current_coeff,
            _project_psin_query_to_chebyshev_x(psin[0], projection_domain_code),
        )
        _, right_current = _evaluate_chebyshev_pair(
            heat_coeff,
            current_coeff,
            _project_psin_query_to_chebyshev_x(psin[n - 1], projection_domain_code),
        )
        delta_left = current_source_values[0] - left_current
        delta_right = current_source_values[-1] - right_current
        for i in range(n):
            q = psin[i]
            diff = abs(q - query[i])
            if diff > max_abs_diff:
                max_abs_diff = diff
            query[i] = q
            heat_val, current_val = _evaluate_chebyshev_pair(
                heat_coeff,
                current_coeff,
                _project_psin_query_to_chebyshev_x(q, projection_domain_code),
            )
            out_heat_input[i] = heat_val
            out_current_input[i] = current_val + (1.0 - blend[i]) * delta_left + blend[i] * delta_right
        return max_abs_diff <= max_residual

    for i in range(n):
        q = psin[i]
        diff = abs(q - query[i])
        if diff > max_abs_diff:
            max_abs_diff = diff
        query[i] = q

        heat_val, current_val = _evaluate_chebyshev_pair(
            heat_coeff,
            current_coeff,
            _project_psin_query_to_chebyshev_x(q, projection_domain_code),
        )
        out_heat_input[i] = heat_val
        out_current_input[i] = current_val

    if endpoint_policy_code == ENDPOINT_POLICY_NONE:
        return max_abs_diff <= max_residual
    if endpoint_policy_code == ENDPOINT_POLICY_RIGHT:
        out_current_input[-1] = current_source_values[-1]
        return max_abs_diff <= max_residual
    if endpoint_policy_code == ENDPOINT_POLICY_BOTH:
        out_current_input[0] = current_source_values[0]
        out_current_input[-1] = current_source_values[-1]
        return max_abs_diff <= max_residual
    raise ValueError("Unsupported endpoint policy code")


@njit(cache=True, fastmath=True, nogil=True)
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
    for i in range(out_psin.shape[0]):
        psin_value = psin_fields[0, i]
        out_psin[i] = psin_value
        out_psin_r[i] = psin_fields[1, i]
        out_psin_rr[i] = psin_fields[2, i]
        out_source_psin_query[i] = psin_value
        out_parameter_query[i] = psin_value

    if parameterization_code == SOURCE_PARAMETERIZATION_CODE_SQRT_PSIN:
        for i in range(out_parameter_query.shape[0]):
            value = out_parameter_query[i]
            if value < 0.0:
                value = 0.0
            out_parameter_query[i] = np.sqrt(value)
    elif parameterization_code != SOURCE_PARAMETERIZATION_CODE_IDENTITY:
        raise ValueError("Unsupported source parameterization code")

    _linear_uniform_interpolate_pair(
        out_heat_input,
        out_current_input,
        heat_input,
        current_input,
        out_parameter_query,
    )


@njit(cache=True, fastmath=True, nogil=True)
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
        if order <= c_active_order:
            slot = c_source_slots[order]
            if slot >= 0:
                for d in range(out_c_fields.shape[1]):
                    for i in range(out_c_fields.shape[2]):
                        out_c_fields[order, d, i] = active_u_fields[slot, d, i]
            else:
                for d in range(out_c_fields.shape[1]):
                    for i in range(out_c_fields.shape[2]):
                        out_c_fields[order, d, i] = base_c_fields[order, d, i]
        else:
            for d in range(out_c_fields.shape[1]):
                for i in range(out_c_fields.shape[2]):
                    out_c_fields[order, d, i] = 0.0

    for d in range(out_s_fields.shape[1]):
        for i in range(out_s_fields.shape[2]):
            out_s_fields[0, d, i] = base_s_fields[0, d, i]
    for order in range(1, out_s_fields.shape[0]):
        if order <= s_active_order:
            slot = s_source_slots[order]
            if slot >= 0:
                for d in range(out_s_fields.shape[1]):
                    for i in range(out_s_fields.shape[2]):
                        out_s_fields[order, d, i] = active_u_fields[slot, d, i]
            else:
                for d in range(out_s_fields.shape[1]):
                    for i in range(out_s_fields.shape[2]):
                        out_s_fields[order, d, i] = base_s_fields[order, d, i]
        else:
            for d in range(out_s_fields.shape[1]):
                for i in range(out_s_fields.shape[2]):
                    out_s_fields[order, d, i] = 0.0


@njit(cache=True, fastmath=True, nogil=True)
def _evaluate_chebyshev_scalar(coeff: np.ndarray, x: float) -> float:
    if coeff.size == 0:
        return 0.0
    if coeff.size == 1:
        return coeff[0]
    b_kplus1 = 0.0
    b_kplus2 = 0.0
    for idx in range(coeff.size - 1, 0, -1):
        b_k = 2.0 * x * b_kplus1 - b_kplus2 + coeff[idx]
        b_kplus2 = b_kplus1
        b_kplus1 = b_k
    return x * b_kplus1 - b_kplus2 + coeff[0]


@njit(cache=True, fastmath=True, nogil=True)
def _project_psin_query_to_chebyshev_x(q: float, projection_domain_code: int) -> float:
    if q < 0.0:
        q = 0.0
    elif q > 1.0:
        q = 1.0
    if projection_domain_code == PROJECTION_DOMAIN_SQRT_PSIN:
        q = np.sqrt(q)
    elif projection_domain_code != PROJECTION_DOMAIN_PSIN:
        raise ValueError("Unsupported projection domain code")
    return 2.0 * q - 1.0


@njit(cache=True, fastmath=True, nogil=True)
def _evaluate_chebyshev_pair(coeff0: np.ndarray, coeff1: np.ndarray, x: float) -> tuple[float, float]:
    size0 = coeff0.size
    size1 = coeff1.size
    max_size = size0 if size0 >= size1 else size1
    if max_size == 0:
        return 0.0, 0.0
    if max_size == 1:
        return (
            coeff0[0] if size0 > 0 else 0.0,
            coeff1[0] if size1 > 0 else 0.0,
        )

    b0_kplus1 = 0.0
    b0_kplus2 = 0.0
    b1_kplus1 = 0.0
    b1_kplus2 = 0.0
    for idx in range(max_size - 1, 0, -1):
        c0 = coeff0[idx] if idx < size0 else 0.0
        c1 = coeff1[idx] if idx < size1 else 0.0
        b0_k = 2.0 * x * b0_kplus1 - b0_kplus2 + c0
        b1_k = 2.0 * x * b1_kplus1 - b1_kplus2 + c1
        b0_kplus2 = b0_kplus1
        b0_kplus1 = b0_k
        b1_kplus2 = b1_kplus1
        b1_kplus1 = b1_k
    return (
        x * b0_kplus1 - b0_kplus2 + (coeff0[0] if size0 > 0 else 0.0),
        x * b1_kplus1 - b1_kplus2 + (coeff1[0] if size1 > 0 else 0.0),
    )


@njit(cache=True, fastmath=True, nogil=True)
def _linear_uniform_interpolate_pair(
    out0: np.ndarray,
    out1: np.ndarray,
    values0: np.ndarray,
    values1: np.ndarray,
    query: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    source_sample_count = values0.shape[0]
    if source_sample_count == 1:
        value0 = values0[0]
        value1 = values1[0]
        for i in range(out0.shape[0]):
            out0[i] = value0
            out1[i] = value1
        return out0, out1

    step = 1.0 / (source_sample_count - 1.0)
    for i in range(out0.shape[0]):
        q = query[i]
        if q < 0.0:
            q = 0.0
        elif q > 1.0:
            q = 1.0

        if q >= 1.0:
            out0[i] = values0[-1]
            out1[i] = values1[-1]
            continue

        position = q / step
        left = int(position)
        right = left + 1
        frac = position - left
        out0[i] = (1.0 - frac) * values0[left] + frac * values0[right]
        out1[i] = (1.0 - frac) * values1[left] + frac * values1[right]
    return out0, out1


@njit(cache=True, fastmath=True, nogil=True)
def _local_barycentric_interpolate_pair(
    out0: np.ndarray,
    out1: np.ndarray,
    values0: np.ndarray,
    values1: np.ndarray,
    query: np.ndarray,
    weights: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    source_sample_count = values0.shape[0]
    if source_sample_count == 1:
        value0 = values0[0]
        value1 = values1[0]
        for i in range(out0.shape[0]):
            out0[i] = value0
            out1[i] = value1
        return out0, out1

    local_size = weights.shape[0]
    denom_scale = source_sample_count - 1.0
    for i in range(out0.shape[0]):
        q = query[i]
        if q < 0.0:
            q = 0.0
        elif q > 1.0:
            q = 1.0

        start = _local_uniform_stencil_start(q, source_sample_count, local_size)
        hit = -1
        for local_j in range(local_size):
            j = start + local_j
            xj = j / denom_scale
            if abs(q - xj) <= 1e-14:
                hit = j
                break
        if hit >= 0:
            out0[i] = values0[hit]
            out1[i] = values1[hit]
            continue

        denominator = 0.0
        numerator0 = 0.0
        numerator1 = 0.0
        for local_j in range(local_size):
            j = start + local_j
            term = weights[local_j] / (q - j / denom_scale)
            denominator += term
            numerator0 += term * values0[j]
            numerator1 += term * values1[j]
        out0[i] = numerator0 / denominator
        out1[i] = numerator1 / denominator
    return out0, out1


@njit(cache=True, fastmath=True, nogil=True)
def _uniform_barycentric_weights(source_sample_count: int) -> np.ndarray:
    weights = np.empty(source_sample_count, dtype=np.float64)
    weights[0] = 1.0
    for j in range(1, source_sample_count):
        weights[j] = -weights[j - 1] * (source_sample_count - j) / j
    return weights


uniform_barycentric_weights = _uniform_barycentric_weights


def _build_uniform_barycentric_matrix(
    query: np.ndarray,
    source_sample_count: int,
    stencil_size: int,
    weights: np.ndarray,
) -> np.ndarray:
    matrix = np.empty((query.shape[0], source_sample_count), dtype=np.float64)
    if source_sample_count == 1:
        matrix[:, 0] = 1.0
        return matrix

    for i, q in enumerate(query):
        for j in range(source_sample_count):
            matrix[i, j] = 0.0
        start = _local_uniform_stencil_start(q, source_sample_count, stencil_size)
        hit = False
        for local_j in range(stencil_size):
            j = start + local_j
            diff = q - j / (source_sample_count - 1.0)
            if abs(diff) <= 1e-14:
                matrix[i, j] = 1.0
                hit = True
                break
        if hit:
            continue

        denominator = 0.0
        for local_j in range(stencil_size):
            j = start + local_j
            denominator += weights[local_j] / (q - j / (source_sample_count - 1.0))
        for local_j in range(stencil_size):
            j = start + local_j
            matrix[i, j] = (weights[local_j] / (q - j / (source_sample_count - 1.0))) / denominator
    return matrix


@njit(cache=True, fastmath=True, nogil=True)
def _local_uniform_stencil_start(q: float, source_sample_count: int, stencil_size: int) -> int:
    if stencil_size >= source_sample_count:
        return 0
    pos = q * (source_sample_count - 1.0)
    center = int(pos)
    if pos > center:
        center += 1
    start = center - stencil_size // 2
    if start < 0:
        return 0
    max_start = source_sample_count - stencil_size
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

"""
Module: engine.numba_source

Role:
- 负责在 numba backend 下注册 source operators.
- 负责校验 operator/derivative 组合并执行 source kernels.

Public API:
- register_operator
- validate_operator
- bind_source_runner

Notes:
- operator routing 保留在这里.
- operator 层只 bind 一个 source runner, 并把它作为 Stage-C 执行入口.
"""

from dataclasses import dataclass
from typing import Callable

import numpy as np
from numba import njit

MU0 = 4.0 * np.pi * 1e-7

RHO_AXIS = 0
THETA_AXIS = 1

RHO_DERIVATIVE = 0
PSI_DERIVATIVE = 1

DERIVATIVE_NAMES = {
    RHO_DERIVATIVE: "rho",
    PSI_DERIVATIVE: "psi",
}

DERIVATIVE_CODES = {
    "rho": RHO_DERIVATIVE,
    "psi": PSI_DERIVATIVE,
}


@dataclass(frozen=True, slots=True)
class _SourceSpec:
    name: str
    family: str
    supported_derivatives: tuple[int, ...]
    implementation: Callable


OPERATOR_REGISTRY: dict[str, _SourceSpec] = {}


def register_operator(
    name: str,
    *,
    family: str,
    supported_derivatives: tuple[int, ...] = (RHO_DERIVATIVE, PSI_DERIVATIVE),
) -> Callable:
    """注册一个 source operator kernel."""

    def decorator(func: Callable) -> Callable:
        existing = OPERATOR_REGISTRY.get(name)
        if existing is not None:
            raise ValueError(f"_SourceSpec {name!r} is already registered")

        OPERATOR_REGISTRY[name] = _SourceSpec(
            name=name,
            family=family,
            supported_derivatives=supported_derivatives,
            implementation=func,
        )
        return func

    return decorator


def validate_operator(name: str, derivative: str) -> _SourceSpec:
    """校验 operator/derivative 组合并返回注册规格."""

    derivative_code = _normalize_derivative(derivative)
    try:
        spec = OPERATOR_REGISTRY[name]
    except KeyError as exc:
        supported = ", ".join(OPERATOR_REGISTRY)
        raise KeyError(f"Unknown operator {name!r}. Supported operators: {supported}") from exc

    if derivative_code not in spec.supported_derivatives:
        raise ValueError(
            f"_SourceSpec {name!r} does not support derivative={derivative!r}. Supported: {spec.supported_derivatives}"
        )

    return spec


def bind_source_runner(name: str, derivative: str) -> Callable:
    """绑定 Stage-C source runner."""

    derivative_code = _normalize_derivative(derivative)
    spec = validate_operator(name, derivative)
    operator_kernel = spec.implementation

    def runner(
        out_psin_r: np.ndarray,
        out_psin_rr: np.ndarray,
        out_FFn_r: np.ndarray,
        out_Pn_r: np.ndarray,
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
    ) -> tuple[float, float]:
        return operator_kernel(
            out_psin_r,
            out_psin_rr,
            out_FFn_r,
            out_Pn_r,
            heat_input,
            current_input,
            derivative_code,
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


@register_operator("PF", family="strict")
@njit(cache=True, nogil=True)
def update_PF(
    out_psin_r: np.ndarray,
    out_psin_rr: np.ndarray,
    out_FFn_r: np.ndarray,
    out_Pn_r: np.ndarray,
    heat_input: np.ndarray,
    current_input: np.ndarray,
    derivative: int,
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
    """根据 PF 约束更新 source root fields."""
    has_Ip = not np.isnan(Ip)
    has_beta = not np.isnan(beta)

    pressure_scale = MU0 if not has_Ip and not has_beta else 1.0
    integrand = np.empty_like(out_psin_r)

    if derivative == RHO_DERIVATIVE:
        _fill_pf_rho_integrand(integrand, Kn, current_input, Ln_r, V_r, heat_input, pressure_scale)
        corrected_integration(out_psin_r, integrand, integration_matrix, 1, rho, differentiation_matrix)
        out_psin_r *= -2.0
        out_psin_r[:] = np.sqrt(out_psin_r)
        out_psin_r /= Kn
    else:
        _fill_pf_psi_integrand(integrand, current_input, Ln_r, V_r, heat_input, pressure_scale)
        corrected_integration(out_psin_r, integrand, integration_matrix, 1, rho, differentiation_matrix)
        out_psin_r *= -1.0
        out_psin_r /= Kn

    prof = out_psin_r
    integral_prof = quadrature(prof, weights)
    out_psin_r /= integral_prof
    full_differentiation(out_psin_rr, out_psin_r, differentiation_matrix)

    if (not has_Ip) and (not has_beta):
        alpha2 = integral_prof
        if derivative == RHO_DERIVATIVE:
            alpha1 = -MU0 / alpha2 * quadrature(heat_input, weights)
            _fill_scaled_vector(out_Pn_r, heat_input, MU0 / (alpha1 * alpha2))
            _fill_scaled_vector(out_FFn_r, current_input, 1.0 / (alpha1 * alpha2))
        else:
            pressure_profile = np.empty_like(out_psin_r)
            current_profile = np.empty_like(out_psin_r)
            _fill_pointwise_product(pressure_profile, heat_input, prof)
            _fill_pointwise_product(current_profile, current_input, prof)
            alpha1 = -MU0 * quadrature(pressure_profile, weights)
            _fill_scaled_vector(out_Pn_r, pressure_profile, MU0 / alpha1)
            _fill_scaled_vector(out_FFn_r, current_profile, 1.0 / alpha1)
    else:
        c2 = integral_prof**2 if derivative == RHO_DERIVATIVE else integral_prof
        _copy_vector(out_Pn_r, heat_input)
        _copy_vector(out_FFn_r, current_input)
        if derivative != RHO_DERIVATIVE:
            _mul_vector_inplace(out_Pn_r, out_psin_r)
            _mul_vector_inplace(out_FFn_r, out_psin_r)

        if has_Ip and (not has_beta):
            psin_r_safe = _maximum_floor(out_psin_r, 1e-10)
            g1n_integrand = np.empty_like(R)
            _fill_g1n_integrand(g1n_integrand, JdivR, out_FFn_r, R, out_Pn_r, psin_r_safe)
            G1n_integral = quadrature(g1n_integrand, weights)
            alpha1 = -MU0 * Ip / G1n_integral
        elif has_beta and (not has_Ip):
            Pn = _compute_Pn(out_Pn_r, integration_matrix, weights)
            c1 = 0.5 * beta * B0**2 * quadrature(V_r, weights) / quadrature(Pn * V_r, weights)
            alpha1 = np.sqrt(c1 / c2)
        else:
            alpha1 = np.nan

        alpha2 = c2 * alpha1

    return alpha1, alpha2


@register_operator("PP", family="strict")
@njit(cache=True, nogil=True)
def update_PP(
    out_psin_r: np.ndarray,
    out_psin_rr: np.ndarray,
    out_FFn_r: np.ndarray,
    out_Pn_r: np.ndarray,
    heat_input: np.ndarray,
    current_input: np.ndarray,
    derivative: int,
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
    """根据 PP 约束更新 source root fields."""
    has_Ip = not np.isnan(Ip)
    has_beta = not np.isnan(beta)

    if has_Ip:
        _copy_vector(out_psin_r, current_input)
        alpha2 = MU0 * Ip / (2.0 * np.pi * Kn[-1] * out_psin_r[-1])
    else:
        alpha2 = quadrature(current_input, weights)
        _fill_scaled_vector(out_psin_r, current_input, 1.0 / alpha2)

    full_differentiation(out_psin_rr, out_psin_r, differentiation_matrix)

    if has_beta:
        _copy_vector(out_Pn_r, heat_input)
        if derivative != RHO_DERIVATIVE:
            _mul_vector_inplace(out_Pn_r, out_psin_r)
        Pn = _compute_Pn(out_Pn_r, integration_matrix, weights)
        alpha1 = 0.5 * beta * B0**2 / alpha2 * quadrature(V_r, weights) / quadrature(Pn * V_r, weights)
    else:
        P_r = np.empty_like(out_psin_r)
        if derivative == RHO_DERIVATIVE:
            _copy_vector(P_r, heat_input)
        else:
            _fill_scaled_product(P_r, heat_input, out_psin_r, alpha2)
        alpha1 = -MU0 / alpha2 * quadrature(P_r, weights)
        _fill_scaled_vector(out_Pn_r, P_r, MU0 / (alpha1 * alpha2))

    _fill_pp_ffn(out_FFn_r, out_psin_r, Kn_r, Kn, out_psin_rr, V_r, out_Pn_r, Ln_r, alpha2 / alpha1)
    return alpha1, alpha2


@register_operator("PI", family="strict")
@njit(cache=True, nogil=True)
def update_PI(
    out_psin_r: np.ndarray,
    out_psin_rr: np.ndarray,
    out_FFn_r: np.ndarray,
    out_Pn_r: np.ndarray,
    heat_input: np.ndarray,
    current_input: np.ndarray,
    derivative: int,
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
    """根据 PI 约束更新 source root fields."""
    has_Ip = not np.isnan(Ip)
    has_beta = not np.isnan(beta)
    Itor = np.empty_like(current_input)

    if has_Ip:
        _fill_scaled_vector(Itor, current_input, Ip / current_input[-1])
    else:
        _copy_vector(Itor, current_input)

    itor_over_kn = np.empty_like(current_input)
    _fill_scaled_ratio(itor_over_kn, Itor, Kn, MU0 / (2.0 * np.pi))
    alpha2 = quadrature(itor_over_kn, weights)

    _fill_scaled_ratio(out_psin_r, Itor, Kn, MU0 / (2.0 * np.pi * alpha2))
    full_differentiation(out_psin_rr, out_psin_r, differentiation_matrix)
    Itor_r = np.empty_like(Itor)
    full_differentiation(Itor_r, Itor, differentiation_matrix)

    if has_beta:
        _copy_vector(out_Pn_r, heat_input)
        if derivative != RHO_DERIVATIVE:
            _mul_vector_inplace(out_Pn_r, out_psin_r)
        Pn = _compute_Pn(out_Pn_r, integration_matrix, weights)
        alpha1 = 0.5 * beta * B0**2 / alpha2 * quadrature(V_r, weights) / quadrature(Pn * V_r, weights)
    else:
        P_r = np.empty_like(out_psin_r)
        if derivative == RHO_DERIVATIVE:
            _copy_vector(P_r, heat_input)
        else:
            _fill_scaled_product(P_r, heat_input, out_psin_r, alpha2)
        alpha1 = -MU0 / alpha2 * quadrature(P_r, weights)
        _fill_scaled_vector(out_Pn_r, P_r, MU0 / (alpha1 * alpha2))

    _fill_pi_ffn(out_FFn_r, Itor_r, out_psin_r, V_r, out_Pn_r, Ln_r, MU0 / (2.0 * np.pi * alpha1))
    return alpha1, alpha2


@register_operator("PJ1", family="strict")
@njit(cache=True, nogil=True)
def update_PJ(
    out_psin_r: np.ndarray,
    out_psin_rr: np.ndarray,
    out_FFn_r: np.ndarray,
    out_Pn_r: np.ndarray,
    heat_input: np.ndarray,
    current_input: np.ndarray,
    derivative: int,
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
    """根据 PJ1 约束更新 source root fields."""
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

    if has_beta:
        _copy_vector(out_Pn_r, heat_input)
        if derivative != RHO_DERIVATIVE:
            _mul_vector_inplace(out_Pn_r, out_psin_r)
        Pn = _compute_Pn(out_Pn_r, integration_matrix, weights)
        alpha1 = 0.5 * beta * B0**2 / alpha2 * quadrature(V_r, weights) / quadrature(Pn * V_r, weights)
    else:
        P_r = np.empty_like(out_psin_r)
        if derivative == RHO_DERIVATIVE:
            _copy_vector(P_r, heat_input)
        else:
            _fill_scaled_product(P_r, heat_input, out_psin_r, alpha2)
        alpha1 = -MU0 / alpha2 * quadrature(P_r, weights)
        _fill_scaled_vector(out_Pn_r, P_r, MU0 / (alpha1 * alpha2))

    _fill_pj_ffn(out_FFn_r, jtor, S_r, out_psin_r, V_r, out_Pn_r, Ln_r, MU0 / (2.0 * np.pi * alpha1))
    return alpha1, alpha2


@register_operator("PJ2", family="robust")
@njit(cache=True, nogil=True)
def update_PJ2(
    out_psin_r: np.ndarray,
    out_psin_rr: np.ndarray,
    out_FFn_r: np.ndarray,
    out_Pn_r: np.ndarray,
    heat_input: np.ndarray,
    current_input: np.ndarray,
    derivative: int,
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
    """根据 PJ2 约束更新 source root fields."""
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

    if has_beta:
        _copy_vector(out_Pn_r, heat_input)
        if derivative != RHO_DERIVATIVE:
            _mul_vector_inplace(out_Pn_r, out_psin_r)
        Pn = _compute_Pn(out_Pn_r, integration_matrix, weights)
        alpha1 = 0.5 * beta * B0**2 / alpha2 * quadrature(V_r, weights) / quadrature(Pn * V_r, weights)
    else:
        P_r = np.empty_like(out_psin_r)
        if derivative == RHO_DERIVATIVE:
            _copy_vector(P_r, heat_input)
        else:
            _fill_scaled_product(P_r, heat_input, out_psin_r, alpha2)
        alpha1 = -MU0 / alpha2 * quadrature(P_r, weights)
        _fill_scaled_vector(out_Pn_r, P_r, MU0 / (alpha1 * alpha2))

    F_r = np.empty_like(F)
    full_differentiation(F_r, F, differentiation_matrix)
    _fill_scaled_product(out_FFn_r, F, F_r, 1.0 / (alpha1 * alpha2))
    return alpha1, alpha2


@register_operator("PQ", family="robust")
@njit(cache=True, nogil=True)
def update_PQ(
    out_psin_r: np.ndarray,
    out_psin_rr: np.ndarray,
    out_FFn_r: np.ndarray,
    out_Pn_r: np.ndarray,
    heat_input: np.ndarray,
    current_input: np.ndarray,
    derivative: int,
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
    """根据 PQ 约束更新 source root fields."""
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

    if has_beta:
        _copy_vector(out_Pn_r, heat_input)
        if derivative != RHO_DERIVATIVE:
            _mul_vector_inplace(out_Pn_r, out_psin_r)
        Pn = _compute_Pn(out_Pn_r, integration_matrix, weights)
        alpha1 = 0.5 * beta * B0**2 / alpha2 * quadrature(V_r, weights) / quadrature(Pn * V_r, weights)
    else:
        P_r = np.empty_like(out_psin_r)
        if derivative == RHO_DERIVATIVE:
            _copy_vector(P_r, heat_input)
        else:
            _fill_scaled_product(P_r, heat_input, out_psin_r, alpha2)
        alpha1 = -MU0 / alpha2 * quadrature(P_r, weights)
        _fill_scaled_vector(out_Pn_r, P_r, MU0 / (alpha1 * alpha2))

    F_r = np.empty_like(F)
    full_differentiation(F_r, F, differentiation_matrix)
    _fill_scaled_product(out_FFn_r, F, F_r, 1.0 / (alpha1 * alpha2))
    return alpha1, alpha2


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
def _fill_pf_psi_integrand(
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
def _fill_g1n_integrand(
    out: np.ndarray,
    JdivR: np.ndarray,
    FFn_r: np.ndarray,
    R: np.ndarray,
    Pn_r: np.ndarray,
    psin_r_safe: np.ndarray,
) -> np.ndarray:
    nr, nt = out.shape
    for i in range(nr):
        inv_psin = 1.0 / psin_r_safe[i]
        ffn_i = FFn_r[i]
        pn_i = Pn_r[i]
        for j in range(nt):
            out[i, j] = JdivR[i, j] * (ffn_i + R[i, j] * R[i, j] * pn_i) * inv_psin
    return out


@njit(cache=True, fastmath=True, nogil=True)
def _fill_pp_ffn(
    out: np.ndarray,
    psin_r: np.ndarray,
    Kn_r: np.ndarray,
    Kn: np.ndarray,
    psin_rr: np.ndarray,
    V_r: np.ndarray,
    Pn_r: np.ndarray,
    Ln_r: np.ndarray,
    alpha_ratio: float,
) -> np.ndarray:
    pressure_factor = 1.0 / (4.0 * np.pi**2)
    for i in range(out.shape[0]):
        term0 = alpha_ratio * psin_r[i] * (Kn_r[i] * psin_r[i] + Kn[i] * psin_rr[i])
        term1 = V_r[i] * Pn_r[i] * pressure_factor
        out[i] = -(term0 + term1) / Ln_r[i]
    return out


@njit(cache=True, fastmath=True, nogil=True)
def _fill_pi_ffn(
    out: np.ndarray,
    Itor_r: np.ndarray,
    psin_r: np.ndarray,
    V_r: np.ndarray,
    Pn_r: np.ndarray,
    Ln_r: np.ndarray,
    current_scale: float,
) -> np.ndarray:
    pressure_factor = 1.0 / (4.0 * np.pi**2)
    for i in range(out.shape[0]):
        term0 = current_scale * Itor_r[i] * psin_r[i]
        term1 = V_r[i] * Pn_r[i] * pressure_factor
        out[i] = -(term0 + term1) / Ln_r[i]
    return out


@njit(cache=True, fastmath=True, nogil=True)
def _fill_pj_ffn(
    out: np.ndarray,
    jtor: np.ndarray,
    S_r: np.ndarray,
    psin_r: np.ndarray,
    V_r: np.ndarray,
    Pn_r: np.ndarray,
    Ln_r: np.ndarray,
    current_scale: float,
) -> np.ndarray:
    pressure_factor = 1.0 / (4.0 * np.pi**2)
    for i in range(out.shape[0]):
        term0 = current_scale * jtor[i] * S_r[i] * psin_r[i]
        term1 = V_r[i] * Pn_r[i] * pressure_factor
        out[i] = -(term0 + term1) / Ln_r[i]
    return out


@njit(cache=True, nogil=True)
def _maximum_floor(arr: np.ndarray, floor: float) -> np.ndarray:
    out = np.empty_like(arr)
    for i in range(arr.shape[0]):
        value = arr[i]
        out[i] = value if value > floor else floor
    return out


def _normalize_derivative(value: str) -> int:
    try:
        return DERIVATIVE_CODES[value]
    except KeyError as exc:
        raise ValueError(f"Unsupported derivative {value!r}") from exc

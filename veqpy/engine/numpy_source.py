"""
Module: engine.numpy_source

Role:
- 负责在 numpy backend 下注册 source operators.
- 负责校验 operator/derivative 组合并执行 source kernels.

Public API:
- register_operator
- validate_operator
- bind_source_runner

Notes:
- 这个文件同时作为 Stage-C source routing 与 field update 的 vectorized reference.
"""

import warnings
from dataclasses import dataclass
from typing import Callable

import numpy as np

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


def validate_operator(
    name: str,
    derivative: str,
) -> _SourceSpec:
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


def bind_source_runner(
    name: str,
    derivative: str,
) -> Callable:
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


def _normalize_derivative(value: str) -> int:
    try:
        return DERIVATIVE_CODES[value]
    except KeyError as exc:
        raise ValueError(f"Unsupported derivative {value!r}") from exc


@register_operator("PF", family="strict")
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
    if derivative == RHO_DERIVATIVE:
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
        np.sqrt(out_psin_r, out=out_psin_r)
        out_psin_r /= Kn
    else:
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

    if not has_Ip and not has_beta:
        alpha2 = integral_prof
        if derivative == RHO_DERIVATIVE:
            alpha1 = -MU0 / alpha2 * quadrature(heat_input, weights)
            out_Pn_r[:] = MU0 * heat_input / (alpha1 * alpha2)
            out_FFn_r[:] = current_input / (alpha1 * alpha2)
        else:
            alpha1 = -MU0 * quadrature(heat_input * prof, weights)
            out_Pn_r[:] = MU0 * heat_input * prof / alpha1
            out_FFn_r[:] = current_input * prof / alpha1

    else:
        c2 = integral_prof**2 if derivative == RHO_DERIVATIVE else integral_prof
        out_Pn_r[:] = heat_input
        out_FFn_r[:] = current_input
        if derivative != RHO_DERIVATIVE:
            out_Pn_r *= out_psin_r
            out_FFn_r *= out_psin_r

        if has_Ip and not has_beta:
            psin_r_safe = np.maximum(out_psin_r, 1e-10)
            G1n_integral = quadrature(
                JdivR * (out_FFn_r[:, None] + R * R * out_Pn_r[:, None]) / psin_r_safe[:, None],
                weights,
            )
            alpha1 = -MU0 * Ip / G1n_integral
        elif has_beta and not has_Ip:
            Pn = _compute_Pn(out_Pn_r, integration_matrix, weights)
            c1 = 0.5 * beta * B0**2 * quadrature(V_r, weights) / quadrature(Pn * V_r, weights)
            alpha1 = np.sqrt(c1 / c2)

        alpha2 = c2 * alpha1

    return alpha1, alpha2


@register_operator("PP", family="strict")
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
        out_psin_r[:] = current_input
        alpha2 = MU0 * Ip / (2.0 * np.pi * Kn[-1] * out_psin_r[-1])
    else:
        alpha2 = quadrature(current_input, weights)
        out_psin_r[:] = current_input / alpha2

    full_differentiation(out_psin_rr, out_psin_r, differentiation_matrix)

    if has_beta:
        out_Pn_r[:] = heat_input
        if derivative != RHO_DERIVATIVE:
            out_Pn_r *= out_psin_r
        Pn = _compute_Pn(out_Pn_r, integration_matrix, weights)
        alpha1 = 0.5 * beta * B0**2 / alpha2 * quadrature(V_r, weights) / quadrature(Pn * V_r, weights)
    else:
        P_r = heat_input if derivative == RHO_DERIVATIVE else heat_input * out_psin_r * alpha2
        alpha1 = -MU0 / alpha2 * quadrature(P_r, weights)
        out_Pn_r[:] = MU0 * P_r / (alpha1 * alpha2)

    out_FFn_r[:] = (
        -((alpha2 / alpha1) * out_psin_r * (Kn_r * out_psin_r + Kn * out_psin_rr) + V_r * out_Pn_r / (4.0 * np.pi**2))
        / Ln_r
    )
    return alpha1, alpha2


@register_operator("PI", family="strict")
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

    if has_Ip:
        Itor = Ip / current_input[-1] * current_input
    else:
        Itor = current_input

    alpha2 = quadrature(MU0 * Itor / (2.0 * np.pi * Kn), weights)

    out_psin_r[:] = MU0 * Itor / (2.0 * np.pi * alpha2 * Kn)
    full_differentiation(out_psin_rr, out_psin_r, differentiation_matrix)
    Itor_r = np.empty_like(Itor)
    full_differentiation(Itor_r, Itor, differentiation_matrix)

    if has_beta:
        out_Pn_r[:] = heat_input
        if derivative != RHO_DERIVATIVE:
            out_Pn_r *= out_psin_r
        Pn = _compute_Pn(out_Pn_r, integration_matrix, weights)
        alpha1 = 0.5 * beta * B0**2 / alpha2 * quadrature(V_r, weights) / quadrature(Pn * V_r, weights)
    else:
        P_r = heat_input if derivative == RHO_DERIVATIVE else heat_input * out_psin_r * alpha2
        alpha1 = -MU0 / alpha2 * quadrature(P_r, weights)
        out_Pn_r[:] = MU0 * P_r / (alpha1 * alpha2)

    out_FFn_r[:] = -((MU0 / (2.0 * np.pi * alpha1)) * Itor_r * out_psin_r + V_r * out_Pn_r / (4.0 * np.pi**2)) / Ln_r
    return alpha1, alpha2


@register_operator("PJ1", family="strict")
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

    if has_beta:
        out_Pn_r[:] = heat_input
        if derivative != RHO_DERIVATIVE:
            out_Pn_r *= out_psin_r
        Pn = _compute_Pn(out_Pn_r, integration_matrix, weights)
        alpha1 = 0.5 * beta * B0**2 / alpha2 * quadrature(V_r, weights) / quadrature(Pn * V_r, weights)
    else:
        P_r = heat_input if derivative == RHO_DERIVATIVE else heat_input * out_psin_r * alpha2
        alpha1 = -MU0 / alpha2 * quadrature(P_r, weights)
        out_Pn_r[:] = MU0 * P_r / (alpha1 * alpha2)

    out_FFn_r[:] = (
        -((MU0 / (2.0 * np.pi * alpha1)) * jtor * S_r * out_psin_r + V_r * out_Pn_r / (4.0 * np.pi**2)) / Ln_r
    )
    return alpha1, alpha2


@register_operator("PJ2", family="robust")
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

    if has_beta:
        out_Pn_r[:] = heat_input
        if derivative != RHO_DERIVATIVE:
            out_Pn_r *= out_psin_r
        Pn = _compute_Pn(out_Pn_r, integration_matrix, weights)
        alpha1 = 0.5 * beta * B0**2 / alpha2 * quadrature(V_r, weights) / quadrature(Pn * V_r, weights)
    else:
        P_r = heat_input if derivative == RHO_DERIVATIVE else heat_input * out_psin_r * alpha2
        alpha1 = -MU0 / alpha2 * quadrature(P_r, weights)
        out_Pn_r[:] = MU0 * P_r / (alpha1 * alpha2)

    F_r = np.empty_like(F)
    full_differentiation(F_r, F, differentiation_matrix)
    FF_r = F * F_r
    out_FFn_r[:] = FF_r / (alpha1 * alpha2)
    return alpha1, alpha2


@register_operator("PQ", family="robust")
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

    if has_Ip:
        q_scale = (2.0 * np.pi * R0 * B0) / (MU0 * Ip)
        q_prof = current_input * q_scale * (Kn[-1] * Ln_r[-1] / current_input[-1])
    else:
        q_prof = current_input

    integrand_alpha2 = (F * Ln_r) / q_prof
    alpha2 = quadrature(integrand_alpha2, weights)

    out_psin_r[:] = (F * Ln_r) / (alpha2 * q_prof)
    full_differentiation(out_psin_rr, out_psin_r, differentiation_matrix)

    if has_beta:
        out_Pn_r[:] = heat_input
        if derivative != RHO_DERIVATIVE:
            out_Pn_r *= out_psin_r
        Pn = _compute_Pn(out_Pn_r, integration_matrix, weights)
        alpha1 = 0.5 * beta * B0**2 / alpha2 * quadrature(V_r, weights) / quadrature(Pn * V_r, weights)
    else:
        P_r = heat_input if derivative == RHO_DERIVATIVE else heat_input * out_psin_r * alpha2
        alpha1 = -MU0 / alpha2 * quadrature(P_r, weights)
        out_Pn_r[:] = MU0 * P_r / (alpha1 * alpha2)

    F_r = np.empty_like(F)
    full_differentiation(F_r, F, differentiation_matrix)
    FF_r = F * F_r
    out_FFn_r[:] = FF_r / (alpha1 * alpha2)
    return alpha1, alpha2


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

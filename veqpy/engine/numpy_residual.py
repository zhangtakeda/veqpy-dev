"""
Module: engine.numpy_residual

Role:
- 负责在 numpy backend 下生成 residual fields.
- 负责把 residual blocks 组装成 packed residual.

Public API:
- update_residual
- bind_residual_runner

Notes:
- 这个文件同时作为 residual field update 与 packed residual assembly 的 vectorized reference.
"""

from dataclasses import dataclass
from typing import Callable

import numpy as np


@dataclass(frozen=True, slots=True)
class _ResidualBlockSpec:
    name: str
    implementation: Callable


RESIDUAL_BLOCK_REGISTRY: dict[str, _ResidualBlockSpec] = {}


def register_residual_block(name: str) -> Callable:
    def decorator(func: Callable) -> Callable:
        existing = RESIDUAL_BLOCK_REGISTRY.get(name)
        if existing is not None:
            raise ValueError(f"Residual block {name!r} is already registered")
        RESIDUAL_BLOCK_REGISTRY[name] = _ResidualBlockSpec(name=name, implementation=func)
        return func

    return decorator


def bind_residual_runner(
    profile_names: tuple[str, ...],
    coeff_index_rows: np.ndarray,
    lengths: np.ndarray,
    residual_size: int,
) -> Callable:
    kernels: list[Callable] = []
    for name in profile_names:
        try:
            kernels.append(RESIDUAL_BLOCK_REGISTRY[name].implementation)
        except KeyError as exc:
            supported = ", ".join(RESIDUAL_BLOCK_REGISTRY)
            raise KeyError(f"Unknown residual block {name!r}. Supported blocks: {supported}") from exc
    bound_kernels = tuple(kernels)

    def runner(
        G: np.ndarray,
        psin_R: np.ndarray,
        psin_Z: np.ndarray,
        sin_tb: np.ndarray,
        sin_theta: np.ndarray,
        cos_theta: np.ndarray,
        sin_2theta: np.ndarray,
        rho: np.ndarray,
        rho2: np.ndarray,
        y: np.ndarray,
        T: np.ndarray,
        weights: np.ndarray,
        a: float,
        R0: float,
        B0: float,
    ) -> np.ndarray:
        out = np.zeros(residual_size, dtype=np.float64)
        for slot, kernel in enumerate(bound_kernels):
            coeff_size = int(lengths[slot])
            kernel(
                out,
                coeff_index_rows[slot, :coeff_size],
                G,
                psin_R,
                psin_Z,
                sin_tb,
                sin_theta,
                cos_theta,
                sin_2theta,
                rho,
                rho2,
                y,
                T,
                weights,
                a,
                R0,
                B0,
            )
        return out

    return runner


def update_residual(
    out_fields: np.ndarray,
    alpha1: float,
    alpha2: float,
    root_fields: np.ndarray,
    R_fields: np.ndarray,
    Z_fields: np.ndarray,
    J_fields: np.ndarray,
    g_fields: np.ndarray,
) -> None:
    """原地更新 residual 相关二维 fields."""
    out_psin_R = out_fields[0]
    out_psin_Z = out_fields[1]
    out_G = out_fields[2]

    psin_r = root_fields[0]
    psin_rr = root_fields[1]
    FFn_r = root_fields[2]
    Pn_r = root_fields[3]

    R = R_fields[0]
    R_t = R_fields[2]
    Z_t = Z_fields[2]
    J = J_fields[0]
    JdivR = J_fields[6]
    grtdivJR_t = g_fields[2]
    gttdivJR = g_fields[5]
    gttdivJR_r = g_fields[6]

    np.divide(Z_t, J, out=out_psin_R)
    out_psin_R *= -1.0
    out_psin_R *= psin_r[:, None]

    np.divide(R_t, J, out=out_psin_Z)
    out_psin_Z *= psin_r[:, None]

    psin_r_safe = np.maximum(psin_r, 1e-10)

    G1n = JdivR * (FFn_r[:, None] + R**2 * Pn_r[:, None]) / psin_r_safe[:, None]
    G2n = gttdivJR * psin_rr[:, None] + (gttdivJR_r - grtdivJR_t) * psin_r[:, None]

    out_G[:] = alpha1 * G1n + alpha2 * G2n


@register_residual_block("h")
def assemble_h_residual_block(
    out_packed: np.ndarray,
    coeff_indices: np.ndarray,
    G: np.ndarray,
    psin_R: np.ndarray,
    psin_Z: np.ndarray,
    sin_tb: np.ndarray,
    sin_theta: np.ndarray,
    cos_theta: np.ndarray,
    sin_2theta: np.ndarray,
    rho: np.ndarray,
    rho2: np.ndarray,
    y: np.ndarray,
    T: np.ndarray,
    weights: np.ndarray,
    a: float,
    R0: float,
    B0: float,
) -> None:
    """组装 h 通道 residual block."""
    del psin_Z, sin_tb, sin_theta, cos_theta, sin_2theta, rho, rho2, R0, B0
    collapsed = _collapse_g_field(G, psin_R)
    _scale_and_project_rows_two(out_packed, coeff_indices, T, collapsed, y, weights, (2.0 * np.pi / G.shape[1]) * a)


@register_residual_block("v")
def assemble_v_residual_block(
    out_packed: np.ndarray,
    coeff_indices: np.ndarray,
    G: np.ndarray,
    psin_R: np.ndarray,
    psin_Z: np.ndarray,
    sin_tb: np.ndarray,
    sin_theta: np.ndarray,
    cos_theta: np.ndarray,
    sin_2theta: np.ndarray,
    rho: np.ndarray,
    rho2: np.ndarray,
    y: np.ndarray,
    T: np.ndarray,
    weights: np.ndarray,
    a: float,
    R0: float,
    B0: float,
) -> None:
    """组装 v 通道 residual block."""
    del psin_R, sin_tb, sin_theta, cos_theta, sin_2theta, rho, rho2, R0, B0
    collapsed = _collapse_g_field(G, psin_Z)
    _scale_and_project_rows_two(out_packed, coeff_indices, T, collapsed, y, weights, (2.0 * np.pi / G.shape[1]) * a)


@register_residual_block("k")
def assemble_k_residual_block(
    out_packed: np.ndarray,
    coeff_indices: np.ndarray,
    G: np.ndarray,
    psin_R: np.ndarray,
    psin_Z: np.ndarray,
    sin_tb: np.ndarray,
    sin_theta: np.ndarray,
    cos_theta: np.ndarray,
    sin_2theta: np.ndarray,
    rho: np.ndarray,
    rho2: np.ndarray,
    y: np.ndarray,
    T: np.ndarray,
    weights: np.ndarray,
    a: float,
    R0: float,
    B0: float,
) -> None:
    """组装 k 通道 residual block."""
    del psin_R, sin_tb, cos_theta, sin_2theta, rho2, R0, B0
    collapsed = _collapse_g_field_theta(G, psin_Z, sin_theta)
    _scale_and_project_rows_three(
        out_packed, coeff_indices, T, collapsed, rho, y, weights, (2.0 * np.pi / G.shape[1]) * (-a)
    )


@register_residual_block("c0")
def assemble_c0_residual_block(
    out_packed: np.ndarray,
    coeff_indices: np.ndarray,
    G: np.ndarray,
    psin_R: np.ndarray,
    psin_Z: np.ndarray,
    sin_tb: np.ndarray,
    sin_theta: np.ndarray,
    cos_theta: np.ndarray,
    sin_2theta: np.ndarray,
    rho: np.ndarray,
    rho2: np.ndarray,
    y: np.ndarray,
    T: np.ndarray,
    weights: np.ndarray,
    a: float,
    R0: float,
    B0: float,
) -> None:
    """组装 c0 通道 residual block."""
    del psin_Z, sin_theta, cos_theta, sin_2theta, rho2, R0, B0
    collapsed = _collapse_g_two_fields(G, psin_R, sin_tb)
    _scale_and_project_rows_three(
        out_packed, coeff_indices, T, collapsed, rho, y, weights, (2.0 * np.pi / G.shape[1]) * (-a)
    )


@register_residual_block("c1")
def assemble_c1_residual_block(
    out_packed: np.ndarray,
    coeff_indices: np.ndarray,
    G: np.ndarray,
    psin_R: np.ndarray,
    psin_Z: np.ndarray,
    sin_tb: np.ndarray,
    sin_theta: np.ndarray,
    cos_theta: np.ndarray,
    sin_2theta: np.ndarray,
    rho: np.ndarray,
    rho2: np.ndarray,
    y: np.ndarray,
    T: np.ndarray,
    weights: np.ndarray,
    a: float,
    R0: float,
    B0: float,
) -> None:
    """组装 c1 通道 residual block."""
    del psin_Z, sin_theta, sin_2theta, rho, R0, B0
    collapsed = _collapse_g_two_fields_theta(G, psin_R, sin_tb, cos_theta)
    _scale_and_project_rows_three(
        out_packed, coeff_indices, T, collapsed, rho2, y, weights, (2.0 * np.pi / G.shape[1]) * (-a)
    )


@register_residual_block("s1")
def assemble_s1_residual_block(
    out_packed: np.ndarray,
    coeff_indices: np.ndarray,
    G: np.ndarray,
    psin_R: np.ndarray,
    psin_Z: np.ndarray,
    sin_tb: np.ndarray,
    sin_theta: np.ndarray,
    cos_theta: np.ndarray,
    sin_2theta: np.ndarray,
    rho: np.ndarray,
    rho2: np.ndarray,
    y: np.ndarray,
    T: np.ndarray,
    weights: np.ndarray,
    a: float,
    R0: float,
    B0: float,
) -> None:
    """组装 s1 通道 residual block."""
    del psin_Z, cos_theta, sin_2theta, rho, R0, B0
    collapsed = _collapse_g_two_fields_theta(G, psin_R, sin_tb, sin_theta)
    _scale_and_project_rows_three(
        out_packed, coeff_indices, T, collapsed, rho2, y, weights, (2.0 * np.pi / G.shape[1]) * (-a)
    )


@register_residual_block("s2")
def assemble_s2_residual_block(
    out_packed: np.ndarray,
    coeff_indices: np.ndarray,
    G: np.ndarray,
    psin_R: np.ndarray,
    psin_Z: np.ndarray,
    sin_tb: np.ndarray,
    sin_theta: np.ndarray,
    cos_theta: np.ndarray,
    sin_2theta: np.ndarray,
    rho: np.ndarray,
    rho2: np.ndarray,
    y: np.ndarray,
    T: np.ndarray,
    weights: np.ndarray,
    a: float,
    R0: float,
    B0: float,
) -> None:
    """组装 s2 通道 residual block."""
    del psin_Z, sin_theta, cos_theta, R0, B0
    collapsed = _collapse_g_two_fields_theta(G, psin_R, sin_tb, sin_2theta)
    _scale_and_project_rows_four(
        out_packed, coeff_indices, T, collapsed, rho, rho2, y, weights, (2.0 * np.pi / G.shape[1]) * (-a)
    )


@register_residual_block("psin")
def assemble_psin_residual_block(
    out_packed: np.ndarray,
    coeff_indices: np.ndarray,
    G: np.ndarray,
    psin_R: np.ndarray,
    psin_Z: np.ndarray,
    sin_tb: np.ndarray,
    sin_theta: np.ndarray,
    cos_theta: np.ndarray,
    sin_2theta: np.ndarray,
    rho: np.ndarray,
    rho2: np.ndarray,
    y: np.ndarray,
    T: np.ndarray,
    weights: np.ndarray,
    a: float,
    R0: float,
    B0: float,
) -> None:
    """组装 psin 通道 residual block."""
    del psin_R, psin_Z, sin_tb, sin_theta, cos_theta, sin_2theta, rho, a, R0, B0
    collapsed = _collapse_g(G)
    _scale_and_project_rows_three(out_packed, coeff_indices, T, collapsed, rho2, y, weights, 2.0 * np.pi / G.shape[1])


@register_residual_block("F")
def assemble_F_residual_block(
    out_packed: np.ndarray,
    coeff_indices: np.ndarray,
    G: np.ndarray,
    psin_R: np.ndarray,
    psin_Z: np.ndarray,
    sin_tb: np.ndarray,
    sin_theta: np.ndarray,
    cos_theta: np.ndarray,
    sin_2theta: np.ndarray,
    rho: np.ndarray,
    rho2: np.ndarray,
    y: np.ndarray,
    T: np.ndarray,
    weights: np.ndarray,
    a: float,
    R0: float,
    B0: float,
) -> None:
    """组装 F 通道 residual block."""
    del psin_R, psin_Z, sin_tb, sin_theta, cos_theta, sin_2theta, rho, rho2, a
    collapsed = _collapse_g(G)
    _scale_and_project_rows_three(
        out_packed, coeff_indices, T, collapsed, y, y, weights, (2.0 * np.pi / G.shape[1]) * (R0 * B0)
    )


def _collapse_g(G: np.ndarray) -> np.ndarray:
    return np.sum(G, axis=1)


def _collapse_g_field(G: np.ndarray, field: np.ndarray) -> np.ndarray:
    return np.einsum("ij,ij->i", G, field)


def _collapse_g_field_theta(G: np.ndarray, field: np.ndarray, theta_weight: np.ndarray) -> np.ndarray:
    return np.einsum("ij,ij,j->i", G, field, theta_weight)


def _collapse_g_two_fields(G: np.ndarray, field_a: np.ndarray, field_b: np.ndarray) -> np.ndarray:
    return np.einsum("ij,ij,ij->i", G, field_a, field_b)


def _collapse_g_two_fields_theta(
    G: np.ndarray,
    field_a: np.ndarray,
    field_b: np.ndarray,
    theta_weight: np.ndarray,
) -> np.ndarray:
    return np.einsum("ij,ij,ij,j->i", G, field_a, field_b, theta_weight)


def _project_rows_to_packed(
    out_packed: np.ndarray,
    coeff_indices: np.ndarray,
    T: np.ndarray,
    weighted_rho: np.ndarray,
) -> None:
    out_packed[coeff_indices] = T[: coeff_indices.shape[0]] @ weighted_rho


def _scale_and_project_rows_two(
    out_packed: np.ndarray,
    coeff_indices: np.ndarray,
    T: np.ndarray,
    collapsed: np.ndarray,
    weight_a: np.ndarray,
    weight_b: np.ndarray,
    scalar: float,
) -> None:
    collapsed *= weight_a * weight_b * scalar
    _project_rows_to_packed(out_packed, coeff_indices, T, collapsed)


def _scale_and_project_rows_three(
    out_packed: np.ndarray,
    coeff_indices: np.ndarray,
    T: np.ndarray,
    collapsed: np.ndarray,
    weight_a: np.ndarray,
    weight_b: np.ndarray,
    weight_c: np.ndarray,
    scalar: float,
) -> None:
    collapsed *= weight_a * weight_b * weight_c * scalar
    _project_rows_to_packed(out_packed, coeff_indices, T, collapsed)


def _scale_and_project_rows_four(
    out_packed: np.ndarray,
    coeff_indices: np.ndarray,
    T: np.ndarray,
    collapsed: np.ndarray,
    weight_a: np.ndarray,
    weight_b: np.ndarray,
    weight_c: np.ndarray,
    weight_d: np.ndarray,
    scalar: float,
) -> None:
    collapsed *= weight_a * weight_b * weight_c * weight_d * scalar
    _project_rows_to_packed(out_packed, coeff_indices, T, collapsed)

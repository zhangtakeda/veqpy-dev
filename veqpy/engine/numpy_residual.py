"""
Module: engine.numpy_residual

Role:
- 负责在 numpy backend 下生成 residual fields.
- 负责把 residual blocks 组装成 packed residual.

Public API:
- update_residual
- bind_residual_runner
- bind_residual_stage_runner

Notes:
- 这个文件同时作为 residual field update 与 packed residual assembly 的 vectorized reference.
"""

from dataclasses import dataclass
from typing import Callable

import numpy as np

from veqpy.residual_blocks import F2_BLOCK_CODE, decode_residual_block_kind


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


def _decode_residual_block(name: str) -> tuple[str, int, Callable | None]:
    kind, order, fixed_name = decode_residual_block_kind(name)
    if kind != "fixed":
        return (kind, order, None)
    try:
        if fixed_name is None:
            raise KeyError(name)
        return ("fixed", 0, RESIDUAL_BLOCK_REGISTRY[fixed_name].implementation)
    except KeyError as exc:
        supported = ", ".join(sorted(RESIDUAL_BLOCK_REGISTRY))
        raise KeyError(f"Unknown residual block {name!r}. Supported blocks: {supported}, c<k>, s<k>") from exc


def _decode_residual_block_code(code: int, order: int) -> tuple[str, int, Callable | None]:
    code_int = int(code)
    if code_int == 0:
        return ("fixed", 0, RESIDUAL_BLOCK_REGISTRY["h"].implementation)
    if code_int == 1:
        return ("fixed", 0, RESIDUAL_BLOCK_REGISTRY["v"].implementation)
    if code_int == 2:
        return ("fixed", 0, RESIDUAL_BLOCK_REGISTRY["k"].implementation)
    if code_int == 3:
        return ("fixed", 0, RESIDUAL_BLOCK_REGISTRY["c0"].implementation)
    if code_int == 4:
        return ("c_family", int(order), None)
    if code_int == 5:
        return ("s_family", int(order), None)
    if code_int == 6:
        return ("fixed", 0, RESIDUAL_BLOCK_REGISTRY["psin"].implementation)
    if code_int == 7:
        return ("fixed", 0, RESIDUAL_BLOCK_REGISTRY["F"].implementation)
    if code_int == F2_BLOCK_CODE:
        return ("fixed", 0, RESIDUAL_BLOCK_REGISTRY["F2"].implementation)
    raise KeyError(f"Unknown residual block code {code_int}")


def bind_residual_runner(
    profile_names: tuple[str, ...],
    coeff_index_rows: np.ndarray,
    lengths: np.ndarray,
    residual_size: int,
    *,
    block_names: tuple[str, ...] | None = None,
    block_codes: np.ndarray | None = None,
    block_orders: np.ndarray | None = None,
) -> Callable:
    specs: list[tuple[str, int, Callable | None]] = []
    if block_codes is not None and block_orders is not None:
        for code, order in zip(block_codes, block_orders, strict=True):
            specs.append(_decode_residual_block_code(int(code), int(order)))
    else:
        names_for_decode = profile_names if block_names is None else block_names
        for name in names_for_decode:
            specs.append(_decode_residual_block(name))
    bound_specs = tuple(specs)

    def runner(
        G: np.ndarray,
        psin_R: np.ndarray,
        psin_Z: np.ndarray,
        sin_tb: np.ndarray,
        sin_ktheta: np.ndarray,
        cos_ktheta: np.ndarray,
        rho_powers: np.ndarray,
        y: np.ndarray,
        T: np.ndarray,
        weights: np.ndarray,
        a: float,
        R0: float,
        B0: float,
    ) -> np.ndarray:
        out = np.zeros(residual_size, dtype=np.float64)
        for slot, (kind, order, kernel) in enumerate(bound_specs):
            coeff_size = int(lengths[slot])
            coeff_indices = coeff_index_rows[slot, :coeff_size]
            if kind == "fixed":
                kernel(
                    out,
                    coeff_indices,
                    G,
                    psin_R,
                    psin_Z,
                    sin_tb,
                    sin_ktheta,
                    cos_ktheta,
                    rho_powers,
                    y,
                    T,
                    weights,
                    a,
                    R0,
                    B0,
                )
            elif kind == "c_family":
                assemble_c_family_residual_block(
                    out,
                    coeff_indices,
                    order,
                    G,
                    psin_R,
                    sin_tb,
                    cos_ktheta,
                    rho_powers,
                    y,
                    T,
                    weights,
                    a,
                )
            else:
                assemble_s_family_residual_block(
                    out,
                    coeff_indices,
                    order,
                    G,
                    psin_R,
                    sin_tb,
                    sin_ktheta,
                    rho_powers,
                    y,
                    T,
                    weights,
                    a,
                )
        return out

    return runner


def bind_residual_stage_runner(
    profile_names: tuple[str, ...],
    coeff_index_rows: np.ndarray,
    lengths: np.ndarray,
    residual_size: int,
    *,
    block_names: tuple[str, ...] | None = None,
    block_codes: np.ndarray | None = None,
    block_orders: np.ndarray | None = None,
) -> Callable:
    specs: list[tuple[str, int, Callable | None]] = []
    if block_codes is not None and block_orders is not None:
        for code, order in zip(block_codes, block_orders, strict=True):
            specs.append(_decode_residual_block_code(int(code), int(order)))
    else:
        names_for_decode = profile_names if block_names is None else block_names
        for name in names_for_decode:
            specs.append(_decode_residual_block(name))
    bound_specs = tuple(specs)

    def runner(
        out_packed: np.ndarray,
        out_fields: np.ndarray,
        alpha1: float,
        alpha2: float,
        root_fields: np.ndarray,
        R_fields: np.ndarray,
        Z_fields: np.ndarray,
        J_fields: np.ndarray,
        g_fields: np.ndarray,
        sin_tb: np.ndarray,
        sin_ktheta: np.ndarray,
        cos_ktheta: np.ndarray,
        rho_powers: np.ndarray,
        y: np.ndarray,
        T: np.ndarray,
        weights: np.ndarray,
        a: float,
        R0: float,
        B0: float,
    ) -> np.ndarray:
        if out_packed.ndim != 1 or out_packed.shape[0] != residual_size:
            raise ValueError(f"Expected out_packed to have shape ({residual_size},), got {out_packed.shape}")
        update_residual(out_fields, alpha1, alpha2, root_fields, R_fields, Z_fields, J_fields, g_fields)
        out_packed.fill(0.0)
        G = out_fields[2]
        psin_R = out_fields[0]
        psin_Z = out_fields[1]
        for slot, (kind, order, kernel) in enumerate(bound_specs):
            coeff_size = int(lengths[slot])
            coeff_indices = coeff_index_rows[slot, :coeff_size]
            if kind == "fixed":
                kernel(
                    out_packed,
                    coeff_indices,
                    G,
                    psin_R,
                    psin_Z,
                    sin_tb,
                    sin_ktheta,
                    cos_ktheta,
                    rho_powers,
                    y,
                    T,
                    weights,
                    a,
                    R0,
                    B0,
                )
            elif kind == "c_family":
                assemble_c_family_residual_block(
                    out_packed,
                    coeff_indices,
                    order,
                    G,
                    psin_R,
                    sin_tb,
                    cos_ktheta,
                    rho_powers,
                    y,
                    T,
                    weights,
                    a,
                )
            else:
                assemble_s_family_residual_block(
                    out_packed,
                    coeff_indices,
                    order,
                    G,
                    psin_R,
                    sin_tb,
                    sin_ktheta,
                    rho_powers,
                    y,
                    T,
                    weights,
                    a,
                )
        return out_packed

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

    psin_r = root_fields[1]
    psin_rr = root_fields[2]
    FFn_psin = root_fields[3]
    Pn_psin = root_fields[4]

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

    G1n = JdivR * (FFn_psin[:, None] + R**2 * Pn_psin[:, None])
    G2n = gttdivJR * psin_rr[:, None] + (gttdivJR_r - grtdivJR_t) * psin_r[:, None]

    out_G[:] = alpha1 * G1n + alpha2 * G2n


def assemble_c_family_residual_block(
    out_packed: np.ndarray,
    coeff_indices: np.ndarray,
    order: int,
    G: np.ndarray,
    psin_R: np.ndarray,
    sin_tb: np.ndarray,
    cos_ktheta: np.ndarray,
    rho_powers: np.ndarray,
    y: np.ndarray,
    T: np.ndarray,
    weights: np.ndarray,
    a: float,
) -> None:
    collapsed = _collapse_g_two_fields_theta(G, psin_R, sin_tb, cos_ktheta[order])
    _scale_and_project_rows_three(
        out_packed,
        coeff_indices,
        T,
        collapsed,
        rho_powers[order + 1],
        y,
        weights,
        (2.0 * np.pi / G.shape[1]) * (-a),
    )


def assemble_s_family_residual_block(
    out_packed: np.ndarray,
    coeff_indices: np.ndarray,
    order: int,
    G: np.ndarray,
    psin_R: np.ndarray,
    sin_tb: np.ndarray,
    sin_ktheta: np.ndarray,
    rho_powers: np.ndarray,
    y: np.ndarray,
    T: np.ndarray,
    weights: np.ndarray,
    a: float,
) -> None:
    collapsed = _collapse_g_two_fields_theta(G, psin_R, sin_tb, sin_ktheta[order])
    _scale_and_project_rows_three(
        out_packed,
        coeff_indices,
        T,
        collapsed,
        rho_powers[order + 1],
        y,
        weights,
        (2.0 * np.pi / G.shape[1]) * (-a),
    )


@register_residual_block("h")
def assemble_h_residual_block(
    out_packed: np.ndarray,
    coeff_indices: np.ndarray,
    G: np.ndarray,
    psin_R: np.ndarray,
    psin_Z: np.ndarray,
    sin_tb: np.ndarray,
    sin_ktheta: np.ndarray,
    cos_ktheta: np.ndarray,
    rho_powers: np.ndarray,
    y: np.ndarray,
    T: np.ndarray,
    weights: np.ndarray,
    a: float,
    R0: float,
    B0: float,
) -> None:
    """组装 h 通道 residual block."""
    del psin_Z, sin_tb, sin_ktheta, cos_ktheta, rho_powers, R0, B0
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
    sin_ktheta: np.ndarray,
    cos_ktheta: np.ndarray,
    rho_powers: np.ndarray,
    y: np.ndarray,
    T: np.ndarray,
    weights: np.ndarray,
    a: float,
    R0: float,
    B0: float,
) -> None:
    """组装 v 通道 residual block."""
    del psin_R, sin_tb, sin_ktheta, cos_ktheta, rho_powers, R0, B0
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
    sin_ktheta: np.ndarray,
    cos_ktheta: np.ndarray,
    rho_powers: np.ndarray,
    y: np.ndarray,
    T: np.ndarray,
    weights: np.ndarray,
    a: float,
    R0: float,
    B0: float,
) -> None:
    """组装 k 通道 residual block."""
    del psin_R, sin_tb, cos_ktheta, R0, B0
    collapsed = _collapse_g_field_theta(G, psin_Z, sin_ktheta[1])
    _scale_and_project_rows_three(
        out_packed, coeff_indices, T, collapsed, rho_powers[1], y, weights, (2.0 * np.pi / G.shape[1]) * (-a)
    )


@register_residual_block("c0")
def assemble_c0_residual_block(
    out_packed: np.ndarray,
    coeff_indices: np.ndarray,
    G: np.ndarray,
    psin_R: np.ndarray,
    psin_Z: np.ndarray,
    sin_tb: np.ndarray,
    sin_ktheta: np.ndarray,
    cos_ktheta: np.ndarray,
    rho_powers: np.ndarray,
    y: np.ndarray,
    T: np.ndarray,
    weights: np.ndarray,
    a: float,
    R0: float,
    B0: float,
) -> None:
    """组装 c0 通道 residual block."""
    del psin_Z, sin_ktheta, cos_ktheta, R0, B0
    collapsed = _collapse_g_two_fields(G, psin_R, sin_tb)
    _scale_and_project_rows_three(
        out_packed, coeff_indices, T, collapsed, rho_powers[1], y, weights, (2.0 * np.pi / G.shape[1]) * (-a)
    )


@register_residual_block("c1")
def assemble_c1_residual_block(
    out_packed: np.ndarray,
    coeff_indices: np.ndarray,
    G: np.ndarray,
    psin_R: np.ndarray,
    psin_Z: np.ndarray,
    sin_tb: np.ndarray,
    sin_ktheta: np.ndarray,
    cos_ktheta: np.ndarray,
    rho_powers: np.ndarray,
    y: np.ndarray,
    T: np.ndarray,
    weights: np.ndarray,
    a: float,
    R0: float,
    B0: float,
) -> None:
    """组装 c1 通道 residual block."""
    del psin_Z, sin_ktheta, R0, B0
    assemble_c_family_residual_block(
        out_packed, coeff_indices, 1, G, psin_R, sin_tb, cos_ktheta, rho_powers, y, T, weights, a
    )


@register_residual_block("s1")
def assemble_s1_residual_block(
    out_packed: np.ndarray,
    coeff_indices: np.ndarray,
    G: np.ndarray,
    psin_R: np.ndarray,
    psin_Z: np.ndarray,
    sin_tb: np.ndarray,
    sin_ktheta: np.ndarray,
    cos_ktheta: np.ndarray,
    rho_powers: np.ndarray,
    y: np.ndarray,
    T: np.ndarray,
    weights: np.ndarray,
    a: float,
    R0: float,
    B0: float,
) -> None:
    """组装 s1 通道 residual block."""
    del psin_Z, cos_ktheta, R0, B0
    assemble_s_family_residual_block(
        out_packed, coeff_indices, 1, G, psin_R, sin_tb, sin_ktheta, rho_powers, y, T, weights, a
    )


@register_residual_block("s2")
def assemble_s2_residual_block(
    out_packed: np.ndarray,
    coeff_indices: np.ndarray,
    G: np.ndarray,
    psin_R: np.ndarray,
    psin_Z: np.ndarray,
    sin_tb: np.ndarray,
    sin_ktheta: np.ndarray,
    cos_ktheta: np.ndarray,
    rho_powers: np.ndarray,
    y: np.ndarray,
    T: np.ndarray,
    weights: np.ndarray,
    a: float,
    R0: float,
    B0: float,
) -> None:
    """组装 s2 通道 residual block."""
    del psin_Z, cos_ktheta, R0, B0
    assemble_s_family_residual_block(
        out_packed, coeff_indices, 2, G, psin_R, sin_tb, sin_ktheta, rho_powers, y, T, weights, a
    )


@register_residual_block("psin")
def assemble_psin_residual_block(
    out_packed: np.ndarray,
    coeff_indices: np.ndarray,
    G: np.ndarray,
    psin_R: np.ndarray,
    psin_Z: np.ndarray,
    sin_tb: np.ndarray,
    sin_ktheta: np.ndarray,
    cos_ktheta: np.ndarray,
    rho_powers: np.ndarray,
    y: np.ndarray,
    T: np.ndarray,
    weights: np.ndarray,
    a: float,
    R0: float,
    B0: float,
) -> None:
    """组装 psin 通道 residual block."""
    del psin_R, psin_Z, sin_tb, sin_ktheta, cos_ktheta, a, R0, B0
    collapsed = _collapse_g(G)
    _scale_and_project_rows_three(
        out_packed, coeff_indices, T, collapsed, rho_powers[2], y, weights, 2.0 * np.pi / G.shape[1]
    )


@register_residual_block("F")
def assemble_F_residual_block(
    out_packed: np.ndarray,
    coeff_indices: np.ndarray,
    G: np.ndarray,
    psin_R: np.ndarray,
    psin_Z: np.ndarray,
    sin_tb: np.ndarray,
    sin_ktheta: np.ndarray,
    cos_ktheta: np.ndarray,
    rho_powers: np.ndarray,
    y: np.ndarray,
    T: np.ndarray,
    weights: np.ndarray,
    a: float,
    R0: float,
    B0: float,
) -> None:
    """组装 F 通道 residual block."""
    del psin_R, psin_Z, sin_tb, sin_ktheta, cos_ktheta, rho_powers, a
    collapsed = _collapse_g(G)
    _scale_and_project_rows_three(
        out_packed, coeff_indices, T, collapsed, y, y, weights, (2.0 * np.pi / G.shape[1]) * (R0 * B0)
    )


@register_residual_block("F2")
def assemble_F2_residual_block(
    out_packed: np.ndarray,
    coeff_indices: np.ndarray,
    G: np.ndarray,
    psin_R: np.ndarray,
    psin_Z: np.ndarray,
    sin_tb: np.ndarray,
    sin_ktheta: np.ndarray,
    cos_ktheta: np.ndarray,
    rho_powers: np.ndarray,
    y: np.ndarray,
    T: np.ndarray,
    weights: np.ndarray,
    a: float,
    R0: float,
    B0: float,
) -> None:
    """组装 F2-linear 通道 residual block."""
    del psin_R, psin_Z, sin_tb, sin_ktheta, cos_ktheta, rho_powers, a
    collapsed = _collapse_g(G)
    _scale_and_project_rows_two(
        out_packed,
        coeff_indices,
        T,
        collapsed,
        y,
        weights,
        (2.0 * np.pi / G.shape[1]) * (R0 * B0) * (R0 * B0),
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

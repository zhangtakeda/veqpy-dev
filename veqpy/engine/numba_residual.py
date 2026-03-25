"""
Module: engine.numba_residual

Role:
- 负责在 numba backend 下生成 residual fields.
- 负责把 residual blocks 组装成 packed residual.

Public API:
- update_residual
- bind_residual_runner

Notes:
- residual block registration 保留在本模块内.
- operator 层只负责 bind 并调用 residual runner.
"""

from dataclasses import dataclass
from typing import Callable

import numpy as np
from numba import njit


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


_BLOCK_CODE_BY_NAME = {
    "h": 0,
    "v": 1,
    "k": 2,
    "c0": 3,
    "c1": 4,
    "s1": 5,
    "s2": 6,
    "psin": 7,
    "F": 8,
}


def bind_residual_runner(
    profile_names: tuple[str, ...],
    coeff_index_rows: np.ndarray,
    lengths: np.ndarray,
    residual_size: int,
) -> Callable:
    block_codes = np.empty(len(profile_names), dtype=np.int64)
    for i, name in enumerate(profile_names):
        if name not in RESIDUAL_BLOCK_REGISTRY:
            supported = ", ".join(RESIDUAL_BLOCK_REGISTRY)
            raise KeyError(f"Unknown residual block {name!r}. Supported blocks: {supported}")
        try:
            block_codes[i] = _BLOCK_CODE_BY_NAME[name]
        except KeyError as exc:
            supported = ", ".join(_BLOCK_CODE_BY_NAME)
            raise KeyError(f"Unknown residual block {name!r}. Supported blocks: {supported}") from exc

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
        _run_residual_blocks_packed(
            out,
            block_codes,
            coeff_index_rows,
            lengths,
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


@njit(cache=True, fastmath=True, nogil=True)
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

    nr, nt = out_G.shape
    for i in range(nr):
        psin_r_safe = psin_r[i]
        if psin_r_safe < 1e-10:
            psin_r_safe = 1e-10
        for j in range(nt):
            inv_J = 1.0 / J[i, j]
            psin_R = -Z_t[i, j] * inv_J * psin_r[i]
            psin_Z = R_t[i, j] * inv_J * psin_r[i]
            out_psin_R[i, j] = psin_R
            out_psin_Z[i, j] = psin_Z

            G1n = JdivR[i, j] * (FFn_r[i] + R[i, j] * R[i, j] * Pn_r[i]) / psin_r_safe
            G2n = gttdivJR[i, j] * psin_rr[i] + (gttdivJR_r[i, j] - grtdivJR_t[i, j]) * psin_r[i]
            out_G[i, j] = alpha1 * G1n + alpha2 * G2n


@register_residual_block("h")
@njit(cache=True, fastmath=True, nogil=True)
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
    nr = G.shape[0]
    collapsed = np.empty(nr, dtype=G.dtype)
    _collapse_g_field(collapsed, G, psin_R)
    _scale_and_project_rows_two(out_packed, coeff_indices, T, collapsed, y, weights, (2.0 * np.pi / G.shape[1]) * a)


@register_residual_block("v")
@njit(cache=True, fastmath=True, nogil=True)
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
    nr = G.shape[0]
    collapsed = np.empty(nr, dtype=G.dtype)
    _collapse_g_field(collapsed, G, psin_Z)
    _scale_and_project_rows_two(out_packed, coeff_indices, T, collapsed, y, weights, (2.0 * np.pi / G.shape[1]) * a)


@register_residual_block("k")
@njit(cache=True, fastmath=True, nogil=True)
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
    nr, nt = G.shape
    collapsed = np.empty(nr, dtype=G.dtype)
    _collapse_g_field_theta(collapsed, G, psin_Z, sin_theta)
    _scale_and_project_rows_three(out_packed, coeff_indices, T, collapsed, rho, y, weights, (2.0 * np.pi / nt) * (-a))


@register_residual_block("c0")
@njit(cache=True, fastmath=True, nogil=True)
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
    nr, nt = G.shape
    collapsed = np.empty(nr, dtype=G.dtype)
    _collapse_g_two_fields(collapsed, G, psin_R, sin_tb)
    _scale_and_project_rows_three(out_packed, coeff_indices, T, collapsed, rho, y, weights, (2.0 * np.pi / nt) * (-a))


@register_residual_block("c1")
@njit(cache=True, fastmath=True, nogil=True)
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
    nr, nt = G.shape
    collapsed = np.empty(nr, dtype=G.dtype)
    _collapse_g_two_fields_theta(collapsed, G, psin_R, sin_tb, cos_theta)
    _scale_and_project_rows_three(out_packed, coeff_indices, T, collapsed, rho2, y, weights, (2.0 * np.pi / nt) * (-a))


@register_residual_block("s1")
@njit(cache=True, fastmath=True, nogil=True)
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
    nr, nt = G.shape
    collapsed = np.empty(nr, dtype=G.dtype)
    _collapse_g_two_fields_theta(collapsed, G, psin_R, sin_tb, sin_theta)
    _scale_and_project_rows_three(out_packed, coeff_indices, T, collapsed, rho2, y, weights, (2.0 * np.pi / nt) * (-a))


@register_residual_block("s2")
@njit(cache=True, fastmath=True, nogil=True)
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
    nr, nt = G.shape
    collapsed = np.empty(nr, dtype=G.dtype)
    _collapse_g_two_fields_theta(collapsed, G, psin_R, sin_tb, sin_2theta)
    _scale_and_project_rows_four(
        out_packed, coeff_indices, T, collapsed, rho, rho2, y, weights, (2.0 * np.pi / nt) * (-a)
    )


@register_residual_block("psin")
@njit(cache=True, fastmath=True, nogil=True)
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
    nr, nt = G.shape
    collapsed = np.empty(nr, dtype=G.dtype)
    _collapse_g(collapsed, G)
    _scale_and_project_rows_three(out_packed, coeff_indices, T, collapsed, rho2, y, weights, 2.0 * np.pi / nt)


@register_residual_block("F")
@njit(cache=True, fastmath=True, nogil=True)
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
    nr, nt = G.shape
    collapsed = np.empty(nr, dtype=G.dtype)
    _collapse_g(collapsed, G)
    _scale_and_project_rows_three(
        out_packed, coeff_indices, T, collapsed, y, y, weights, (2.0 * np.pi / nt) * (R0 * B0)
    )


@njit(cache=True, fastmath=True, nogil=True)
def _collapse_g(out: np.ndarray, G: np.ndarray) -> None:
    nr, nt = G.shape
    for i in range(nr):
        collapsed = 0.0
        for j in range(nt):
            collapsed += G[i, j]
        out[i] = collapsed


@njit(cache=True, fastmath=True, nogil=True)
def _collapse_g_field(out: np.ndarray, G: np.ndarray, field: np.ndarray) -> None:
    nr, nt = G.shape
    for i in range(nr):
        collapsed = 0.0
        for j in range(nt):
            collapsed += G[i, j] * field[i, j]
        out[i] = collapsed


@njit(cache=True, fastmath=True, nogil=True)
def _collapse_g_field_theta(out: np.ndarray, G: np.ndarray, field: np.ndarray, theta_weight: np.ndarray) -> None:
    nr, nt = G.shape
    for i in range(nr):
        collapsed = 0.0
        for j in range(nt):
            collapsed += G[i, j] * field[i, j] * theta_weight[j]
        out[i] = collapsed


@njit(cache=True, fastmath=True, nogil=True)
def _collapse_g_two_fields(out: np.ndarray, G: np.ndarray, field_a: np.ndarray, field_b: np.ndarray) -> None:
    nr, nt = G.shape
    for i in range(nr):
        collapsed = 0.0
        for j in range(nt):
            collapsed += G[i, j] * field_a[i, j] * field_b[i, j]
        out[i] = collapsed


@njit(cache=True, fastmath=True, nogil=True)
def _collapse_g_two_fields_theta(
    out: np.ndarray,
    G: np.ndarray,
    field_a: np.ndarray,
    field_b: np.ndarray,
    theta_weight: np.ndarray,
) -> None:
    nr, nt = G.shape
    for i in range(nr):
        collapsed = 0.0
        for j in range(nt):
            collapsed += G[i, j] * field_a[i, j] * field_b[i, j] * theta_weight[j]
        out[i] = collapsed


@njit(cache=True, fastmath=True, nogil=True)
def _project_rows_to_packed(
    out_packed: np.ndarray,
    coeff_indices: np.ndarray,
    T: np.ndarray,
    weighted_rho: np.ndarray,
) -> None:
    rows = coeff_indices.shape[0]
    cols = weighted_rho.shape[0]
    for i in range(rows):
        total = 0.0
        for j in range(cols):
            total += T[i, j] * weighted_rho[j]
        out_packed[coeff_indices[i]] = total


@njit(cache=True, fastmath=True, nogil=True)
def _scale_and_project_rows_two(
    out_packed: np.ndarray,
    coeff_indices: np.ndarray,
    T: np.ndarray,
    collapsed: np.ndarray,
    weight_a: np.ndarray,
    weight_b: np.ndarray,
    scalar: float,
) -> None:
    for i in range(collapsed.shape[0]):
        collapsed[i] *= weight_a[i] * weight_b[i] * scalar
    _project_rows_to_packed(out_packed, coeff_indices, T, collapsed)


@njit(cache=True, fastmath=True, nogil=True)
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
    for i in range(collapsed.shape[0]):
        collapsed[i] *= weight_a[i] * weight_b[i] * weight_c[i] * scalar
    _project_rows_to_packed(out_packed, coeff_indices, T, collapsed)


@njit(cache=True, fastmath=True, nogil=True)
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
    for i in range(collapsed.shape[0]):
        collapsed[i] *= weight_a[i] * weight_b[i] * weight_c[i] * weight_d[i] * scalar
    _project_rows_to_packed(out_packed, coeff_indices, T, collapsed)


@njit(cache=True, fastmath=True, nogil=True)
def _run_residual_blocks_packed(
    out_packed: np.ndarray,
    block_codes: np.ndarray,
    coeff_index_rows: np.ndarray,
    lengths: np.ndarray,
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
    for slot in range(block_codes.shape[0]):
        coeff_indices = coeff_index_rows[slot, : lengths[slot]]
        code = block_codes[slot]
        if code == 0:
            assemble_h_residual_block(
                out_packed,
                coeff_indices,
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
        elif code == 1:
            assemble_v_residual_block(
                out_packed,
                coeff_indices,
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
        elif code == 2:
            assemble_k_residual_block(
                out_packed,
                coeff_indices,
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
        elif code == 3:
            assemble_c0_residual_block(
                out_packed,
                coeff_indices,
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
        elif code == 4:
            assemble_c1_residual_block(
                out_packed,
                coeff_indices,
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
        elif code == 5:
            assemble_s1_residual_block(
                out_packed,
                coeff_indices,
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
        elif code == 6:
            assemble_s2_residual_block(
                out_packed,
                coeff_indices,
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
        elif code == 7:
            assemble_psin_residual_block(
                out_packed,
                coeff_indices,
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
        else:
            assemble_F_residual_block(
                out_packed,
                coeff_indices,
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

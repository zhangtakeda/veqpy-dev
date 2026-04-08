"""
Module: engine.numba_residual

Role:
- 负责在 numba backend 下生成 residual fields.
- 负责把 residual blocks 组装成 packed residual.

Public API:
- update_residual
- bind_residual_runner
- bind_residual_stage_runner

Notes:
- residual block registration 保留在本模块内.
- operator 层只负责 bind 并调用 residual runner.
"""

from dataclasses import dataclass
from typing import Callable

import numpy as np
from numba import njit

from veqpy.residual_blocks import decode_residual_block_code


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


def _decode_residual_block_code(name: str) -> tuple[int, int]:
    if not (name.startswith(("c", "s")) and name[1:].isdigit()) and name not in RESIDUAL_BLOCK_REGISTRY:
        supported = ", ".join(sorted(RESIDUAL_BLOCK_REGISTRY))
        raise KeyError(f"Unknown residual block {name!r}. Supported blocks: {supported}, c<k>, s<k>")
    return decode_residual_block_code(name)


def bind_residual_runner(
    profile_names: tuple[str, ...],
    coeff_index_rows: np.ndarray,
    lengths: np.ndarray,
    residual_size: int,
    *,
    block_codes: np.ndarray | None = None,
    block_orders: np.ndarray | None = None,
) -> Callable:
    if block_codes is None or block_orders is None:
        block_codes = np.empty(len(profile_names), dtype=np.int64)
        block_orders = np.zeros(len(profile_names), dtype=np.int64)
        for i, name in enumerate(profile_names):
            block_codes[i], block_orders[i] = _decode_residual_block_code(name)
    else:
        block_codes = np.asarray(block_codes, dtype=np.int64)
        block_orders = np.asarray(block_orders, dtype=np.int64)
    scratch_holder: list[np.ndarray | None] = [None]

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
        scratch = scratch_holder[0]
        if scratch is None or scratch.shape[0] != G.shape[0]:
            scratch = np.empty(G.shape[0], dtype=np.float64)
            scratch_holder[0] = scratch
        _run_residual_blocks_packed(
            out,
            scratch,
            block_codes,
            block_orders,
            coeff_index_rows,
            lengths,
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
        return out

    return runner


def bind_residual_stage_runner(
    profile_names: tuple[str, ...],
    coeff_index_rows: np.ndarray,
    lengths: np.ndarray,
    residual_size: int,
    *,
    block_codes: np.ndarray | None = None,
    block_orders: np.ndarray | None = None,
) -> Callable:
    if block_codes is None or block_orders is None:
        block_codes = np.empty(len(profile_names), dtype=np.int64)
        block_orders = np.zeros(len(profile_names), dtype=np.int64)
        for i, name in enumerate(profile_names):
            block_codes[i], block_orders[i] = _decode_residual_block_code(name)
    else:
        block_codes = np.asarray(block_codes, dtype=np.int64)
        block_orders = np.asarray(block_orders, dtype=np.int64)
    scratch_holder: list[np.ndarray | None] = [None]

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
        scratch = scratch_holder[0]
        nr = out_fields.shape[1]
        if scratch is None or scratch.shape[0] != nr:
            scratch = np.empty(nr, dtype=np.float64)
            scratch_holder[0] = scratch
        _run_residual_blocks_packed(
            out_packed,
            scratch,
            block_codes,
            block_orders,
            coeff_index_rows,
            lengths,
            out_fields[2],
            out_fields[0],
            out_fields[1],
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
        return out_packed

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

    nr, nt = out_G.shape
    for i in range(nr):
        for j in range(nt):
            inv_J = 1.0 / J[i, j]
            psin_R = -Z_t[i, j] * inv_J * psin_r[i]
            psin_Z = R_t[i, j] * inv_J * psin_r[i]
            out_psin_R[i, j] = psin_R
            out_psin_Z[i, j] = psin_Z

            G1n = JdivR[i, j] * (FFn_psin[i] + R[i, j] * R[i, j] * Pn_psin[i])
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
    scratch: np.ndarray,
) -> None:
    """组装 h 通道 residual block."""
    _collapse_g_field(scratch, G, psin_R)
    _scale_and_project_rows_two(out_packed, coeff_indices, T, scratch, y, weights, (2.0 * np.pi / G.shape[1]) * a)


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
    scratch: np.ndarray,
) -> None:
    """组装 v 通道 residual block."""
    _collapse_g_field(scratch, G, psin_Z)
    _scale_and_project_rows_two(out_packed, coeff_indices, T, scratch, y, weights, (2.0 * np.pi / G.shape[1]) * a)


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
    scratch: np.ndarray,
) -> None:
    """组装 k 通道 residual block."""
    nt = G.shape[1]
    _collapse_g_field_theta(scratch, G, psin_Z, sin_theta)
    _scale_and_project_rows_three(out_packed, coeff_indices, T, scratch, rho, y, weights, (2.0 * np.pi / nt) * (-a))


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
    scratch: np.ndarray,
) -> None:
    """组装 c0 通道 residual block."""
    nt = G.shape[1]
    _collapse_g_two_fields(scratch, G, psin_R, sin_tb)
    _scale_and_project_rows_three(out_packed, coeff_indices, T, scratch, rho, y, weights, (2.0 * np.pi / nt) * (-a))


@njit(cache=True, fastmath=True, nogil=True)
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
    scratch: np.ndarray,
) -> None:
    nt = G.shape[1]
    _collapse_g_two_fields_theta(scratch, G, psin_R, sin_tb, cos_ktheta[order])
    _scale_and_project_rows_three(
        out_packed,
        coeff_indices,
        T,
        scratch,
        rho_powers[order + 1],
        y,
        weights,
        (2.0 * np.pi / nt) * (-a),
    )


@njit(cache=True, fastmath=True, nogil=True)
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
    scratch: np.ndarray,
) -> None:
    nt = G.shape[1]
    _collapse_g_two_fields_theta(scratch, G, psin_R, sin_tb, sin_ktheta[order])
    _scale_and_project_rows_three(
        out_packed,
        coeff_indices,
        T,
        scratch,
        rho_powers[order + 1],
        y,
        weights,
        (2.0 * np.pi / nt) * (-a),
    )


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
    scratch: np.ndarray,
) -> None:
    """组装 c1 通道 residual block."""
    nt = G.shape[1]
    _collapse_g_two_fields_theta(scratch, G, psin_R, sin_tb, cos_theta)
    _scale_and_project_rows_three(out_packed, coeff_indices, T, scratch, rho2, y, weights, (2.0 * np.pi / nt) * (-a))


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
    scratch: np.ndarray,
) -> None:
    """组装 s1 通道 residual block."""
    nt = G.shape[1]
    _collapse_g_two_fields_theta(scratch, G, psin_R, sin_tb, sin_theta)
    _scale_and_project_rows_three(out_packed, coeff_indices, T, scratch, rho2, y, weights, (2.0 * np.pi / nt) * (-a))


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
    scratch: np.ndarray,
) -> None:
    """组装 s2 通道 residual block."""
    nt = G.shape[1]
    _collapse_g_two_fields_theta(scratch, G, psin_R, sin_tb, sin_2theta)
    _scale_and_project_rows_four(
        out_packed, coeff_indices, T, scratch, rho, rho2, y, weights, (2.0 * np.pi / nt) * (-a)
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
    scratch: np.ndarray,
) -> None:
    """组装 psin 通道 residual block."""
    nt = G.shape[1]
    _collapse_g(scratch, G)
    _scale_and_project_rows_three(out_packed, coeff_indices, T, scratch, rho2, y, weights, 2.0 * np.pi / nt)


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
    scratch: np.ndarray,
) -> None:
    """组装 F 通道 residual block."""
    nt = G.shape[1]
    _collapse_g(scratch, G)
    _scale_and_project_rows_three(out_packed, coeff_indices, T, scratch, y, y, weights, (2.0 * np.pi / nt) * (R0 * B0))


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
    scratch: np.ndarray,
    block_codes: np.ndarray,
    block_orders: np.ndarray,
    coeff_index_rows: np.ndarray,
    lengths: np.ndarray,
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
    sin_theta = sin_ktheta[1]
    cos_theta = cos_ktheta[1]
    sin_2theta = sin_ktheta[2]
    rho = rho_powers[1]
    rho2 = rho_powers[2]
    for slot in range(block_codes.shape[0]):
        coeff_indices = coeff_index_rows[slot, : lengths[slot]]
        code = block_codes[slot]
        order = block_orders[slot]
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
                scratch,
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
                scratch,
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
                scratch,
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
                scratch,
            )
        elif code == 4:
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
                scratch,
            )
        elif code == 5:
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
                scratch,
            )
        elif code == 6:
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
                scratch,
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
                scratch,
            )

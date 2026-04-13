"""
Module: engine.numba_residual

Role:
- 负责更新 residual surface workspace.
- 负责把预计算 residual 场组装成 packed residual.

Public API:
- update_residual_compact

Notes:
- 保留的是 numba hot path 所需最小接口.
- 旧 staged/binder residual API 已移除.
"""

import numpy as np
from numba import njit


@njit(cache=True, fastmath=True, nogil=True)
def update_residual_compact(
    out_workspace: np.ndarray,
    alpha1: float,
    alpha2: float,
    root_fields: np.ndarray,
    geometry_surface_workspace: np.ndarray,
) -> None:
    """使用 compact geometry fields 原地更新 residual 相关二维 fields."""
    out_G = out_workspace[0]
    out_Gpsin_R = out_workspace[1]
    out_Gpsin_Z = out_workspace[2]
    out_Gpsin_R_sin_tb = out_workspace[3]
    sin_tb_surface = geometry_surface_workspace[0]
    R_surface = geometry_surface_workspace[1]
    R_t_surface = geometry_surface_workspace[2]
    Z_t_surface = geometry_surface_workspace[3]
    J_surface = geometry_surface_workspace[4]
    JdivR_surface = geometry_surface_workspace[5]
    grtdivJR_t_surface = geometry_surface_workspace[6]
    gttdivJR_surface = geometry_surface_workspace[7]
    gttdivJR_r_surface = geometry_surface_workspace[8]

    psin_r = root_fields[1]
    psin_rr = root_fields[2]
    FFn_psin = root_fields[3]
    Pn_psin = root_fields[4]

    nr, nt = out_G.shape
    for i in range(nr):
        psin_r_i = psin_r[i]
        psin_rr_i = psin_rr[i]
        FFn_psin_i = FFn_psin[i]
        Pn_psin_i = Pn_psin[i]
        for j in range(nt):
            inv_J = 1.0 / J_surface[i, j]
            psin_R = -Z_t_surface[i, j] * inv_J * psin_r_i
            psin_Z = R_t_surface[i, j] * inv_J * psin_r_i

            R_ij = R_surface[i, j]
            G1n = JdivR_surface[i, j] * (FFn_psin_i + R_ij * R_ij * Pn_psin_i)
            G2n = gttdivJR_surface[i, j] * psin_rr_i + (gttdivJR_r_surface[i, j] - grtdivJR_t_surface[i, j]) * psin_r_i
            G_ij = alpha1 * G1n + alpha2 * G2n
            out_G[i, j] = G_ij
            Gpsin_R = G_ij * psin_R
            out_Gpsin_R[i, j] = Gpsin_R
            out_Gpsin_Z[i, j] = G_ij * psin_Z
            out_Gpsin_R_sin_tb[i, j] = Gpsin_R * sin_tb_surface[i, j]


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
def _project_scaled2(
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
def _project_scaled3(
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
def _collapse_g(out: np.ndarray, G: np.ndarray) -> None:
    nr, nt = G.shape
    for i in range(nr):
        collapsed = 0.0
        for j in range(nt):
            collapsed += G[i, j]
        out[i] = collapsed


@njit(cache=True, fastmath=True, nogil=True)
def _collapse_field(out: np.ndarray, field: np.ndarray) -> None:
    nr, nt = field.shape
    for i in range(nr):
        collapsed = 0.0
        for j in range(nt):
            collapsed += field[i, j]
        out[i] = collapsed


@njit(cache=True, fastmath=True, nogil=True)
def _collapse_field_theta(out: np.ndarray, field: np.ndarray, theta_weight: np.ndarray) -> None:
    nr, nt = field.shape
    for i in range(nr):
        collapsed = 0.0
        for j in range(nt):
            collapsed += field[i, j] * theta_weight[j]
        out[i] = collapsed


@njit(cache=True, fastmath=True, nogil=True)
def _run_residual_blocks_packed_precomputed(
    out_packed: np.ndarray,
    scratch: np.ndarray,
    block_codes: np.ndarray,
    block_orders: np.ndarray,
    coeff_index_rows: np.ndarray,
    lengths: np.ndarray,
    residual_workspace: np.ndarray,
    sin_ktheta: np.ndarray,
    cos_ktheta: np.ndarray,
    rho_powers: np.ndarray,
    y: np.ndarray,
    T_fields: np.ndarray,
    weights: np.ndarray,
    a: float,
    R0: float,
    B0: float,
) -> None:
    G = residual_workspace[0]
    Gpsin_R = residual_workspace[1]
    Gpsin_Z = residual_workspace[2]
    Gpsin_R_sin_tb = residual_workspace[3]
    T = T_fields[0]
    sin_theta = sin_ktheta[1]
    rho = rho_powers[1]
    rho2 = rho_powers[2]
    nt = G.shape[1]
    base_scale = 2.0 * np.pi / nt
    for slot in range(block_codes.shape[0]):
        coeff_indices = coeff_index_rows[slot, : lengths[slot]]
        code = block_codes[slot]
        order = block_orders[slot]
        if code == 0:
            _collapse_field(scratch, Gpsin_R)
            _project_scaled2(out_packed, coeff_indices, T, scratch, y, weights, base_scale * a)
        elif code == 1:
            _collapse_field(scratch, Gpsin_Z)
            _project_scaled2(out_packed, coeff_indices, T, scratch, y, weights, base_scale * a)
        elif code == 2:
            _collapse_field_theta(scratch, Gpsin_Z, sin_theta)
            _project_scaled3(out_packed, coeff_indices, T, scratch, rho, y, weights, base_scale * (-a))
        elif code == 3:
            _collapse_field(scratch, Gpsin_R_sin_tb)
            _project_scaled3(out_packed, coeff_indices, T, scratch, rho, y, weights, base_scale * (-a))
        elif code == 4:
            _collapse_field_theta(scratch, Gpsin_R_sin_tb, cos_ktheta[order])
            _project_scaled3(
                out_packed, coeff_indices, T, scratch, rho_powers[order + 1], y, weights, base_scale * (-a)
            )
        elif code == 5:
            _collapse_field_theta(scratch, Gpsin_R_sin_tb, sin_ktheta[order])
            _project_scaled3(
                out_packed, coeff_indices, T, scratch, rho_powers[order + 1], y, weights, base_scale * (-a)
            )
        elif code == 6:
            _collapse_g(scratch, G)
            _project_scaled3(out_packed, coeff_indices, T, scratch, rho2, y, weights, base_scale)
        elif code == 7:
            _collapse_g(scratch, G)
            _project_scaled3(out_packed, coeff_indices, T, scratch, y, y, weights, base_scale * (R0 * B0))
        elif code == 8:
            _collapse_g(scratch, G)
            _project_scaled3(
                out_packed,
                coeff_indices,
                T,
                scratch,
                y,
                y,
                weights,
                base_scale * (R0 * B0) * (R0 * B0),
            )
        else:
            raise ValueError("Unknown residual block code")

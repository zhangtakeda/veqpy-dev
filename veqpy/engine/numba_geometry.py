"""
Module: engine.numba_geometry

Role:
- 负责在 numba backend 下物化 geometry fields.
- 负责同步更新 geometry integrals.

Public API:
- update_geometry

Notes:
- 输入和输出都采用 packed fields 语义.
- backend dispatch 与 operator staging 不在这里处理.
"""

import numpy as np
from numba import njit


@njit(cache=True, fastmath=True, nogil=True)
def update_geometry(
    tb_fields: np.ndarray,
    R_fields: np.ndarray,
    Z_fields: np.ndarray,
    J_fields: np.ndarray,
    g_fields: np.ndarray,
    S_r: np.ndarray,
    V_r: np.ndarray,
    Kn: np.ndarray,
    Kn_r: np.ndarray,
    Ln_r: np.ndarray,
    a: float,
    R0: float,
    Z0: float,
    rho: np.ndarray,
    theta: np.ndarray,
    cos_ktheta: np.ndarray,
    sin_ktheta: np.ndarray,
    k_cos_ktheta: np.ndarray,
    k_sin_ktheta: np.ndarray,
    k2_cos_ktheta: np.ndarray,
    k2_sin_ktheta: np.ndarray,
    weights: np.ndarray,
    h_fields: np.ndarray,
    v_fields: np.ndarray,
    k_fields: np.ndarray,
    c_fields: np.ndarray,
    s_fields: np.ndarray,
):
    """原地更新 geometry fields 与 geometry integrals."""
    nr = rho.shape[0]
    nt = theta.shape[0]
    theta_scale = 2.0 * np.pi / nt
    mean_scale = 1.0 / nt
    two_pi = 2.0 * np.pi
    c_limit = min(c_fields.shape[0], cos_ktheta.shape[0])
    s_limit = min(s_fields.shape[0], sin_ktheta.shape[0])

    for i in range(nr):
        rho_i = rho[i]
        h_i = h_fields[0, i]
        h_r_i = h_fields[1, i]
        h_rr_i = h_fields[2, i]
        v_i = v_fields[0, i]
        v_r_i = v_fields[1, i]
        v_rr_i = v_fields[2, i]
        k_i = k_fields[0, i]
        k_r_i = k_fields[1, i]
        k_rr_i = k_fields[2, i]
        c0_i = c_fields[0, 0, i]
        c0_r_i = c_fields[0, 1, i]
        c0_rr_i = c_fields[0, 2, i]

        sum_J = 0.0
        sum_JR = 0.0
        sum_gttdivJR = 0.0
        sum_gttdivJR_r = 0.0
        sum_JdivR = 0.0

        for j in range(nt):
            sin_t = sin_ktheta[1, j]
            cos_t = cos_ktheta[1, j]

            tb_ij = theta[j] + c0_i
            tb_r_ij = c0_r_i
            tb_t_ij = 1.0
            tb_rr_ij = c0_rr_i
            tb_rt_ij = 0.0
            tb_tt_ij = 0.0

            for order in range(1, c_limit):
                cos_kt = cos_ktheta[order, j]
                k_sin_kt = k_sin_ktheta[order, j]
                k2_cos_kt = k2_cos_ktheta[order, j]
                c_i = c_fields[order, 0, i]
                c_r_i = c_fields[order, 1, i]
                c_rr_i = c_fields[order, 2, i]

                tb_ij += c_i * cos_kt
                tb_r_ij += c_r_i * cos_kt
                tb_t_ij -= c_i * k_sin_kt
                tb_rr_ij += c_rr_i * cos_kt
                tb_rt_ij -= c_r_i * k_sin_kt
                tb_tt_ij -= c_i * k2_cos_kt

            for order in range(1, s_limit):
                sin_kt = sin_ktheta[order, j]
                k_cos_kt = k_cos_ktheta[order, j]
                k2_sin_kt = k2_sin_ktheta[order, j]
                s_i = s_fields[order, 0, i]
                s_r_i = s_fields[order, 1, i]
                s_rr_i = s_fields[order, 2, i]

                tb_ij += s_i * sin_kt
                tb_r_ij += s_r_i * sin_kt
                tb_t_ij += s_i * k_cos_kt
                tb_rr_ij += s_rr_i * sin_kt
                tb_rt_ij += s_r_i * k_cos_kt
                tb_tt_ij -= s_i * k2_sin_kt

            cos_tb_ij = np.cos(tb_ij)
            sin_tb_ij = np.sin(tb_ij)

            R_ij = R0 + a * (h_i + rho_i * cos_tb_ij)
            R_r_ij = a * (h_r_i + cos_tb_ij - rho_i * sin_tb_ij * tb_r_ij)
            R_t_ij = -a * rho_i * sin_tb_ij * tb_t_ij
            R_rr_ij = a * (
                h_rr_i - 2.0 * sin_tb_ij * tb_r_ij - rho_i * (cos_tb_ij * tb_r_ij * tb_r_ij + sin_tb_ij * tb_rr_ij)
            )
            R_rt_ij = -a * (sin_tb_ij * tb_t_ij + rho_i * (cos_tb_ij * tb_r_ij * tb_t_ij + sin_tb_ij * tb_rt_ij))
            R_tt_ij = -a * rho_i * (cos_tb_ij * tb_t_ij * tb_t_ij + sin_tb_ij * tb_tt_ij)

            Z_ij = Z0 + a * (v_i - rho_i * k_i * sin_t)
            Z_r_ij = a * (v_r_i - (k_i + rho_i * k_r_i) * sin_t)
            Z_t_ij = -a * rho_i * k_i * cos_t
            Z_rr_ij = a * (v_rr_i - (2.0 * k_r_i + rho_i * k_rr_i) * sin_t)
            Z_rt_ij = -a * (k_i + rho_i * k_r_i) * cos_t
            Z_tt_ij = a * rho_i * k_i * sin_t

            J_ij = R_t_ij * Z_r_ij - R_r_ij * Z_t_ij
            if J_ij < 1e-15:
                J_ij = 1e-15

            J_r_ij = -(R_rr_ij * Z_t_ij - R_rt_ij * Z_r_ij + R_r_ij * Z_rt_ij - R_t_ij * Z_rr_ij)
            J_t_ij = -(R_rt_ij * Z_t_ij - R_tt_ij * Z_r_ij + R_r_ij * Z_tt_ij - R_t_ij * Z_rt_ij)
            JR_ij = J_ij * R_ij
            JR_r_ij = J_r_ij * R_ij + J_ij * R_r_ij
            JR_t_ij = J_t_ij * R_ij + J_ij * R_t_ij
            inv_R = 1.0 / R_ij
            JdivR_ij = J_ij * inv_R
            JdivR_r_ij = (J_r_ij * R_ij - J_ij * R_r_ij) * inv_R * inv_R

            grt_ij = R_r_ij * R_t_ij + Z_r_ij * Z_t_ij
            grt_t_ij = R_rt_ij * R_t_ij + R_r_ij * R_tt_ij + Z_rt_ij * Z_t_ij + Z_r_ij * Z_tt_ij
            gtt_ij = R_t_ij * R_t_ij + Z_t_ij * Z_t_ij
            gtt_r_ij = 2.0 * (R_t_ij * R_rt_ij + Z_t_ij * Z_rt_ij)
            inv_JR = 1.0 / JR_ij
            grtdivJR_t_ij = (grt_t_ij - grt_ij * JR_t_ij * inv_JR) * inv_JR
            gttdivJR_ij = gtt_ij * inv_JR
            gttdivJR_r_ij = gtt_r_ij * inv_JR - gtt_ij * JR_r_ij * inv_JR * inv_JR

            tb_fields[0, i, j] = tb_ij
            tb_fields[1, i, j] = tb_r_ij
            tb_fields[2, i, j] = tb_t_ij
            tb_fields[3, i, j] = tb_rr_ij
            tb_fields[4, i, j] = tb_rt_ij
            tb_fields[5, i, j] = tb_tt_ij
            tb_fields[6, i, j] = cos_tb_ij
            tb_fields[7, i, j] = sin_tb_ij

            R_fields[0, i, j] = R_ij
            R_fields[1, i, j] = R_r_ij
            R_fields[2, i, j] = R_t_ij
            R_fields[3, i, j] = R_rr_ij
            R_fields[4, i, j] = R_rt_ij
            R_fields[5, i, j] = R_tt_ij

            Z_fields[0, i, j] = Z_ij
            Z_fields[1, i, j] = Z_r_ij
            Z_fields[2, i, j] = Z_t_ij
            Z_fields[3, i, j] = Z_rr_ij
            Z_fields[4, i, j] = Z_rt_ij
            Z_fields[5, i, j] = Z_tt_ij

            J_fields[0, i, j] = J_ij
            J_fields[1, i, j] = J_r_ij
            J_fields[2, i, j] = J_t_ij

            J_fields[3, i, j] = JR_ij
            J_fields[4, i, j] = JR_r_ij
            J_fields[5, i, j] = JR_t_ij
            J_fields[6, i, j] = JdivR_ij
            J_fields[7, i, j] = JdivR_r_ij

            g_fields[0, i, j] = grt_ij
            g_fields[1, i, j] = grt_t_ij
            g_fields[2, i, j] = grtdivJR_t_ij
            g_fields[3, i, j] = gtt_ij
            g_fields[4, i, j] = gtt_r_ij
            g_fields[5, i, j] = gttdivJR_ij
            g_fields[6, i, j] = gttdivJR_r_ij

            sum_J += J_ij
            sum_JR += JR_ij
            sum_gttdivJR += gttdivJR_ij
            sum_gttdivJR_r += gttdivJR_r_ij
            sum_JdivR += JdivR_ij

        S_r[i] = sum_J * theta_scale
        V_r[i] = sum_JR * theta_scale * two_pi
        Kn[i] = sum_gttdivJR * mean_scale
        Kn_r[i] = sum_gttdivJR_r * mean_scale
        Ln_r[i] = sum_JdivR * mean_scale

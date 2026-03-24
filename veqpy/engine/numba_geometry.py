"""
engine 层 Numba 几何核.
负责以循环形式生成二维几何场和 theta 积分量, 与 NumPy 实现保持数值语义一致.
不负责 backend 选择, packed layout/codec, solver 状态编排.
"""

import numpy as np
from numba import njit


@njit(cache=True, fastmath=True, nogil=True)
def update_geometry(
    # 输出缓冲区
    tb: np.ndarray,
    cos_tb: np.ndarray,
    sin_tb: np.ndarray,
    tb_r: np.ndarray,
    tb_t: np.ndarray,
    tb_rr: np.ndarray,
    tb_rt: np.ndarray,
    tb_tt: np.ndarray,
    R: np.ndarray,
    R_r: np.ndarray,
    R_t: np.ndarray,
    R_rr: np.ndarray,
    R_rt: np.ndarray,
    R_tt: np.ndarray,
    Z: np.ndarray,
    Z_r: np.ndarray,
    Z_t: np.ndarray,
    Z_rr: np.ndarray,
    Z_rt: np.ndarray,
    Z_tt: np.ndarray,
    J: np.ndarray,
    J_r: np.ndarray,
    J_t: np.ndarray,
    JR: np.ndarray,
    JR_r: np.ndarray,
    JR_t: np.ndarray,
    JdivR: np.ndarray,
    JdivR_r: np.ndarray,
    grt: np.ndarray,
    grt_t: np.ndarray,
    gtt: np.ndarray,
    gtt_r: np.ndarray,
    gttdivJR: np.ndarray,
    gttdivJR_r: np.ndarray,
    grtdivJR_t: np.ndarray,
    S_r: np.ndarray,
    V_r: np.ndarray,
    Kn: np.ndarray,
    Kn_r: np.ndarray,
    Ln_r: np.ndarray,
    # 输入 profile 与网格
    a: float,
    R0: float,
    Z0: float,
    rho: np.ndarray,
    theta: np.ndarray,
    cos_theta: np.ndarray,
    sin_theta: np.ndarray,
    weights: np.ndarray,
    h: np.ndarray,
    h_r: np.ndarray,
    h_rr: np.ndarray,
    v: np.ndarray,
    v_r: np.ndarray,
    v_rr: np.ndarray,
    k: np.ndarray,
    k_r: np.ndarray,
    k_rr: np.ndarray,
    c0: np.ndarray,
    c0_r: np.ndarray,
    c0_rr: np.ndarray,
    c1: np.ndarray,
    c1_r: np.ndarray,
    c1_rr: np.ndarray,
    s1: np.ndarray,
    s1_r: np.ndarray,
    s1_rr: np.ndarray,
    s2: np.ndarray,
    s2_r: np.ndarray,
    s2_rr: np.ndarray,
):
    """
    原地更新几何场及其径向积分量.

    Args:
        tb, cos_tb, sin_tb, tb_r, tb_t, ..., gttdivJR_r, grtdivJR_t: 调用方持有的二维输出缓冲区, shape=(nr, nt).
        S_r, V_r, Kn, Kn_r, Ln_r: 调用方持有的一维输出缓冲区, shape=(nr,).
        a, R0, Z0: 几何尺度与平移参数, 单位与 R, Z 保持一致.
        rho, theta, cos_theta, sin_theta, weights: 径向和极向网格及求积权重, shape=(nr,) 或 (nt,).
        h, h_r, h_rr, v, v_r, v_rr, ...: 当前 grid 上的几何参数化 profile 及其一阶, 二阶导数, shape=(nr,).

    Returns:
        返回 None. 所有二维几何场与一维 theta 积分量都会原地写入调用方缓冲区.
    """
    nr = rho.shape[0]
    nt = theta.shape[0]
    theta_scale = 2.0 * np.pi / nt
    mean_scale = 1.0 / nt
    two_pi = 2.0 * np.pi

    for i in range(nr):
        rho_i = rho[i]
        h_i = h[i]
        h_r_i = h_r[i]
        h_rr_i = h_rr[i]
        v_i = v[i]
        v_r_i = v_r[i]
        v_rr_i = v_rr[i]
        k_i = k[i]
        k_r_i = k_r[i]
        k_rr_i = k_rr[i]
        c0_i = c0[i]
        c0_r_i = c0_r[i]
        c0_rr_i = c0_rr[i]
        c1_i = c1[i]
        c1_r_i = c1_r[i]
        c1_rr_i = c1_rr[i]
        s1_i = s1[i]
        s1_r_i = s1_r[i]
        s1_rr_i = s1_rr[i]
        s2_i = s2[i]
        s2_r_i = s2_r[i]
        s2_rr_i = s2_rr[i]

        sum_J = 0.0
        sum_JR = 0.0
        sum_gttdivJR = 0.0
        sum_gttdivJR_r = 0.0
        sum_JdivR = 0.0

        for j in range(nt):
            cos_t = cos_theta[j]
            sin_t = sin_theta[j]
            cos_2t = cos_t * cos_t - sin_t * sin_t
            sin_2t = 2.0 * sin_t * cos_t

            tb_ij = theta[j] + c0_i + c1_i * cos_t + s1_i * sin_t + s2_i * sin_2t
            tb_r_ij = c0_r_i + c1_r_i * cos_t + s1_r_i * sin_t + s2_r_i * sin_2t
            tb_t_ij = 1.0 - c1_i * sin_t + s1_i * cos_t + 2.0 * s2_i * cos_2t
            tb_rr_ij = c0_rr_i + c1_rr_i * cos_t + s1_rr_i * sin_t + s2_rr_i * sin_2t
            tb_rt_ij = -c1_r_i * sin_t + s1_r_i * cos_t + 2.0 * s2_r_i * cos_2t
            tb_tt_ij = -c1_i * cos_t - s1_i * sin_t - 4.0 * s2_i * sin_2t

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

            tb[i, j] = tb_ij
            tb_r[i, j] = tb_r_ij
            tb_t[i, j] = tb_t_ij
            tb_rr[i, j] = tb_rr_ij
            tb_rt[i, j] = tb_rt_ij
            tb_tt[i, j] = tb_tt_ij
            cos_tb[i, j] = cos_tb_ij
            sin_tb[i, j] = sin_tb_ij

            R[i, j] = R_ij
            R_r[i, j] = R_r_ij
            R_t[i, j] = R_t_ij
            R_rr[i, j] = R_rr_ij
            R_rt[i, j] = R_rt_ij
            R_tt[i, j] = R_tt_ij
            Z[i, j] = Z_ij
            Z_r[i, j] = Z_r_ij
            Z_t[i, j] = Z_t_ij
            Z_rr[i, j] = Z_rr_ij
            Z_rt[i, j] = Z_rt_ij
            Z_tt[i, j] = Z_tt_ij
            J[i, j] = J_ij
            J_r[i, j] = J_r_ij
            J_t[i, j] = J_t_ij
            JR[i, j] = JR_ij
            JR_r[i, j] = JR_r_ij
            JR_t[i, j] = JR_t_ij
            JdivR[i, j] = JdivR_ij
            JdivR_r[i, j] = JdivR_r_ij
            grt[i, j] = grt_ij
            grt_t[i, j] = grt_t_ij
            gtt[i, j] = gtt_ij
            gtt_r[i, j] = gtt_r_ij
            gttdivJR[i, j] = gttdivJR_ij
            gttdivJR_r[i, j] = gttdivJR_r_ij
            grtdivJR_t[i, j] = grtdivJR_t_ij

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

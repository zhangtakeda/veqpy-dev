"""
Module: engine.numpy_geometry

Role:
- 负责在 numpy backend 下物化 geometry fields.
- 负责同步更新 geometry integrals.

Public API:
- update_geometry

Notes:
- 输入和输出都采用 packed fields 语义.
- 这个文件同时作为 geometry stage 的 vectorized reference.
"""

import numpy as np


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
    tb = tb_fields[0]
    tb_r = tb_fields[1]
    tb_t = tb_fields[2]
    tb_rr = tb_fields[3]
    tb_rt = tb_fields[4]
    tb_tt = tb_fields[5]
    cos_tb = tb_fields[6]
    sin_tb = tb_fields[7]

    R = R_fields[0]
    R_r = R_fields[1]
    R_t = R_fields[2]
    R_rr = R_fields[3]
    R_rt = R_fields[4]
    R_tt = R_fields[5]

    Z = Z_fields[0]
    Z_r = Z_fields[1]
    Z_t = Z_fields[2]
    Z_rr = Z_fields[3]
    Z_rt = Z_fields[4]
    Z_tt = Z_fields[5]

    J = J_fields[0]
    J_r = J_fields[1]
    J_t = J_fields[2]
    JR = J_fields[3]
    JR_r = J_fields[4]
    JR_t = J_fields[5]
    JdivR = J_fields[6]
    JdivR_r = J_fields[7]
    grt = g_fields[0]
    grt_t = g_fields[1]
    grtdivJR_t = g_fields[2]
    gtt = g_fields[3]
    gtt_r = g_fields[4]
    gttdivJR = g_fields[5]
    gttdivJR_r = g_fields[6]

    rho_2d = rho[:, None]
    theta_2d = theta[None, :]
    cos_t = cos_ktheta[1][None, :]
    sin_t = sin_ktheta[1][None, :]

    h = h_fields[0]
    h_r = h_fields[1]
    h_rr = h_fields[2]
    v = v_fields[0]
    v_r = v_fields[1]
    v_rr = v_fields[2]
    k = k_fields[0]
    k_r = k_fields[1]
    k_rr = k_fields[2]
    c0 = c_fields[0]

    tb[:] = theta_2d + c0[0][:, None]
    tb_r[:] = c0[1][:, None]
    tb_t.fill(1.0)
    tb_rr[:] = c0[2][:, None]
    tb_rt.fill(0.0)
    tb_tt.fill(0.0)

    c_limit = min(c_fields.shape[0], cos_ktheta.shape[0])
    for order in range(1, c_limit):
        cos_kt = cos_ktheta[order][None, :]
        k_sin_kt = k_sin_ktheta[order][None, :]
        k2_cos_kt = k2_cos_ktheta[order][None, :]
        c = c_fields[order, 0][:, None]
        c_r = c_fields[order, 1][:, None]
        c_rr = c_fields[order, 2][:, None]

        tb += c * cos_kt
        tb_r += c_r * cos_kt
        tb_t -= c * k_sin_kt
        tb_rr += c_rr * cos_kt
        tb_rt -= c_r * k_sin_kt
        tb_tt -= c * k2_cos_kt

    s_limit = min(s_fields.shape[0], sin_ktheta.shape[0])
    for order in range(1, s_limit):
        sin_kt = sin_ktheta[order][None, :]
        k_cos_kt = k_cos_ktheta[order][None, :]
        k2_sin_kt = k2_sin_ktheta[order][None, :]
        s = s_fields[order, 0][:, None]
        s_r = s_fields[order, 1][:, None]
        s_rr = s_fields[order, 2][:, None]

        tb += s * sin_kt
        tb_r += s_r * sin_kt
        tb_t += s * k_cos_kt
        tb_rr += s_rr * sin_kt
        tb_rt += s_r * k_cos_kt
        tb_tt -= s * k2_sin_kt

    np.cos(tb, out=cos_tb)
    np.sin(tb, out=sin_tb)

    R[:] = R0 + a * (h[:, None] + rho_2d * cos_tb)
    R_r[:] = a * (h_r[:, None] + cos_tb - rho_2d * sin_tb * tb_r)
    R_t[:] = -a * rho_2d * sin_tb * tb_t
    R_rr[:] = a * (h_rr[:, None] - 2.0 * sin_tb * tb_r - rho_2d * (cos_tb * tb_r**2 + sin_tb * tb_rr))
    R_rt[:] = -a * (sin_tb * tb_t + rho_2d * (cos_tb * tb_r * tb_t + sin_tb * tb_rt))
    R_tt[:] = -a * rho_2d * (cos_tb * tb_t**2 + sin_tb * tb_tt)

    Z[:] = Z0 + a * (v[:, None] - rho_2d * k[:, None] * sin_t)
    Z_r[:] = a * (v_r[:, None] - (k[:, None] + rho_2d * k_r[:, None]) * sin_t)
    Z_t[:] = -a * rho_2d * k[:, None] * cos_t
    Z_rr[:] = a * (v_rr[:, None] - (2.0 * k_r[:, None] + rho_2d * k_rr[:, None]) * sin_t)
    Z_rt[:] = -a * (k[:, None] + rho_2d * k_r[:, None]) * cos_t
    Z_tt[:] = a * rho_2d * k[:, None] * sin_t

    J[:] = R_t * Z_r - R_r * Z_t
    np.maximum(J, 1e-15, out=J)
    J_r[:] = -(R_rr * Z_t - R_rt * Z_r + R_r * Z_rt - R_t * Z_rr)
    J_t[:] = -(R_rt * Z_t - R_tt * Z_r + R_r * Z_tt - R_t * Z_rt)
    JR[:] = J * R
    JR_r[:] = J_r * R + J * R_r
    JR_t[:] = J_t * R + J * R_t
    JdivR[:] = J / R
    JdivR_r[:] = (J_r * R - J * R_r) / (R * R)

    grt[:] = R_r * R_t + Z_r * Z_t
    grt_t[:] = R_rt * R_t + R_r * R_tt + Z_rt * Z_t + Z_r * Z_tt
    gtt[:] = R_t * R_t + Z_t * Z_t
    gtt_r[:] = 2.0 * (R_t * R_rt + Z_t * Z_rt)
    grtdivJR_t[:] = (grt_t - grt * JR_t / JR) / JR
    gttdivJR[:] = gtt / JR
    gttdivJR_r[:] = gtt_r / JR - gtt * JR_r / (JR * JR)

    _theta_integral(S_r, J, weights)
    _theta_integral(V_r, JR, weights)
    V_r *= 2.0 * np.pi

    _theta_integral(Kn, gttdivJR, weights)
    _theta_integral(Kn_r, gttdivJR_r, weights)
    _theta_integral(Ln_r, JdivR, weights)
    Kn /= 2.0 * np.pi
    Kn_r /= 2.0 * np.pi
    Ln_r /= 2.0 * np.pi


def _theta_integral(
    out: np.ndarray,
    arr: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    """按 theta 方向积分并写回 out."""
    np.sum(arr, axis=1, out=out)
    out *= 2.0 * np.pi / arr.shape[1]
    return out

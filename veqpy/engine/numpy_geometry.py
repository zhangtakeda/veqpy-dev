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
    cos_theta: np.ndarray,
    sin_theta: np.ndarray,
    cos_2theta: np.ndarray,
    sin_2theta: np.ndarray,
    weights: np.ndarray,
    h_fields: np.ndarray,
    v_fields: np.ndarray,
    k_fields: np.ndarray,
    c0_fields: np.ndarray,
    c1_fields: np.ndarray,
    s1_fields: np.ndarray,
    s2_fields: np.ndarray,
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
    cos_t = cos_theta[None, :]
    sin_t = sin_theta[None, :]
    cos_2t = cos_2theta[None, :]
    sin_2t = sin_2theta[None, :]

    h = h_fields[0]
    h_r = h_fields[1]
    h_rr = h_fields[2]
    v = v_fields[0]
    v_r = v_fields[1]
    v_rr = v_fields[2]
    k = k_fields[0]
    k_r = k_fields[1]
    k_rr = k_fields[2]
    c0 = c0_fields[0]
    c0_r = c0_fields[1]
    c0_rr = c0_fields[2]
    c1 = c1_fields[0]
    c1_r = c1_fields[1]
    c1_rr = c1_fields[2]
    s1 = s1_fields[0]
    s1_r = s1_fields[1]
    s1_rr = s1_fields[2]
    s2 = s2_fields[0]
    s2_r = s2_fields[1]
    s2_rr = s2_fields[2]

    tb[:] = theta_2d + c0[:, None]
    tb_r[:] = c0_r[:, None]
    tb_t.fill(1.0)
    tb_rr[:] = c0_rr[:, None]
    tb_rt.fill(0.0)
    tb_tt.fill(0.0)

    tb += c1[:, None] * cos_t
    tb_r += c1_r[:, None] * cos_t
    tb_t -= c1[:, None] * sin_t
    tb_rr += c1_rr[:, None] * cos_t
    tb_rt -= c1_r[:, None] * sin_t
    tb_tt -= c1[:, None] * cos_t

    tb += s1[:, None] * sin_t
    tb_r += s1_r[:, None] * sin_t
    tb_t += s1[:, None] * cos_t
    tb_rr += s1_rr[:, None] * sin_t
    tb_rt += s1_r[:, None] * cos_t
    tb_tt -= s1[:, None] * sin_t

    tb += s2[:, None] * sin_2t
    tb_r += s2_r[:, None] * sin_2t
    tb_t += 2.0 * s2[:, None] * cos_2t
    tb_rr += s2_rr[:, None] * sin_2t
    tb_rt += 2.0 * s2_r[:, None] * cos_2t
    tb_tt -= 4.0 * s2[:, None] * sin_2t

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

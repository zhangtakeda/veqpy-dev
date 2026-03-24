"""
engine 层 NumPy 几何核.
负责从径向参数化 profile 生成二维几何场和 theta 积分量.
不负责 backend 选择, packed layout/codec, solver 状态编排.
"""

import numpy as np


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
    """
    rho_2d = rho[:, None]
    theta_2d = theta[None, :]
    cos_t = cos_theta[None, :]
    sin_t = sin_theta[None, :]
    cos_2t = cos_t * cos_t - sin_t * sin_t
    sin_2t = 2.0 * sin_t * cos_t

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
    """按 theta 方向做全角积分并写回调用方缓冲区."""

    np.sum(arr, axis=1, out=out)
    out *= 2.0 * np.pi / arr.shape[1]
    return out

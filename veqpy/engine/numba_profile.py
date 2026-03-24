"""
engine 层 Numba profile 核.
负责以循环形式组合静态基函数, 输出 profile 值及其径向导数.
不负责 profile 参数求解, backend 选择, solver 状态管理.
"""

import numpy as np
from numba import njit


@njit(cache=True, fastmath=True, nogil=True)
def update_profile(
    out_u: np.ndarray,
    out_u_r: np.ndarray,
    out_u_rr: np.ndarray,
    T: np.ndarray,
    T_r: np.ndarray,
    T_rr: np.ndarray,
    rp: np.ndarray,
    rp_r: np.ndarray,
    rp_rr: np.ndarray,
    env: np.ndarray,
    env_r: np.ndarray,
    env_rr: np.ndarray,
    offset: float,
    coeff: np.ndarray | None,
) -> None:
    """
    原地更新一维 profile 值, 一阶导数和二阶导数.

    Args:
        out_u, out_u_r, out_u_rr: 调用方持有的输出缓冲区, shape=(nr,).
        T, T_r, T_rr: 基函数及其径向导数, shape=(n_basis, nr).
        rp, rp_r, rp_rr: 基础径向包络及其导数, shape=(nr,).
        env, env_r, env_rr: 系数调制包络及其导数, shape=(nr,).
        offset: profile 常数偏移项.
        coeff: profile 系数向量, shape=(n_active,). None 表示只保留 offset 项.

    Returns:
        返回 None. 所有结果都会原地写入 out_u, out_u_r, out_u_rr.
    """
    nr = out_u.shape[0]
    if coeff is None:
        for i in range(nr):
            out_u[i] = offset * rp[i]
            out_u_r[i] = offset * rp_r[i]
            out_u_rr[i] = offset * rp_rr[i]
        return

    coeff_size = coeff.size
    for i in range(nr):
        series = 0.0
        series_r = 0.0
        series_rr = 0.0
        for k in range(coeff_size):
            c = coeff[k]
            series += c * T[k, i]
            series_r += c * T_r[k, i]
            series_rr += c * T_rr[k, i]

        base = env[i] * series
        base_r = env_r[i] * series + env[i] * series_r
        base_rr = env_rr[i] * series + 2.0 * env_r[i] * series_r + env[i] * series_rr
        amp = offset + base

        out_u[i] = rp[i] * amp
        out_u_r[i] = rp_r[i] * amp + rp[i] * base_r
        out_u_rr[i] = rp_rr[i] * amp + 2.0 * rp_r[i] * base_r + rp[i] * base_rr

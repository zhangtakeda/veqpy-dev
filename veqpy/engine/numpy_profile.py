"""
engine 层 NumPy profile 核.
负责把静态基函数和系数组合成 profile 值及其径向导数.
不负责 profile 参数求解, backend 选择, solver 状态管理.
"""

import numpy as np


def update_profile(
    # 输出缓冲区
    out_u: np.ndarray,
    out_u_r: np.ndarray,
    out_u_rr: np.ndarray,
    # 输入 profile 项
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
    """
    if coeff is None:
        out_u[:] = offset * rp
        out_u_r[:] = offset * rp_r
        out_u_rr[:] = offset * rp_rr
        return

    T_slice = T[: coeff.size]
    T_r_slice = T_r[: coeff.size]
    T_rr_slice = T_rr[: coeff.size]

    series = coeff @ T_slice
    series_r = coeff @ T_r_slice
    series_rr = coeff @ T_rr_slice

    base = env * series
    base_r = env_r * series + env * series_r
    base_rr = env_rr * series + 2.0 * env_r * series_r + env * series_rr
    amp = offset + base

    out_u[:] = rp * amp
    out_u_r[:] = rp_r * amp + rp * base_r
    out_u_rr[:] = rp_rr * amp + 2.0 * rp_r * base_r + rp * base_rr

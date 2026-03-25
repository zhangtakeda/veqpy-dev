"""
engine 层 Numba profile 核.
负责以循环形式组合静态基函数, 输出 profile 值及其径向导数.
不负责 profile 参数求解, backend 选择, solver 状态管理.
"""

import numpy as np
from numba import njit


@njit(cache=True, fastmath=True, nogil=True)
def update_profile(
    out_fields: np.ndarray,
    T_fields: np.ndarray,
    rp_fields: np.ndarray,
    env_fields: np.ndarray,
    offset: float,
    coeff: np.ndarray | None,
) -> None:
    """
    原地更新一维 profile 值, 一阶导数和二阶导数.

    Args:
        out_fields: 调用方持有的输出缓冲区, shape=(3, nr).
        T_fields: 基函数及其径向导数表, shape=(3, n_basis, nr).
        rp_fields: 基础径向包络及其导数, shape=(3, nr).
        env_fields: 系数调制包络及其导数, shape=(3, nr).
        offset: profile 常数偏移项.
        coeff: profile 系数向量, shape=(n_active,). None 表示只保留 offset 项.
    """
    nr = out_fields.shape[1]

    if coeff is None:
        for i in range(nr):
            out_fields[0, i] = offset * rp_fields[0, i]
            out_fields[1, i] = offset * rp_fields[1, i]
            out_fields[2, i] = offset * rp_fields[2, i]
        return

    coeff_size = coeff.size
    for i in range(nr):
        series = 0.0
        series_r = 0.0
        series_rr = 0.0
        for k in range(coeff_size):
            c = coeff[k]
            series += c * T_fields[0, k, i]
            series_r += c * T_fields[1, k, i]
            series_rr += c * T_fields[2, k, i]

        env = env_fields[0, i]
        env_r = env_fields[1, i]
        env_rr = env_fields[2, i]
        base = env * series
        base_r = env_r * series + env * series_r
        base_rr = env_rr * series + 2.0 * env_r * series_r + env * series_rr
        amp = offset + base

        rp = rp_fields[0, i]
        rp_r = rp_fields[1, i]
        rp_rr = rp_fields[2, i]
        out_fields[0, i] = rp * amp
        out_fields[1, i] = rp_r * amp + rp * base_r
        out_fields[2, i] = rp_rr * amp + 2.0 * rp_r * base_r + rp * base_rr


@njit(cache=True, fastmath=True, nogil=True)
def update_profile_packed(
    out_fields: np.ndarray,
    T_fields: np.ndarray,
    rp_fields: np.ndarray,
    env_fields: np.ndarray,
    offset: float,
    x: np.ndarray,
    coeff_indices: np.ndarray,
) -> None:
    """
    直接从 packed 状态向量读取系数并更新一维 profile 值及其径向导数.

    Args:
        out_fields: 调用方持有的输出缓冲区, shape=(3, nr).
        T_fields: 基函数及其径向导数表, shape=(3, n_basis, nr).
        rp_fields: 基础径向包络及其导数, shape=(3, nr).
        env_fields: 系数调制包络及其导数, shape=(3, nr).
        offset: profile 常数偏移项.
        x: 当前 packed 状态向量.
        coeff_indices: 当前 profile 在 packed 状态中的索引向量. 空数组表示只保留 offset 项.
    """
    nr = out_fields.shape[1]

    if coeff_indices.size == 0:
        for i in range(nr):
            out_fields[0, i] = offset * rp_fields[0, i]
            out_fields[1, i] = offset * rp_fields[1, i]
            out_fields[2, i] = offset * rp_fields[2, i]
        return

    coeff_size = coeff_indices.size
    for i in range(nr):
        series = 0.0
        series_r = 0.0
        series_rr = 0.0
        for k in range(coeff_size):
            c = x[coeff_indices[k]]
            series += c * T_fields[0, k, i]
            series_r += c * T_fields[1, k, i]
            series_rr += c * T_fields[2, k, i]

        env = env_fields[0, i]
        env_r = env_fields[1, i]
        env_rr = env_fields[2, i]
        base = env * series
        base_r = env_r * series + env * series_r
        base_rr = env_rr * series + 2.0 * env_r * series_r + env * series_rr
        amp = offset + base

        rp = rp_fields[0, i]
        rp_r = rp_fields[1, i]
        rp_rr = rp_fields[2, i]
        out_fields[0, i] = rp * amp
        out_fields[1, i] = rp_r * amp + rp * base_r
        out_fields[2, i] = rp_rr * amp + 2.0 * rp_r * base_r + rp * base_rr


@njit(cache=True, fastmath=True, nogil=True)
def update_profiles_packed_bulk(
    out_fields_all: np.ndarray,
    T_fields: np.ndarray,
    rp_fields_all: np.ndarray,
    env_fields_all: np.ndarray,
    offsets: np.ndarray,
    scales: np.ndarray,
    x: np.ndarray,
    coeff_index_rows: np.ndarray,
    lengths: np.ndarray,
) -> None:
    """
    批量从 packed 状态向量读取所有 active profiles 的系数并刷新输出字段.

    Args:
        out_fields_all: 批量输出缓冲区, shape=(n_active, 3, nr).
        T_fields: 基函数及其导数表, shape=(3, n_basis, nr).
        rp_fields_all: 每个 active profile 的 power 缓存, shape=(n_active, 3, nr).
        env_fields_all: 每个 active profile 的 envelope 缓存, shape=(n_active, 3, nr).
        offsets: 每个 active profile 的 offset.
        scales: 每个 active profile 的 scale.
        x: 当前 packed 状态向量.
        coeff_index_rows: 每个 active profile 对应的 packed 索引行, shape=(n_active, max_len).
        lengths: 每个 active profile 实际使用的系数长度.
    """
    n_active = out_fields_all.shape[0]
    nr = out_fields_all.shape[2]

    for p in range(n_active):
        coeff_size = lengths[p]
        offset = offsets[p]
        scale = scales[p]

        for i in range(nr):
            series = 0.0
            series_r = 0.0
            series_rr = 0.0

            for k in range(coeff_size):
                c = x[coeff_index_rows[p, k]]
                series += c * T_fields[0, k, i]
                series_r += c * T_fields[1, k, i]
                series_rr += c * T_fields[2, k, i]

            env = env_fields_all[p, 0, i]
            env_r = env_fields_all[p, 1, i]
            env_rr = env_fields_all[p, 2, i]
            base = env * series
            base_r = env_r * series + env * series_r
            base_rr = env_rr * series + 2.0 * env_r * series_r + env * series_rr
            amp = offset + base

            rp = rp_fields_all[p, 0, i]
            rp_r = rp_fields_all[p, 1, i]
            rp_rr = rp_fields_all[p, 2, i]
            out_fields_all[p, 0, i] = scale * (rp * amp)
            out_fields_all[p, 1, i] = scale * (rp_r * amp + rp * base_r)
            out_fields_all[p, 2, i] = scale * (rp_rr * amp + 2.0 * rp_r * base_r + rp * base_rr)

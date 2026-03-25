"""
engine 层 NumPy profile 核.
负责把静态基函数和系数组合成 profile 值及其径向导数.
不负责 profile 参数求解, backend 选择, solver 状态管理.
"""

import numpy as np


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
    out_u = out_fields[0]
    out_u_r = out_fields[1]
    out_u_rr = out_fields[2]

    T = T_fields[0]
    T_r = T_fields[1]
    T_rr = T_fields[2]

    rp = rp_fields[0]
    rp_r = rp_fields[1]
    rp_rr = rp_fields[2]
    env = env_fields[0]
    env_r = env_fields[1]
    env_rr = env_fields[2]

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
    if coeff_indices.size == 0:
        update_profile(out_fields, T_fields, rp_fields, env_fields, offset, None)
        return
    coeff = np.asarray(x, dtype=np.float64)[coeff_indices]
    update_profile(out_fields, T_fields, rp_fields, env_fields, offset, coeff)


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
    x_arr = np.asarray(x, dtype=np.float64)
    n_active = out_fields_all.shape[0]
    for p in range(n_active):
        coeff_size = int(lengths[p])
        if coeff_size == 0:
            update_profile(
                out_fields_all[p],
                T_fields,
                rp_fields_all[p],
                env_fields_all[p],
                float(offsets[p]),
                None,
            )
        else:
            coeff = x_arr[coeff_index_rows[p, :coeff_size]]
            update_profile(
                out_fields_all[p],
                T_fields,
                rp_fields_all[p],
                env_fields_all[p],
                float(offsets[p]),
                coeff,
            )
        scale = float(scales[p])
        if scale != 1.0:
            np.multiply(out_fields_all[p], scale, out=out_fields_all[p])

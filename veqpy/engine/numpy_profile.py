"""
Module: engine.numpy_profile

Role:
- 负责在 numpy backend 下计算 profile fields.
- 输入使用 basis tables 与 profile coefficients.

Public API:
- update_profile
- update_profiles_packed_bulk

Notes:
- 这个文件同时作为 profile evaluation 的 vectorized reference.
- 这个文件同时作为 Stage-A bulk update 的 vectorized reference.
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
    """原地更新单个 profile 的 fields."""
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
    """批量从 packed x 刷新所有 active profile fields."""
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

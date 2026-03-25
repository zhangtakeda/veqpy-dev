"""
Module: engine.numba_profile

Role:
- 负责在 numba backend 下计算 profile fields.
- 输入使用 basis tables 与 profile coefficients.

Public API:
- update_profile
- update_profiles_packed_bulk

Notes:
- update_profile 用于单个 explicit-coeff profile.
- update_profiles_packed_bulk 用于 Stage-A 的 packed runtime 更新.
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
    """原地更新单个 profile 的 fields."""
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

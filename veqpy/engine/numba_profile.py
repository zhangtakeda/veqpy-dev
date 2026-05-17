"""
Module: engine.numba_profile

Role:
- Compute profile fields.
- Inputs use basis tables and profile coefficients.

Public API:
- update_profile
- update_profiles_packed_bulk

Notes:
- update_profile is used for one explicit-coefficient profile.
- update_profiles_packed_bulk is used for packed runtime updates in Stage A.
"""

from __future__ import annotations

import numpy as np
from numba import njit


@njit(cache=True, fastmath=True, nogil=True)
def update_profile(
    out_fields: np.ndarray,
    T: np.ndarray,
    T_r: np.ndarray,
    T_rr: np.ndarray,
    rp_fields: np.ndarray,
    env_fields: np.ndarray,
    offset: float,
    coeff: np.ndarray | None,
) -> None:
    """Update one profile field set in place."""
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
            series += c * T[k, i]
            series_r += c * T_r[k, i]
            series_rr += c * T_rr[k, i]

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
    active_u_fields: np.ndarray,
    active_rp_fields: np.ndarray,
    active_env_fields: np.ndarray,
    T: np.ndarray,
    T_r: np.ndarray,
    T_rr: np.ndarray,
    offsets: np.ndarray,
    scales: np.ndarray,
    x: np.ndarray,
    coeff_index_rows: np.ndarray,
    lengths: np.ndarray,
) -> None:
    """Refresh all active profile fields from packed x in bulk."""
    n_active = active_u_fields.shape[0]
    nr = active_u_fields.shape[2]

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
                series += c * T[k, i]
                series_r += c * T_r[k, i]
                series_rr += c * T_rr[k, i]

            env = active_env_fields[p, 0, i]
            env_r = active_env_fields[p, 1, i]
            env_rr = active_env_fields[p, 2, i]
            base = env * series
            base_r = env_r * series + env * series_r
            base_rr = env_rr * series + 2.0 * env_r * series_r + env * series_rr
            amp = offset + base

            rp = active_rp_fields[p, 0, i]
            rp_r = active_rp_fields[p, 1, i]
            rp_rr = active_rp_fields[p, 2, i]
            active_u_fields[p, 0, i] = scale * (rp * amp)
            active_u_fields[p, 1, i] = scale * (rp_r * amp + rp * base_r)
            active_u_fields[p, 2, i] = scale * (rp_rr * amp + 2.0 * rp_r * base_r + rp * base_rr)

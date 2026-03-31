"""
Module: operator.stage_helpers

Role:
- 收敛 profile/geometry stage 的共享 Python 装配逻辑.
- 避免 operator.py 混入过多 stage runtime 视图绑定细节.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from veqpy.engine import update_fourier_family_fields, update_geometry, update_profiles_packed_bulk
from veqpy.model import Profile


def refresh_stage_a_runtime(
    *,
    active_profile_ids: np.ndarray,
    profile_names: tuple[str, ...],
    profiles_by_name: dict[str, Profile],
    profile_L: np.ndarray,
    coeff_index: np.ndarray,
    active_u_fields: np.ndarray,
    active_rp_fields: np.ndarray,
    active_env_fields: np.ndarray,
    active_offsets: np.ndarray,
    active_scales: np.ndarray,
    active_lengths: np.ndarray,
    active_coeff_index_rows: np.ndarray,
) -> None:
    if active_profile_ids.size == 0:
        return

    for slot, p in enumerate(active_profile_ids):
        p_int = int(p)
        profile_name = profile_names[p_int]
        profile = profiles_by_name[profile_name]
        L = int(profile_L[p_int])
        coeff_indices = coeff_index[p_int, : L + 1]

        profile.u_fields = active_u_fields[slot]
        active_rp_fields[slot] = profile.rp_fields
        active_env_fields[slot] = profile.env_fields
        active_offsets[slot] = profile.offset
        active_scales[slot] = profile.scale
        active_lengths[slot] = coeff_indices.size
        if active_coeff_index_rows.shape[1] > 0:
            active_coeff_index_rows[slot].fill(-1)
            active_coeff_index_rows[slot, : coeff_indices.size] = coeff_indices


def build_profile_stage_runner(
    *,
    active_profile_ids: np.ndarray,
    active_u_fields: np.ndarray,
    T_fields: np.ndarray,
    active_rp_fields: np.ndarray,
    active_env_fields: np.ndarray,
    active_offsets: np.ndarray,
    active_scales: np.ndarray,
    active_coeff_index_rows: np.ndarray,
    active_lengths: np.ndarray,
) -> Callable[[np.ndarray], None]:
    if active_profile_ids.size == 0:
        return lambda x: None

    def runner(x: np.ndarray) -> None:
        update_profiles_packed_bulk(
            active_u_fields,
            T_fields,
            active_rp_fields,
            active_env_fields,
            active_offsets,
            active_scales,
            x,
            active_coeff_index_rows,
            active_lengths,
        )

    return runner


def build_geometry_stage_runner(
    *,
    c_family_fields: np.ndarray,
    s_family_fields: np.ndarray,
    c_family_base_fields: np.ndarray,
    s_family_base_fields: np.ndarray,
    active_u_fields: np.ndarray,
    c_family_source_slots: np.ndarray,
    s_family_source_slots: np.ndarray,
    c_effective_order: int,
    s_effective_order: int,
    h_fields: np.ndarray,
    v_fields: np.ndarray,
    k_fields: np.ndarray,
    a: float,
    R0: float,
    Z0: float,
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
    rho: np.ndarray,
    theta: np.ndarray,
    cos_ktheta: np.ndarray,
    sin_ktheta: np.ndarray,
    k_cos_ktheta: np.ndarray,
    k_sin_ktheta: np.ndarray,
    k2_cos_ktheta: np.ndarray,
    k2_sin_ktheta: np.ndarray,
    weights: np.ndarray,
) -> Callable[[], None]:
    c_effective_order = int(c_effective_order)
    s_effective_order = int(s_effective_order)
    a = float(a)
    R0 = float(R0)
    Z0 = float(Z0)

    def runner() -> None:
        update_fourier_family_fields(
            c_family_fields,
            s_family_fields,
            c_family_base_fields,
            s_family_base_fields,
            active_u_fields,
            c_family_source_slots,
            s_family_source_slots,
            c_effective_order,
            s_effective_order,
        )
        update_geometry(
            tb_fields,
            R_fields,
            Z_fields,
            J_fields,
            g_fields,
            S_r,
            V_r,
            Kn,
            Kn_r,
            Ln_r,
            a,
            R0,
            Z0,
            rho,
            theta,
            cos_ktheta,
            sin_ktheta,
            k_cos_ktheta,
            k_sin_ktheta,
            k2_cos_ktheta,
            k2_sin_ktheta,
            weights,
            h_fields,
            v_fields,
            k_fields,
            c_family_fields,
            s_family_fields,
            c_effective_order,
            s_effective_order,
        )

    return runner


def refresh_fourier_family_base_fields(
    *,
    M_max: int,
    profile_index: dict[str, int],
    profiles_by_name: dict[str, Profile],
    c_family_base_fields: np.ndarray,
    s_family_base_fields: np.ndarray,
) -> None:
    c_family_base_fields.fill(0.0)
    s_family_base_fields.fill(0.0)
    for order in range(int(M_max) + 1):
        c_name = f"c{order}"
        if c_name in profile_index:
            np.copyto(c_family_base_fields[order], profiles_by_name[c_name].u_fields)
        if order == 0:
            continue
        s_name = f"s{order}"
        if s_name in profile_index:
            np.copyto(s_family_base_fields[order], profiles_by_name[s_name].u_fields)


def refresh_fourier_family_metadata(
    *,
    c_profile_names: tuple[str, ...],
    s_profile_names: tuple[str, ...],
    profile_coeffs: dict[str, list[float] | None],
    c_offsets: np.ndarray | None,
    s_offsets: np.ndarray | None,
    c_family_fields: np.ndarray,
    s_family_fields: np.ndarray,
) -> tuple[int, int]:
    c_effective_order = effective_family_order(c_profile_names, profile_coeffs, c_offsets, minimum_order=0)
    s_effective_order = effective_family_order(s_profile_names, profile_coeffs, s_offsets, minimum_order=0)

    if c_effective_order + 1 < c_family_fields.shape[0]:
        c_family_fields[c_effective_order + 1 :].fill(0.0)
    if s_effective_order + 1 < s_family_fields.shape[0]:
        s_family_fields[s_effective_order + 1 :].fill(0.0)
    return c_effective_order, s_effective_order


def effective_family_order(
    profile_names: tuple[str, ...],
    profile_coeffs: dict[str, list[float] | None],
    offsets: np.ndarray | None,
    *,
    minimum_order: int,
) -> int:
    effective_order = int(minimum_order)
    for name in profile_names:
        order = int(name[1:])
        if profile_coeffs.get(name) is not None:
            effective_order = max(effective_order, order)
            continue
        if abs(_offset_from_array(offsets, order)) > 1e-14:
            effective_order = max(effective_order, order)
    return effective_order


def _offset_from_array(offsets: np.ndarray | None, order: int) -> float:
    if offsets is None or order >= offsets.shape[0]:
        return 0.0
    return float(offsets[order])

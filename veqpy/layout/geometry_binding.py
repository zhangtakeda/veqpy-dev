"""
Module: layout.geometry_binding

Role:
- Bind geometry stage runners from already-built layout/workspace state.
- Keep Python closure wiring separate from geometry runtime memory ownership.

Notes:
- This module binds preallocated arrays and engine callables; it does not allocate memory.
- Numerical kernels remain in ``veqpy.engine``.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from veqpy.engine.numba_geometry import update_geometry_hot
from veqpy.engine.numba_source import update_fourier_family_fields


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
    surface_fields: np.ndarray,
    radial_fields: np.ndarray,
    rho: np.ndarray,
    theta: np.ndarray,
    cos_mtheta: np.ndarray,
    sin_mtheta: np.ndarray,
    m_cos_mtheta: np.ndarray,
    m_sin_mtheta: np.ndarray,
    m2_cos_mtheta: np.ndarray,
    m2_sin_mtheta: np.ndarray,
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
        update_geometry_hot(
            surface_fields,
            radial_fields,
            a,
            R0,
            Z0,
            rho,
            theta,
            cos_mtheta,
            sin_mtheta,
            m_cos_mtheta,
            m_sin_mtheta,
            m2_cos_mtheta,
            m2_sin_mtheta,
            h_fields,
            v_fields,
            k_fields,
            c_family_fields,
            s_family_fields,
            c_effective_order,
            s_effective_order,
        )

    return runner


__all__ = ["build_geometry_stage_runner"]

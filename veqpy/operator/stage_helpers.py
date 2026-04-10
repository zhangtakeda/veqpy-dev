"""
Module: operator.stage_helpers

Role:
- 收敛 geometry stage 的共享 Python 装配逻辑.
- 避免 operator.py 混入 geometry runner 绑定细节.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from veqpy.engine import update_fourier_family_fields
from veqpy.engine.numba_geometry import update_geometry_hot


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
    surface_workspace: np.ndarray,
    radial_workspace: np.ndarray,
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
        sin_tb = surface_workspace[0]
        R_surface = surface_workspace[1]
        R_t_surface = surface_workspace[2]
        Z_t_surface = surface_workspace[3]
        J_surface = surface_workspace[4]
        JdivR_surface = surface_workspace[5]
        grtdivJR_t_surface = surface_workspace[6]
        gttdivJR_surface = surface_workspace[7]
        gttdivJR_r_surface = surface_workspace[8]
        S_r = radial_workspace[0]
        V_r = radial_workspace[1]
        Kn = radial_workspace[2]
        Kn_r = radial_workspace[3]
        Ln_r = radial_workspace[4]
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
            surface_workspace,
            radial_workspace,
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
            h_fields,
            v_fields,
            k_fields,
            c_family_fields,
            s_family_fields,
            c_effective_order,
            s_effective_order,
        )

    return runner

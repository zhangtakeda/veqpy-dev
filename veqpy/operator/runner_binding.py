"""
Module: operator.runner_binding

Role:
- 收敛 operator 侧 residual runner 的 Python 装配逻辑.
- 让 operator.py 更聚焦于 state ownership 和流程协调.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from veqpy.engine import bind_fused_residual_runner


def build_bound_residual_pack_stage_runner(
    *,
    residual_pack_runner: Callable,
    G: np.ndarray,
    psin_R: np.ndarray,
    psin_Z: np.ndarray,
    sin_tb: np.ndarray,
    sin_ktheta: np.ndarray,
    cos_ktheta: np.ndarray,
    rho_powers: np.ndarray,
    y: np.ndarray,
    T: np.ndarray,
    weights: np.ndarray,
    a: float,
    R0: float,
    B0: float,
) -> Callable[[], np.ndarray]:
    def runner() -> np.ndarray:
        return residual_pack_runner(
            G,
            psin_R,
            psin_Z,
            sin_tb,
            sin_ktheta,
            cos_ktheta,
            rho_powers,
            y,
            T,
            weights,
            a,
            R0,
            B0,
        )

    return runner


def build_bound_residual_full_stage_runner(
    *,
    residual_stage_runner: Callable,
    packed_residual: np.ndarray,
    residual_fields: np.ndarray,
    alpha_state: np.ndarray,
    root_fields: np.ndarray,
    R_fields: np.ndarray,
    Z_fields: np.ndarray,
    J_fields: np.ndarray,
    g_fields: np.ndarray,
    sin_tb: np.ndarray,
    sin_ktheta: np.ndarray,
    cos_ktheta: np.ndarray,
    rho_powers: np.ndarray,
    y: np.ndarray,
    T: np.ndarray,
    weights: np.ndarray,
    a: float,
    R0: float,
    B0: float,
) -> Callable[[], np.ndarray]:
    def runner() -> np.ndarray:
        return residual_stage_runner(
            packed_residual,
            residual_fields,
            float(alpha_state[0]),
            float(alpha_state[1]),
            root_fields,
            R_fields,
            Z_fields,
            J_fields,
            g_fields,
            sin_tb,
            sin_ktheta,
            cos_ktheta,
            rho_powers,
            y,
            T,
            weights,
            a,
            R0,
            B0,
        )

    return runner


def build_fused_residual_runner(
    *,
    residual_plan,
    psin_profile_u_fields: np.ndarray | None,
    evaluate_residual: Callable[[np.ndarray], np.ndarray],
    fused_common_kwargs: dict[str, object],
) -> tuple[Callable[[np.ndarray], np.ndarray], bool]:
    if residual_plan.requires_psin_profile_fields and psin_profile_u_fields is None:
        return evaluate_residual, False
    if not residual_plan.supports_fused_residual:
        return evaluate_residual, False
    return bind_fused_residual_runner(**fused_common_kwargs), True

"""
Module: operator.runner_binding

Role:
- 收敛 operator 侧 fused residual runner 的最小绑定逻辑.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from veqpy.engine import bind_fused_residual_runner


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

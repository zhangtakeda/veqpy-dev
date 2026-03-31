"""
Module: operator.residual_binding

Role:
- 收敛 residual binder 的共享 Python 装配逻辑.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from veqpy.operator.layouts import ResidualBindingLayout


def build_residual_binding_runner(
    bind_runner: Callable,
    *,
    binding_layout: ResidualBindingLayout,
    active_coeff_index_rows: np.ndarray,
    active_lengths: np.ndarray,
    x_size: int,
) -> Callable:
    profile_names = binding_layout.active_profile_names
    try:
        return bind_runner(
            profile_names,
            active_coeff_index_rows,
            active_lengths,
            x_size,
            block_codes=binding_layout.active_residual_block_codes,
            block_orders=binding_layout.active_residual_block_orders,
        )
    except KeyError as exc:
        raise ValueError(f"Unsupported active residual block set {profile_names!r}") from exc

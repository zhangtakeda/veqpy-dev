"""
Module: operator.layout_builders

Role:
- 收敛 operator 布局对象的构造逻辑.
- 避免 operator.py 直接承载 Static/Setup/ResidualBinding layout 的装配细节.
"""

from __future__ import annotations

import numpy as np

from veqpy.model import Grid
from veqpy.operator.layouts import ResidualBindingLayout, SetupLayout, StaticLayout
from veqpy.residual_blocks import build_residual_block_metadata


def build_static_layout(grid: Grid) -> StaticLayout:
    return StaticLayout(
        Nr=int(grid.Nr),
        Nt=int(grid.Nt),
        M_max=int(grid.M_max),
        T_fields=grid.T_fields,
        rho=grid.rho,
        theta=grid.theta,
        cos_ktheta=grid.cos_ktheta,
        sin_ktheta=grid.sin_ktheta,
        k_cos_ktheta=grid.k_cos_ktheta,
        k_sin_ktheta=grid.k_sin_ktheta,
        k2_cos_ktheta=grid.k2_cos_ktheta,
        k2_sin_ktheta=grid.k2_sin_ktheta,
        weights=grid.weights,
        differentiation_matrix=grid.differentiation_matrix,
        integration_matrix=grid.integration_matrix,
        rho_powers=grid.rho_powers,
        y=grid.y,
    )


def build_setup_layout(
    *,
    case_name: str,
    coordinate: str,
    nodes: str,
    prefix_profile_names: tuple[str, ...],
    shape_profile_names: tuple[str, ...],
    profile_names: tuple[str, ...],
    profile_index: dict[str, int],
    c_profile_names: tuple[str, ...],
    s_profile_names: tuple[str, ...],
    profile_L: np.ndarray,
    coeff_index: np.ndarray,
    order_offsets: np.ndarray,
    active_profile_mask: np.ndarray,
    active_profile_ids: np.ndarray,
    x_size: int,
) -> SetupLayout:
    return SetupLayout(
        case_name=case_name,
        coordinate=coordinate,
        nodes=nodes,
        prefix_profile_names=prefix_profile_names,
        shape_profile_names=shape_profile_names,
        profile_names=profile_names,
        profile_index=profile_index,
        c_profile_names=c_profile_names,
        s_profile_names=s_profile_names,
        profile_L=profile_L,
        coeff_index=coeff_index,
        order_offsets=order_offsets,
        active_profile_mask=active_profile_mask,
        active_profile_ids=active_profile_ids,
        x_size=int(x_size),
    )


def build_residual_binding_layout(
    *,
    profile_names: tuple[str, ...],
    active_profile_ids: np.ndarray,
) -> ResidualBindingLayout:
    active_profile_names = tuple(profile_names[int(p)] for p in active_profile_ids)
    active_residual_block_codes, active_residual_block_orders = build_residual_block_metadata(active_profile_names)
    return ResidualBindingLayout(
        active_profile_names=active_profile_names,
        active_residual_block_codes=active_residual_block_codes,
        active_residual_block_orders=active_residual_block_orders,
    )

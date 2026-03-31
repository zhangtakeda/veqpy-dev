"""
Module: operator.layouts

Role:
- 定义 operator 运行期使用的三层 layout 容器.
- 显式区分 grid 静态表, case/setup 拓扑, 与 residual 热路径缓冲区.

Public API:
- StaticLayout
- ResidualBindingLayout
- SetupLayout
- RuntimeLayout
- FieldRuntimeState
- ExecutionState
- SourceRuntimeState

Notes:
- 这里定义的是 layout ownership 和生命周期边界.
- 不负责具体数值核计算或 solver 迭代控制.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable

import numpy as np

if TYPE_CHECKING:
    from veqpy.model.geometry import Geometry
    from veqpy.model.profile import Profile


@dataclass(frozen=True, slots=True)
class StaticLayout:
    """绑定到固定 grid/backend 的只读静态表."""

    Nr: int
    Nt: int
    M_max: int
    T_fields: np.ndarray
    rho: np.ndarray
    theta: np.ndarray
    cos_ktheta: np.ndarray
    sin_ktheta: np.ndarray
    k_cos_ktheta: np.ndarray
    k_sin_ktheta: np.ndarray
    k2_cos_ktheta: np.ndarray
    k2_sin_ktheta: np.ndarray
    weights: np.ndarray
    differentiation_matrix: np.ndarray
    integration_matrix: np.ndarray
    rho_powers: np.ndarray
    y: np.ndarray


@dataclass(frozen=True, slots=True)
class SetupLayout:
    """绑定到当前 operator case/setup 的拓扑与索引布局."""

    case_name: str
    coordinate: str
    nodes: str
    prefix_profile_names: tuple[str, ...]
    shape_profile_names: tuple[str, ...]
    profile_names: tuple[str, ...]
    profile_index: dict[str, int]
    c_profile_names: tuple[str, ...]
    s_profile_names: tuple[str, ...]
    profile_L: np.ndarray
    coeff_index: np.ndarray
    order_offsets: np.ndarray
    active_profile_mask: np.ndarray
    active_profile_ids: np.ndarray
    x_size: int


@dataclass(frozen=True, slots=True)
class ResidualBindingLayout:
    """绑定到 residual binder 的只读 metadata."""

    active_profile_names: tuple[str, ...]
    active_residual_block_codes: np.ndarray
    active_residual_block_orders: np.ndarray


@dataclass(slots=True)
class RuntimeLayout:
    """绑定到 residual 热路径的可变 runtime 缓冲区."""

    geometry: Geometry
    profiles_by_name: dict[str, Profile]
    active_profile_slab: np.ndarray
    family_field_slab: np.ndarray
    source_vector_slab: np.ndarray
    geometry_surface_slab: np.ndarray
    geometry_radial_slab: np.ndarray
    residual_fields: np.ndarray
    root_fields: np.ndarray
    packed_residual: np.ndarray
    active_u_fields: np.ndarray
    active_rp_fields: np.ndarray
    active_env_fields: np.ndarray
    active_offsets: np.ndarray
    active_scales: np.ndarray
    active_lengths: np.ndarray
    active_coeff_index_rows: np.ndarray
    c_family_fields: np.ndarray
    s_family_fields: np.ndarray
    c_family_base_fields: np.ndarray
    s_family_base_fields: np.ndarray
    active_slot_by_profile_id: np.ndarray
    c_family_source_slots: np.ndarray
    s_family_source_slots: np.ndarray
    source_barycentric_weights: np.ndarray
    source_fixed_remap_matrix: np.ndarray
    source_psin_query: np.ndarray
    source_parameter_query: np.ndarray
    source_heat_projection_fit_matrix: np.ndarray
    source_current_projection_fit_matrix: np.ndarray
    source_heat_projection_coeff: np.ndarray
    source_current_projection_coeff: np.ndarray
    source_projection_query: np.ndarray
    source_endpoint_blend: np.ndarray
    materialized_heat_input: np.ndarray
    materialized_current_input: np.ndarray
    source_scratch_1d: np.ndarray
    source_target_root_fields: np.ndarray


@dataclass(slots=True)
class FieldRuntimeState:
    """绑定到 root/residual 场缓存的可变 runtime 状态."""

    residual_fields: np.ndarray
    root_fields: np.ndarray
    packed_residual: np.ndarray
    psin_R: np.ndarray
    psin_Z: np.ndarray
    G: np.ndarray
    psin: np.ndarray
    psin_r: np.ndarray
    psin_rr: np.ndarray
    FFn_psin: np.ndarray
    Pn_psin: np.ndarray


@dataclass(slots=True)
class ExecutionState:
    """绑定到 operator 执行策略的可变 runner/state 容器."""

    profile_stage_runner: Callable
    geometry_stage_runner: Callable
    source_stage_runner: Callable
    residual_pack_stage_runner: Callable
    residual_full_stage_runner: Callable
    residual_pack_runner: Callable
    residual_stage_runner: Callable
    fused_residual_runner: Callable
    fused_alpha_state: np.ndarray
    supports_fused_residual: bool


@dataclass(slots=True)
class SourceRuntimeState:
    """绑定到 source materialization/remap 的可变 runtime 状态."""

    cache_key: tuple[str, str, int] | None
    barycentric_weights: np.ndarray
    fixed_remap_matrix: np.ndarray
    psin_query: np.ndarray
    parameter_query: np.ndarray
    heat_projection_fit_matrix: np.ndarray
    current_projection_fit_matrix: np.ndarray
    heat_projection_coeff: np.ndarray
    current_projection_coeff: np.ndarray
    projection_query: np.ndarray
    endpoint_blend: np.ndarray
    materialized_heat_input: np.ndarray
    materialized_current_input: np.ndarray
    scratch_1d: np.ndarray
    target_root_fields: np.ndarray

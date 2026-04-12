"""
Module: operator.runtime_allocation

Role:
- 收敛 operator runtime/state 的一次性分配逻辑.
- 避免 operator.py 直接承载大块 slab/state 初始化细节.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from veqpy.model import Grid, Profile
from veqpy.operator.layouts import FieldRuntimeState, RuntimeLayout, SourceRuntimeState, StaticLayout


@dataclass(slots=True)
class RuntimeAllocationBundle:
    profiles_by_name: dict[str, Profile]
    field_runtime_state: FieldRuntimeState
    source_runtime_state: SourceRuntimeState
    active_profile_slab: np.ndarray
    active_u_fields: np.ndarray
    active_rp_fields: np.ndarray
    active_env_fields: np.ndarray
    active_offsets: np.ndarray
    active_scales: np.ndarray
    active_lengths: np.ndarray
    active_coeff_index_rows: np.ndarray
    family_field_slab: np.ndarray
    c_family_fields: np.ndarray
    s_family_fields: np.ndarray
    c_family_base_fields: np.ndarray
    s_family_base_fields: np.ndarray
    active_slot_by_profile_id: np.ndarray
    c_family_source_slots: np.ndarray
    s_family_source_slots: np.ndarray
    runtime_layout: RuntimeLayout


def allocate_runtime_state(
    *,
    grid: Grid,
    static_layout: StaticLayout,
    source_plan,
    profile_names: tuple[str, ...],
    profile_index: dict[str, int],
    active_profile_ids: np.ndarray,
    profile_L: np.ndarray,
    x_size: int,
    make_profile: Callable[[str], Profile],
) -> RuntimeAllocationBundle:
    nr = static_layout.Nr
    nt = static_layout.Nt
    n_active = int(active_profile_ids.size)
    max_active_len = 0
    if n_active > 0:
        max_active_len = max(int(profile_L[int(p)]) + 1 for p in active_profile_ids)

    # solve-time surface workspace rows:
    # 0=sin_tb, 1=R, 2=R_t, 3=Z_t, 4=J, 5=JdivR, 6=grtdivJR_t, 7=gttdivJR, 8=gttdivJR_r
    geometry_surface_workspace = np.empty((9, nr, nt), dtype=np.float64)
    # solve-time radial workspace rows:
    # 0=S_r, 1=V_r, 2=Kn, 3=Kn_r, 4=Ln_r
    geometry_radial_workspace = np.empty((5, nr), dtype=np.float64)
    # solve-time residual workspace rows:
    # 0=G, 1=G*psin_R, 2=G*psin_Z, 3=G*psin_R*sin_tb
    residual_surface_workspace = np.empty((4, nr, nt), dtype=np.float64)

    profiles_by_name: dict[str, Profile] = {}
    for name in profile_names:
        profiles_by_name[name] = make_profile(name)

    root_fields = np.empty((5, nr), dtype=np.float64)
    field_runtime_state = FieldRuntimeState(
        root_fields=root_fields,
        packed_residual=np.empty(x_size, dtype=np.float64),
        psin=root_fields[0],
        psin_r=root_fields[1],
        psin_rr=root_fields[2],
        FFn_psin=root_fields[3],
        Pn_psin=root_fields[4],
    )

    materialized_heat_input = np.empty(nr, dtype=np.float64)
    materialized_current_input = np.empty(nr, dtype=np.float64)
    needs_psin_query = bool(source_plan.is_profile_owned_psin or source_plan.is_fixed_point_psin)
    psin_query = np.empty(nr, dtype=np.float64) if needs_psin_query else np.empty(0, dtype=np.float64)
    if source_plan.is_psin_coordinate and source_plan.parameterization != "identity":
        parameter_query = np.empty(nr, dtype=np.float64)
    else:
        parameter_query = psin_query
    if source_plan.is_profile_owned_psin or source_plan.is_fixed_point_psin:
        target_root_fields = np.empty((3, nr), dtype=np.float64)
    else:
        target_root_fields = np.empty((3, 0), dtype=np.float64)
    source_runtime_state = SourceRuntimeState(
        cache_key=None,
        barycentric_weights=np.empty(0, dtype=np.float64),
        fixed_remap_matrix=np.empty((0, 0), dtype=np.float64),
        materialized_heat_input=materialized_heat_input,
        materialized_current_input=materialized_current_input,
        psin_query=psin_query,
        parameter_query=parameter_query,
        heat_projection_fit_matrix=np.empty((0, 0), dtype=np.float64),
        current_projection_fit_matrix=np.empty((0, 0), dtype=np.float64),
        heat_projection_coeff=np.empty(0, dtype=np.float64),
        current_projection_coeff=np.empty(0, dtype=np.float64),
        endpoint_blend=np.linspace(0.0, 1.0, nr, dtype=np.float64),
        target_root_fields=target_root_fields,
        scratch_1d=np.empty((6, nr), dtype=np.float64),
    )

    active_profile_slab = np.empty((3, n_active, 3, nr), dtype=np.float64)
    active_u_fields = active_profile_slab[0]
    active_rp_fields = active_profile_slab[1]
    active_env_fields = active_profile_slab[2]
    active_offsets = np.empty(n_active, dtype=np.float64)
    active_scales = np.empty(n_active, dtype=np.float64)
    active_lengths = np.empty(n_active, dtype=np.int64)
    active_coeff_index_rows = np.full((n_active, max_active_len), -1, dtype=np.int64)

    family_field_slab = np.empty((4, grid.M_max + 1, 3, nr), dtype=np.float64)
    c_family_fields = family_field_slab[0]
    s_family_fields = family_field_slab[1]
    c_family_base_fields = family_field_slab[2]
    s_family_base_fields = family_field_slab[3]
    s_family_fields.fill(0.0)
    c_family_base_fields.fill(0.0)
    s_family_base_fields.fill(0.0)

    active_slot_by_profile_id = np.full(len(profile_names), -1, dtype=np.int64)
    for slot, p in enumerate(active_profile_ids):
        active_slot_by_profile_id[int(p)] = int(slot)

    c_family_source_slots = np.full(grid.M_max + 1, -1, dtype=np.int64)
    s_family_source_slots = np.full(grid.M_max + 1, -1, dtype=np.int64)
    for order in range(grid.M_max + 1):
        c_name = f"c{order}"
        if c_name in profile_index:
            c_family_source_slots[order] = active_slot_by_profile_id[profile_index[c_name]]
        if order == 0:
            continue
        s_name = f"s{order}"
        if s_name in profile_index:
            s_family_source_slots[order] = active_slot_by_profile_id[profile_index[s_name]]

    runtime_layout = RuntimeLayout(
        profiles_by_name=profiles_by_name,
        active_profile_slab=active_profile_slab,
        family_field_slab=family_field_slab,
        source_runtime_state=source_runtime_state,
        geometry_surface_workspace=geometry_surface_workspace,
        geometry_radial_workspace=geometry_radial_workspace,
        residual_surface_workspace=residual_surface_workspace,
        root_fields=field_runtime_state.root_fields,
        packed_residual=field_runtime_state.packed_residual,
        active_u_fields=active_u_fields,
        active_rp_fields=active_rp_fields,
        active_env_fields=active_env_fields,
        active_offsets=active_offsets,
        active_scales=active_scales,
        active_lengths=active_lengths,
        active_coeff_index_rows=active_coeff_index_rows,
        c_family_fields=c_family_fields,
        s_family_fields=s_family_fields,
        c_family_base_fields=c_family_base_fields,
        s_family_base_fields=s_family_base_fields,
        active_slot_by_profile_id=active_slot_by_profile_id,
        c_family_source_slots=c_family_source_slots,
        s_family_source_slots=s_family_source_slots,
    )

    return RuntimeAllocationBundle(
        profiles_by_name=profiles_by_name,
        field_runtime_state=field_runtime_state,
        source_runtime_state=source_runtime_state,
        active_profile_slab=active_profile_slab,
        active_u_fields=active_u_fields,
        active_rp_fields=active_rp_fields,
        active_env_fields=active_env_fields,
        active_offsets=active_offsets,
        active_scales=active_scales,
        active_lengths=active_lengths,
        active_coeff_index_rows=active_coeff_index_rows,
        family_field_slab=family_field_slab,
        c_family_fields=c_family_fields,
        s_family_fields=s_family_fields,
        c_family_base_fields=c_family_base_fields,
        s_family_base_fields=s_family_base_fields,
        active_slot_by_profile_id=active_slot_by_profile_id,
        c_family_source_slots=c_family_source_slots,
        s_family_source_slots=s_family_source_slots,
        runtime_layout=runtime_layout,
    )

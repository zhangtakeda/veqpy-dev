"""
Module: operator.runtime_views

Role:
- 收敛 operator 到 RuntimeLayout 的视图同步逻辑.
- 避免 operator.py 混入大段重复字段转发.
"""

from __future__ import annotations

from veqpy.operator.layouts import RuntimeLayout


def refresh_runtime_layout_views(
    runtime: RuntimeLayout,
    *,
    geometry,
    profiles_by_name,
    active_profile_slab,
    family_field_slab,
    source_vector_slab,
    geometry_surface_slab,
    geometry_radial_slab,
    residual_fields,
    root_fields,
    packed_residual,
    active_u_fields,
    active_rp_fields,
    active_env_fields,
    active_offsets,
    active_scales,
    active_lengths,
    active_coeff_index_rows,
    c_family_fields,
    s_family_fields,
    c_family_base_fields,
    s_family_base_fields,
    active_slot_by_profile_id,
    c_family_source_slots,
    s_family_source_slots,
    source_barycentric_weights,
    source_fixed_remap_matrix,
    source_psin_query,
    source_parameter_query,
    source_heat_projection_fit_matrix,
    source_current_projection_fit_matrix,
    source_heat_projection_coeff,
    source_current_projection_coeff,
    source_projection_query,
    source_endpoint_blend,
    materialized_heat_input,
    materialized_current_input,
    source_scratch_1d,
    source_target_root_fields,
) -> None:
    runtime.geometry = geometry
    runtime.profiles_by_name = profiles_by_name
    runtime.active_profile_slab = active_profile_slab
    runtime.family_field_slab = family_field_slab
    runtime.source_vector_slab = source_vector_slab
    runtime.geometry_surface_slab = geometry_surface_slab
    runtime.geometry_radial_slab = geometry_radial_slab
    runtime.residual_fields = residual_fields
    runtime.root_fields = root_fields
    runtime.packed_residual = packed_residual
    runtime.active_u_fields = active_u_fields
    runtime.active_rp_fields = active_rp_fields
    runtime.active_env_fields = active_env_fields
    runtime.active_offsets = active_offsets
    runtime.active_scales = active_scales
    runtime.active_lengths = active_lengths
    runtime.active_coeff_index_rows = active_coeff_index_rows
    runtime.c_family_fields = c_family_fields
    runtime.s_family_fields = s_family_fields
    runtime.c_family_base_fields = c_family_base_fields
    runtime.s_family_base_fields = s_family_base_fields
    runtime.active_slot_by_profile_id = active_slot_by_profile_id
    runtime.c_family_source_slots = c_family_source_slots
    runtime.s_family_source_slots = s_family_source_slots
    runtime.source_barycentric_weights = source_barycentric_weights
    runtime.source_fixed_remap_matrix = source_fixed_remap_matrix
    runtime.source_psin_query = source_psin_query
    runtime.source_parameter_query = source_parameter_query
    runtime.source_heat_projection_fit_matrix = source_heat_projection_fit_matrix
    runtime.source_current_projection_fit_matrix = source_current_projection_fit_matrix
    runtime.source_heat_projection_coeff = source_heat_projection_coeff
    runtime.source_current_projection_coeff = source_current_projection_coeff
    runtime.source_projection_query = source_projection_query
    runtime.source_endpoint_blend = source_endpoint_blend
    runtime.materialized_heat_input = materialized_heat_input
    runtime.materialized_current_input = materialized_current_input
    runtime.source_scratch_1d = source_scratch_1d
    runtime.source_target_root_fields = source_target_root_fields

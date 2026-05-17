"""Operator workspace allocation."""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable

import numpy as np

from veqpy.workspace.geometry_workspace import GeometryWorkspace
from veqpy.workspace.operator_workspace import OperatorWorkspace, RuntimeAllocationBundle
from veqpy.workspace.profile_workspace import ProfileWorkspace
from veqpy.workspace.residual_workspace import ResidualWorkspace
from veqpy.workspace.source_workspace import SourceWorkspace

if TYPE_CHECKING:
    from veqpy.model.profile import Profile
    from veqpy.workspace.grid_workspace import GridWorkspace


def allocate_runtime_state(
    *,
    static_layout: GridWorkspace,
    source_execution,
    profile_names: tuple[str, ...],
    profile_index: dict[str, int],
    active_profile_ids: np.ndarray,
    profile_L: np.ndarray,
    x_size: int,
    make_profile: Callable[[str], Profile],
) -> RuntimeAllocationBundle:
    """Allocate mutable arrays/state required by ``Operator`` runtime."""

    nr = static_layout.Nr
    nt = static_layout.Nt
    m_max = static_layout.M_max
    n_active = int(active_profile_ids.size)
    max_active_len = 0
    if n_active > 0:
        max_active_len = max(int(profile_L[int(p)]) + 1 for p in active_profile_ids)

    profiles_by_name = {name: make_profile(name) for name in profile_names}

    profile_workspace = _allocate_profile_workspace(
        nr=nr,
        m_max=m_max,
        profile_names=profile_names,
        profile_index=profile_index,
        active_profile_ids=active_profile_ids,
        profile_L=profile_L,
        n_active=n_active,
        max_active_len=max_active_len,
    )
    geometry_workspace = GeometryWorkspace(
        surface_fields=np.empty((9, nr, nt), dtype=np.float64),
        radial_fields=np.empty((5, nr), dtype=np.float64),
        h_fields=np.empty((0, nr), dtype=np.float64),
        v_fields=np.empty((0, nr), dtype=np.float64),
        k_fields=np.empty((0, nr), dtype=np.float64),
    )
    source_workspace = _allocate_source_workspace(
        nr=nr,
        nt=nt,
        source_execution=source_execution,
    )
    residual_workspace = _allocate_residual_workspace(
        nr=nr,
        nt=nt,
        x_size=x_size,
        radial_weights=np.asarray(static_layout.weights, dtype=np.float64),
    )
    workspace = OperatorWorkspace(
        profile=profile_workspace,
        geometry=geometry_workspace,
        source=source_workspace,
        residual=residual_workspace,
    )
    return RuntimeAllocationBundle(
        profiles_by_name=profiles_by_name,
        workspace=workspace,
        profile_workspace=profile_workspace,
        geometry_workspace=geometry_workspace,
        source_workspace=source_workspace,
        residual_workspace=residual_workspace,
    )


def _allocate_profile_workspace(
    *,
    nr: int,
    m_max: int,
    profile_names: tuple[str, ...],
    profile_index: dict[str, int],
    active_profile_ids: np.ndarray,
    profile_L: np.ndarray,
    n_active: int,
    max_active_len: int,
) -> ProfileWorkspace:
    active_profile_slab = np.empty((3, n_active, 3, nr), dtype=np.float64)
    family_field_slab = np.empty((4, m_max + 1, 3, nr), dtype=np.float64)
    family_field_slab[1:].fill(0.0)

    active_slot_by_profile_id = np.full(len(profile_names), -1, dtype=np.int64)
    for slot, p in enumerate(active_profile_ids):
        active_slot_by_profile_id[int(p)] = int(slot)

    c_family_source_slots = np.full(m_max + 1, -1, dtype=np.int64)
    s_family_source_slots = np.full(m_max + 1, -1, dtype=np.int64)
    for order in range(m_max + 1):
        c_name = f"c{order}"
        if c_name in profile_index:
            c_family_source_slots[order] = active_slot_by_profile_id[profile_index[c_name]]
        if order == 0:
            continue
        s_name = f"s{order}"
        if s_name in profile_index:
            s_family_source_slots[order] = active_slot_by_profile_id[profile_index[s_name]]

    return ProfileWorkspace(
        active_profile_slab=active_profile_slab,
        family_field_slab=family_field_slab,
        active_u_fields=active_profile_slab[0],
        active_rp_fields=active_profile_slab[1],
        active_env_fields=active_profile_slab[2],
        active_offsets=np.empty(n_active, dtype=np.float64),
        active_scales=np.empty(n_active, dtype=np.float64),
        active_lengths=np.empty(n_active, dtype=np.int64),
        active_coeff_index_rows=np.full((n_active, max_active_len), -1, dtype=np.int64),
        c_family_fields=family_field_slab[0],
        s_family_fields=family_field_slab[1],
        c_family_base_fields=family_field_slab[2],
        s_family_base_fields=family_field_slab[3],
        active_slot_by_profile_id=active_slot_by_profile_id,
        c_family_source_slots=c_family_source_slots,
        s_family_source_slots=s_family_source_slots,
    )


def _allocate_source_workspace(*, nr: int, nt: int, source_execution) -> SourceWorkspace:
    needs_psin_query = bool(source_execution.requires_psin_query_workspace)
    psin_query = (
        np.empty(nr, dtype=np.float64) if needs_psin_query else np.empty(0, dtype=np.float64)
    )
    parameter_query = (
        np.empty(nr, dtype=np.float64)
        if source_execution.requires_source_parameter_query
        else psin_query
    )
    target_root_fields = (
        np.empty((3, nr), dtype=np.float64)
        if source_execution.requires_target_root_fields
        else np.empty((3, 0), dtype=np.float64)
    )
    return SourceWorkspace(
        cache_key=None,
        barycentric_weights=np.empty(0, dtype=np.float64),
        fixed_remap_matrix=np.empty((0, 0), dtype=np.float64),
        endpoint_blend=np.linspace(0.0, 1.0, nr, dtype=np.float64),
        heat_spline_coeff=np.empty((0, 4), dtype=np.float64),
        current_spline_coeff=np.empty((0, 4), dtype=np.float64),
        psin_query=psin_query,
        parameter_query=parameter_query,
        materialized_heat_input=np.empty(nr, dtype=np.float64),
        materialized_current_input=np.empty(nr, dtype=np.float64),
        scratch_1d=np.empty((7 + nr, nr), dtype=np.float64),
        scratch_2d=np.empty((1, nr, nt), dtype=np.float64),
        target_root_fields=target_root_fields,
        alpha_state=np.zeros(2, dtype=np.float64),
        f_u=np.empty(0, dtype=np.float64),
        f_fields=np.empty((0, nr), dtype=np.float64),
        psin_u=np.empty(0, dtype=np.float64),
        psin_fields=np.empty((0, nr), dtype=np.float64),
    )


def _allocate_residual_workspace(
    *, nr: int, nt: int, x_size: int, radial_weights: np.ndarray
) -> ResidualWorkspace:
    if radial_weights.ndim != 1 or radial_weights.size != nr:
        raise ValueError(f"Invalid radial weights shape {radial_weights.shape}")
    return ResidualWorkspace(
        root_fields=np.empty((5, nr), dtype=np.float64),
        packed_residual=np.empty(x_size, dtype=np.float64),
        surface_fields=np.empty((4, nr, nt), dtype=np.float64),
        pack_scratch=np.empty(nr, dtype=np.float64),
        collocation_sqrt_weights=np.sqrt(radial_weights / max(nt, 1)),
    )

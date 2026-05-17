"""
Module: workspace.runtime

Role:
- Own all operator runtime memory.
- Allocate workspace arrays from model-derived layout/plan dimensions.
- Keep large compute-dense buffers grouped by physical meaning and use timing.

Notes:
- ``OperatorWorkspace`` is the semantic/debug owner of runtime arrays.
- Hot kernels and bound runners must capture concrete arrays or ABI objects at bind time,
  not access workspace properties inside tight loops.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable

import numpy as np

if TYPE_CHECKING:
    from veqpy.model.profile import Profile
    from veqpy.operator.runtime_layout import ResidualBindingLayout, StaticLayout


# ---- workspace array conventions -------------------------------------------
# ProfileWorkspace.active_profile_slab: (3, n_active, 3, Nr)
#   [0] active_u_fields   — profile value/r/r-r fields updated from packed x
#   [1] active_rp_fields  — radial-prefactor value/r/r-r fields
#   [2] active_env_fields — envelope value/r/r-r fields
#   derivative axis [0, 1, 2] means value, radial first derivative,
#   radial second derivative.
#
# ProfileWorkspace.family_field_slab: (4, M_max + 1, 3, Nr)
#   [0] c_family_fields
#   [1] s_family_fields
#   [2] c_family_base_fields
#   [3] s_family_base_fields
#
# ResidualWorkspace.root_fields: (5, Nr)
#   [0] psin
#   [1] psin_r
#   [2] psin_rr
#   [3] FFn_psin
#   [4] Pn_psin
#
# GeometryWorkspace.surface_workspace: (9, Nr, Nt)
#   [0] sin_tb
#   [1] R
#   [2] R_t
#   [3] Z_t
#   [4] J
#   [5] JdivR
#   [6] grtdivJR_t
#   [7] gttdivJR
#   [8] gttdivJR_r
#
# GeometryWorkspace.radial_workspace: (5, Nr)
#   [0] S_r
#   [1] V_r
#   [2] Kn
#   [3] Kn_r
#   [4] Ln_r
#
# ResidualWorkspace.surface_workspace: (4, Nr, Nt)
#   [0] G
#   [1] G*psin_R
#   [2] G*psin_Z
#   [3] G*psin_R*sin_tb


@dataclass(slots=True)
class ProfileWorkspace:
    """Profile stage memory owner."""

    active_profile_slab: np.ndarray
    family_field_slab: np.ndarray
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


@dataclass(slots=True)
class GeometryWorkspace:
    """Geometry stage memory owner."""

    surface_workspace: np.ndarray
    radial_workspace: np.ndarray


@dataclass(slots=True)
class SourceWorkspace:
    """Source stage memory owner."""

    runtime_state: SourceRuntimeState

    @property
    def const_state(self) -> SourceConstState:
        return self.runtime_state.const_state

    @property
    def work_state(self) -> SourceWorkState:
        return self.runtime_state.work_state

    @property
    def aux_state(self) -> SourceAuxState:
        return self.runtime_state.aux_state


@dataclass(slots=True)
class ResidualWorkspace:
    """Residual/root stage memory owner."""

    root_fields: np.ndarray
    packed_residual: np.ndarray
    surface_workspace: np.ndarray
    pack_scratch: np.ndarray
    collocation_sqrt_weights: np.ndarray


@dataclass(slots=True)
class OperatorWorkspace:
    """Operator 运行期数组内存 owner.

    Workspace 的大小与可选区域由 model 降解后的 layout/plan 决定。热计算密集区域
    按数组物理性质和使用时机合并成少数大 slab，再切成 view 传给 engine。

    数组切片语义由本文件顶部 workspace array conventions 约束；hot path 必须
    使用 bind-time 捕获的 arrays/ABI，不要在循环内访问 workspace 字段。
    """

    profile: ProfileWorkspace
    geometry: GeometryWorkspace
    source: SourceWorkspace
    residual: ResidualWorkspace
    h_fields: np.ndarray
    v_fields: np.ndarray
    k_fields: np.ndarray
    F_profile_u: np.ndarray
    F_profile_fields: np.ndarray
    psin_profile_u: np.ndarray
    psin_profile_fields: np.ndarray



@dataclass(slots=True)
class FieldRuntimeState:
    """绑定到 root/residual 场缓存的可变 runtime 状态."""

    root_fields: np.ndarray
    packed_residual: np.ndarray
    psin: np.ndarray
    psin_r: np.ndarray
    psin_rr: np.ndarray
    FFn_psin: np.ndarray
    Pn_psin: np.ndarray


@dataclass(slots=True)
class SourceConstState:
    """绑定到 source route 的只读 const arrays."""

    barycentric_weights: np.ndarray
    fixed_remap_matrix: np.ndarray
    endpoint_blend: np.ndarray


@dataclass(slots=True)
class SourceWorkState:
    """绑定到 source materialization 的可复用 work buffers."""

    psin_query: np.ndarray
    parameter_query: np.ndarray
    materialized_heat_input: np.ndarray
    materialized_current_input: np.ndarray
    scratch_1d: np.ndarray
    scratch_2d: np.ndarray


@dataclass(slots=True)
class SourceAuxState:
    """绑定到 source refresh/stage 的 aux outputs."""

    heat_spline_coeff: np.ndarray
    current_spline_coeff: np.ndarray
    target_root_fields: np.ndarray


@dataclass(slots=True)
class SourceRuntimeState:
    """绑定到 source materialization/remap 的 runtime ownership 容器."""

    cache_key: tuple[str, str, int, str] | None
    const_state: SourceConstState
    work_state: SourceWorkState
    aux_state: SourceAuxState


@dataclass(slots=True)
class BackendState:
    """绑定到 backend ABI 的 arrays-only state 聚合入口."""

    static_layout: StaticLayout
    residual_binding_layout: ResidualBindingLayout
    workspace: OperatorWorkspace
    field_runtime_state: FieldRuntimeState
    source_runtime_state: SourceRuntimeState


@dataclass(slots=True)
class RuntimeAllocationBundle:
    """Operator runtime 一次性分配结果."""

    profiles_by_name: dict[str, Profile]
    workspace: OperatorWorkspace
    profile_workspace: ProfileWorkspace
    geometry_workspace: GeometryWorkspace
    source_workspace: SourceWorkspace
    residual_workspace: ResidualWorkspace
    field_runtime_state: FieldRuntimeState
    source_runtime_state: SourceRuntimeState


def allocate_runtime_state(
    *,
    static_layout: StaticLayout,
    source_execution,
    profile_names: tuple[str, ...],
    profile_index: dict[str, int],
    active_profile_ids: np.ndarray,
    profile_L: np.ndarray,
    x_size: int,
    make_profile: Callable[[str], Profile],
) -> RuntimeAllocationBundle:
    """分配并绑定 Operator runtime 所需的 mutable arrays/state."""
    nr = static_layout.Nr
    nt = static_layout.Nt
    M_max = static_layout.M_max
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
    residual_pack_scratch = np.empty(nr, dtype=np.float64)
    radial_weights = np.asarray(static_layout.weights, dtype=np.float64)
    if radial_weights.ndim != 1 or radial_weights.size != nr:
        raise ValueError(f"Invalid radial weights shape {radial_weights.shape}")
    collocation_sqrt_weights = np.sqrt(radial_weights / max(nt, 1))

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
    needs_psin_query = bool(source_execution.requires_psin_query_workspace)
    psin_query = (
        np.empty(nr, dtype=np.float64) if needs_psin_query else np.empty(0, dtype=np.float64)
    )
    if source_execution.requires_source_parameter_query:
        parameter_query = np.empty(nr, dtype=np.float64)
    else:
        parameter_query = psin_query
    if source_execution.requires_target_root_fields:
        target_root_fields = np.empty((3, nr), dtype=np.float64)
    else:
        target_root_fields = np.empty((3, 0), dtype=np.float64)
    source_const_state = SourceConstState(
        barycentric_weights=np.empty(0, dtype=np.float64),
        fixed_remap_matrix=np.empty((0, 0), dtype=np.float64),
        endpoint_blend=np.linspace(0.0, 1.0, nr, dtype=np.float64),
    )
    source_work_state = SourceWorkState(
        psin_query=psin_query,
        parameter_query=parameter_query,
        materialized_heat_input=materialized_heat_input,
        materialized_current_input=materialized_current_input,
        scratch_1d=np.empty((7 + nr, nr), dtype=np.float64),
        scratch_2d=np.empty((1, nr, nt), dtype=np.float64),
    )
    source_aux_state = SourceAuxState(
        heat_spline_coeff=np.empty((0, 4), dtype=np.float64),
        current_spline_coeff=np.empty((0, 4), dtype=np.float64),
        target_root_fields=target_root_fields,
    )
    source_runtime_state = SourceRuntimeState(
        cache_key=None,
        const_state=source_const_state,
        work_state=source_work_state,
        aux_state=source_aux_state,
    )

    active_profile_slab = np.empty((3, n_active, 3, nr), dtype=np.float64)
    active_u_fields = active_profile_slab[0]
    active_rp_fields = active_profile_slab[1]
    active_env_fields = active_profile_slab[2]
    active_offsets = np.empty(n_active, dtype=np.float64)
    active_scales = np.empty(n_active, dtype=np.float64)
    active_lengths = np.empty(n_active, dtype=np.int64)
    active_coeff_index_rows = np.full((n_active, max_active_len), -1, dtype=np.int64)

    family_field_slab = np.empty((4, M_max + 1, 3, nr), dtype=np.float64)
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

    c_family_source_slots = np.full(M_max + 1, -1, dtype=np.int64)
    s_family_source_slots = np.full(M_max + 1, -1, dtype=np.int64)
    for order in range(M_max + 1):
        c_name = f"c{order}"
        if c_name in profile_index:
            c_family_source_slots[order] = active_slot_by_profile_id[profile_index[c_name]]
        if order == 0:
            continue
        s_name = f"s{order}"
        if s_name in profile_index:
            s_family_source_slots[order] = active_slot_by_profile_id[profile_index[s_name]]

    profile_workspace = ProfileWorkspace(
        active_profile_slab=active_profile_slab,
        family_field_slab=family_field_slab,
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
    geometry_workspace = GeometryWorkspace(
        surface_workspace=geometry_surface_workspace,
        radial_workspace=geometry_radial_workspace,
    )
    source_workspace = SourceWorkspace(runtime_state=source_runtime_state)
    residual_workspace = ResidualWorkspace(
        root_fields=field_runtime_state.root_fields,
        packed_residual=field_runtime_state.packed_residual,
        surface_workspace=residual_surface_workspace,
        pack_scratch=residual_pack_scratch,
        collocation_sqrt_weights=collocation_sqrt_weights,
    )
    workspace = OperatorWorkspace(
        profile=profile_workspace,
        geometry=geometry_workspace,
        source=source_workspace,
        residual=residual_workspace,
        h_fields=np.empty((0, nr), dtype=np.float64),
        v_fields=np.empty((0, nr), dtype=np.float64),
        k_fields=np.empty((0, nr), dtype=np.float64),
        F_profile_u=np.empty(0, dtype=np.float64),
        F_profile_fields=np.empty((0, nr), dtype=np.float64),
        psin_profile_u=np.empty(0, dtype=np.float64),
        psin_profile_fields=np.empty((0, nr), dtype=np.float64),
    )
    return RuntimeAllocationBundle(
        profiles_by_name=profiles_by_name,
        workspace=workspace,
        profile_workspace=profile_workspace,
        geometry_workspace=geometry_workspace,
        source_workspace=source_workspace,
        residual_workspace=residual_workspace,
        field_runtime_state=field_runtime_state,
        source_runtime_state=source_runtime_state,
    )

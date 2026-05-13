"""
Module: operator.runtime_layout

Role:
- 定义 operator 运行期使用的三层 layout 容器.
- 收敛 operator runtime/state 的一次性分配逻辑.
- 显式区分 grid 静态表, case/setup 拓扑, 与 residual 热路径缓冲区.

Public API:
- StaticLayout
- ResidualBindingLayout
- RuntimeLayout
- BackendState
- FieldRuntimeState
- ExecutionState
- SourceRuntimeState
- SourceConstState
- SourceWorkState
- SourceAuxState
- RuntimeAllocationBundle
- allocate_runtime_state

Notes:
- 这里定义的是 layout ownership 和生命周期边界.
- 不负责具体数值核计算或 solver 迭代控制.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable

import numpy as np

if TYPE_CHECKING:
    from veqpy.model.grid import Grid
    from veqpy.model.profile import Profile


# ---- block layout convention -----------------------------------------------
#   radial_block (8+K_max+3*L_max, Nr)
#     rho           (Nr,)
#     x             (Nr,)
#     y             (Nr,)
#     rho_powers    (K_max+2, Nr)
#     T             (L_max+1, Nr)
#     T_r           (L_max+1, Nr)
#     T_rr          (L_max+1, Nr)
#
#   poloidal_block (7+6*M_max, Nt)
#     theta         (Nt,)
#     cos_mtheta    (M_max+1, Nt)
#     sin_mtheta    (M_max+1, Nt)
#     m_cos_mtheta  (M_max+1, Nt)
#     m_sin_mtheta  (M_max+1, Nt)
#     m2_cos_mtheta (M_max+1, Nt)
#     m2_sin_mtheta (M_max+1, Nt)
#


def _pack_radial_block(
    rho: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    rho_powers: np.ndarray,
    T: np.ndarray,
    T_r: np.ndarray,
    T_rr: np.ndarray,
) -> np.ndarray:
    """按规范压合径向场为 (R, Nr) 2D 只读数组."""
    if abs(rho[0]) < 1e-10:
        raise ValueError("rho[0] is too close to zero")

    Nr = rho.shape[0]
    K_max = rho_powers.shape[0] - 2
    L_max = T.shape[0] - 1

    block = np.empty((8 + K_max + 3 * L_max, Nr), dtype=np.float64)
    block[0] = rho
    block[1] = x
    block[2] = y
    block[3 : 5 + K_max] = rho_powers
    block[5 + K_max : 6 + K_max + L_max] = T
    block[6 + K_max + L_max : 7 + K_max + 2 * L_max] = T_r
    block[7 + K_max + 2 * L_max : 8 + K_max + 3 * L_max] = T_rr
    block.flags.writeable = False
    return block


def _pack_poloidal_block(
    theta: np.ndarray,
    cos_mtheta: np.ndarray,
    sin_mtheta: np.ndarray,
    m_cos_mtheta: np.ndarray,
    m_sin_mtheta: np.ndarray,
    m2_cos_mtheta: np.ndarray,
    m2_sin_mtheta: np.ndarray,
) -> np.ndarray:
    """按规范压合极向场为 (P, Nt) 2D 只读数组."""
    Nt = theta.shape[0]
    M_max = cos_mtheta.shape[0] - 1

    block = np.empty((7 + 6 * M_max, Nt), dtype=np.float64)
    block[0] = theta
    block[1 : 2 + M_max] = cos_mtheta
    block[2 + M_max : 3 + 2 * M_max] = sin_mtheta
    block[3 + 2 * M_max : 4 + 3 * M_max] = m_cos_mtheta
    block[4 + 3 * M_max : 5 + 4 * M_max] = m_sin_mtheta
    block[5 + 4 * M_max : 6 + 5 * M_max] = m2_cos_mtheta
    block[6 + 5 * M_max : 7 + 6 * M_max] = m2_sin_mtheta
    block.flags.writeable = False
    return block


@dataclass(frozen=True, slots=True)
class StaticLayout:
    """operator 层 Grid 快照 — engine ABI 所需的数组+元数据.

    K_max 已规范化：Grid.K_max=None 映射为 M_max.
    """

    Nr: int
    Nt: int
    M_max: int
    L_max: int
    K_max: int
    scheme: str
    calculus: str
    K_values: np.ndarray
    quadrature: np.ndarray
    differentiator: np.ndarray
    accumulator: np.ndarray
    radial_block: np.ndarray  # (8+K_max+3*L_max, Nr)
    poloidal_block: np.ndarray  # (7+6*M_max, Nt)

    @property
    def rho(self) -> np.ndarray:
        return self.radial_block[0]

    @property
    def x(self) -> np.ndarray:
        return self.radial_block[1]

    @property
    def y(self) -> np.ndarray:
        return self.radial_block[2]

    @property
    def rho_powers(self) -> np.ndarray:
        return self.radial_block[3 : 5 + self.K_max]

    @property
    def T(self) -> np.ndarray:
        return self.radial_block[5 + self.K_max : 6 + self.K_max + self.L_max]

    @property
    def T_r(self) -> np.ndarray:
        return self.radial_block[6 + self.K_max + self.L_max : 7 + self.K_max + 2 * self.L_max]

    @property
    def T_rr(self) -> np.ndarray:
        return self.radial_block[7 + self.K_max + 2 * self.L_max : 8 + self.K_max + 3 * self.L_max]

    @property
    def theta(self) -> np.ndarray:
        return self.poloidal_block[0]

    @property
    def cos_mtheta(self) -> np.ndarray:
        return self.poloidal_block[1 : 2 + self.M_max]

    @property
    def sin_mtheta(self) -> np.ndarray:
        return self.poloidal_block[2 + self.M_max : 3 + 2 * self.M_max]

    @property
    def m_cos_mtheta(self) -> np.ndarray:
        return self.poloidal_block[3 + 2 * self.M_max : 4 + 3 * self.M_max]

    @property
    def m_sin_mtheta(self) -> np.ndarray:
        return self.poloidal_block[4 + 3 * self.M_max : 5 + 4 * self.M_max]

    @property
    def m2_cos_mtheta(self) -> np.ndarray:
        return self.poloidal_block[5 + 4 * self.M_max : 6 + 5 * self.M_max]

    @property
    def m2_sin_mtheta(self) -> np.ndarray:
        return self.poloidal_block[6 + 5 * self.M_max : 7 + 6 * self.M_max]

    def resolve_fourier_power(self, order: int) -> int:
        """返回 Fourier shape profile 的径向 prefactor 幂次."""
        order = int(order)
        if order <= 0:
            return 0
        if order <= self.M_max:
            return int(self.K_values[order])
        return min(order, self.K_max)

    def to_grid(self) -> Grid:
        """从快照重建完整 Grid（用于 Equilibrium 物化等需要真实 Grid 的场景）."""
        from veqpy.model.grid import Grid

        return Grid(
            Nr=self.Nr,
            Nt=self.Nt,
            L_max=self.L_max,
            M_max=self.M_max,
            K_max=self.K_max,
            scheme=self.scheme,
            calculus=self.calculus,
        )


@dataclass(frozen=True, slots=True)
class ResidualBindingLayout:
    """绑定到 residual binder 的只读 metadata."""

    active_profile_names: tuple[str, ...]
    active_residual_block_codes: np.ndarray
    active_residual_block_orders: np.ndarray
    active_residual_block_radial_powers: np.ndarray


@dataclass(slots=True)
class RuntimeLayout:
    """绑定到 residual 热路径的可变 runtime 缓冲区."""

    active_profile_slab: np.ndarray
    family_field_slab: np.ndarray
    source_runtime_state: SourceRuntimeState
    geometry_surface_workspace: np.ndarray
    geometry_radial_workspace: np.ndarray
    residual_surface_workspace: np.ndarray
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
class ExecutionState:
    """绑定到 operator 执行策略的可变 runner/state 容器."""

    profile_stage_runner: Callable
    profile_postprocess_runner: Callable
    geometry_stage_runner: Callable
    source_eval_runner: Callable
    source_stage_runner: Callable
    residual_pack_stage_runner: Callable
    residual_full_stage_runner: Callable
    fused_residual_runner: Callable
    fused_alpha_state: np.ndarray


@dataclass(slots=True)
class SourceConstState:
    """绑定到 source route 的只读 const arrays."""

    barycentric_weights: np.ndarray
    fixed_remap_matrix: np.ndarray
    heat_projection_fit_matrix: np.ndarray
    current_projection_fit_matrix: np.ndarray
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

    heat_projection_coeff: np.ndarray
    current_projection_coeff: np.ndarray
    heat_spline_coeff: np.ndarray
    current_spline_coeff: np.ndarray
    target_root_fields: np.ndarray


@dataclass(slots=True)
class SourceRuntimeState:
    """绑定到 source materialization/remap 的 runtime ownership 容器."""

    cache_key: tuple[str, str, int] | None
    const_state: SourceConstState
    work_state: SourceWorkState
    aux_state: SourceAuxState


@dataclass(slots=True)
class BackendState:
    """绑定到 backend ABI 的 arrays-only state 聚合入口."""

    static_layout: StaticLayout
    residual_binding_layout: ResidualBindingLayout
    runtime_layout: RuntimeLayout
    field_runtime_state: FieldRuntimeState
    source_runtime_state: SourceRuntimeState


@dataclass(slots=True)
class RuntimeAllocationBundle:
    """Operator runtime 一次性分配结果."""

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
        heat_projection_fit_matrix=np.empty((0, 0), dtype=np.float64),
        current_projection_fit_matrix=np.empty((0, 0), dtype=np.float64),
        endpoint_blend=np.linspace(0.0, 1.0, nr, dtype=np.float64),
    )
    source_work_state = SourceWorkState(
        psin_query=psin_query,
        parameter_query=parameter_query,
        materialized_heat_input=materialized_heat_input,
        materialized_current_input=materialized_current_input,
        scratch_1d=np.empty((7, nr), dtype=np.float64),
        scratch_2d=np.empty((1, nr, nt), dtype=np.float64),
    )
    source_aux_state = SourceAuxState(
        heat_projection_coeff=np.empty(0, dtype=np.float64),
        current_projection_coeff=np.empty(0, dtype=np.float64),
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

    runtime_layout = RuntimeLayout(
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

"""
Module: operator.operator

Role:
- 负责连接 case, grid, model runtime, engine kernels 与 packed layout.
- 负责暴露稳定的 residual 求值接口.

Public API:
- Operator

Notes:
- `Operator` 是默认 fused 算子.
- 不负责 solver 迭代策略, backend 选择, 或 benchmark 编排.
"""

from __future__ import annotations

from dataclasses import InitVar, dataclass, field

import numpy as np

from veqpy.engine import numba_residual
from veqpy.layout import OperatorLayout, build_operator_layout
from veqpy.model.equilibrium import Equilibrium
from veqpy.model.grid import Grid
from veqpy.model.profile import Profile
from veqpy.operator.build_plan import (
    OperatorBuildPlan,
    build_operator_plan,
    refresh_operator_plan_for_case,
)
from veqpy.operator.operator_case import OperatorCase
from veqpy.operator.packed_layout import (
    decode_packed_blocks,
    encode_packed_state,
)
from veqpy.operator.profile_runtime import (
    make_profile,
    refresh_fourier_family_base_fields,
    refresh_fourier_family_metadata,
    refresh_profile_runtime,
    refresh_stage_a_runtime,
    validate_case_compatibility,
)
from veqpy.operator.source_plan import (
    validate_source_inputs,
    validate_source_plan_profile_support,
)
from veqpy.operator.source_runtime import refresh_source_runtime
from veqpy.workspace import (
    BackendState,
    FieldRuntimeState,
    OperatorWorkspace,
    allocate_runtime_state,
)


@dataclass(slots=True)
class Operator:
    """封装固定 case, grid 与 runtime 的 residual 求值器."""

    grid: InitVar[Grid]
    case: OperatorCase = field(repr=False)
    fix_rho: float = 0.05
    source_interpolation_kind: str = "barycentric"
    plan: OperatorBuildPlan = field(init=False, repr=False)
    workspace: OperatorWorkspace = field(init=False, repr=False)
    layout: OperatorLayout = field(init=False, repr=False)
    backend_state: BackendState = field(init=False, repr=False)

    h_profile: Profile = field(init=False)
    v_profile: Profile = field(init=False)
    k_profile: Profile = field(init=False)
    psin_profile: Profile = field(init=False)
    F_profile: Profile = field(init=False)
    profiles_by_name: dict[str, Profile] = field(init=False, repr=False)


    c_effective_order: int = field(init=False, repr=False)
    s_effective_order: int = field(init=False, repr=False)
    field_runtime_state: FieldRuntimeState = field(init=False, repr=False)

    def __post_init__(self, grid: Grid) -> None:
        """完成 layout 构造, 运行时缓冲区分配和 case 绑定.

        grid 在构造时被降低为 StaticLayout 快照，之后 Operator 不再读取实时 Grid.
        """
        self._apply_plan(
            build_operator_plan(
                grid=grid,
                case=self.case,
                source_interpolation_kind=self.source_interpolation_kind,
            )
        )
        self._validate_runtime_profile_support()

        self.layout = OperatorLayout.empty(self.plan.x_size)
        self._setup_runtime_state()
        self._refresh_runtime_state()

    def _apply_plan(self, plan: OperatorBuildPlan) -> None:
        """Install the operator topology/configuration plan."""

        self.plan = plan

    def __call__(self, x: np.ndarray, *args, **kwargs) -> np.ndarray:
        """调用 variational residual 求值主入口."""
        return self.residual_var(x, *args, **kwargs)

    @property
    def alpha1(self) -> float:
        return float(self.layout.alpha_state[0])

    @alpha1.setter
    def alpha1(self, value: float) -> None:
        self.layout.alpha_state[0] = float(value)

    @property
    def alpha2(self) -> float:
        return float(self.layout.alpha_state[1])

    @alpha2.setter
    def alpha2(self, value: float) -> None:
        self.layout.alpha_state[1] = float(value)

    # Solver-facing plan accessors kept as the public facade; Operator does not
    # mirror these fields as mutable state.
    @property
    def x_size(self) -> int:
        return self.plan.x_size

    @property
    def profile_names(self) -> tuple[str, ...]:
        return self.plan.profile_names

    @property
    def active_profile_ids(self) -> np.ndarray:
        return self.plan.active_profile_ids

    def residual_block_lengths(self) -> np.ndarray:
        """Return packed residual block lengths for solver normalization.

        This is a narrow solver-facing view; raw workspace indexing arrays remain
        owned by ``ProfileWorkspace``.
        """
        return self.workspace.profile.active_lengths.copy()

    def active_profile_blocks(self) -> tuple[tuple[int, str, np.ndarray, float, float], ...]:
        """Return solver-scale metadata for active packed profile blocks.

        Each item is ``(profile_id, profile_name, coeff_indices, offset, scale)``.
        Coefficient index arrays are copies so callers do not depend on workspace
        storage layout.
        """

        profile_workspace = self.workspace.profile
        blocks: list[tuple[int, str, np.ndarray, float, float]] = []
        for slot, profile_id in enumerate(self.plan.active_profile_ids):
            length = int(profile_workspace.active_lengths[slot])
            if length <= 0:
                continue
            p = int(profile_id)
            blocks.append(
                (
                    p,
                    self.plan.profile_names[p],
                    profile_workspace.active_coeff_index_rows[slot, :length].copy(),
                    float(profile_workspace.active_offsets[slot]),
                    float(profile_workspace.active_scales[slot]),
                )
            )
        return tuple(blocks)

    def build_boundary_slope_initial_state(
        self, *, boundary_slope_factor: float = 1.0
    ) -> np.ndarray:
        """Build a boundary-scaled packed x0 for active c/s Fourier profiles."""

        x = np.zeros(self.plan.x_size, dtype=np.float64)
        target_factor = float(boundary_slope_factor)
        active_slot_by_profile_id = self.workspace.profile.active_slot_by_profile_id
        active_lengths = self.workspace.profile.active_lengths
        active_coeff_index_rows = self.workspace.profile.active_coeff_index_rows
        for profile_id, name in enumerate(self.plan.profile_names):
            if not (name.startswith("c") or name.startswith("s")):
                continue
            slot = int(active_slot_by_profile_id[int(profile_id)])
            if slot < 0 or int(active_lengths[slot]) <= 0:
                continue
            profile = self.profiles_by_name[name]
            power = int(profile.power)
            offset = float(profile.offset)
            if power <= 0 or abs(offset) <= 1.0e-14:
                continue
            coeff_index = int(active_coeff_index_rows[slot, 0])
            x[coeff_index] = 0.5 * (float(power) - target_factor) * offset
        return x

    def _validate_runtime_profile_support(self) -> None:
        """校验当前 source route 对 psin profile ownership 的要求."""
        validate_source_plan_profile_support(
            source_plan=self.plan.source_plan,
            source_execution=self.plan.source_execution,
            case=self.case,
        )
        return None

    def replace_case(self, case: OperatorCase) -> None:
        """在不改变 packed layout 的前提下替换当前 case."""
        validate_case_compatibility(
            case,
            profile_names=self.plan.profile_names,
            prefix_profile_names=self.plan.prefix_profile_names,
            profile_L=self.plan.profile_L,
            coeff_index=self.plan.coeff_index,
            order_offsets=self.plan.order_offsets,
            validate_source_inputs=lambda next_case: validate_source_inputs(
                next_case, self.plan.static_layout.Nr
            ),
        )
        self.case = case
        self._refresh_runtime_state()

    def encode_initial_state(self) -> np.ndarray:
        """把当前 case 中的 profile 系数编码成 packed 初值."""
        return encode_packed_state(
            self.case.profile_coeffs,
            self.plan.profile_L,
            self.plan.coeff_index,
            profile_names=self.plan.profile_names,
        )

    def residual_var(
        self,
        x: np.ndarray,
    ) -> np.ndarray:
        """返回 variational/Galerkin residual 向量."""
        out = np.empty(self.plan.x_size, dtype=np.float64)
        self.residual_var_into(x, out)
        return out

    def residual_var_into(self, x: np.ndarray, out: np.ndarray) -> None:
        """把 variational/Galerkin residual 写入调用方提供的 ``out``."""
        x_eval = self.coerce_x(x)
        if not isinstance(out, np.ndarray):
            raise TypeError("Expected out to be a numpy.ndarray")
        out_eval = out
        if out_eval.dtype != np.float64:
            raise TypeError(f"Expected out dtype float64, got {out_eval.dtype}")
        if out_eval.ndim != 1 or out_eval.shape[0] != self.plan.x_size:
            raise ValueError(
                f"Expected out to have shape ({self.plan.x_size},), got {out_eval.shape}"
            )
        if not out_eval.flags.c_contiguous:
            raise ValueError("Expected out to be C-contiguous")
        self.layout.run_fused_residual_into(x_eval, out_eval)

    def residual_collocation(self, x: np.ndarray) -> np.ndarray:
        """返回 DESC-style 点值 force-balance collocation residual.

        该 residual 不是把 Galerkin/弱形式残差追加到外部目标, 而是在每个
        collocation node 上直接约束与 Grad-Shafranov residual `G` 对应的
        force-balance components ``G*psin_R`` 和 ``G*psin_Z``.
        这里 `G = J/R * GS_residual`; 这两个分量对应体积化后的
        pointwise force-balance residual, radial/poloidal quadrature 的
        平方根权重用于离散 least-squares 标度.
        返回形状为 ``(2 * Nr * Nt,)`` 的向量.
        """
        out = np.empty(
            2 * self.plan.static_layout.Nr * self.plan.static_layout.Nt, dtype=np.float64
        )
        self.residual_collocation_into(x, out)
        return out

    def residual_collocation_into(self, x: np.ndarray, out: np.ndarray) -> None:
        """把 DESC-style collocation residual 写入调用方提供的 ``out``."""
        expected_size = 2 * self.plan.static_layout.Nr * self.plan.static_layout.Nt
        if not isinstance(out, np.ndarray):
            raise TypeError("Expected out to be a numpy.ndarray")
        out_eval = out
        if out_eval.dtype != np.float64:
            raise TypeError(f"Expected out dtype float64, got {out_eval.dtype}")
        if out_eval.ndim != 1 or out_eval.shape[0] != expected_size:
            raise ValueError(f"Expected out to have shape ({expected_size},), got {out_eval.shape}")
        if not out_eval.flags.c_contiguous:
            raise ValueError("Expected out to be C-contiguous")
        self._evaluate_collocation_workspace(x)
        block_size = self.plan.static_layout.Nr * self.plan.static_layout.Nt
        residual_workspace = self.workspace.residual
        numba_residual.write_weighted_collocation_field_into(
            out_eval,
            residual_workspace.surface_workspace[1],
            residual_workspace.collocation_sqrt_weights,
            0,
        )
        numba_residual.write_weighted_collocation_field_into(
            out_eval,
            residual_workspace.surface_workspace[2],
            residual_workspace.collocation_sqrt_weights,
            block_size,
        )

    def _evaluate_collocation_workspace(self, x: np.ndarray) -> None:
        x_eval = self.coerce_x(x)
        self.stage_a_profile(x_eval)
        self.stage_b_geometry()
        self.stage_c_source()
        self._update_residual_surface_workspace()

    def _evaluate_residual(self, x_eval: np.ndarray) -> np.ndarray:
        out = np.empty(self.plan.x_size, dtype=np.float64)
        self._evaluate_residual_into(x_eval, out)
        return out

    def _evaluate_residual_into(self, x_eval: np.ndarray, out: np.ndarray) -> None:
        self.stage_a_profile(x_eval)
        self.stage_b_geometry()
        self.stage_c_source()
        self.stage_d_residual_into(out)

    def build_coeffs(
        self, x: np.ndarray, *, include_none: bool = True
    ) -> dict[str, list[float] | None]:
        """把 packed 状态向量还原成 profile 系数字典."""
        blocks = decode_packed_blocks(
            x, self.plan.profile_L, self.plan.coeff_index, profile_names=self.plan.profile_names
        )
        coeffs: dict[str, list[float] | None] = {}
        for name, block in zip(self.plan.profile_names, blocks, strict=True):
            if include_none or block is not None:
                coeffs[name] = None if block is None else block.tolist()
        return coeffs

    def build_equilibrium(self, x: np.ndarray) -> Equilibrium:
        """从 packed 状态向量构造完整 Equilibrium 快照."""
        x_eval = self.coerce_x(x)
        self.residual_var(x_eval)
        return self._snapshot_equilibrium_from_runtime(x_eval)

    def stage_a_profile(self, x: np.ndarray) -> None:
        """执行 profile 阶段并刷新 active profile fields."""
        self.layout.run_profile(x)

    def stage_b_geometry(self) -> None:
        """执行 geometry 阶段并刷新 geometry fields."""
        self.layout.run_geometry()

    def stage_c_source(self) -> None:
        """执行 source 阶段并刷新 root fields 与缩放系数."""
        self.layout.run_source()

    def stage_d_residual(self) -> np.ndarray:
        """执行 residual 阶段并返回 packed 残差."""
        out = np.empty(self.plan.x_size, dtype=np.float64)
        self.stage_d_residual_into(out)
        return out

    def stage_d_residual_into(self, out: np.ndarray) -> None:
        """执行 residual 阶段并把 packed 残差写入 ``out``."""
        if not isinstance(out, np.ndarray):
            raise TypeError("Expected out to be a numpy.ndarray")
        out_eval = out
        if out_eval.dtype != np.float64:
            raise TypeError(f"Expected out dtype float64, got {out_eval.dtype}")
        if out_eval.ndim != 1 or out_eval.shape[0] != self.plan.x_size:
            raise ValueError(
                f"Expected out to have shape ({self.plan.x_size},), got {out_eval.shape}"
            )
        if not out_eval.flags.c_contiguous:
            raise ValueError("Expected out to be C-contiguous")
        self.layout.run_residual_into(out_eval)

    def _update_residual_surface_workspace(self) -> None:
        residual_workspace = self.workspace.residual
        numba_residual.update_residual_compact(
            residual_workspace.surface_workspace,
            self.alpha1,
            self.alpha2,
            residual_workspace.root_fields,
            self.workspace.geometry.surface_workspace,
        )

    def coerce_x(self, x: np.ndarray) -> np.ndarray:
        """校验完整 packed 状态向量形状."""
        arr = np.asarray(x, dtype=np.float64)
        if arr.ndim != 1 or arr.shape[0] != self.plan.x_size:
            raise ValueError(f"Expected x to have shape ({self.plan.x_size},), got {arr.shape}")
        return arr

    def _refresh_workspace_views(self) -> None:
        workspace = self.workspace
        workspace.h_fields = self.h_profile.u_fields
        workspace.v_fields = self.v_profile.u_fields
        workspace.k_fields = self.k_profile.u_fields
        workspace.F_profile_u = self.F_profile.u
        workspace.F_profile_fields = self.F_profile.u_fields
        workspace.psin_profile_u = self.psin_profile.u
        workspace.psin_profile_fields = self.psin_profile.u_fields

    def _setup_runtime_state(self) -> None:
        bundle = allocate_runtime_state(
            static_layout=self.plan.static_layout,
            source_execution=self.plan.source_execution,
            profile_names=self.plan.profile_names,
            profile_index=self.plan.profile_index,
            active_profile_ids=self.plan.active_profile_ids,
            profile_L=self.plan.profile_L,
            x_size=self.plan.x_size,
            make_profile=lambda name: make_profile(
                case=self.case,
                operator_grid=self.plan.static_layout,
                name=name,
                profile_L=self.plan.profile_L,
                profile_names=self.plan.profile_names,
                profile_index=self.plan.profile_index,
                profile_static_kwargs_by_name=self.plan.profile_static_kwargs_by_name,
                profile_offset_specs=self.plan.profile_offset_specs,
            ),
        )
        self.profiles_by_name = bundle.profiles_by_name
        for name, profile in self.profiles_by_name.items():
            if hasattr(type(self), f"{name}_profile"):
                setattr(self, f"{name}_profile", profile)
        self.workspace = bundle.workspace
        self.field_runtime_state = bundle.field_runtime_state

    def _refresh_runtime_state(self) -> None:
        self._apply_plan(
            refresh_operator_plan_for_case(
                self.plan,
                case=self.case,
                source_interpolation_kind=self.source_interpolation_kind,
            )
        )
        self._validate_runtime_profile_support()
        self._refresh_profile_runtime()
        self._refresh_fourier_family_metadata()
        refresh_source_runtime(
            case=self.case,
            grid_rho=self.plan.static_layout.rho,
            source_plan=self.plan.source_plan,
            source_execution=self.plan.source_execution,
            source_runtime_state=self.workspace.source.runtime_state,
            psin=self.workspace.residual.root_fields[0],
        )
        self._refresh_stage_a_runtime()
        self._refresh_workspace_views()
        self._refresh_backend_state()
        self._refresh_runtime_bindings()

    def _refresh_profile_runtime(self) -> None:
        refresh_profile_runtime(
            case=self.case,
            operator_grid=self.plan.static_layout,
            profile_names=self.plan.profile_names,
            profile_index=self.plan.profile_index,
            profile_L=self.plan.profile_L,
            profiles_by_name=self.profiles_by_name,
            profile_static_kwargs_by_name=self.plan.profile_static_kwargs_by_name,
            profile_offset_specs=self.plan.profile_offset_specs,
            refresh_fourier_family_base_fields=lambda: refresh_fourier_family_base_fields(
                M_max=self.plan.static_layout.M_max,
                profile_index=self.plan.profile_index,
                profiles_by_name=self.profiles_by_name,
                c_family_base_fields=self.workspace.profile.c_family_base_fields,
                s_family_base_fields=self.workspace.profile.s_family_base_fields,
            ),
        )

    def _refresh_runtime_bindings(self) -> None:
        alpha_state = self.layout.alpha_state
        self.layout = build_operator_layout(
            plan=self.plan,
            case=self.case,
            workspace=self.workspace,
            backend_state=self.backend_state,
            alpha_state=alpha_state,
            c_effective_order=self.c_effective_order,
            s_effective_order=self.s_effective_order,
            fix_rho=self.fix_rho,
            psin_profile_fields_available=self.psin_profile.u_fields is not None,
            fallback_residual_runner_into=self._evaluate_residual_into,
        )
        fixed_profile_ids = np.flatnonzero(~self.plan.active_profile_mask).astype(
            np.int64, copy=False
        )
        for p in fixed_profile_ids:
            self.profiles_by_name[self.plan.profile_names[int(p)]].update()
        f_profile_id = self.plan.profile_index.get("F", -1)
        if f_profile_id >= 0 and not bool(self.plan.active_profile_mask[f_profile_id]):
            self.layout.profile.run_postprocess()

    def _refresh_backend_state(self) -> None:
        self.backend_state = BackendState(
            static_layout=self.plan.static_layout,
            residual_binding_layout=self.plan.residual_binding_layout,
            workspace=self.workspace,
            field_runtime_state=self.field_runtime_state,
            source_runtime_state=self.workspace.source.runtime_state,
        )

    def _refresh_stage_a_runtime(self) -> None:
        profile_workspace = self.workspace.profile
        refresh_stage_a_runtime(
            active_profile_ids=self.plan.active_profile_ids,
            profile_names=self.plan.profile_names,
            profiles_by_name=self.profiles_by_name,
            profile_L=self.plan.profile_L,
            coeff_index=self.plan.coeff_index,
            active_u_fields=profile_workspace.active_u_fields,
            active_rp_fields=profile_workspace.active_rp_fields,
            active_env_fields=profile_workspace.active_env_fields,
            active_offsets=profile_workspace.active_offsets,
            active_scales=profile_workspace.active_scales,
            active_lengths=profile_workspace.active_lengths,
            active_coeff_index_rows=profile_workspace.active_coeff_index_rows,
        )

    def _refresh_fourier_family_metadata(self) -> None:
        self.c_effective_order, self.s_effective_order = refresh_fourier_family_metadata(
            c_profile_names=self.plan.c_profile_names,
            s_profile_names=self.plan.s_profile_names,
            profile_coeffs=self.case.profile_coeffs,
            c_offsets=self.case.c_offsets,
            s_offsets=self.case.s_offsets,
            c_family_fields=self.workspace.profile.c_family_fields,
            s_family_fields=self.workspace.profile.s_family_fields,
        )

    def invalidate_source_state(self) -> None:
        if tuple(self.plan.source_execution.route_key) == ("PJ2", "psin", "uniform"):
            self.workspace.source.work_state.psin_query.fill(-1.0)

    def _snapshot_equilibrium_from_runtime(self, x: np.ndarray) -> Equilibrium:
        root_fields = self.workspace.residual.root_fields
        return snapshot_equilibrium_from_runtime(
            x,
            case=self.case,
            grid=self.plan.static_layout.to_grid(),
            profile_L=self.plan.profile_L,
            coeff_index=self.plan.coeff_index,
            profile_names=self.plan.profile_names,
            shape_profile_names=self.plan.shape_profile_names,
            profile_index=self.plan.profile_index,
            profiles_by_name=self.profiles_by_name,
            psin=root_fields[0],
            FFn_psin=root_fields[3],
            Pn_psin=root_fields[4],
            psin_r=root_fields[1],
            psin_rr=root_fields[2],
            alpha1=self.alpha1,
            alpha2=self.alpha2,
        )


def snapshot_equilibrium_from_runtime(
    x: np.ndarray,
    *,
    case: OperatorCase,
    grid: Grid,
    profile_L: np.ndarray,
    coeff_index: np.ndarray,
    profile_names: tuple[str, ...],
    shape_profile_names: tuple[str, ...],
    profile_index: dict[str, int],
    profiles_by_name: dict[str, Profile],
    psin: np.ndarray,
    FFn_psin: np.ndarray,
    Pn_psin: np.ndarray,
    psin_r: np.ndarray,
    psin_rr: np.ndarray,
    alpha1: float,
    alpha2: float,
) -> Equilibrium:
    """Materialize an Equilibrium snapshot from current Operator runtime arrays."""
    coeff_blocks = decode_packed_blocks(x, profile_L, coeff_index, profile_names=profile_names)
    shape_profiles = snapshot_equilibrium_profiles(
        coeff_blocks,
        shape_profile_names=shape_profile_names,
        profile_index=profile_index,
        profiles_by_name=profiles_by_name,
    )
    return Equilibrium(
        R0=case.R0,
        Z0=case.Z0,
        B0=case.B0,
        a=case.a,
        grid=grid,
        shape_profiles=shape_profiles,
        psin=psin.copy(),
        FFn_psin=np.asarray(FFn_psin, dtype=np.float64).copy(),
        Pn_psin=Pn_psin.copy(),
        psin_r=psin_r.copy(),
        psin_rr=psin_rr.copy(),
        alpha1=float(alpha1),
        alpha2=float(alpha2),
    )


def snapshot_equilibrium_profiles(
    coeff_blocks: tuple[np.ndarray | None, ...],
    *,
    shape_profile_names: tuple[str, ...],
    profile_index: dict[str, int],
    profiles_by_name: dict[str, Profile],
) -> dict[str, Profile]:
    return {
        name: snapshot_profile(profiles_by_name[name], coeff_blocks[profile_index[name]])
        for name in shape_profile_names
    }


def snapshot_profile(profile: Profile, coeff_block: np.ndarray | None) -> Profile:
    copied = profile.copy()
    copied.coeff = None if coeff_block is None else coeff_block.copy()
    return copied

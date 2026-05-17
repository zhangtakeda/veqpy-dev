"""
Operator layout binding.

This module owns Python closure wiring for executable operator layouts.  The
Operator facade refreshes plan/workspace/backend state, then calls this module
to bind hot-path callables against those already-refreshed objects.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable

import numpy as np

from veqpy.engine import numba_operator, numba_profile, numba_residual
from veqpy.layout.geometry_binding import build_geometry_stage_runner
from veqpy.layout.source_binding import build_bound_source_stage_runner
from veqpy.workspace import BackendState, OperatorWorkspace

if TYPE_CHECKING:
    from veqpy.operator.build_plan import OperatorBuildPlan
    from veqpy.operator.operator_case import OperatorCase

from .runtime import OperatorLayout


def build_operator_layout(
    *,
    plan: "OperatorBuildPlan",
    case: "OperatorCase",
    workspace: OperatorWorkspace,
    backend_state: BackendState,
    alpha_state: np.ndarray,
    c_effective_order: int,
    s_effective_order: int,
    fix_rho: float,
    psin_profile_fields_available: bool,
    fallback_residual_runner_into: Callable[[np.ndarray, np.ndarray], None],
) -> OperatorLayout:
    """Bind a full executable ``OperatorLayout`` from refreshed runtime state."""

    profile_stage_runner = _build_profile_stage_runner(plan=plan, workspace=workspace)
    profile_postprocess_runner = _build_profile_postprocess_runner(workspace=workspace)
    geometry_stage_runner = _build_geometry_stage_runner(
        plan=plan,
        case=case,
        workspace=workspace,
        c_effective_order=c_effective_order,
        s_effective_order=s_effective_order,
    )
    source_eval_runner = numba_operator.bind_source_eval_runner(
        source_plan=plan.source_plan,
        backend_state=backend_state,
        B0=case.B0,
        fix_rho=fix_rho,
    )
    source_stage_runner = _build_bound_source_stage_runner(
        plan=plan,
        case=case,
        workspace=workspace,
        fix_rho=fix_rho,
        source_eval_runner=source_eval_runner,
    )
    residual_full_stage_runner_into = _build_bound_residual_full_stage_runner_into(
        plan=plan,
        case=case,
        workspace=workspace,
        alpha_state=alpha_state,
    )
    residual_full_stage_runner = _build_bound_residual_full_stage_runner(
        plan=plan,
        runner_into=residual_full_stage_runner_into,
    )
    fused_residual_runner_into = _build_fused_residual_runner_into(
        plan=plan,
        case=case,
        backend_state=backend_state,
        alpha_state=alpha_state,
        c_effective_order=c_effective_order,
        s_effective_order=s_effective_order,
        fix_rho=fix_rho,
        psin_profile_fields_available=psin_profile_fields_available,
        fallback_residual_runner_into=fallback_residual_runner_into,
    )
    fused_residual_runner = _build_fused_residual_runner(
        plan=plan,
        runner_into=fused_residual_runner_into,
    )
    return OperatorLayout.from_callables(
        profile_stage_runner=profile_stage_runner,
        profile_postprocess_runner=profile_postprocess_runner,
        geometry_stage_runner=geometry_stage_runner,
        source_eval_runner=source_eval_runner,
        source_stage_runner=source_stage_runner,
        residual_full_stage_runner_into=residual_full_stage_runner_into,
        residual_full_stage_runner=residual_full_stage_runner,
        fused_residual_runner_into=fused_residual_runner_into,
        fused_residual_runner=fused_residual_runner,
        alpha_state=alpha_state,
    )


def _build_profile_stage_runner(
    *,
    plan: "OperatorBuildPlan",
    workspace: OperatorWorkspace,
) -> Callable[[np.ndarray], None]:
    from veqpy.operator.profile_runtime import build_profile_stage_runner

    profile_workspace = workspace.profile
    return build_profile_stage_runner(
        active_profile_ids=plan.active_profile_ids,
        active_profile_slab=profile_workspace.active_profile_slab,
        T=plan.static_layout.T,
        T_r=plan.static_layout.T_r,
        T_rr=plan.static_layout.T_rr,
        active_offsets=profile_workspace.active_offsets,
        active_scales=profile_workspace.active_scales,
        active_coeff_index_rows=profile_workspace.active_coeff_index_rows,
        active_lengths=profile_workspace.active_lengths,
        update_profiles_packed_bulk=numba_profile.update_profiles_packed_bulk,
    )


def _build_profile_postprocess_runner(
    *,
    workspace: OperatorWorkspace,
) -> Callable[[], None]:
    eps = 1.0e-10

    def runner() -> None:
        numba_operator.convert_f_squared_fields_to_f(workspace.F_profile_fields, eps=eps)

    return runner


def _build_geometry_stage_runner(
    *,
    plan: "OperatorBuildPlan",
    case: "OperatorCase",
    workspace: OperatorWorkspace,
    c_effective_order: int,
    s_effective_order: int,
) -> Callable[[], None]:
    profile_workspace = workspace.profile
    geometry_workspace = workspace.geometry
    return build_geometry_stage_runner(
        c_family_fields=profile_workspace.c_family_fields,
        s_family_fields=profile_workspace.s_family_fields,
        c_family_base_fields=profile_workspace.c_family_base_fields,
        s_family_base_fields=profile_workspace.s_family_base_fields,
        active_u_fields=profile_workspace.active_u_fields,
        c_family_source_slots=profile_workspace.c_family_source_slots,
        s_family_source_slots=profile_workspace.s_family_source_slots,
        c_effective_order=c_effective_order,
        s_effective_order=s_effective_order,
        h_fields=workspace.h_fields,
        v_fields=workspace.v_fields,
        k_fields=workspace.k_fields,
        a=case.a,
        R0=case.R0,
        Z0=case.Z0,
        surface_workspace=geometry_workspace.surface_workspace,
        radial_workspace=geometry_workspace.radial_workspace,
        rho=plan.static_layout.rho,
        theta=plan.static_layout.theta,
        cos_mtheta=plan.static_layout.cos_mtheta,
        sin_mtheta=plan.static_layout.sin_mtheta,
        m_cos_mtheta=plan.static_layout.m_cos_mtheta,
        m_sin_mtheta=plan.static_layout.m_sin_mtheta,
        m2_cos_mtheta=plan.static_layout.m2_cos_mtheta,
        m2_sin_mtheta=plan.static_layout.m2_sin_mtheta,
    )


def _build_bound_source_stage_runner(
    *,
    plan: "OperatorBuildPlan",
    case: "OperatorCase",
    workspace: OperatorWorkspace,
    fix_rho: float,
    source_eval_runner: Callable,
) -> Callable[[], tuple[float, float]]:
    return build_bound_source_stage_runner(
        plan=plan,
        case=case,
        workspace=workspace,
        fix_rho=fix_rho,
        source_eval_runner=source_eval_runner,
    )


def _build_bound_residual_full_stage_runner_into(
    *,
    plan: "OperatorBuildPlan",
    case: "OperatorCase",
    workspace: OperatorWorkspace,
    alpha_state: np.ndarray,
) -> Callable[[np.ndarray], None]:
    root_fields = workspace.residual.root_fields
    surface_workspace = workspace.geometry.surface_workspace
    residual_workspace = workspace.residual.surface_workspace
    residual_pack_scratch = workspace.residual.pack_scratch
    profile_workspace = workspace.profile
    sin_mtheta = plan.static_layout.sin_mtheta
    cos_mtheta = plan.static_layout.cos_mtheta
    rho_powers = plan.static_layout.rho_powers
    y = plan.static_layout.y
    T = plan.static_layout.T
    weights = plan.static_layout.weights
    a = case.a
    R0 = case.R0
    B0 = case.B0

    def runner(out: np.ndarray) -> None:
        numba_residual.update_residual_compact(
            residual_workspace,
            float(alpha_state[0]),
            float(alpha_state[1]),
            root_fields,
            surface_workspace,
        )
        out.fill(0.0)
        numba_residual.run_residual_blocks_packed_precomputed(
            out,
            residual_pack_scratch,
            plan.residual_binding_layout.active_residual_block_codes,
            plan.residual_binding_layout.active_residual_block_orders,
            plan.residual_binding_layout.active_residual_block_radial_powers,
            profile_workspace.active_coeff_index_rows,
            profile_workspace.active_lengths,
            residual_workspace,
            sin_mtheta,
            cos_mtheta,
            rho_powers,
            y,
            T,
            weights,
            a,
            R0,
            B0,
        )

    return runner


def _build_bound_residual_full_stage_runner(
    *,
    plan: "OperatorBuildPlan",
    runner_into: Callable[[np.ndarray], None],
) -> Callable[[], np.ndarray]:
    def runner() -> np.ndarray:
        out = np.empty(plan.x_size, dtype=np.float64)
        runner_into(out)
        return out

    return runner


def _build_fused_residual_runner_into(
    *,
    plan: "OperatorBuildPlan",
    case: "OperatorCase",
    backend_state: BackendState,
    alpha_state: np.ndarray,
    c_effective_order: int,
    s_effective_order: int,
    fix_rho: float,
    psin_profile_fields_available: bool,
    fallback_residual_runner_into: Callable[[np.ndarray, np.ndarray], None],
) -> Callable[[np.ndarray, np.ndarray], None]:
    if plan.source_execution.requires_optimized_psin_profile and not psin_profile_fields_available:
        return fallback_residual_runner_into
    return numba_operator.bind_fused_residual_runner_into(
        source_plan=plan.source_plan,
        source_execution=plan.source_execution,
        backend_state=backend_state,
        alpha_state=alpha_state,
        c_active_order=int(c_effective_order),
        s_active_order=int(s_effective_order),
        a=float(case.a),
        R0=float(case.R0),
        Z0=float(case.Z0),
        B0=float(case.B0),
        fix_rho=fix_rho,
    )


def _build_fused_residual_runner(
    *,
    plan: "OperatorBuildPlan",
    runner_into: Callable[[np.ndarray, np.ndarray], None],
) -> Callable[[np.ndarray], np.ndarray]:
    def runner(x_eval: np.ndarray) -> np.ndarray:
        out = np.empty(plan.x_size, dtype=np.float64)
        runner_into(x_eval, out)
        return out

    return runner

"""
Module: engine.__init__

Role:
- 负责导出 engine 层稳定入口.

Public API:
- update_profile
- update_profiles_packed_bulk
- update_geometry
- update_residual
- bind_fused_residual_runner
- bind_fused_single_pass_residual_runner
- bind_fused_profile_owned_psin_residual_runner
- bind_fused_fixed_point_psin_residual_runner
- bind_residual_runner
- bind_residual_stage_runner
- build_source_remap_cache
- materialize_profile_owned_psin_source
- update_fourier_family_fields
- resolve_source_inputs
- resolve_source_scratch_kernel
- validate_route

Notes:
- operator layout 与 solver orchestration 保留在上层.
- source 输入解析正式入口是 `resolve_source_inputs`.
"""

import os

VALID_BACKENDS = ("numba",)
BACKEND = os.environ.get("VEQPY_BACKEND", "numba")

if BACKEND not in VALID_BACKENDS:
    raise ValueError(f"Unsupported VEQPY_BACKEND={BACKEND!r}. Supported backends: {VALID_BACKENDS}")

if BACKEND == "numba":
    from veqpy.engine.numba_operator import (
        bind_fused_fixed_point_psin_residual_runner,
        bind_fused_profile_owned_psin_residual_runner,
        bind_fused_residual_runner,
        bind_fused_single_pass_residual_runner,
    )
    from veqpy.engine.numba_profile import (
        update_profile,
        update_profiles_packed_bulk,
    )
    from veqpy.engine.numba_residual import (
        bind_residual_runner,
        bind_residual_stage_runner,
        update_residual,
    )
    from veqpy.engine.numba_source import (
        COORDINATE_NAMES,
        PSIN_COORDINATE,
        RHO_AXIS,
        RHO_COORDINATE,
        THETA_AXIS,
        build_source_remap_cache,
        corrected_even_derivative,
        corrected_integration,
        corrected_linear_derivative,
        full_differentiation,
        full_integration,
        materialize_profile_owned_psin_source,
        materialize_projected_source_inputs,
        quadrature,
        resolve_source_inputs,
        resolve_source_scratch_kernel,
        theta_reduction,
        update_fixed_point_psin_query,
        update_fourier_family_fields,
        validate_route,
    )

__all__ = [
    "update_profile",
    "update_profiles_packed_bulk",
    "update_geometry",
    "update_residual",
    "bind_fused_residual_runner",
    "bind_fused_single_pass_residual_runner",
    "bind_fused_profile_owned_psin_residual_runner",
    "bind_fused_fixed_point_psin_residual_runner",
    "bind_residual_runner",
    "bind_residual_stage_runner",
    "RHO_AXIS",
    "THETA_AXIS",
    "COORDINATE_NAMES",
    "PSIN_COORDINATE",
    "RHO_COORDINATE",
    "build_source_remap_cache",
    "validate_route",
    "materialize_profile_owned_psin_source",
    "update_fourier_family_fields",
    "resolve_source_inputs",
    "resolve_source_scratch_kernel",
    "full_differentiation",
    "theta_reduction",
    "quadrature",
    "materialize_projected_source_inputs",
    "update_fixed_point_psin_query",
    "full_integration",
    "corrected_integration",
    "corrected_linear_derivative",
    "corrected_even_derivative",
]

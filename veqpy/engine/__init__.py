"""
Module: engine.__init__

Role:
- 导出 numba engine 的稳定入口.

Notes:
- 求解后端已统一为 numba.
- operator layout 与 solver orchestration 保留在上层.
"""

from veqpy.engine.numba_operator import (
    bind_fused_residual_runner,
    bind_source_eval_runner,
)
from veqpy.engine.numba_profile import update_profile, update_profiles_packed_bulk
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
    "bind_fused_residual_runner",
    "bind_source_eval_runner",
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

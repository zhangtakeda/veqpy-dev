"""
Module: veqpy.orchestration

Role:
- Centralize Python-level stage orchestration shared by operator/runtime code.
- Keep route policy, source materialization, and residual metadata out of
  numeric kernels.

Notes:
- This module assembles and refreshes runtime stages; it does not implement
  numba kernels.
- ABI dataclasses remain in ``veqpy.engine.backend_abi`` because they describe
  backend binding shapes rather than orchestration policy.
"""

from __future__ import annotations

from veqpy.layout.geometry_binding import build_geometry_stage_runner  # noqa: E402
from veqpy.layout.source_binding import build_bound_source_stage_runner  # noqa: E402

# Compatibility re-exports for legacy ``veqpy.orchestration`` imports.
from veqpy.operator.packed_layout import (  # noqa: E402
    ALL_PROFILE_FAMILIES,
    PACKED_PROFILE_FAMILY_ORDER,
    PREFIX_PROFILE_FAMILIES,
    PROFILE_OFFSET_SPECS,
    PROFILE_STATIC_KWARGS,
    RESIDUAL_BLOCK_CODE_BY_NAME,
    SHAPE_PROFILE_FAMILIES,
    build_fourier_profile_names,
    build_profile_names,
    build_residual_block_metadata,
    build_residual_block_radial_powers,
    build_shape_profile_names,
    expand_profile_family,
    get_prefix_profile_names,
    validate_profile_family_order,
)
from veqpy.operator.source_plan import (  # noqa: E402
    SourcePlan,
    build_source_plan,
    validate_source_inputs,
    validate_source_plan_profile_support,
)
from veqpy.operator.source_runtime import refresh_source_runtime  # noqa: E402

__all__ = [
    "ALL_PROFILE_FAMILIES",
    "PACKED_PROFILE_FAMILY_ORDER",
    "PREFIX_PROFILE_FAMILIES",
    "PROFILE_OFFSET_SPECS",
    "PROFILE_STATIC_KWARGS",
    "RESIDUAL_BLOCK_CODE_BY_NAME",
    "SHAPE_PROFILE_FAMILIES",
    "SourcePlan",
    "build_bound_source_stage_runner",
    "build_fourier_profile_names",
    "build_geometry_stage_runner",
    "build_profile_names",
    "build_residual_block_metadata",
    "build_residual_block_radial_powers",
    "build_shape_profile_names",
    "build_source_plan",
    "expand_profile_family",
    "get_prefix_profile_names",
    "refresh_source_runtime",
    "validate_profile_family_order",
    "validate_source_inputs",
    "validate_source_plan_profile_support",
]

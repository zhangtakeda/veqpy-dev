"""
Module: operator.__init__

Role:
- Expose the stable operator-layer public API.

Public API:
- Operator
- OperatorCase
- Packed-state naming/layout helpers used by callers that need to prepare coefficient vectors

Notes:
- Runtime layout containers stay in ``veqpy.operator.runtime_layout`` as implementation data structures.
- Engine backend selection, solver driving, and document/demo orchestration live outside this package surface.
"""

from veqpy.operator.operator import Operator
from veqpy.operator.operator_case import OperatorCase
from veqpy.operator.packed_layout import (
    build_active_profile_metadata,
    build_fourier_profile_names,
    build_profile_index,
    build_profile_layout,
    build_profile_names,
    build_shape_profile_names,
    get_prefix_profile_names,
    packed_size,
    validate_packed_state,
)

__all__ = [
    "Operator",
    "OperatorCase",
    "build_active_profile_metadata",
    "build_fourier_profile_names",
    "build_profile_index",
    "build_profile_layout",
    "build_profile_names",
    "build_shape_profile_names",
    "get_prefix_profile_names",
    "packed_size",
    "validate_packed_state",
]

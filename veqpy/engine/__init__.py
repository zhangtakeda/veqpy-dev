"""
Module: engine.__init__

Role:
- Expose supported numerical engine primitives.

Notes:
- Numba is the only supported execution backend.
- High-level Python orchestration lives in :mod:`veqpy.orchestration`.
"""

from veqpy.engine import backend_abi
from veqpy.engine.numba_source import (
    COORDINATE_NAMES,
    PSIN_COORDINATE,
    RHO_AXIS,
    RHO_COORDINATE,
    THETA_AXIS,
    corrected_even_derivative,
    corrected_integration,
    corrected_linear_derivative,
    full_differentiation,
    full_integration,
    quadrature,
    theta_reduction,
    validate_route,
)

__all__ = [
    "backend_abi",
    "RHO_AXIS",
    "THETA_AXIS",
    "COORDINATE_NAMES",
    "PSIN_COORDINATE",
    "RHO_COORDINATE",
    "validate_route",
    "full_differentiation",
    "theta_reduction",
    "quadrature",
    "full_integration",
    "corrected_integration",
    "corrected_linear_derivative",
    "corrected_even_derivative",
]

"""
Module: engine.__init__

Role:
- Expose supported numerical engine primitives used by public model/operator code.

Notes:
- Numba is the only supported execution backend.
- Backend ABI builders and residual runner wiring are submodule implementation details.
- High-level Python orchestration lives in :mod:`veqpy.orchestration`.
"""

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

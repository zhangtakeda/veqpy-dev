"""
Module: math

Role:
- Host pure mathematical construction utilities shared by model and engine layers.

Public API:
- barycentric_log_weights
- corrected_even_derivative_matrix
- corrected_integration_matrix
- corrected_linear_derivative_matrix
- differentiation_matrix
- interpolation_matrix
- quadrature_generator
- uniform_differentiation_matrix
- uniform_variable_limit_integration_matrix
- variable_limit_integration_matrix
"""

# -----------------------------------------------------------------------------
# Public interface
# -----------------------------------------------------------------------------

from veqpy.math.differentiate import (
    corrected_even_derivative_matrix,
    corrected_linear_derivative_matrix,
    differentiation_matrix,
    uniform_differentiation_matrix,
)
from veqpy.math.integrate import (
    corrected_integration_matrix,
    uniform_variable_limit_integration_matrix,
    variable_limit_integration_matrix,
)
from veqpy.math.interpolate import barycentric_log_weights, interpolation_matrix
from veqpy.math.quadrature import quadrature_generator

__all__ = [
    "barycentric_log_weights",
    "corrected_even_derivative_matrix",
    "corrected_integration_matrix",
    "corrected_linear_derivative_matrix",
    "differentiation_matrix",
    "interpolation_matrix",
    "quadrature_generator",
    "uniform_differentiation_matrix",
    "uniform_variable_limit_integration_matrix",
    "variable_limit_integration_matrix",
]

"""
Module: math

Role:
- Host pure mathematical construction utilities shared by model and engine layers.
"""

from veqpy.math.calculus import (
    corrected_differentiation_matrix,
    corrected_integration_matrix,
    make_calculus,
)
from veqpy.math.interpolate import barycentric_log_weights, interpolation_matrix
from veqpy.math.quadrature import available_quadrature_schemes, make_quadrature

__all__ = [
    "available_quadrature_schemes",
    "barycentric_log_weights",
    "corrected_differentiation_matrix",
    "corrected_integration_matrix",
    "interpolation_matrix",
    "make_calculus",
    "make_quadrature",
]

"""
Module: math

Role:
- Host pure mathematical construction utilities shared by model and engine layers.
"""

from veqpy.math.calculus import (
    cfd33_differentiation_matrix,
    cfd33_integration_matrix,
    cfd33_matrices,
    corrected_even_derivative_matrix,
    corrected_integration_matrix,
    corrected_linear_derivative_matrix,
    make_calculus,
    spectral_differentiation_matrix,
    spectral_integration_matrix,
    uniform_spectral_differentiation_matrix,
    uniform_spectral_integration_matrix,
)
from veqpy.math.interpolate import barycentric_log_weights, interpolation_matrix
from veqpy.math.quadrature import available_quadrature_schemes, make_quadrature

__all__ = [
    "available_quadrature_schemes",
    "barycentric_log_weights",
    "cfd33_differentiation_matrix",
    "cfd33_integration_matrix",
    "cfd33_matrices",
    "corrected_even_derivative_matrix",
    "corrected_integration_matrix",
    "corrected_linear_derivative_matrix",
    "interpolation_matrix",
    "make_calculus",
    "make_quadrature",
    "spectral_differentiation_matrix",
    "spectral_integration_matrix",
    "uniform_spectral_differentiation_matrix",
    "uniform_spectral_integration_matrix",
]

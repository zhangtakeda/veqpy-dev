"""
Module: math

Role:
- Host pure mathematical construction utilities shared by model and engine layers.
"""

from veqpy.math.calculus import make_calculus, make_ffn_projection_calculus
from veqpy.math.interpolate import barycentric_log_weights, interpolation_matrix
from veqpy.math.quadrature import make_quadrature

__all__ = [
    "barycentric_log_weights",
    "interpolation_matrix",
    "make_calculus",
    "make_ffn_projection_calculus",
    "make_quadrature",
]

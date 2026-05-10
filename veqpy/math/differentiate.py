"""
Module: math.differentiate

Role:
- Provide pure derivative matrix builders.

Public API:
- corrected_even_derivative_matrix
- corrected_linear_derivative_matrix
- differentiation_matrix
- uniform_differentiation_matrix
"""

from collections.abc import Callable

import numpy as np

from veqpy.math.interpolate import barycentric_log_weights

# -----------------------------------------------------------------------------
# Public interface
# -----------------------------------------------------------------------------


def differentiation_matrix(nodes: np.ndarray) -> np.ndarray:
    """Build the polynomial collocation first-derivative matrix."""

    nodes = np.asarray(nodes, dtype=np.float64)
    n = len(nodes)
    signs, log_weights = barycentric_log_weights(nodes)
    diff = nodes[:, None] - nodes[None, :]
    np.fill_diagonal(diff, 1.0)
    log_ratio = log_weights[None, :] - log_weights[:, None]
    mag_ratio = np.exp(np.clip(log_ratio, -700.0, 700.0))
    sign_ratio = signs[None, :] * signs[:, None]
    matrix = sign_ratio * mag_ratio / diff
    np.fill_diagonal(matrix, 0.0)
    matrix[np.diag_indices(n)] = -np.sum(matrix, axis=1)
    return matrix


def uniform_differentiation_matrix(n: int) -> np.ndarray:
    """Build the finite-difference first-derivative matrix on a uniform grid."""

    h = 1.0 / (n - 1)
    matrix = np.zeros((n, n), dtype=np.float64)
    matrix[0, 0] = -3.0 / (2.0 * h)
    matrix[0, 1] = 4.0 / (2.0 * h)
    matrix[0, 2] = -1.0 / (2.0 * h)
    for i in range(1, n - 1):
        matrix[i, i - 1] = -1.0 / (2.0 * h)
        matrix[i, i + 1] = 1.0 / (2.0 * h)
    matrix[-1, -3] = 1.0 / (2.0 * h)
    matrix[-1, -2] = -4.0 / (2.0 * h)
    matrix[-1, -1] = 3.0 / (2.0 * h)
    return matrix


def corrected_linear_derivative_matrix(
    nodes: np.ndarray, base_differentiation_matrix: np.ndarray
) -> np.ndarray:
    """Build the precomputed corrected derivative matrix for linear-axis data."""

    nodes = np.asarray(nodes, dtype=np.float64)
    return _linear_operator_matrix(
        nodes.shape[0],
        lambda out, arr: _corrected_linear_derivative(out, arr, base_differentiation_matrix, nodes),
    )


def corrected_even_derivative_matrix(
    nodes: np.ndarray, base_differentiation_matrix: np.ndarray
) -> np.ndarray:
    """Build the precomputed corrected derivative matrix for even-axis data."""

    nodes = np.asarray(nodes, dtype=np.float64)
    return _linear_operator_matrix(
        nodes.shape[0],
        lambda out, arr: _corrected_even_derivative(out, arr, base_differentiation_matrix, nodes),
    )


# -----------------------------------------------------------------------------
# Private implementation
# -----------------------------------------------------------------------------


linear_operator = Callable[[np.ndarray, np.ndarray], None]


def _linear_operator_matrix(n: int, operator: linear_operator) -> np.ndarray:
    matrix = np.empty((n, n), dtype=np.float64)
    basis = np.zeros(n, dtype=np.float64)
    out = np.empty(n, dtype=np.float64)
    for j in range(n):
        basis.fill(0.0)
        basis[j] = 1.0
        operator(out, basis)
        matrix[:, j] = out
    return matrix


def _corrected_linear_derivative(
    out: np.ndarray,
    arr: np.ndarray,
    base_differentiation_matrix: np.ndarray,
    nodes: np.ndarray,
) -> np.ndarray:
    n = arr.shape[0]
    if n == 0:
        return out
    if n == 1:
        out[0] = 0.0
        return out

    reduced = np.empty_like(arr)
    for i in range(n):
        reduced[i] = arr[i] / nodes[i] if nodes[i] > 1e-10 else 0.0
    reduced[0] = reduced[1]
    _enforce_axis_even_profile(reduced, nodes)

    reduced_r = base_differentiation_matrix @ reduced
    _enforce_axis_linear_profile(reduced_r, nodes)

    out[:] = reduced + nodes * reduced_r
    out[0] = reduced[0]
    return out


def _corrected_even_derivative(
    out: np.ndarray,
    arr: np.ndarray,
    base_differentiation_matrix: np.ndarray,
    nodes: np.ndarray,
) -> np.ndarray:
    n = arr.shape[0]
    if n == 0:
        return out
    if n == 1:
        out[0] = 0.0
        return out

    smooth = np.array(arr, dtype=np.float64, copy=True)
    _enforce_axis_even_profile(smooth, nodes)
    base = smooth[0]

    reduced = np.empty_like(arr)
    for i in range(n):
        rho2 = nodes[i] * nodes[i]
        reduced[i] = (smooth[i] - base) / rho2 if rho2 > 1e-10 else 0.0
    reduced[0] = reduced[1]
    _enforce_axis_even_profile(reduced, nodes)

    reduced_r = base_differentiation_matrix @ reduced
    _enforce_axis_linear_profile(reduced_r, nodes)

    out[:] = 2.0 * nodes * reduced + (nodes * nodes) * reduced_r
    out[0] = 0.0
    _enforce_axis_linear_profile(out, nodes)
    return out


def _enforce_axis_linear_profile(values: np.ndarray, nodes: np.ndarray) -> np.ndarray:
    if values.shape[0] < 2:
        return values
    if abs(nodes[1]) < 1e-14:
        return values
    if values.shape[0] >= 3 and abs(nodes[2]) >= 1e-14:
        slope = values[2] / nodes[2]
        values[0] = slope * nodes[0]
        values[1] = slope * nodes[1]
        return values
    values[0] = values[1] * nodes[0] / nodes[1]
    return values


def _enforce_axis_even_profile(values: np.ndarray, nodes: np.ndarray) -> np.ndarray:
    if values.shape[0] < 3:
        return values
    x1 = nodes[1] * nodes[1]
    x2 = nodes[2] * nodes[2]
    if abs(x2 - x1) < 1e-14:
        return values
    slope = (values[2] - values[1]) / (x2 - x1)
    intercept = values[1] - slope * x1
    values[0] = intercept + slope * nodes[0] * nodes[0]
    values[1] = intercept + slope * x1
    return values

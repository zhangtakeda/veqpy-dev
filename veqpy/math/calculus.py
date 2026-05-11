"""
Module: math.calculus

Role:
- Build radial calculus matrices and corrected calculus operators.
- Own the registry that selects base integration/differentiation schemes.

Public API:
- CalculusBuilder
- make_calculus
- cfd33_matrices
- cfd33_differentiation_matrix
- cfd33_integration_matrix
- corrected_even_derivative_matrix
- corrected_integration_matrix
- corrected_linear_derivative_matrix
- spectral_differentiation_matrix
- spectral_integration_matrix
- uniform_spectral_differentiation_matrix
- uniform_spectral_integration_matrix
"""

import math
from collections.abc import Callable

import numpy as np

from veqpy.base.registry import Registry
from veqpy.math.interpolate import barycentric_log_weights, interpolation_matrix

CalculusBuilder = Callable[[np.ndarray], tuple[np.ndarray, np.ndarray]]
LinearOperator = Callable[[np.ndarray, np.ndarray], None]

calculus_generator: Registry[str, CalculusBuilder] = Registry(str, Callable)


def make_calculus(
    nodes: np.ndarray,
    *,
    calculus: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Build ``(integration_matrix, differentiation_matrix)`` for a calculus scheme."""

    calculus = calculus.lower()
    if calculus not in calculus_generator:
        available = ", ".join(sorted(calculus_generator.registry))
        raise ValueError(f"Unknown calculus scheme: {calculus}. Available schemes: {available}")
    return calculus_generator[calculus](np.asarray(nodes, dtype=np.float64))


@calculus_generator("compact", "cfd33")
def compact_cfd33_calculus(nodes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Build dense CFD33 compact integration and differentiation matrices."""

    return cfd33_integration_matrix(nodes), cfd33_differentiation_matrix(nodes)


@calculus_generator("spectral")
def spectral_calculus(nodes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Build the legacy polynomial/spectral calculus matrices."""

    if _has_uniform_spacing(nodes):
        return (
            uniform_spectral_integration_matrix(nodes.shape[0]),
            uniform_spectral_differentiation_matrix(nodes.shape[0]),
        )
    return spectral_integration_matrix(nodes), spectral_differentiation_matrix(nodes)


def corrected_integration_matrix(
    nodes: np.ndarray,
    base_differentiation_matrix: np.ndarray,
    *,
    p: int,
) -> np.ndarray:
    """Build the precomputed corrected variable-limit integration matrix."""

    nodes = np.asarray(nodes, dtype=np.float64)
    return _linear_operator_matrix(
        nodes.shape[0],
        lambda out, arr: _corrected_integration(out, arr, p, nodes, base_differentiation_matrix),
    )


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


def cfd33_matrices(nodes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Build non-uniform CFD33 matrices ``A`` and ``B`` for ``A @ u_r == B @ u``."""

    nodes = _validated_nodes(nodes)
    n = nodes.shape[0]
    a_matrix = np.zeros((n, n), dtype=np.float64)
    b_matrix = np.zeros((n, n), dtype=np.float64)

    a_matrix[0, 0] = 1.0
    b_matrix[0, :4] = _finite_difference_weights(nodes[:4], nodes[0], derivative_order=1)

    for i in range(1, n - 1):
        h_left = nodes[i] - nodes[i - 1]
        h_right = nodes[i + 1] - nodes[i]
        h_sum = h_left + h_right

        a_matrix[i, i - 1] = (h_right / h_sum) ** 2
        a_matrix[i, i] = 1.0
        a_matrix[i, i + 1] = (h_left / h_sum) ** 2

        b_matrix[i, i - 1] = -(2.0 * h_right * h_right * (2.0 * h_left + h_right)) / (
            h_left * h_sum**3
        )
        b_matrix[i, i] = 2.0 * (h_right - h_left) / (h_right * h_left)
        b_matrix[i, i + 1] = (2.0 * h_left * h_left * (h_left + 2.0 * h_right)) / (
            h_right * h_sum**3
        )

    a_matrix[-1, -1] = 1.0
    b_matrix[-1, -4:] = _finite_difference_weights(nodes[-4:], nodes[-1], derivative_order=1)
    return a_matrix, b_matrix


def cfd33_differentiation_matrix(nodes: np.ndarray) -> np.ndarray:
    """Build the dense pre-eliminated CFD33 derivative matrix."""

    a_matrix, b_matrix = cfd33_matrices(nodes)
    return np.linalg.solve(a_matrix, b_matrix)


def cfd33_integration_matrix(nodes: np.ndarray) -> np.ndarray:
    """Build the dense CFD33 variable-limit integration matrix with ``v(0) == 0``."""

    nodes = _validated_nodes(nodes)
    a_matrix, b_matrix = cfd33_matrices(nodes)
    system = b_matrix.copy()
    rhs_matrix = a_matrix.copy()

    constraint_row = int(np.argmin(np.abs(nodes)))
    system[constraint_row, :] = interpolation_matrix(nodes, np.array([0.0], dtype=np.float64))[0]
    rhs_matrix[constraint_row, :] = 0.0
    return np.linalg.solve(system, rhs_matrix)


def spectral_differentiation_matrix(nodes: np.ndarray) -> np.ndarray:
    """Build the polynomial collocation first-derivative matrix."""

    nodes = np.asarray(nodes, dtype=np.float64)
    signs, log_weights = barycentric_log_weights(nodes)
    diff = nodes[:, None] - nodes[None, :]
    np.fill_diagonal(diff, 1.0)
    log_ratio = log_weights[None, :] - log_weights[:, None]
    mag_ratio = np.exp(np.clip(log_ratio, -700.0, 700.0))
    matrix = signs[None, :] * signs[:, None] * mag_ratio / diff
    np.fill_diagonal(matrix, 0.0)
    matrix[np.diag_indices_from(matrix)] = -np.sum(matrix, axis=1)
    return matrix


def uniform_spectral_differentiation_matrix(n: int) -> np.ndarray:
    """Build the legacy explicit first-derivative matrix on a uniform grid."""

    h = 1.0 / (n - 1)
    matrix = np.zeros((n, n), dtype=np.float64)
    matrix[0, :3] = [-3.0 / (2.0 * h), 4.0 / (2.0 * h), -1.0 / (2.0 * h)]
    for i in range(1, n - 1):
        matrix[i, i - 1] = -1.0 / (2.0 * h)
        matrix[i, i + 1] = 1.0 / (2.0 * h)
    matrix[-1, -3:] = [1.0 / (2.0 * h), -4.0 / (2.0 * h), 3.0 / (2.0 * h)]
    return matrix


def spectral_integration_matrix(nodes: np.ndarray) -> np.ndarray:
    """Build the polynomial matrix for integrals from zero to each node."""

    nodes = np.asarray(nodes, dtype=np.float64)
    n = nodes.shape[0]
    xg = 2.0 * nodes - 1.0
    full_legendre = np.polynomial.legendre.legvander(xg, n)
    legendre = full_legendre[:, :n]

    antiderivative = np.zeros((n, n), dtype=np.float64)
    antiderivative[:, 0] = 0.5 * (xg + 1.0)
    degrees = np.arange(1, n, dtype=np.float64)
    antiderivative[:, 1:] = (
        0.5 * (full_legendre[:, 2 : n + 1] - full_legendre[:, : n - 1]) / (2.0 * degrees + 1.0)
    )

    lower_legendre = np.polynomial.legendre.legvander(np.array([-1.0]), n)[0]
    lower_antiderivative = np.zeros(n, dtype=np.float64)
    lower_antiderivative[1:] = (
        0.5 * (lower_legendre[2 : n + 1] - lower_legendre[: n - 1]) / (2.0 * degrees + 1.0)
    )

    rhs = (antiderivative - lower_antiderivative[None, :]).T
    cond = np.linalg.cond(legendre)
    if np.isfinite(cond) and cond <= 1.0 / np.sqrt(np.finfo(np.float64).eps):
        return np.linalg.solve(legendre.T, rhs).T
    return np.linalg.lstsq(legendre.T, rhs, rcond=None)[0].T


def uniform_spectral_integration_matrix(n: int) -> np.ndarray:
    """Build the legacy trapezoidal variable-limit integration matrix."""

    h = 1.0 / (n - 1)
    matrix = np.zeros((n, n), dtype=np.float64)
    for i in range(1, n):
        matrix[i, 0] = 0.5 * h
        matrix[i, i] = 0.5 * h
        if i > 1:
            matrix[i, 1:i] = h
    return matrix


def _validated_nodes(nodes: np.ndarray) -> np.ndarray:
    nodes = np.asarray(nodes, dtype=np.float64)
    if nodes.ndim != 1:
        raise ValueError("nodes must be a one-dimensional array")
    if nodes.shape[0] < 4:
        raise ValueError("CFD33 requires at least four nodes")
    if not np.all(np.isfinite(nodes)):
        raise ValueError("nodes must be finite")
    if not np.all(np.diff(nodes) > 0.0):
        raise ValueError("nodes must be strictly increasing")
    return nodes


def _finite_difference_weights(
    stencil_nodes: np.ndarray, target: float, *, derivative_order: int
) -> np.ndarray:
    offsets = np.asarray(stencil_nodes, dtype=np.float64) - float(target)
    n = offsets.shape[0]
    powers = offsets[None, :] ** np.arange(n, dtype=np.float64)[:, None]
    rhs = np.zeros(n, dtype=np.float64)
    rhs[derivative_order] = float(math.factorial(derivative_order))
    return np.linalg.solve(powers, rhs)


def _has_uniform_spacing(nodes: np.ndarray) -> bool:
    nodes = np.asarray(nodes, dtype=np.float64)
    if nodes.ndim != 1 or nodes.shape[0] < 2:
        return False
    spacing = np.diff(nodes)
    return bool(np.all(np.abs(spacing - spacing[0]) < 1.0e-6))


def _linear_operator_matrix(n: int, operator: LinearOperator) -> np.ndarray:
    matrix = np.empty((n, n), dtype=np.float64)
    basis = np.zeros(n, dtype=np.float64)
    out = np.empty(n, dtype=np.float64)
    for j in range(n):
        basis.fill(0.0)
        basis[j] = 1.0
        operator(out, basis)
        matrix[:, j] = out
    return matrix


def _corrected_integration(
    out: np.ndarray,
    arr: np.ndarray,
    p: int,
    nodes: np.ndarray,
    base_differentiation_matrix: np.ndarray,
) -> np.ndarray:
    rho_safe = np.where(nodes > 1e-10, nodes, 1e-10)
    q_solution = np.linalg.solve(
        nodes[:, None] * base_differentiation_matrix
        + np.eye(nodes.shape[0], dtype=np.float64) * float(p + 1),
        arr / (rho_safe**p),
    )
    out[:] = q_solution * (nodes ** (p + 1))
    return out


def _corrected_linear_derivative(
    out: np.ndarray,
    arr: np.ndarray,
    base_differentiation_matrix: np.ndarray,
    nodes: np.ndarray,
) -> np.ndarray:
    if arr.shape[0] <= 1:
        if arr.shape[0] == 1:
            out[0] = 0.0
        return out

    reduced = np.divide(arr, nodes, out=np.zeros_like(arr), where=nodes > 1e-10)
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
    if arr.shape[0] <= 1:
        if arr.shape[0] == 1:
            out[0] = 0.0
        return out

    smooth = np.array(arr, dtype=np.float64, copy=True)
    _enforce_axis_even_profile(smooth, nodes)
    rho2 = nodes * nodes

    reduced = np.divide(smooth - smooth[0], rho2, out=np.zeros_like(arr), where=rho2 > 1e-10)
    reduced[0] = reduced[1]
    _enforce_axis_even_profile(reduced, nodes)

    reduced_r = base_differentiation_matrix @ reduced
    _enforce_axis_linear_profile(reduced_r, nodes)

    out[:] = 2.0 * nodes * reduced + rho2 * reduced_r
    out[0] = 0.0
    _enforce_axis_linear_profile(out, nodes)
    return out


def _enforce_axis_linear_profile(values: np.ndarray, nodes: np.ndarray) -> np.ndarray:
    if values.shape[0] < 2 or abs(nodes[1]) < 1e-14:
        return values
    if values.shape[0] >= 3 and abs(nodes[2]) >= 1e-14:
        values[:2] = values[2] / nodes[2] * nodes[:2]
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
    values[:2] = intercept + slope * nodes[:2] * nodes[:2]
    return values

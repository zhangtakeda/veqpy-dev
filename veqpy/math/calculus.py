"""
Module: math.calculus

Role:
- Build radial calculus matrices.
- Own the registry that selects base integration/differentiation schemes.

Public API:
- make_calculus
"""

import math
from collections.abc import Callable

import numpy as np

from veqpy.base.registry import Registry
from veqpy.math.interpolate import barycentric_log_weights, interpolation_matrix

# -----------------------------------------------------------------------------
# Public interface
# -----------------------------------------------------------------------------

CalculusBuilder = Callable[[np.ndarray], tuple[np.ndarray, np.ndarray]]
calculus_generator: Registry[str, CalculusBuilder] = Registry(str, Callable)


def make_calculus(
    nodes: np.ndarray,
    *,
    calculus: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Build ``(accumulator, differentiator)`` for a calculus scheme."""

    calculus = calculus.lower()
    if calculus not in calculus_generator:
        available = ", ".join(sorted(calculus_generator.registry))
        raise ValueError(f"Unknown calculus scheme: {calculus}. Available schemes: {available}")
    return calculus_generator[calculus](nodes)


@calculus_generator("compact", "cfd33")
def compact_cfd33_calculus(nodes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Build dense CFD33 compact integration and differentiation matrices."""

    _validate_nodes(nodes, min_size=4)
    return (
        _cfd33_accumulator(nodes),
        _cfd33_differentiator(nodes),
    )


@calculus_generator("spectral")
def spectral_calculus(nodes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Build base integration/differentiation matrices for the spectral scheme."""

    _validate_nodes(nodes, min_size=4)
    if _has_uniform_spacing(nodes):
        return (
            _uniform_accumulator(nodes.shape[0]),
            _cfd33_differentiator(nodes),
        )
    return (
        _spectral_accumulator(nodes),
        _spectral_differentiator(nodes),
    )


# -----------------------------------------------------------------------------
# Private implementation
# -----------------------------------------------------------------------------


def _cfd33_matrices(nodes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Build non-uniform CFD33 matrices ``A`` and ``B`` for ``A @ u_r == B @ u``."""

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


def _cfd33_differentiator(nodes: np.ndarray) -> np.ndarray:
    """Build the dense pre-eliminated CFD33 derivative matrix."""

    a_matrix, b_matrix = _cfd33_matrices(nodes)
    return np.linalg.solve(a_matrix, b_matrix)


def _cfd33_accumulator(nodes: np.ndarray) -> np.ndarray:
    """Build the dense CFD33 variable-limit integration matrix with ``v(0) == 0``."""

    a_matrix, b_matrix = _cfd33_matrices(nodes)
    system = b_matrix.copy()
    rhs_matrix = a_matrix.copy()

    constraint_row = int(np.argmin(np.abs(nodes)))
    system[constraint_row, :] = interpolation_matrix(nodes, np.array([0.0], dtype=np.float64))[0]
    rhs_matrix[constraint_row, :] = 0.0
    return np.linalg.solve(system, rhs_matrix)


def _spectral_differentiator(nodes: np.ndarray) -> np.ndarray:
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


def _spectral_accumulator(nodes: np.ndarray) -> np.ndarray:
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


def _uniform_accumulator(n: int) -> np.ndarray:
    """Build the trapezoidal variable-limit integration matrix."""

    h = 1.0 / (n - 1)
    matrix = np.zeros((n, n), dtype=np.float64)
    for i in range(1, n):
        matrix[i, 0] = 0.5 * h
        matrix[i, i] = 0.5 * h
        if i > 1:
            matrix[i, 1:i] = h
    return matrix


def _validate_nodes(
    nodes: np.ndarray,
    *,
    min_size: int = 4,
):
    nodes = np.asarray(nodes, dtype=np.float64)
    if nodes.ndim != 1:
        raise ValueError("nodes must be a one-dimensional array")
    if nodes.shape[0] < min_size:
        raise ValueError(f"require at least {min_size} nodes")
    if not np.all(np.isfinite(nodes)):
        raise ValueError("nodes must be finite")
    if not np.all(np.diff(nodes) > 0.0):
        raise ValueError("nodes must be strictly increasing")


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

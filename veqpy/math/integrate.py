"""
Module: math.integrate

Role:
- Provide pure variable-limit integration matrix builders.

Public API:
- corrected_integration_matrix
- uniform_variable_limit_integration_matrix
- variable_limit_integration_matrix
"""

from collections.abc import Callable

import numpy as np

# -----------------------------------------------------------------------------
# Public interface
# -----------------------------------------------------------------------------


def variable_limit_integration_matrix(nodes: np.ndarray) -> np.ndarray:
    """Build the polynomial matrix for integrals from zero to each node."""

    nodes = np.asarray(nodes, dtype=np.float64)
    n = len(nodes)
    xg = 2.0 * nodes - 1.0
    full_legendre = np.polynomial.legendre.legvander(xg, n)
    legendre = full_legendre[:, :n]

    antiderivative = np.zeros((n, n), dtype=np.float64)
    antiderivative[:, 0] = 0.5 * (xg + 1.0)
    degrees = np.arange(1, n, dtype=np.float64)
    antiderivative[:, 1:] = (
        0.5 * (full_legendre[:, 2 : n + 1] - full_legendre[:, : n - 1])
        / (2.0 * degrees + 1.0)
    )

    lower_x = -1.0
    lower_legendre = np.polynomial.legendre.legvander(np.array([lower_x]), n)[0]
    lower_antiderivative = np.zeros(n, dtype=np.float64)
    lower_antiderivative[0] = 0.5 * (lower_x + 1.0)
    lower_antiderivative[1:] = (
        0.5 * (lower_legendre[2 : n + 1] - lower_legendre[: n - 1])
        / (2.0 * degrees + 1.0)
    )

    rhs = (antiderivative - lower_antiderivative[None, :]).T
    cond = np.linalg.cond(legendre)
    eps = np.finfo(np.float64).eps

    if np.isfinite(cond) and cond <= 1.0 / np.sqrt(eps):
        return np.linalg.solve(legendre.T, rhs).T

    return np.linalg.lstsq(legendre.T, rhs, rcond=None)[0].T


def uniform_variable_limit_integration_matrix(n: int) -> np.ndarray:
    """Build the trapezoidal variable-limit integration matrix on a uniform grid."""

    h = 1.0 / (n - 1)
    matrix = np.zeros((n, n), dtype=np.float64)
    for i in range(1, n):
        matrix[i, 0] = 0.5 * h
        matrix[i, i] = 0.5 * h
        if i > 1:
            matrix[i, 1:i] = h
    return matrix


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
        lambda out, arr: _corrected_integration(
            out,
            arr,
            p,
            nodes,
            base_differentiation_matrix,
        ),
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


def _corrected_integration(
    out: np.ndarray,
    arr: np.ndarray,
    p: int,
    nodes: np.ndarray,
    base_differentiation_matrix: np.ndarray,
) -> np.ndarray:
    rho_safe = np.where(nodes > 1e-10, nodes, 1e-10)
    q_int = arr / (rho_safe**p)
    system = nodes[:, None] * base_differentiation_matrix
    system[np.diag_indices_from(system)] += float(p + 1)
    q_solution = np.linalg.solve(system, q_int)
    out[:] = q_solution * (nodes ** (p + 1))
    return out

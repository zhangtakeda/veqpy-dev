"""
Module: math.interpolate

Role:
- Provide pure interpolation matrix builders.

Public API:
- barycentric_log_weights
- interpolation_matrix

Notes:
- Matrices use source nodes as columns and evaluation nodes as rows.
"""

import numpy as np

# -----------------------------------------------------------------------------
# Public interface
# -----------------------------------------------------------------------------


def barycentric_log_weights(nodes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Build signed logarithmic barycentric weights for distinct nodes."""

    nodes = np.asarray(nodes, dtype=np.float64)
    diff = nodes[:, None] - nodes[None, :]
    mask = ~np.eye(len(nodes), dtype=bool)
    if np.any(np.isclose(diff[mask], 0.0)):
        raise ValueError("nodes must be distinct")

    abs_diff = np.abs(diff)
    sign_diff = np.sign(diff)
    np.fill_diagonal(abs_diff, 1.0)
    np.fill_diagonal(sign_diff, 1.0)

    log_abs_prod = np.sum(np.log(abs_diff), axis=1)
    signs = np.prod(sign_diff, axis=1)
    return signs, -log_abs_prod


def interpolation_matrix(source_nodes: np.ndarray, evaluation_nodes: np.ndarray) -> np.ndarray:
    """Build the Lagrange interpolation matrix from source to evaluation nodes."""

    source_nodes = np.asarray(source_nodes, dtype=np.float64)
    evaluation_nodes = np.asarray(evaluation_nodes, dtype=np.float64)
    signs, log_weights = barycentric_log_weights(source_nodes)
    matrix = np.empty((evaluation_nodes.size, source_nodes.size), dtype=np.float64)

    for i, node in enumerate(evaluation_nodes):
        diff = node - source_nodes
        exact = np.flatnonzero(np.isclose(diff, 0.0))
        if exact.size:
            matrix[i].fill(0.0)
            matrix[i, exact[0]] = 1.0
            continue

        log_terms = log_weights - np.log(np.abs(diff))
        scale = np.max(log_terms)
        terms = signs * np.sign(diff) * np.exp(log_terms - scale)
        matrix[i] = terms / np.sum(terms)

    return matrix

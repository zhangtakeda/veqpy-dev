from __future__ import annotations

import numpy as np

from veqpy.math.calculus import _cfd33_matrices, _compact_matrices


def test_cfd33_uses_generic_compact_moment_coefficients() -> None:
    nodes = np.array([0.0, 0.02, 0.07, 0.16, 0.30, 0.43, 0.61, 0.72, 0.86])

    a_cfd33, b_cfd33 = _cfd33_matrices(nodes)
    a_generic, b_generic = _compact_matrices(nodes, implicit_width=3, explicit_width=3)

    np.testing.assert_allclose(a_cfd33, a_generic, rtol=1.0e-12, atol=1.0e-12)
    np.testing.assert_allclose(b_cfd33, b_generic, rtol=1.0e-12, atol=1.0e-12)
    assert np.count_nonzero(np.abs(b_cfd33[0]) > 1.0e-12) == 5
    assert np.count_nonzero(np.abs(b_cfd33[-1]) > 1.0e-12) == 5


def test_cfd33_interior_matches_documented_nonuniform_formula() -> None:
    nodes = np.array([0.0, 0.02, 0.07, 0.16, 0.30, 0.43, 0.61, 0.72, 0.86])
    row = 4

    a_matrix, b_matrix = _cfd33_matrices(nodes)
    h_left = nodes[row] - nodes[row - 1]
    h_right = nodes[row + 1] - nodes[row]
    h_sum = h_left + h_right

    expected_a = np.array(
        [
            (h_right / h_sum) ** 2,
            1.0,
            (h_left / h_sum) ** 2,
        ]
    )
    expected_b = np.array(
        [
            -(2.0 * h_right * h_right * (2.0 * h_left + h_right)) / (h_left * h_sum**3),
            2.0 * (h_right - h_left) / (h_right * h_left),
            (2.0 * h_left * h_left * (h_left + 2.0 * h_right)) / (h_right * h_sum**3),
        ]
    )

    np.testing.assert_allclose(a_matrix[row, row - 1 : row + 2], expected_a)
    np.testing.assert_allclose(b_matrix[row, row - 1 : row + 2], expected_b)

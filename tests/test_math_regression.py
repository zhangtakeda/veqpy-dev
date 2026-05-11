import numpy as np

import veqpy.math.calculus as math_calculus
from veqpy.engine import (
    corrected_even_derivative,
    corrected_integration,
    corrected_linear_derivative,
)
from veqpy.math.interpolate import interpolation_matrix
from veqpy.math.quadrature import available_quadrature_schemes, legendre_quadrature, make_quadrature

HIGH_ORDER_NODE_COUNT = 129


def test_cfd33_differentiation_matrix_preserves_low_order_polynomials():
    nodes = np.array([0.0, 0.05, 0.2, 0.45, 0.7, 1.0], dtype=np.float64)
    matrix = math_calculus.cfd33_differentiation_matrix(nodes)

    assert np.all(np.isfinite(matrix))
    assert np.allclose(matrix @ np.ones_like(nodes), 0.0, atol=1.0e-13)
    assert np.allclose(matrix @ nodes, 1.0, atol=1.0e-13)
    assert np.allclose(matrix @ (nodes**2), 2.0 * nodes, atol=1.0e-13)


def test_cfd33_uniform_matrices_match_documented_interior_stencil():
    n = 8
    h = 1.0 / (n - 1)

    a_matrix, b_matrix = math_calculus.cfd33_matrices(np.linspace(0.0, 1.0, n))

    assert np.allclose(a_matrix[3, 2:5], [0.25, 1.0, 0.25])
    assert np.allclose(b_matrix[3, 2:5], [-3.0 / (4.0 * h), 0.0, 3.0 / (4.0 * h)])


def test_cfd33_integration_matrix_enforces_axis_constraint_without_axis_node():
    nodes, _ = legendre_quadrature(16)
    matrix = math_calculus.cfd33_integration_matrix(nodes)
    constant_integral = matrix @ np.ones_like(nodes)
    axis_value = interpolation_matrix(nodes, np.array([0.0])) @ constant_integral

    assert np.all(np.isfinite(matrix))
    assert np.allclose(constant_integral, nodes, rtol=1.0e-12, atol=1.0e-12)
    assert np.allclose(axis_value, 0.0, atol=1.0e-13)


def test_cfd33_differentiation_recovers_cfd33_integral_on_interior():
    nodes = np.linspace(0.0, 1.0, 33)
    values = np.sin(nodes)
    diff = math_calculus.cfd33_differentiation_matrix(nodes)
    integ = math_calculus.cfd33_integration_matrix(nodes)
    recovered = diff @ (integ @ values)

    assert np.allclose(recovered[2:-2], values[2:-2], rtol=1.0e-6, atol=1.0e-6)


def test_calculus_registry_selects_compact_and_uniform_spectral_matrices():
    nodes = np.linspace(0.0, 1.0, 8)

    compact_integration, compact_differentiation = math_calculus.make_calculus(
        nodes,
        calculus="compact",
    )
    spectral_integration, spectral_differentiation = math_calculus.make_calculus(
        nodes,
        calculus="spectral",
    )

    assert np.allclose(compact_differentiation @ np.ones_like(nodes), 0.0, atol=1.0e-13)
    assert np.allclose(compact_integration @ np.ones_like(nodes), nodes, atol=1.0e-13)
    assert np.allclose(
        spectral_integration,
        math_calculus.uniform_spectral_integration_matrix(nodes.shape[0]),
    )
    assert np.allclose(
        spectral_differentiation,
        math_calculus.uniform_spectral_differentiation_matrix(nodes.shape[0]),
    )


def test_calculus_registry_selects_nonuniform_spectral_matrices_from_nodes():
    nodes, _ = legendre_quadrature(8)

    spectral_integration, spectral_differentiation = math_calculus.make_calculus(
        nodes,
        calculus="spectral",
    )

    assert np.allclose(spectral_integration, math_calculus.spectral_integration_matrix(nodes))
    assert np.allclose(
        spectral_differentiation,
        math_calculus.spectral_differentiation_matrix(nodes),
    )


def test_interpolation_matrix_preserves_polynomial_values():
    source = np.array([0.0, 0.2, 0.7, 1.0], dtype=np.float64)
    evaluation = np.array([0.1, 0.4, 0.9], dtype=np.float64)
    values = source**3 - 2.0 * source + 1.0
    expected = evaluation**3 - 2.0 * evaluation + 1.0

    matrix = interpolation_matrix(source, evaluation)

    assert np.allclose(matrix @ values, expected)


def test_high_order_interpolation_matrix_preserves_polynomial_values():
    source, _ = legendre_quadrature(HIGH_ORDER_NODE_COUNT)
    evaluation = np.linspace(0.0, 1.0, 41, dtype=np.float64)
    values = source**5 - 2.0 * source**3 + 0.25 * source + 1.0
    expected = evaluation**5 - 2.0 * evaluation**3 + 0.25 * evaluation + 1.0

    matrix = interpolation_matrix(source, evaluation)

    assert np.all(np.isfinite(matrix))
    assert np.allclose(matrix @ values, expected, rtol=1.0e-10, atol=1.0e-10)


def test_spectral_differentiation_matrix_preserves_polynomial_derivative():
    nodes, _ = legendre_quadrature(6)
    values = nodes**4 - 3.0 * nodes**2 + nodes
    expected = 4.0 * nodes**3 - 6.0 * nodes + 1.0

    matrix = math_calculus.spectral_differentiation_matrix(nodes)

    assert np.allclose(matrix @ values, expected)


def test_high_order_spectral_differentiation_matrix_preserves_polynomial_derivative():
    nodes, _ = legendre_quadrature(HIGH_ORDER_NODE_COUNT)
    values = nodes**5 - 2.0 * nodes**3 + 0.25 * nodes
    expected = 5.0 * nodes**4 - 6.0 * nodes**2 + 0.25

    matrix = math_calculus.spectral_differentiation_matrix(nodes)

    assert np.all(np.isfinite(matrix))
    assert np.allclose(matrix @ values, expected, rtol=1.0e-9, atol=1.0e-9)


def test_spectral_integration_matrix_preserves_polynomial_integral():
    nodes, _ = legendre_quadrature(6)
    values = nodes**3 - 2.0 * nodes
    expected = 0.25 * nodes**4 - nodes**2

    matrix = math_calculus.spectral_integration_matrix(nodes)

    assert np.allclose(matrix @ values, expected)


def test_high_order_spectral_integration_matrix_preserves_polynomial_integral():
    nodes, _ = legendre_quadrature(HIGH_ORDER_NODE_COUNT)
    values = nodes**5 - 2.0 * nodes**3 + 0.25 * nodes
    expected = nodes**6 / 6.0 - 0.5 * nodes**4 + 0.125 * nodes**2

    matrix = math_calculus.spectral_integration_matrix(nodes)

    assert np.all(np.isfinite(matrix))
    assert np.allclose(matrix @ values, expected, rtol=1.0e-12, atol=1.0e-12)


def test_high_order_quadrature_rules_preserve_expected_moments():
    exact = {
        0: 1.0,
        1: 0.5,
        2: 1.0 / 3.0,
        5: 1.0 / 6.0,
        20: 1.0 / 21.0,
        128: 1.0 / 129.0,
    }

    for scheme in available_quadrature_schemes():
        nodes, weights = make_quadrature(scheme, HIGH_ORDER_NODE_COUNT)
        assert np.all(np.isfinite(nodes))
        assert np.all(np.isfinite(weights))
        assert np.all(np.diff(nodes) > 0.0)
        assert np.all(weights > 0.0)
        assert np.isclose(np.sum(weights), 1.0, rtol=1.0e-13, atol=1.0e-13)

        if scheme == "uniform":
            assert np.isclose(np.dot(weights, nodes), exact[1], rtol=1.0e-13, atol=1.0e-13)
            continue

        for degree, expected in exact.items():
            assert np.isclose(
                np.dot(weights, nodes**degree),
                expected,
                rtol=1.0e-12,
                atol=1.0e-12,
            )


def test_uniform_spectral_integration_matrix_matches_trapezoid_for_linear_data():
    n = 7
    nodes = np.linspace(0.0, 1.0, n)
    values = 3.0 * nodes - 2.0
    expected = 1.5 * nodes**2 - 2.0 * nodes

    matrix = math_calculus.uniform_spectral_integration_matrix(n)

    assert np.allclose(matrix @ values, expected)


def test_corrected_integration_matrix_matches_engine_kernel():
    nodes, _ = legendre_quadrature(6)
    values = nodes**2 + nodes
    base_integration = math_calculus.spectral_integration_matrix(nodes)
    base_differentiation = math_calculus.spectral_differentiation_matrix(nodes)
    expected = np.empty_like(values)

    corrected_integration(
        expected,
        values,
        base_integration,
        2,
        nodes,
        base_differentiation,
    )
    matrix = math_calculus.corrected_integration_matrix(
        nodes,
        base_differentiation,
        p=2,
    )

    assert np.allclose(matrix @ values, expected)


def test_high_order_corrected_integration_matrix_matches_engine_kernel():
    nodes, _ = legendre_quadrature(HIGH_ORDER_NODE_COUNT)
    values = nodes**3 + nodes
    base_integration = math_calculus.spectral_integration_matrix(nodes)
    base_differentiation = math_calculus.spectral_differentiation_matrix(nodes)
    expected = np.empty_like(values)

    corrected_integration(
        expected,
        values,
        base_integration,
        2,
        nodes,
        base_differentiation,
    )
    matrix = math_calculus.corrected_integration_matrix(
        nodes,
        base_differentiation,
        p=2,
    )

    assert np.all(np.isfinite(matrix))
    assert np.allclose(matrix @ values, expected, rtol=1.0e-10, atol=1.0e-10)


def test_corrected_derivative_matrices_match_engine_kernels():
    nodes, _ = legendre_quadrature(6)
    values = nodes**4 - nodes**2 + 0.25 * nodes
    base_differentiation = math_calculus.spectral_differentiation_matrix(nodes)
    expected_linear = np.empty_like(values)
    expected_even = np.empty_like(values)

    corrected_linear_derivative(expected_linear, values, base_differentiation, rho=nodes)
    corrected_even_derivative(expected_even, values, base_differentiation, rho=nodes)

    linear_matrix = math_calculus.corrected_linear_derivative_matrix(nodes, base_differentiation)
    even_matrix = math_calculus.corrected_even_derivative_matrix(nodes, base_differentiation)

    assert np.allclose(linear_matrix @ values, expected_linear)
    assert np.allclose(even_matrix @ values, expected_even)


def test_high_order_corrected_derivative_matrices_match_engine_kernels():
    nodes, _ = legendre_quadrature(HIGH_ORDER_NODE_COUNT)
    values = nodes**4 - nodes**2 + 0.25 * nodes
    base_differentiation = math_calculus.spectral_differentiation_matrix(nodes)
    expected_linear = np.empty_like(values)
    expected_even = np.empty_like(values)

    corrected_linear_derivative(expected_linear, values, base_differentiation, rho=nodes)
    corrected_even_derivative(expected_even, values, base_differentiation, rho=nodes)

    linear_matrix = math_calculus.corrected_linear_derivative_matrix(nodes, base_differentiation)
    even_matrix = math_calculus.corrected_even_derivative_matrix(nodes, base_differentiation)

    assert np.all(np.isfinite(linear_matrix))
    assert np.all(np.isfinite(even_matrix))
    assert np.allclose(linear_matrix @ values, expected_linear, rtol=1.0e-10, atol=1.0e-10)
    assert np.allclose(even_matrix @ values, expected_even, rtol=1.0e-10, atol=1.0e-10)

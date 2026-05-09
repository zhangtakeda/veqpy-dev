import numpy as np

import veqpy.math.differentiate as math_differentiate
import veqpy.math.integrate as math_integrate
from veqpy.engine import (
    corrected_even_derivative,
    corrected_integration,
    corrected_linear_derivative,
)
from veqpy.math.interpolate import interpolation_matrix
from veqpy.math.quadrature import legendre_quadrature, quadrature_generator

HIGH_ORDER_NODE_COUNT = 129


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


def test_differentiation_matrix_preserves_polynomial_derivative():
    nodes, _ = legendre_quadrature(6)
    values = nodes**4 - 3.0 * nodes**2 + nodes
    expected = 4.0 * nodes**3 - 6.0 * nodes + 1.0

    matrix = math_differentiate.differentiation_matrix(nodes)

    assert np.allclose(matrix @ values, expected)


def test_high_order_differentiation_matrix_preserves_polynomial_derivative():
    nodes, _ = legendre_quadrature(HIGH_ORDER_NODE_COUNT)
    values = nodes**5 - 2.0 * nodes**3 + 0.25 * nodes
    expected = 5.0 * nodes**4 - 6.0 * nodes**2 + 0.25

    matrix = math_differentiate.differentiation_matrix(nodes)

    assert np.all(np.isfinite(matrix))
    assert np.allclose(matrix @ values, expected, rtol=1.0e-9, atol=1.0e-9)


def test_variable_limit_integration_matrix_preserves_polynomial_integral():
    nodes, _ = legendre_quadrature(6)
    values = nodes**3 - 2.0 * nodes
    expected = 0.25 * nodes**4 - nodes**2

    matrix = math_integrate.variable_limit_integration_matrix(nodes)

    assert np.allclose(matrix @ values, expected)


def test_high_order_variable_limit_integration_matrix_preserves_polynomial_integral():
    nodes, _ = legendre_quadrature(HIGH_ORDER_NODE_COUNT)
    values = nodes**5 - 2.0 * nodes**3 + 0.25 * nodes
    expected = nodes**6 / 6.0 - 0.5 * nodes**4 + 0.125 * nodes**2

    matrix = math_integrate.variable_limit_integration_matrix(nodes)

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

    for scheme in quadrature_generator:
        nodes, weights = quadrature_generator[scheme](HIGH_ORDER_NODE_COUNT)
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


def test_uniform_variable_limit_integration_matrix_matches_trapezoid_for_linear_data():
    n = 7
    nodes = np.linspace(0.0, 1.0, n)
    values = 3.0 * nodes - 2.0
    expected = 1.5 * nodes**2 - 2.0 * nodes

    matrix = math_integrate.uniform_variable_limit_integration_matrix(n)

    assert np.allclose(matrix @ values, expected)


def test_corrected_integration_matrix_matches_engine_kernel():
    nodes, _ = legendre_quadrature(6)
    values = nodes**2 + nodes
    base_integration = math_integrate.variable_limit_integration_matrix(nodes)
    base_differentiation = math_differentiate.differentiation_matrix(nodes)
    expected = np.empty_like(values)

    corrected_integration(
        expected,
        values,
        base_integration,
        2,
        nodes,
        base_differentiation,
    )
    matrix = math_integrate.corrected_integration_matrix(
        nodes,
        base_differentiation,
        p=2,
    )

    assert np.allclose(matrix @ values, expected)


def test_high_order_corrected_integration_matrix_matches_engine_kernel():
    nodes, _ = legendre_quadrature(HIGH_ORDER_NODE_COUNT)
    values = nodes**3 + nodes
    base_integration = math_integrate.variable_limit_integration_matrix(nodes)
    base_differentiation = math_differentiate.differentiation_matrix(nodes)
    expected = np.empty_like(values)

    corrected_integration(
        expected,
        values,
        base_integration,
        2,
        nodes,
        base_differentiation,
    )
    matrix = math_integrate.corrected_integration_matrix(
        nodes,
        base_differentiation,
        p=2,
    )

    assert np.all(np.isfinite(matrix))
    assert np.allclose(matrix @ values, expected, rtol=1.0e-10, atol=1.0e-10)


def test_corrected_derivative_matrices_match_engine_kernels():
    nodes, _ = legendre_quadrature(6)
    values = nodes**4 - nodes**2 + 0.25 * nodes
    base_differentiation = math_differentiate.differentiation_matrix(nodes)
    expected_linear = np.empty_like(values)
    expected_even = np.empty_like(values)

    corrected_linear_derivative(expected_linear, values, base_differentiation, rho=nodes)
    corrected_even_derivative(expected_even, values, base_differentiation, rho=nodes)

    linear_matrix = math_differentiate.corrected_linear_derivative_matrix(
        nodes, base_differentiation
    )
    even_matrix = math_differentiate.corrected_even_derivative_matrix(
        nodes, base_differentiation
    )

    assert np.allclose(linear_matrix @ values, expected_linear)
    assert np.allclose(even_matrix @ values, expected_even)


def test_high_order_corrected_derivative_matrices_match_engine_kernels():
    nodes, _ = legendre_quadrature(HIGH_ORDER_NODE_COUNT)
    values = nodes**4 - nodes**2 + 0.25 * nodes
    base_differentiation = math_differentiate.differentiation_matrix(nodes)
    expected_linear = np.empty_like(values)
    expected_even = np.empty_like(values)

    corrected_linear_derivative(expected_linear, values, base_differentiation, rho=nodes)
    corrected_even_derivative(expected_even, values, base_differentiation, rho=nodes)

    linear_matrix = math_differentiate.corrected_linear_derivative_matrix(
        nodes, base_differentiation
    )
    even_matrix = math_differentiate.corrected_even_derivative_matrix(
        nodes, base_differentiation
    )

    assert np.all(np.isfinite(linear_matrix))
    assert np.all(np.isfinite(even_matrix))
    assert np.allclose(linear_matrix @ values, expected_linear, rtol=1.0e-10, atol=1.0e-10)
    assert np.allclose(even_matrix @ values, expected_even, rtol=1.0e-10, atol=1.0e-10)

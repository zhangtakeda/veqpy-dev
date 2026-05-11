import numpy as np

import veqpy.math as math_api
from veqpy.engine import (
    corrected_even_derivative,
    corrected_integration,
    corrected_linear_derivative,
)
from veqpy.math.interpolate import interpolation_matrix
from veqpy.math.quadrature import available_quadrature_schemes, legendre_quadrature, make_quadrature

HIGH_ORDER_NODE_COUNT = 129


def _compact_calculus(nodes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return math_api.make_calculus(nodes, calculus="compact")


def _spectral_calculus(nodes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return math_api.make_calculus(nodes, calculus="spectral")


def test_math_public_api_hides_base_calculus_builders():
    assert not hasattr(math_api, "cfd33_matrices")
    assert not hasattr(math_api, "cfd33_differentiation_matrix")
    assert not hasattr(math_api, "cfd33_integration_matrix")
    assert not hasattr(math_api, "spectral_differentiation_matrix")
    assert not hasattr(math_api, "spectral_integration_matrix")
    assert not hasattr(math_api, "uniform_spectral_differentiation_matrix")
    assert not hasattr(math_api, "uniform_spectral_integration_matrix")


def test_make_calculus_rejects_non_1d_nodes():
    nodes = np.eye(4, dtype=np.float64)

    try:
        math_api.make_calculus(nodes, calculus="spectral")
    except ValueError as exc:
        assert "one-dimensional" in str(exc)
    else:
        raise AssertionError("Expected make_calculus to reject non-1D nodes")


def test_compact_calculus_requires_at_least_four_nodes():
    nodes = np.array([0.0, 0.5, 1.0], dtype=np.float64)

    try:
        _compact_calculus(nodes)
    except ValueError as exc:
        assert "at least 4 nodes" in str(exc)
    else:
        raise AssertionError("Expected compact calculus to require at least four nodes")


def test_compact_calculus_differentiation_preserves_low_order_polynomials():
    nodes = np.array([0.0, 0.05, 0.2, 0.45, 0.7, 1.0], dtype=np.float64)
    _, differentiation_matrix = _compact_calculus(nodes)

    assert np.all(np.isfinite(differentiation_matrix))
    assert np.allclose(differentiation_matrix @ np.ones_like(nodes), 0.0, atol=1.0e-13)
    assert np.allclose(differentiation_matrix @ nodes, 1.0, atol=1.0e-13)
    assert np.allclose(differentiation_matrix @ (nodes**2), 2.0 * nodes, atol=1.0e-13)


def test_compact_calculus_integration_enforces_axis_constraint_without_axis_node():
    nodes, _ = legendre_quadrature(16)
    integration_matrix, _ = _compact_calculus(nodes)
    constant_integral = integration_matrix @ np.ones_like(nodes)
    axis_value = interpolation_matrix(nodes, np.array([0.0])) @ constant_integral

    assert np.all(np.isfinite(integration_matrix))
    assert np.allclose(constant_integral, nodes, rtol=1.0e-12, atol=1.0e-12)
    assert np.allclose(axis_value, 0.0, atol=1.0e-13)


def test_compact_calculus_differentiation_recovers_compact_integral_on_interior():
    nodes = np.linspace(0.0, 1.0, 33)
    values = np.sin(nodes)
    integration_matrix, differentiation_matrix = _compact_calculus(nodes)
    recovered = differentiation_matrix @ (integration_matrix @ values)

    assert np.allclose(recovered[2:-2], values[2:-2], rtol=1.0e-6, atol=1.0e-6)


def test_calculus_registry_selects_compact_and_uniform_spectral_matrices():
    nodes = np.linspace(0.0, 1.0, 8)

    compact_integration, compact_differentiation = _compact_calculus(nodes)
    spectral_integration, spectral_differentiation = _spectral_calculus(nodes)

    assert np.allclose(compact_differentiation @ np.ones_like(nodes), 0.0, atol=1.0e-13)
    assert np.allclose(compact_integration @ np.ones_like(nodes), nodes, atol=1.0e-13)
    assert np.allclose(spectral_differentiation @ nodes, 1.0, atol=1.0e-13)
    assert np.allclose(spectral_integration @ (3.0 * nodes - 2.0), 1.5 * nodes**2 - 2.0 * nodes)


def test_calculus_registry_selects_nonuniform_spectral_matrices_from_nodes():
    nodes, _ = legendre_quadrature(8)

    spectral_integration, spectral_differentiation = _spectral_calculus(nodes)
    values = nodes**4 - 2.0 * nodes + 1.0
    derivative = 4.0 * nodes**3 - 2.0
    integral = nodes**4 - 2.0 * nodes

    assert np.allclose(spectral_differentiation @ values, derivative)
    assert np.allclose(spectral_integration @ derivative, integral)


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


def test_spectral_calculus_differentiation_preserves_polynomial_derivative():
    nodes, _ = legendre_quadrature(6)
    values = nodes**4 - 3.0 * nodes**2 + nodes
    expected = 4.0 * nodes**3 - 6.0 * nodes + 1.0

    _, matrix = _spectral_calculus(nodes)

    assert np.allclose(matrix @ values, expected)


def test_high_order_spectral_calculus_differentiation_preserves_polynomial_derivative():
    nodes, _ = legendre_quadrature(HIGH_ORDER_NODE_COUNT)
    values = nodes**5 - 2.0 * nodes**3 + 0.25 * nodes
    expected = 5.0 * nodes**4 - 6.0 * nodes**2 + 0.25

    _, matrix = _spectral_calculus(nodes)

    assert np.all(np.isfinite(matrix))
    assert np.allclose(matrix @ values, expected, rtol=1.0e-9, atol=1.0e-9)


def test_spectral_calculus_integration_preserves_polynomial_integral():
    nodes, _ = legendre_quadrature(6)
    values = nodes**3 - 2.0 * nodes
    expected = 0.25 * nodes**4 - nodes**2

    matrix, _ = _spectral_calculus(nodes)

    assert np.allclose(matrix @ values, expected)


def test_high_order_spectral_calculus_integration_preserves_polynomial_integral():
    nodes, _ = legendre_quadrature(HIGH_ORDER_NODE_COUNT)
    values = nodes**5 - 2.0 * nodes**3 + 0.25 * nodes
    expected = nodes**6 / 6.0 - 0.5 * nodes**4 + 0.125 * nodes**2

    matrix, _ = _spectral_calculus(nodes)

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


def test_uniform_spectral_calculus_matches_trapezoid_for_linear_data():
    n = 7
    nodes = np.linspace(0.0, 1.0, n)
    values = 3.0 * nodes - 2.0
    expected = 1.5 * nodes**2 - 2.0 * nodes

    matrix, _ = _spectral_calculus(nodes)

    assert np.allclose(matrix @ values, expected)


def test_corrected_integration_matrix_matches_engine_kernel():
    nodes, _ = legendre_quadrature(6)
    values = nodes**2 + nodes
    _, base_differentiation = _spectral_calculus(nodes)
    matrix = math_api.corrected_integration_matrix(
        nodes,
        base_differentiation,
        p=2,
    )
    expected = np.empty_like(values)

    corrected_integration(
        expected,
        values,
        matrix,
        2,
        nodes,
        base_differentiation,
    )

    assert np.allclose(matrix @ values, expected)


def test_high_order_corrected_integration_matrix_matches_engine_kernel():
    nodes, _ = legendre_quadrature(HIGH_ORDER_NODE_COUNT)
    values = nodes**3 + nodes
    _, base_differentiation = _spectral_calculus(nodes)
    matrix = math_api.corrected_integration_matrix(
        nodes,
        base_differentiation,
        p=2,
    )
    expected = np.empty_like(values)

    corrected_integration(
        expected,
        values,
        matrix,
        2,
        nodes,
        base_differentiation,
    )

    assert np.all(np.isfinite(matrix))
    assert np.allclose(matrix @ values, expected, rtol=1.0e-10, atol=1.0e-10)


def test_corrected_derivative_matrix_builds_distinct_power_routes():
    nodes, _ = legendre_quadrature(6)
    _, base_differentiation = _spectral_calculus(nodes)
    linear_matrix = math_api.corrected_differentiation_matrix(nodes, base_differentiation, p=1)
    even_matrix = math_api.corrected_differentiation_matrix(nodes, base_differentiation, p=2)

    assert linear_matrix.shape == base_differentiation.shape
    assert even_matrix.shape == base_differentiation.shape
    assert not np.allclose(linear_matrix, even_matrix)
    assert not hasattr(math_api, "odd_differentiation_matrix")
    assert not hasattr(math_api, "even_differentiation_matrix")


def test_corrected_integration_matrix_solves_once_when_precomputing(monkeypatch):
    nodes = np.array([0.25, 0.5, 0.75, 1.0], dtype=np.float64)
    base_differentiation = np.eye(nodes.shape[0], dtype=np.float64)
    original_solve = np.linalg.solve
    calls = {"solve": 0}

    def _track_solve(*args, **kwargs):
        calls["solve"] += 1
        return original_solve(*args, **kwargs)

    monkeypatch.setattr(np.linalg, "solve", _track_solve)

    matrix = math_api.corrected_integration_matrix(nodes, base_differentiation, p=3)

    assert calls["solve"] == 1
    assert matrix.shape == (nodes.shape[0], nodes.shape[0])


def test_corrected_derivative_matrices_match_engine_kernels():
    nodes, _ = legendre_quadrature(6)
    values = nodes**4 - nodes**2 + 0.25 * nodes
    _, base_differentiation = _spectral_calculus(nodes)
    expected_linear = np.empty_like(values)
    expected_even = np.empty_like(values)

    corrected_linear_derivative(expected_linear, values, base_differentiation, rho=nodes)
    corrected_even_derivative(expected_even, values, base_differentiation, rho=nodes)

    linear_matrix = math_api.corrected_differentiation_matrix(nodes, base_differentiation, p=1)
    even_matrix = math_api.corrected_differentiation_matrix(nodes, base_differentiation, p=2)

    assert np.allclose(linear_matrix @ values, expected_linear)
    assert np.allclose(even_matrix @ values, expected_even)


def test_corrected_differentiation_matrix_validates_shape():
    nodes, _ = legendre_quadrature(6)
    _, base_differentiation = _spectral_calculus(nodes)

    try:
        math_api.corrected_differentiation_matrix(nodes, base_differentiation[:, :-1], p=1)
    except ValueError as exc:
        assert "shape" in str(exc)
    else:
        raise AssertionError("Expected corrected_differentiation_matrix to validate matrix shape")


def test_high_order_corrected_derivative_matrices_match_engine_kernels():
    nodes, _ = legendre_quadrature(HIGH_ORDER_NODE_COUNT)
    values = nodes**4 - nodes**2 + 0.25 * nodes
    _, base_differentiation = _spectral_calculus(nodes)
    expected_linear = np.empty_like(values)
    expected_even = np.empty_like(values)

    corrected_linear_derivative(expected_linear, values, base_differentiation, rho=nodes)
    corrected_even_derivative(expected_even, values, base_differentiation, rho=nodes)

    linear_matrix = math_api.corrected_differentiation_matrix(nodes, base_differentiation, p=1)
    even_matrix = math_api.corrected_differentiation_matrix(nodes, base_differentiation, p=2)

    assert np.all(np.isfinite(linear_matrix))
    assert np.all(np.isfinite(even_matrix))
    assert np.allclose(linear_matrix @ values, expected_linear, rtol=1.0e-10, atol=1.0e-10)
    assert np.allclose(even_matrix @ values, expected_even, rtol=1.0e-10, atol=1.0e-10)

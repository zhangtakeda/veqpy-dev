import numpy as np

import veqpy.math as math_api
from veqpy.math.interpolate import interpolation_matrix
from veqpy.math.quadrature import legendre_quadrature, make_quadrature

HIGH_ORDER_NODE_COUNT = 129


def _compact_calculus(nodes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return math_api.make_calculus(nodes, calculus="compact")


def _cfd35_calculus(nodes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return math_api.make_calculus(nodes, calculus="cfd35")


def _cfd55_calculus(nodes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return math_api.make_calculus(nodes, calculus="cfd55")


def _spectral_calculus(nodes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return math_api.make_calculus(nodes, calculus="spectral")


def test_math_public_api_hides_base_calculus_builders():
    assert not hasattr(math_api, "cfd33_matrices")
    assert not hasattr(math_api, "cfd33_differentiator")
    assert not hasattr(math_api, "cfd33_accumulator")
    assert not hasattr(math_api, "spectral_differentiator")
    assert not hasattr(math_api, "spectral_accumulator")
    assert not hasattr(math_api, "uniform_spectral_differentiator")
    assert not hasattr(math_api, "uniform_spectral_accumulator")
    assert not hasattr(math_api, "cfd35_differentiator")
    assert not hasattr(math_api, "cfd55_differentiator")


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
    _, differentiator = _compact_calculus(nodes)

    assert np.all(np.isfinite(differentiator))
    assert np.allclose(differentiator @ np.ones_like(nodes), 0.0, atol=1.0e-13)
    assert np.allclose(differentiator @ nodes, 1.0, atol=1.0e-13)
    assert np.allclose(differentiator @ (nodes**2), 2.0 * nodes, atol=1.0e-13)


def test_compact_calculus_integration_enforces_axis_constraint_without_axis_node():
    nodes, _ = legendre_quadrature(16)
    accumulator, _ = _compact_calculus(nodes)
    constant_integral = accumulator @ np.ones_like(nodes)
    axis_value = interpolation_matrix(nodes, np.array([0.0])) @ constant_integral

    assert np.all(np.isfinite(accumulator))
    assert np.allclose(constant_integral, nodes, rtol=1.0e-12, atol=1.0e-12)
    assert np.allclose(axis_value, 0.0, atol=1.0e-13)


def test_compact_calculus_differentiation_recovers_compact_integral_on_interior():
    nodes = np.linspace(0.0, 1.0, 33)
    values = np.sin(nodes)
    accumulator, differentiator = _compact_calculus(nodes)
    recovered = differentiator @ (accumulator @ values)

    assert np.allclose(recovered[2:-2], values[2:-2], rtol=1.0e-6, atol=1.0e-6)


def test_higher_order_compact_calculus_requires_at_least_five_nodes():
    nodes = np.array([0.0, 0.3, 0.6, 1.0], dtype=np.float64)

    for calculus in ("cfd35", "cfd55"):
        try:
            math_api.make_calculus(nodes, calculus=calculus)
        except ValueError as exc:
            assert "at least 5 nodes" in str(exc)
        else:
            raise AssertionError(f"Expected {calculus} calculus to require at least five nodes")


def test_higher_order_compact_calculus_preserves_polynomial_derivatives():
    nodes = np.linspace(0.0, 1.0, 17)

    for calculus, exact_degree in (("cfd35", 6), ("cfd55", 8)):
        _, differentiator = math_api.make_calculus(nodes, calculus=calculus)

        assert np.all(np.isfinite(differentiator))
        for degree in range(exact_degree + 1):
            values = nodes**degree
            expected = np.zeros_like(nodes) if degree == 0 else degree * nodes ** (degree - 1)
            assert np.allclose(
                differentiator @ values,
                expected,
                rtol=1.0e-9,
                atol=1.0e-9,
            ), calculus


def test_higher_order_compact_calculus_integrates_constant_and_linear_data():
    nodes, _ = legendre_quadrature(17)

    for calculus in ("cfd35", "cfd55"):
        accumulator, differentiator = math_api.make_calculus(nodes, calculus=calculus)

        assert np.all(np.isfinite(accumulator))
        assert np.allclose(accumulator @ np.ones_like(nodes), nodes, rtol=1.0e-10, atol=1.0e-10)
        assert np.allclose(accumulator @ nodes, 0.5 * nodes**2, rtol=1.0e-10, atol=1.0e-10)
        assert np.allclose(differentiator @ nodes, 1.0, rtol=1.0e-11, atol=1.0e-11)


def test_calculus_registry_selects_compact_and_uniform_spectral_matrices():
    nodes = np.linspace(0.0, 1.0, 8)

    compact_integration, compact_differentiation = _compact_calculus(nodes)
    spectral_integration, spectral_differentiation = _spectral_calculus(nodes)

    assert np.allclose(compact_differentiation @ np.ones_like(nodes), 0.0, atol=1.0e-13)
    assert np.allclose(compact_integration @ np.ones_like(nodes), nodes, atol=1.0e-13)
    assert np.allclose(spectral_differentiation @ nodes, 1.0, atol=1.0e-13)
    assert np.allclose(spectral_integration @ (3.0 * nodes - 2.0), 1.5 * nodes**2 - 2.0 * nodes)


def test_calculus_registry_selects_cfd35_and_cfd55_matrices():
    nodes = np.linspace(0.0, 1.0, 9)
    cfd33_integration, cfd33_differentiation = math_api.make_calculus(nodes, calculus="cfd33")
    cfd35_integration, cfd35_differentiation = _cfd35_calculus(nodes)
    cfd55_integration, cfd55_differentiation = _cfd55_calculus(nodes)

    assert np.allclose(cfd35_integration @ np.ones_like(nodes), nodes, atol=1.0e-12)
    assert np.allclose(cfd55_integration @ np.ones_like(nodes), nodes, atol=1.0e-12)
    assert np.allclose(cfd35_differentiation @ nodes, 1.0, atol=1.0e-12)
    assert np.allclose(cfd55_differentiation @ nodes, 1.0, atol=1.0e-12)
    assert not np.allclose(cfd35_differentiation, cfd33_differentiation)
    assert not np.allclose(cfd55_differentiation, cfd33_differentiation)


def test_calculus_registry_selects_nonuniform_spectral_matrices_from_nodes():
    nodes, _ = legendre_quadrature(8)

    spectral_integration, spectral_differentiation = _spectral_calculus(nodes)
    values = nodes**4 - 2.0 * nodes + 1.0
    derivative = 4.0 * nodes**3 - 2.0
    integral = nodes**4 - 2.0 * nodes

    assert np.allclose(spectral_differentiation @ values, derivative)
    assert np.allclose(spectral_integration @ derivative, integral)


def test_ffn_projection_calculus_builds_unanchored_matrix_operator():
    nodes = np.linspace(0.0, 1.0, 17)
    filter_matrix = math_api.make_filter(nodes, degree=4)

    assert filter_matrix.shape == (nodes.shape[0], nodes.shape[0])
    assert np.all(np.isfinite(filter_matrix))

    values = np.exp(nodes) + 0.2 * (-1.0) ** np.arange(nodes.shape[0])
    projected = filter_matrix @ values

    assert not np.allclose(projected[0], values[0])
    assert not np.allclose(projected[-1], values[-1])


def test_ffn_projection_calculus_preserves_affine_nullspace_and_damps_roughness():
    nodes = np.linspace(0.0, 1.0, 21)
    filter_matrix = math_api.make_filter(nodes, degree=3)

    affine = 1.25 - 0.5 * nodes
    rough = affine + 0.2 * (-1.0) ** np.arange(nodes.shape[0])
    projected = filter_matrix @ rough

    assert np.allclose(filter_matrix @ affine, affine, rtol=1.0e-12, atol=1.0e-12)
    assert np.linalg.norm(np.diff(projected, n=2)) < np.linalg.norm(np.diff(rough, n=2))


def test_ffn_projection_calculus_rejects_invalid_nodes_and_accepts_disabled_degree():
    empty_filter = math_api.make_filter(
        np.linspace(0.0, 1.0, 4),
        degree=-1,
    )

    assert empty_filter.shape == (0, 0)

    try:
        math_api.make_filter(np.eye(3), degree=1)
    except ValueError as exc:
        assert "one-dimensional" in str(exc)
    else:
        raise AssertionError("Expected FFn projection calculus to reject non-1D nodes")


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

    for scheme in ["legendre", "chebyshev", "lobatto", "radau", "uniform"]:
        nodes, weights = make_quadrature(HIGH_ORDER_NODE_COUNT, quadrature=scheme)
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

import numpy as np

from veqpy.engine.numba_source import (
    _uniform_spline_interpolate_pair,
    resolve_source_inputs,
    uniform_barycentric_weights,
)
from veqpy.math.interpolate import (
    build_uniform_source_interpolation_coefficients,
    build_uniform_source_interpolation_matrix,
)
from veqpy.model import Boundary, Grid
from veqpy.operator import Operator, OperatorCase, build_profile_names
from veqpy.solver.solver import (
    _build_boundary_flat_initial_state,
    _build_boundary_homothetic_initial_state,
)


def _evaluate(coeff: np.ndarray, query: np.ndarray) -> np.ndarray:
    out0 = np.empty_like(query, dtype=np.float64)
    out1 = np.empty_like(query, dtype=np.float64)
    _uniform_spline_interpolate_pair(out0, out1, coeff, coeff, query)
    np.testing.assert_allclose(out0, out1)
    return out0


def test_local_polynomial_variants_reproduce_source_nodes():
    values = np.sin(np.linspace(0.0, 1.0, 9))
    query = np.linspace(0.0, 1.0, values.size)

    for kind in ("1", "2", "3"):
        coeff = build_uniform_source_interpolation_coefficients(values, kind=kind)
        actual = _evaluate(coeff, query)
        np.testing.assert_allclose(actual, values, atol=1.0e-12, rtol=1.0e-12)


def test_local_polynomial_variants_preserve_matching_polynomial_degree():
    source = np.linspace(0.0, 1.0, 11)
    query = np.linspace(0.0, 1.0, 37)

    cases = (
        ("linear", source + 2.0, query + 2.0),
        ("quadratic", source**2 - 0.25 * source + 1.0, query**2 - 0.25 * query + 1.0),
        ("cubic", source**3 - 2.0 * source**2 + source, query**3 - 2.0 * query**2 + query),
    )
    for kind, values, expected in cases:
        coeff = build_uniform_source_interpolation_coefficients(values, kind=kind)
        actual = _evaluate(coeff, query)
        np.testing.assert_allclose(actual, expected, atol=1.0e-12, rtol=1.0e-12)


def test_default_source_interpolation_kind_is_local_cubic(monkeypatch):
    monkeypatch.delenv("VEQPY_SOURCE_INTERP_KIND", raising=False)
    values = np.cos(np.linspace(0.0, 1.0, 8))

    default_coeff = build_uniform_source_interpolation_coefficients(values)
    explicit_coeff = build_uniform_source_interpolation_coefficients(values, kind="cubic")

    np.testing.assert_allclose(default_coeff, explicit_coeff, atol=0.0, rtol=0.0)


def test_uniform_remap_matrix_uses_selected_interpolation_kind():
    source = np.linspace(0.0, 1.0, 5)
    query = np.array([0.125, 0.375, 0.625, 0.875], dtype=np.float64)
    values = source**2

    linear_matrix = build_uniform_source_interpolation_matrix(query, values.size, kind="linear")
    quadratic_matrix = build_uniform_source_interpolation_matrix(
        query, values.size, kind="quadatic"
    )

    assert not np.allclose(linear_matrix @ values, quadratic_matrix @ values)
    np.testing.assert_allclose(quadratic_matrix @ values, query**2, atol=1.0e-12, rtol=1.0e-12)


def test_psin_uniform_can_use_barycentric_interpolation():
    source = np.linspace(0.0, 1.0, 9)
    query = np.array([0.1, 0.31, 0.58, 0.93], dtype=np.float64)
    heat = np.sin(source)
    current = np.cos(source)
    out_heat = np.empty_like(query)
    out_current = np.empty_like(query)
    coeff_heat = build_uniform_source_interpolation_coefficients(heat, kind="3")
    coeff_current = build_uniform_source_interpolation_coefficients(current, kind="3")
    weights = uniform_barycentric_weights(8)

    resolve_source_inputs(
        out_heat,
        out_current,
        heat,
        current,
        1,
        heat.size,
        weights,
        np.empty((0, 0), dtype=np.float64),
        coeff_heat,
        coeff_current,
        query,
        True,
    )

    bary_matrix = build_uniform_source_interpolation_matrix(query, heat.size, kind="barycentric")
    np.testing.assert_allclose(out_heat, bary_matrix @ heat, atol=1.0e-12, rtol=1.0e-12)
    np.testing.assert_allclose(out_current, bary_matrix @ current, atol=1.0e-12, rtol=1.0e-12)


def _operator_case_for_interpolation(nodes: str) -> tuple[Grid, OperatorCase]:
    grid = Grid(Nr=6, Nt=8, quadrature_scheme="legendre", M_max=2)
    profile_coeffs = {name: None for name in build_profile_names(grid.M_max)}
    profile_coeffs.update({"psin": [0.0, 1.0], "h": [0.0]})
    sample_count = grid.Nr if nodes == "grid" else 5
    source_axis = np.linspace(0.0, 1.0, sample_count)
    return grid, OperatorCase(
        route="PF",
        coordinate="rho",
        nodes=nodes,
        profile_coeffs=profile_coeffs,
        boundary=Boundary(
            a=1.0,
            R0=1.7,
            Z0=0.0,
            B0=2.0,
            c_offsets=np.zeros(grid.M_max + 1),
            s_offsets=np.zeros(grid.M_max + 1),
        ),
        heat_input=source_axis**2,
        current_input=source_axis**2,
    )


def test_operator_selects_uniform_rho_interpolation_kind():
    grid, case = _operator_case_for_interpolation("uniform")
    operator = Operator(grid=grid, case=case, source_interpolation_kind="quadratic")

    expected = operator.static_layout.rho**2
    np.testing.assert_allclose(
        operator.source_runtime_state.work_state.materialized_heat_input,
        expected,
        atol=1.0e-12,
        rtol=1.0e-12,
    )


def test_operator_interpolation_kind_is_noop_for_grid_nodes():
    grid, case = _operator_case_for_interpolation("grid")
    operator = Operator(grid=grid, case=case, source_interpolation_kind="not-a-real-kind")

    assert operator.source_runtime_state.const_state.fixed_remap_matrix.shape == (0, 0)
    np.testing.assert_allclose(
        operator.source_runtime_state.work_state.materialized_heat_input,
        case.heat_input,
        atol=0.0,
        rtol=0.0,
    )


def test_boundary_flat_initial_state_flattens_active_boundary_modes():
    grid, case = _operator_case_for_interpolation("uniform")
    case.profile_coeffs.update({"c1": [0.0], "s2": [0.0]})
    case.boundary.c_offsets[1] = 0.4
    case.boundary.s_offsets[2] = -0.3
    operator = Operator(grid=grid, case=case)

    coeffs = operator.build_coeffs(_build_boundary_flat_initial_state(operator), include_none=False)

    np.testing.assert_allclose(coeffs["c1"], [0.2])
    np.testing.assert_allclose(coeffs["s2"], [-0.3])



def test_boundary_homothetic_initial_state_preserves_scaled_boundary_slopes():
    grid, case = _operator_case_for_interpolation("uniform")
    case.profile_coeffs.update({"c1": [0.0], "s2": [0.0]})
    case.boundary.c_offsets[1] = 0.4
    case.boundary.s_offsets[2] = -0.3
    operator = Operator(grid=grid, case=case)

    coeffs = operator.build_coeffs(
        _build_boundary_homothetic_initial_state(operator), include_none=False
    )

    np.testing.assert_allclose(coeffs["c1"], [0.0])
    np.testing.assert_allclose(coeffs["s2"], [-0.15])

    relaxed_coeffs = operator.build_coeffs(
        _build_boundary_homothetic_initial_state(operator, boundary_slope_factor=0.75),
        include_none=False,
    )
    np.testing.assert_allclose(relaxed_coeffs["c1"], [0.05])
    np.testing.assert_allclose(relaxed_coeffs["s2"], [-0.1875])


def test_operator_runs_psin_uniform_barycentric_source_path():
    grid, case = _operator_case_for_interpolation("uniform")
    case.coordinate = "psin"
    case.heat_input = np.sin(np.linspace(0.0, 1.0, case.heat_input.size))
    case.current_input = np.cos(np.linspace(0.0, 1.0, case.current_input.size))
    operator = Operator(grid=grid, case=case, source_interpolation_kind="barycentric")

    residual = operator.residual_var(operator.encode_initial_state())

    assert operator.source_plan.interpolation_kind == "barycentric"
    assert operator.source_plan.uses_barycentric_interpolation
    assert np.all(np.isfinite(residual))

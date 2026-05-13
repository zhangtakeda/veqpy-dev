import ast
from pathlib import Path

import numpy as np

from veqpy.engine.numba_source import (
    _regularize_ffn_psin,
    _regularize_psin_r,
    _uniform_spline_interpolate_pair,
    _update_fixed_point_psin_query_and_spline_uniform_inputs_impl,
    build_uniform_not_a_knot_spline_coefficients,
)


def test_psin_coordinate_update_uses_base_accumulator():
    """Route kernels should integrate psin_r with the base grid integration matrix."""

    source = Path("veqpy/engine/numba_source.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    call_args: list[str] = []

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if not isinstance(node.func, ast.Name):
            continue
        if node.func.id != "_update_psin_coordinate":
            continue
        assert len(node.args) == 3
        call_args.append(ast.unparse(node.args[2]))

    assert call_args
    assert set(call_args) == {"accumulator"}


def test_uniform_source_interpolation_is_cubic_for_smooth_profiles():
    axis = np.linspace(0.0, 1.0, 6, dtype=np.float64)
    values0 = axis**3 - 0.25 * axis
    values1 = -2.0 * axis**3 + axis * axis
    coeff0 = build_uniform_not_a_knot_spline_coefficients(values0)
    coeff1 = build_uniform_not_a_knot_spline_coefficients(values1)
    query = np.array([0.0, 0.03, 0.11, 0.31, 0.58, 0.83, 1.0], dtype=np.float64)
    out0 = np.empty_like(query)
    out1 = np.empty_like(query)

    _uniform_spline_interpolate_pair(out0, out1, coeff0, coeff1, query)

    assert np.allclose(out0, query**3 - 0.25 * query)
    assert np.allclose(out1, -2.0 * query**3 + query * query)


def test_fixed_point_uniform_source_update_uses_cubic_interpolation():
    axis = np.linspace(0.0, 1.0, 6, dtype=np.float64)
    heat = axis**3 - 0.25 * axis
    current = -2.0 * axis**3 + axis * axis
    heat_coeff = build_uniform_not_a_knot_spline_coefficients(heat)
    current_coeff = build_uniform_not_a_knot_spline_coefficients(current)
    query = np.full(7, -1.0, dtype=np.float64)
    psin = np.array([0.0, 0.03, 0.11, 0.31, 0.58, 0.83, 1.0], dtype=np.float64)
    out_heat = np.empty_like(psin)
    out_current = np.empty_like(psin)

    converged = _update_fixed_point_psin_query_and_spline_uniform_inputs_impl(
        query,
        psin,
        1.0e-14,
        out_heat,
        out_current,
        heat,
        current,
        heat_coeff,
        current_coeff,
    )

    assert not converged
    assert np.allclose(query, psin)
    assert np.allclose(out_heat, psin**3 - 0.25 * psin)
    assert np.allclose(out_current, -2.0 * psin**3 + psin * psin)


def test_axis_psin_r_regularizer_precomputed_n_fix():
    rho = np.array(
        [0.002, 0.012, 0.035, 0.062, 0.095, 0.14, 0.20],
        dtype=np.float64,
    )
    polluted_slopes = np.array([0.0, 12.0, 5.0, 2.38, 2.35, 2.352, 2.353], dtype=np.float64)
    psin_r = rho * polluted_slopes

    # n_fix = count of rho < 0.05 = 3 (indices 0, 1, 2)
    n_fix = 3
    _regularize_psin_r(psin_r, rho, n_fix)

    anchor0, anchor1 = n_fix, n_fix + 1
    x0 = rho[anchor0] * rho[anchor0]
    x1 = rho[anchor1] * rho[anchor1]
    slope_gradient = (polluted_slopes[anchor1] - polluted_slopes[anchor0]) / (x1 - x0)
    expected_head_slopes = polluted_slopes[anchor0] + slope_gradient * (
        rho[:anchor0] * rho[:anchor0] - x0
    )

    assert np.allclose(psin_r[:anchor0] / rho[:anchor0], expected_head_slopes)
    assert np.allclose(psin_r[anchor0:] / rho[anchor0:], polluted_slopes[anchor0:])
    assert np.ptp(psin_r[:anchor0] / rho[:anchor0]) > 1e-4

    # n_fix=0 should be a no-op (all points already in clean region)
    psin_r2 = rho * polluted_slopes
    _regularize_psin_r(psin_r2, rho, 0)
    assert np.allclose(psin_r2, rho * polluted_slopes)


def test_axis_psin_r_regularizer_applies_global_floor():
    rho = np.array([0.002, 0.012, 0.035, 0.062], dtype=np.float64)
    psin_r = np.array([0.0, -1.0e-12, 2.0e-10, 4.0e-10], dtype=np.float64)

    _regularize_psin_r(psin_r, rho, 0)

    assert np.all(psin_r >= 1.0e-10)
    assert psin_r[2] == 2.0e-10


def test_ffn_psin_regularizer_uses_even_axis_form():
    rho = np.array([0.002, 0.012, 0.035, 0.062, 0.095], dtype=np.float64)
    ffn = np.array([10.0, -3.0, 4.0, 2.38, 2.35], dtype=np.float64)
    n_fix = 3

    _regularize_ffn_psin(ffn, rho, n_fix)

    x0 = rho[n_fix] * rho[n_fix]
    x1 = rho[n_fix + 1] * rho[n_fix + 1]
    value_gradient = (2.35 - 2.38) / (x1 - x0)
    expected_head = 2.38 + value_gradient * (rho[:n_fix] * rho[:n_fix] - x0)
    assert np.allclose(ffn[:n_fix], expected_head)
    assert np.allclose(ffn[n_fix:], [2.38, 2.35])


def test_numba_source_uses_regularized_psin_r_directly():
    source = Path("veqpy/engine/numba_source.py").read_text(encoding="utf-8")

    assert "psin_r_safe" not in source
    assert "_SLOT_PSIN_R_SAFE" not in source
    assert "maximum_floor_into(psin_r" not in source
    assert "not (psin_r[i] >=" not in source
    assert "np.isnan(psin_r" not in source
    assert "np.isnan(FFn_psin" not in source
    assert "_regularize_psin_r(Itor_r" not in source


def test_all_source_routes_regularize_psin_r_after_writes_and_ffn_after_final_writes():
    source = Path("veqpy/engine/numba_source.py").read_text(encoding="utf-8")
    tree = ast.parse(source)

    route_functions = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name.startswith("_update_")
        and node.name.endswith("_with_scratch")
    ]
    assert route_functions

    for function in route_functions:
        statements = function.body
        psin_r_regularizations = [
            statement
            for statement in ast.walk(function)
            if isinstance(statement, ast.Call)
            and isinstance(statement.func, ast.Name)
            and statement.func.id == "_regularize_psin_r"
            and ast.unparse(statement.args[0]) == "out_psin_r"
        ]
        psin_r_differentiations = [
            statement
            for statement in ast.walk(function)
            if isinstance(statement, ast.Call)
            and isinstance(statement.func, ast.Name)
            and statement.func.id == "full_differentiation"
            and [ast.unparse(arg) for arg in statement.args[:2]] == ["out_psin_rr", "out_psin_r"]
        ]
        assert len(psin_r_regularizations) == 1, function.name
        assert len(psin_r_differentiations) == 1, function.name

        if function.name.startswith("_update_pf_"):
            for index, statement in enumerate(statements):
                if not isinstance(statement, ast.Assign):
                    continue
                if [ast.unparse(target) for target in statement.targets] != ["prof"]:
                    continue
                previous = statements[index - 1]
                assert isinstance(previous, ast.Expr), function.name
                assert ast.unparse(previous.value) == (
                    "_regularize_psin_r(out_psin_r, rho, n_axis_fix)"
                )

        for index, statement in enumerate(statements):
            if not isinstance(statement, ast.Expr) or not isinstance(statement.value, ast.Call):
                continue
            call = statement.value
            if not isinstance(call.func, ast.Name) or call.func.id != "full_differentiation":
                continue
            args = [ast.unparse(arg) for arg in call.args]
            if args[:2] != ["out_psin_rr", "out_psin_r"]:
                continue
            if function.name.startswith("_update_pf_"):
                continue
            previous = statements[index - 1]
            assert isinstance(previous, ast.Expr), function.name
            assert ast.unparse(previous.value) == "_regularize_psin_r(out_psin_r, rho, n_axis_fix)"

        ffn_regularizations = [
            statement
            for statement in ast.walk(function)
            if isinstance(statement, ast.Call)
            and isinstance(statement.func, ast.Name)
            and statement.func.id == "_regularize_ffn_psin"
            and ast.unparse(statement.args[0]) == "out_FFn_psin"
        ]
        assert ffn_regularizations, function.name

        def _is_out_ffn_write(node: ast.stmt) -> bool:
            if not isinstance(node, ast.Expr) or not isinstance(node.value, ast.Call):
                return False
            if (
                isinstance(node.value.func, ast.Name)
                and node.value.func.id == "_regularize_ffn_psin"
            ):
                return False
            return any(ast.unparse(arg) == "out_FFn_psin" for arg in node.value.args[:1])

        for index, statement in enumerate(statements):
            if not _is_out_ffn_write(statement):
                continue
            if index + 1 >= len(statements) or _is_out_ffn_write(statements[index + 1]):
                continue
            following = statements[index + 1]
            assert isinstance(following, ast.Expr), function.name
            assert ast.unparse(following.value) == (
                "_regularize_ffn_psin(out_FFn_psin, rho, n_axis_fix)"
            )


def test_profile_owned_psin_materialization_regularizes_psin_r():
    source = Path("veqpy/engine/numba_source.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    materializer = next(
        node
        for node in tree.body
        if (
            isinstance(node, ast.FunctionDef)
            and node.name == "_materialize_profile_owned_psin_source_impl"
        )
    )
    calls = [
        ast.unparse(node)
        for node in ast.walk(materializer)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    ]

    assert "_regularize_psin_r(out_psin_r, rho, n_axis_fix)" in calls
    assert "full_differentiation(out_psin_rr, out_psin_r, differentiator)" in calls
    assert "_update_psin_coordinate(out_psin, out_psin_r, accumulator)" in calls

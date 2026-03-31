import importlib.util
import numpy as np
from pathlib import Path
import sys

from veqpy.engine import PSIN_COORDINATE, RHO_COORDINATE
from veqpy.engine.numba_geometry import update_geometry as numba_update_geometry
from veqpy.engine.numba_source import build_source_remap_cache as build_numba_source_remap_cache
from veqpy.engine.numba_residual import bind_residual_runner as bind_numba_residual_runner
from veqpy.engine.numba_source import resolve_source_inputs as numba_resolve_source_inputs
from veqpy.engine.numpy_geometry import update_geometry as numpy_update_geometry
from veqpy.engine.numpy_source import build_source_remap_cache as build_numpy_source_remap_cache
from veqpy.engine.numpy_residual import bind_residual_runner as bind_numpy_residual_runner
from veqpy.engine.numpy_source import resolve_source_inputs as numpy_resolve_source_inputs
from veqpy.model import Boundary, Geometry, Grid
from veqpy.operator import Operator
from veqpy.operator.layout import build_profile_names
from veqpy.operator.operator_case import OperatorCase

TEST_SOURCE_SAMPLE_COUNT = 21


def _poly_fields(grid: Grid, c0: float, c1: float = 0.0, c2: float = 0.0) -> np.ndarray:
    rho = grid.rho
    out = np.empty((3, grid.Nr), dtype=np.float64)
    out[0] = c0 + c1 * rho + c2 * rho * rho
    out[1] = c1 + 2.0 * c2 * rho
    out[2] = 2.0 * c2
    return out


def _allocate_outputs(grid: Grid) -> tuple[np.ndarray, ...]:
    nr = grid.Nr
    nt = grid.Nt
    return (
        np.empty((8, nr, nt), dtype=np.float64),
        np.empty((6, nr, nt), dtype=np.float64),
        np.empty((6, nr, nt), dtype=np.float64),
        np.empty((8, nr, nt), dtype=np.float64),
        np.empty((7, nr, nt), dtype=np.float64),
        np.empty(nr, dtype=np.float64),
        np.empty(nr, dtype=np.float64),
        np.empty(nr, dtype=np.float64),
        np.empty(nr, dtype=np.float64),
        np.empty(nr, dtype=np.float64),
    )


def _build_family_fields(grid: Grid) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    h_fields = _poly_fields(grid, 0.05, -0.02, 0.01)
    v_fields = _poly_fields(grid, -0.03, 0.01, -0.02)
    k_fields = _poly_fields(grid, 1.6, 0.05, -0.01)
    c_fields = np.zeros((grid.M_max + 1, 3, grid.Nr), dtype=np.float64)
    s_fields = np.zeros((grid.M_max + 1, 3, grid.Nr), dtype=np.float64)
    c_fields[0] = _poly_fields(grid, 0.02, 0.01, -0.005)
    c_fields[2] = _poly_fields(grid, -0.04, 0.03, 0.0)
    c_fields[3] = _poly_fields(grid, 0.01, -0.02, 0.01)
    s_fields[1] = _poly_fields(grid, 0.05, -0.01, 0.0)
    s_fields[4] = _poly_fields(grid, -0.03, 0.02, -0.005)
    return h_fields, v_fields, k_fields, c_fields, s_fields


def _expected_tb(grid: Grid, c_fields: np.ndarray, s_fields: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    tb = grid.theta[None, :] + c_fields[0, 0][:, None]
    tb_t = np.ones((grid.Nr, grid.Nt), dtype=np.float64)
    tb_tt = np.zeros((grid.Nr, grid.Nt), dtype=np.float64)

    for order in range(1, c_fields.shape[0]):
        tb += c_fields[order, 0][:, None] * grid.cos_ktheta[order][None, :]
        tb_t -= c_fields[order, 0][:, None] * grid.k_sin_ktheta[order][None, :]
        tb_tt -= c_fields[order, 0][:, None] * grid.k2_cos_ktheta[order][None, :]

    for order in range(1, s_fields.shape[0]):
        tb += s_fields[order, 0][:, None] * grid.sin_ktheta[order][None, :]
        tb_t += s_fields[order, 0][:, None] * grid.k_cos_ktheta[order][None, :]
        tb_tt -= s_fields[order, 0][:, None] * grid.k2_sin_ktheta[order][None, :]

    return tb, tb_t, tb_tt


def _build_operator_with_high_order_profiles() -> Operator:
    grid = Grid(Nr=8, Nt=16, scheme="uniform", M_max=4)
    profile_coeffs = {name: None for name in build_profile_names(grid.M_max)}
    profile_coeffs.update(
        {
            "psin": [0.0, 1.0],
            "F": [1.0],
            "h": [0.0],
            "v": [0.0],
            "k": [0.0],
            "c0": [0.0],
            "c3": [0.05],
            "s4": [-0.03],
        }
    )
    case = OperatorCase(
        name="PF",
        coordinate="rho",
        nodes="uniform",
        profile_coeffs=profile_coeffs,
        boundary=Boundary(
            a=1.1,
            R0=1.7,
            Z0=0.2,
            B0=3.0,
            c_offsets=np.zeros(grid.M_max + 1),
            s_offsets=np.zeros(grid.M_max + 1),
        ),
        heat_input=np.zeros(TEST_SOURCE_SAMPLE_COUNT),
        current_input=np.zeros(TEST_SOURCE_SAMPLE_COUNT),
    )
    return Operator(grid=grid, case=case)


def test_geometry_update_uses_high_order_fourier_family_terms():
    grid = Grid(Nr=8, Nt=16, scheme="uniform", M_max=4)
    geometry = Geometry(grid=grid)
    h_fields, v_fields, k_fields, c_fields, s_fields = _build_family_fields(grid)

    geometry.update(
        1.2,
        1.8,
        -0.1,
        grid,
        h_fields,
        v_fields,
        k_fields,
        c_fields,
        s_fields,
    )

    expected_tb, expected_tb_t, expected_tb_tt = _expected_tb(grid, c_fields, s_fields)

    assert np.allclose(geometry.tb, expected_tb)
    assert np.allclose(geometry.tb_t, expected_tb_t)
    assert np.allclose(geometry.tb_tt, expected_tb_tt)
    assert not np.allclose(geometry.tb, grid.theta[None, :] + c_fields[0, 0][:, None])


def test_numpy_and_numba_geometry_match_for_high_order_terms():
    grid = Grid(Nr=8, Nt=16, scheme="uniform", M_max=4)
    h_fields, v_fields, k_fields, c_fields, s_fields = _build_family_fields(grid)
    numpy_outputs = _allocate_outputs(grid)
    numba_outputs = _allocate_outputs(grid)

    numpy_update_geometry(
        *numpy_outputs,
        1.2,
        1.8,
        -0.1,
        grid.rho,
        grid.theta,
        grid.cos_ktheta,
        grid.sin_ktheta,
        grid.k_cos_ktheta,
        grid.k_sin_ktheta,
        grid.k2_cos_ktheta,
        grid.k2_sin_ktheta,
        grid.weights,
        h_fields,
        v_fields,
        k_fields,
        c_fields,
        s_fields,
        grid.M_max,
        grid.M_max,
    )
    numba_update_geometry(
        *numba_outputs,
        1.2,
        1.8,
        -0.1,
        grid.rho,
        grid.theta,
        grid.cos_ktheta,
        grid.sin_ktheta,
        grid.k_cos_ktheta,
        grid.k_sin_ktheta,
        grid.k2_cos_ktheta,
        grid.k2_sin_ktheta,
        grid.weights,
        h_fields,
        v_fields,
        k_fields,
        c_fields,
        s_fields,
        grid.M_max,
        grid.M_max,
    )

    for numpy_arr, numba_arr in zip(numpy_outputs, numba_outputs, strict=True):
        assert np.allclose(numpy_arr, numba_arr, atol=1e-11, rtol=1e-11)


def test_numpy_and_numba_residual_runners_match_for_high_order_blocks():
    grid = Grid(Nr=8, Nt=16, scheme="uniform", M_max=4)
    rng = np.random.default_rng(0)
    profile_names = ("c3", "s4", "k", "c0", "psin", "F")
    coeff_index_rows = np.array(
        [
            [0, -1],
            [1, -1],
            [2, 3],
            [4, -1],
            [5, 6],
            [7, -1],
        ],
        dtype=np.int64,
    )
    lengths = np.array([1, 1, 2, 1, 2, 1], dtype=np.int64)
    residual_size = 8

    G = rng.normal(size=(grid.Nr, grid.Nt))
    psin_R = rng.normal(size=(grid.Nr, grid.Nt))
    psin_Z = rng.normal(size=(grid.Nr, grid.Nt))
    sin_tb = rng.normal(size=(grid.Nr, grid.Nt))

    numpy_runner = bind_numpy_residual_runner(profile_names, coeff_index_rows, lengths, residual_size)
    numba_runner = bind_numba_residual_runner(profile_names, coeff_index_rows, lengths, residual_size)

    numpy_out = numpy_runner(
        G,
        psin_R,
        psin_Z,
        sin_tb,
        grid.sin_ktheta,
        grid.cos_ktheta,
        grid.rho_powers,
        grid.y,
        grid.T_fields[0],
        grid.weights,
        1.2,
        1.8,
        3.0,
    )
    numba_out = numba_runner(
        G,
        psin_R,
        psin_Z,
        sin_tb,
        grid.sin_ktheta,
        grid.cos_ktheta,
        grid.rho_powers,
        grid.y,
        grid.T_fields[0],
        grid.weights,
        1.2,
        1.8,
        3.0,
    )

    assert np.allclose(numpy_out, numba_out, atol=1e-11, rtol=1e-11)


def test_operator_runtime_propagates_high_order_geometry_and_residual():
    grid = Grid(Nr=8, Nt=16, scheme="uniform", M_max=4)
    profile_coeffs = {name: None for name in build_profile_names(grid.M_max)}
    profile_coeffs["psin"] = [0.0]
    case = OperatorCase(
        name="PF",
        coordinate="rho",
        nodes="uniform",
        profile_coeffs=profile_coeffs,
        boundary=Boundary(
            a=1.1,
            R0=1.7,
            Z0=0.2,
            B0=3.0,
            c_offsets=np.array([0.03, 0.0, -0.02, 0.05, 0.0]),
            s_offsets=np.array([0.0, 0.01, 0.0, 0.0, -0.04]),
        ),
        heat_input=np.zeros(TEST_SOURCE_SAMPLE_COUNT),
        current_input=np.zeros(TEST_SOURCE_SAMPLE_COUNT),
    )
    operator = Operator(grid=grid, case=case)

    operator.stage_b_geometry()
    expected_tb, _, _ = _expected_tb(grid, operator.c_family_fields, operator.s_family_fields)

    assert np.max(np.abs(operator.c_family_fields[3, 0])) > 0.0
    assert np.max(np.abs(operator.s_family_fields[4, 0])) > 0.0
    assert np.allclose(operator.geometry.tb, expected_tb)

    residual_operator = _build_operator_with_high_order_profiles()
    rng = np.random.default_rng(1)
    residual_operator.G[:] = rng.normal(size=residual_operator.G.shape)
    residual_operator.psin_R[:] = rng.normal(size=residual_operator.psin_R.shape)
    residual_operator.psin_Z[:] = rng.normal(size=residual_operator.psin_Z.shape)
    residual_operator.geometry.tb_fields[7] = rng.normal(size=residual_operator.geometry.sin_tb.shape)

    residual = residual_operator._assemble_residual()

    assert residual.shape == (residual_operator.x_size,)
    assert np.all(np.isfinite(residual))


def test_operator_residual_returns_independent_arrays_between_calls():
    operator = _build_operator_with_high_order_profiles()
    x = operator.encode_initial_state()

    residual_first = operator.residual(x)
    residual_first_snapshot = residual_first.copy()
    residual_second = operator.residual(x)

    assert residual_first.shape == residual_second.shape
    assert not np.shares_memory(residual_first, residual_second)
    assert np.allclose(residual_first, residual_first_snapshot)


def test_numpy_and_numba_source_resolution_match_for_rho_and_psin_coordinates():
    grid = Grid(Nr=8, Nt=8, scheme="uniform")
    heat_input = np.linspace(-1.0, 2.0, TEST_SOURCE_SAMPLE_COUNT)
    current_input = np.linspace(0.5, -0.25, TEST_SOURCE_SAMPLE_COUNT)
    psin_query = np.clip(grid.rho * grid.rho + 0.02, 0.0, 1.0)
    numpy_rho_plan = build_numpy_source_remap_cache("rho", TEST_SOURCE_SAMPLE_COUNT, rho=grid.rho)
    numba_rho_plan = build_numba_source_remap_cache("rho", TEST_SOURCE_SAMPLE_COUNT, rho=grid.rho)
    numpy_psin_plan = build_numpy_source_remap_cache("psin", TEST_SOURCE_SAMPLE_COUNT)
    numba_psin_plan = build_numba_source_remap_cache("psin", TEST_SOURCE_SAMPLE_COUNT)
    numpy_rho_stencil, numpy_rho_weights, numpy_rho_matrix = numpy_rho_plan
    numba_rho_stencil, numba_rho_weights, numba_rho_matrix = numba_rho_plan
    numpy_psin_stencil, numpy_psin_weights, numpy_psin_matrix = numpy_psin_plan
    numba_psin_stencil, numba_psin_weights, numba_psin_matrix = numba_psin_plan

    numpy_heat_rho = np.empty(grid.Nr, dtype=np.float64)
    numpy_current_rho = np.empty(grid.Nr, dtype=np.float64)
    numba_heat_rho = np.empty(grid.Nr, dtype=np.float64)
    numba_current_rho = np.empty(grid.Nr, dtype=np.float64)
    numpy_heat_psin = np.empty(grid.Nr, dtype=np.float64)
    numpy_current_psin = np.empty(grid.Nr, dtype=np.float64)
    numba_heat_psin = np.empty(grid.Nr, dtype=np.float64)
    numba_current_psin = np.empty(grid.Nr, dtype=np.float64)

    numpy_resolve_source_inputs(
        numpy_heat_rho,
        numpy_current_rho,
        heat_input,
        current_input,
        RHO_COORDINATE,
        TEST_SOURCE_SAMPLE_COUNT,
        numpy_rho_weights,
        numpy_rho_matrix,
        psin_query,
    )
    numba_resolve_source_inputs(
        numba_heat_rho,
        numba_current_rho,
        heat_input,
        current_input,
        RHO_COORDINATE,
        TEST_SOURCE_SAMPLE_COUNT,
        numba_rho_weights,
        numba_rho_matrix,
        psin_query,
    )
    numpy_resolve_source_inputs(
        numpy_heat_psin,
        numpy_current_psin,
        heat_input,
        current_input,
        PSIN_COORDINATE,
        TEST_SOURCE_SAMPLE_COUNT,
        numpy_psin_weights,
        numpy_psin_matrix,
        psin_query,
    )
    numba_resolve_source_inputs(
        numba_heat_psin,
        numba_current_psin,
        heat_input,
        current_input,
        PSIN_COORDINATE,
        TEST_SOURCE_SAMPLE_COUNT,
        numba_psin_weights,
        numba_psin_matrix,
        psin_query,
    )

    assert numpy_rho_stencil > 0
    assert numba_rho_stencil > 0
    assert numpy_rho_matrix.shape == (grid.Nr, TEST_SOURCE_SAMPLE_COUNT)
    assert numba_rho_matrix.shape == (grid.Nr, TEST_SOURCE_SAMPLE_COUNT)
    assert numpy_psin_stencil > 0
    assert numba_psin_stencil > 0
    assert numpy_psin_matrix.shape == (0, 0)
    assert numba_psin_matrix.shape == (0, 0)
    assert np.allclose(numpy_heat_rho, numba_heat_rho, atol=1e-12, rtol=1e-12)
    assert np.allclose(numpy_current_rho, numba_current_rho, atol=1e-12, rtol=1e-12)
    assert np.allclose(numpy_heat_psin, numba_heat_psin, atol=1e-12, rtol=1e-12)
    assert np.allclose(numpy_current_psin, numba_current_psin, atol=1e-12, rtol=1e-12)


def test_psin_source_resolution_uses_monotone_linear_remap_without_axis_overshoot():
    n_src = 9
    psin_query = np.array([0.0, 0.002, 0.01, 0.04, 0.16, 0.36, 0.64, 1.0], dtype=np.float64)
    src_axis = np.linspace(0.0, 1.0, n_src, dtype=np.float64)
    heat_input = np.linspace(1.0, 3.0, n_src, dtype=np.float64)
    current_input = np.linspace(0.2, 2.0, n_src, dtype=np.float64)

    numpy_plan = build_numpy_source_remap_cache("psin", n_src)
    numba_plan = build_numba_source_remap_cache("psin", n_src)
    _, numpy_weights, numpy_matrix = numpy_plan
    _, numba_weights, numba_matrix = numba_plan

    numpy_heat = np.empty(psin_query.shape[0], dtype=np.float64)
    numpy_current = np.empty(psin_query.shape[0], dtype=np.float64)
    numba_heat = np.empty(psin_query.shape[0], dtype=np.float64)
    numba_current = np.empty(psin_query.shape[0], dtype=np.float64)

    numpy_resolve_source_inputs(
        numpy_heat,
        numpy_current,
        heat_input,
        current_input,
        PSIN_COORDINATE,
        n_src,
        numpy_weights,
        numpy_matrix,
        psin_query,
    )
    numba_resolve_source_inputs(
        numba_heat,
        numba_current,
        heat_input,
        current_input,
        PSIN_COORDINATE,
        n_src,
        numba_weights,
        numba_matrix,
        psin_query,
    )

    expected_heat = np.interp(psin_query, src_axis, heat_input)
    expected_current = np.interp(psin_query, src_axis, current_input)

    assert np.allclose(numpy_heat, expected_heat, atol=1e-12, rtol=1e-12)
    assert np.allclose(numpy_current, expected_current, atol=1e-12, rtol=1e-12)
    assert np.allclose(numba_heat, expected_heat, atol=1e-12, rtol=1e-12)
    assert np.allclose(numba_current, expected_current, atol=1e-12, rtol=1e-12)
    assert np.all(numpy_current >= current_input[0])
    assert np.all(numba_current >= current_input[0])


def test_pq_psin_uniform_benchmark_cases_stay_within_shape_tolerance():
    benchmark_path = Path(__file__).with_name("benchmark.py")
    spec = importlib.util.spec_from_file_location("veqpy_tests_benchmark", benchmark_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load benchmark module from {benchmark_path}")
    benchmark = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = benchmark
    spec.loader.exec_module(benchmark)

    original_repeat = benchmark.BENCHMARK_REPEAT_COUNT
    try:
        benchmark.BENCHMARK_REPEAT_COUNT = 1
        reference = benchmark._solve_reference()
        for constraint in ("Ip_beta", "Ip", "beta", "null"):
            spec_case = benchmark.BenchmarkCaseSpec("PQ", "psin", constraint, "uniform")
            row = benchmark._benchmark_case_result(spec_case, reference)
            assert row.shape_error <= benchmark.SHAPE_MATCH_TOL, (spec_case.case_name, row.shape_error)
    finally:
        benchmark.BENCHMARK_REPEAT_COUNT = original_repeat

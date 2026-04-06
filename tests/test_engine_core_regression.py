import importlib.util
import sys
import warnings
from pathlib import Path

import numpy as np
import pytest

from veqpy.engine import PSIN_COORDINATE, RHO_COORDINATE
from veqpy.engine.numba_geometry import update_geometry as numba_update_geometry
from veqpy.engine.numba_residual import bind_residual_runner as bind_numba_residual_runner
from veqpy.engine.numba_source import build_source_remap_cache as build_numba_source_remap_cache
from veqpy.engine.numba_source import resolve_source_inputs as numba_resolve_source_inputs
from veqpy.engine.numpy_geometry import update_geometry as numpy_update_geometry
from veqpy.engine.numpy_residual import bind_residual_runner as bind_numpy_residual_runner
from veqpy.engine.numpy_source import build_source_remap_cache as build_numpy_source_remap_cache
from veqpy.engine.numpy_source import resolve_source_inputs as numpy_resolve_source_inputs
from veqpy.model import Boundary, Geometry, Grid
from veqpy.model.equilibrium import MU0
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
        route="PF",
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
        route="PF",
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

    residual = residual_operator.execution_state.residual_pack_stage_runner()

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


def test_pq_psin_uniform_benchmark_cases_stay_within_tight_shape_tolerance():
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
        thresholds = {
            "Ip_beta": 1.0e-2,
            "Ip": 1.0e-2,
            "beta": 5.0e-3,
            "null": 5.0e-3,
        }
        for constraint in ("Ip_beta", "Ip", "beta", "null"):
            spec_case = benchmark.BenchmarkCaseSpec("PQ", "psin", constraint, "uniform")
            row = benchmark._benchmark_case_result(spec_case, reference)
            assert row.shape_error <= thresholds[constraint], (spec_case.case_name, row.shape_error)
    finally:
        benchmark.BENCHMARK_REPEAT_COUNT = original_repeat


def test_pq_psin_uniform_ip_cases_improve_shape_and_psi_r_error():
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
        thresholds = {
            "Ip_beta": (9.25e-3, 2.60e-2),
            "Ip": (8.95e-3, 2.52e-2),
        }
        for constraint, (shape_threshold, psi_r_threshold) in thresholds.items():
            spec_case = benchmark.BenchmarkCaseSpec("PQ", "psin", constraint, "uniform")
            row = benchmark._benchmark_case_result(spec_case, reference)
            assert row.shape_error <= shape_threshold, (spec_case.case_name, row.shape_error)
            assert row.psi_r_rel_rms_error <= psi_r_threshold, (
                spec_case.case_name,
                row.psi_r_rel_rms_error,
            )
    finally:
        benchmark.BENCHMARK_REPEAT_COUNT = original_repeat


def test_pi_psin_uniform_benchmark_ffn_r_is_axis_monotone():
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
            spec_case = benchmark.BenchmarkCaseSpec("PI", "psin", constraint, "uniform")
            row = benchmark._benchmark_case_result(spec_case, reference)
            ffn_head = np.asarray(row.equilibrium.FFn_r[:8], dtype=np.float64)
            assert np.all(np.diff(ffn_head) <= 1.0e-8), (spec_case.case_name, ffn_head.tolist())
            assert float(np.max(ffn_head[:3])) <= 1.0e-3, (spec_case.case_name, ffn_head.tolist())
    finally:
        benchmark.BENCHMARK_REPEAT_COUNT = original_repeat


def test_pi_psin_uniform_benchmark_cases_stay_within_shape_tolerance():
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
            spec_case = benchmark.BenchmarkCaseSpec("PI", "psin", constraint, "uniform")
            row = benchmark._benchmark_case_result(spec_case, reference)
            assert row.shape_error <= benchmark.SHAPE_MATCH_TOL, (spec_case.case_name, row.shape_error)
    finally:
        benchmark.BENCHMARK_REPEAT_COUNT = original_repeat


def test_pp_psin_uniform_benchmark_ffn_r_is_axis_monotone():
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
            spec_case = benchmark.BenchmarkCaseSpec("PP", "psin", constraint, "uniform")
            row = benchmark._benchmark_case_result(spec_case, reference)
            ffn_head = np.asarray(row.equilibrium.FFn_r[:20], dtype=np.float64)
            head_signs = np.sign(np.diff(ffn_head))
            head_signs = head_signs[head_signs != 0.0]
            head_turns = int(np.sum(head_signs[1:] * head_signs[:-1] < 0.0)) if head_signs.size >= 2 else 0
            assert head_turns <= 1, (spec_case.case_name, ffn_head.tolist())
            assert float(np.max(ffn_head[:3])) <= 1.0e-3, (spec_case.case_name, ffn_head.tolist())
    finally:
        benchmark.BENCHMARK_REPEAT_COUNT = original_repeat


def test_pp_psin_uniform_benchmark_ff_psi_stays_bounded():
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
            spec_case = benchmark.BenchmarkCaseSpec("PP", "psin", constraint, "uniform")
            row = benchmark._benchmark_case_result(spec_case, reference)
            assert row.ff_psi_rel_rms_error < 1.2e-1, (spec_case.case_name, row.ff_psi_rel_rms_error)
    finally:
        benchmark.BENCHMARK_REPEAT_COUNT = original_repeat


def test_pq_psin_uniform_benchmark_jpara_is_finite():
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
            jpara_head = np.asarray(row.equilibrium.jpara[:12], dtype=np.float64)
            assert np.all(np.isfinite(jpara_head)), (spec_case.case_name, jpara_head.tolist())
    finally:
        benchmark.BENCHMARK_REPEAT_COUNT = original_repeat


def test_pi_pp_psin_uniform_benchmark_boundary_profiles_are_finite():
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
        for mode in ("PI", "PP"):
            for constraint in ("Ip_beta", "Ip", "beta", "null"):
                spec_case = benchmark.BenchmarkCaseSpec(mode, "psin", constraint, "uniform")
                row = benchmark._benchmark_case_result(spec_case, reference)
                ff_r_tail = np.asarray(row.equilibrium.FF_r[-8:], dtype=np.float64)
                jtor_tail = np.asarray(row.equilibrium.jtor[-8:], dtype=np.float64)
                assert np.all(np.isfinite(ff_r_tail)), (spec_case.case_name, ff_r_tail.tolist())
                assert np.all(np.isfinite(jtor_tail)), (spec_case.case_name, jtor_tail.tolist())
    finally:
        benchmark.BENCHMARK_REPEAT_COUNT = original_repeat


def test_pp_rho_uniform_benchmark_jtor_head_is_finite():
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
            spec_case = benchmark.BenchmarkCaseSpec("PP", "rho", constraint, "uniform")
            row = benchmark._benchmark_case_result(spec_case, reference)
            jtor_head = np.asarray(row.equilibrium.jtor[:8], dtype=np.float64)
            assert np.all(np.isfinite(jtor_head)), (spec_case.case_name, jtor_head.tolist())
    finally:
        benchmark.BENCHMARK_REPEAT_COUNT = original_repeat


def test_pi_psin_uniform_benchmark_jtor_head_is_finite():
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
            spec_case = benchmark.BenchmarkCaseSpec("PI", "psin", constraint, "uniform")
            row = benchmark._benchmark_case_result(spec_case, reference)
            jtor_head = np.asarray(row.equilibrium.jtor[:12], dtype=np.float64)
            assert np.all(np.isfinite(jtor_head)), (spec_case.case_name, jtor_head.tolist())
    finally:
        benchmark.BENCHMARK_REPEAT_COUNT = original_repeat


def test_pj2_psin_uniform_benchmark_ff_psi_snapshot_stays_bounded():
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
        ff_thresholds = {
            "Ip_beta": 7.0e-3,
            "Ip": 1.2e-2,
            "beta": 2.8e-2,
            "null": 8.5e-3,
        }
        for constraint in ("Ip_beta", "Ip", "beta", "null"):
            spec_case = benchmark.BenchmarkCaseSpec("PJ2", "psin", constraint, "uniform")
            row = benchmark._benchmark_case_result(spec_case, reference)
            assert row.ff_psi_rel_rms_error <= ff_thresholds[constraint], (
                spec_case.case_name,
                row.ff_psi_rel_rms_error,
            )
            assert row.shape_error <= 6.0e-3, (spec_case.case_name, row.shape_error)
    finally:
        benchmark.BENCHMARK_REPEAT_COUNT = original_repeat


def test_pi_rho_grid_benchmark_source_profile_errors_stay_low():
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
            spec_case = benchmark.BenchmarkCaseSpec("PI", "rho", constraint, "grid")
            row = benchmark._benchmark_case_result(spec_case, reference)
            assert row.shape_error <= 5.0e-4, (spec_case.case_name, row.shape_error)
            assert row.ff_psi_rel_rms_error <= 2.3e-2, (spec_case.case_name, row.ff_psi_rel_rms_error)
            assert row.mu0_p_psi_rel_rms_error <= 6.0e-4, (spec_case.case_name, row.mu0_p_psi_rel_rms_error)
    finally:
        benchmark.BENCHMARK_REPEAT_COUNT = original_repeat


def test_rho_uniform_benchmark_cases_track_rho_grid_cases_closely():
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
        for mode in benchmark.BENCHMARK_MODES:
            for constraint in benchmark.BENCHMARK_MODE_CONSTRAINTS[mode]:
                uniform_case = benchmark.BenchmarkCaseSpec(mode, "rho", constraint, "uniform")
                grid_case = benchmark.BenchmarkCaseSpec(mode, "rho", constraint, "grid")
                uniform_row = benchmark._benchmark_case_result(uniform_case, reference)
                grid_row = benchmark._benchmark_case_result(grid_case, reference)

                assert abs(uniform_row.shape_error - grid_row.shape_error) <= 1.5e-4, (
                    uniform_case.case_name,
                    uniform_row.shape_error,
                    grid_row.shape_error,
                )
                assert abs(uniform_row.psi_r_rel_rms_error - grid_row.psi_r_rel_rms_error) <= 1.1e-4, (
                    uniform_case.case_name,
                    uniform_row.psi_r_rel_rms_error,
                    grid_row.psi_r_rel_rms_error,
                )
                assert abs(uniform_row.ff_psi_rel_rms_error - grid_row.ff_psi_rel_rms_error) <= 7.0e-4, (
                    uniform_case.case_name,
                    uniform_row.ff_psi_rel_rms_error,
                    grid_row.ff_psi_rel_rms_error,
                )
                assert abs(uniform_row.mu0_p_psi_rel_rms_error - grid_row.mu0_p_psi_rel_rms_error) <= 3.5e-4, (
                    uniform_case.case_name,
                    uniform_row.mu0_p_psi_rel_rms_error,
                    grid_row.mu0_p_psi_rel_rms_error,
                )
    finally:
        benchmark.BENCHMARK_REPEAT_COUNT = original_repeat


def test_psin_grid_benchmark_cases_track_rho_grid_cases_reasonably():
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
        for mode in benchmark.BENCHMARK_MODES:
            for constraint in benchmark.BENCHMARK_MODE_CONSTRAINTS[mode]:
                psin_case = benchmark.BenchmarkCaseSpec(mode, "psin", constraint, "grid")
                rho_case = benchmark.BenchmarkCaseSpec(mode, "rho", constraint, "grid")
                psin_row = benchmark._benchmark_case_result(psin_case, reference)
                rho_row = benchmark._benchmark_case_result(rho_case, reference)

                assert abs(psin_row.shape_error - rho_row.shape_error) <= 2.0e-4, (
                    psin_case.case_name,
                    psin_row.shape_error,
                    rho_row.shape_error,
                )
                assert abs(psin_row.psi_r_rel_rms_error - rho_row.psi_r_rel_rms_error) <= 1.2e-4, (
                    psin_case.case_name,
                    psin_row.psi_r_rel_rms_error,
                    rho_row.psi_r_rel_rms_error,
                )
                assert abs(psin_row.ff_psi_rel_rms_error - rho_row.ff_psi_rel_rms_error) <= 1.2e-2, (
                    psin_case.case_name,
                    psin_row.ff_psi_rel_rms_error,
                    rho_row.ff_psi_rel_rms_error,
                )
                assert abs(psin_row.mu0_p_psi_rel_rms_error - rho_row.mu0_p_psi_rel_rms_error) <= 6.0e-4, (
                    psin_case.case_name,
                    psin_row.mu0_p_psi_rel_rms_error,
                    rho_row.mu0_p_psi_rel_rms_error,
                )
    finally:
        benchmark.BENCHMARK_REPEAT_COUNT = original_repeat


def test_pq_psin_uniform_benchmark_successful_fallback_is_silent():
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
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            row = benchmark._benchmark_case_result(benchmark.BenchmarkCaseSpec("PQ", "psin", "Ip_beta", "uniform"), reference)
        retry_warnings = [
            warning
            for warning in caught
            if issubclass(warning.category, RuntimeWarning) and "Retrying with" in str(warning.message)
        ]
        assert not retry_warnings, [str(warning.message) for warning in retry_warnings]
        assert row.result.success, row.result.message
    finally:
        benchmark.BENCHMARK_REPEAT_COUNT = original_repeat


def test_pq_psin_uniform_ip_beta_benchmark_prefers_dogbox_fallback():
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
        row = benchmark._benchmark_case_result(benchmark.BenchmarkCaseSpec("PQ", "psin", "Ip_beta", "uniform"), reference)
        assert row.result.success, row.result.message
        if "selected method=" in row.result.message:
            assert "selected method=least_squares/dogbox [cold-fallback]" in row.result.message, row.result.message
        assert int(row.result.nfev) < 300, row.result.nfev
        assert row.shape_error <= benchmark.SHAPE_MATCH_TOL, row.shape_error
    finally:
        benchmark.BENCHMARK_REPEAT_COUNT = original_repeat


def test_pj1_pq_routes_benchmark_jpara_tail_is_finite():
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
        target_specs = (
            benchmark.BenchmarkCaseSpec("PJ1", "psin", "Ip_beta", "grid"),
            benchmark.BenchmarkCaseSpec("PJ1", "rho", "Ip_beta", "uniform"),
            benchmark.BenchmarkCaseSpec("PJ1", "rho", "Ip_beta", "grid"),
            benchmark.BenchmarkCaseSpec("PQ", "psin", "Ip_beta", "uniform"),
        )
        for spec_case in target_specs:
            row = benchmark._benchmark_case_result(spec_case, reference)
            jpara_tail = np.asarray(row.equilibrium.jpara[-12:], dtype=np.float64)
            assert np.all(np.isfinite(jpara_tail)), (spec_case.case_name, jpara_tail.tolist())
    finally:
        benchmark.BENCHMARK_REPEAT_COUNT = original_repeat


def test_eq_diagnostics_align_with_direct_formulas():
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
        row = benchmark._benchmark_case_result(benchmark.BenchmarkCaseSpec("PQ", "psin", "Ip_beta", "uniform"), reference)
        equilibrium = row.equilibrium
        grid = equilibrium.grid

        expected_ff_int = grid.integrate(equilibrium.FF_r, p=1)
        expected_f2 = (equilibrium.R0 * equilibrium.B0) ** 2 + 2.0 * (expected_ff_int - expected_ff_int[-1])
        assert np.allclose(equilibrium.F2, expected_f2, atol=1.0e-12, rtol=1.0e-12)

        expected_p = grid.integrate(equilibrium.P_r, p=1)
        expected_p -= expected_p[-1]
        assert np.allclose(equilibrium.P, expected_p, atol=1.0e-12, rtol=1.0e-12)

        expected_q = equilibrium.F * equilibrium.Ln_r / (equilibrium.alpha2 * equilibrium.psin_r)
        assert np.allclose(equilibrium.q, expected_q, atol=1.0e-12, rtol=1.0e-12)

        expected_q_r = grid.corrected_even_derivative(expected_q)
        expected_s = equilibrium.rho * expected_q_r / expected_q
        assert np.allclose(equilibrium.s, expected_s, atol=1.0e-12, rtol=1.0e-12)

        expected_jtor = -equilibrium.alpha1 / (MU0 * equilibrium.S_r) * (
            2.0 * np.pi * equilibrium.FFn_psin * equilibrium.Ln_r + equilibrium.V_r * equilibrium.Pn_psin / (2.0 * np.pi)
        )
        assert np.allclose(equilibrium.jtor, expected_jtor, atol=1.0e-12, rtol=1.0e-12)

        expected_f_r = grid.corrected_even_derivative(equilibrium.F)
        expected_jpara = equilibrium.alpha2 / MU0 * equilibrium.F / equilibrium.Ln_r * (
            equilibrium.Kn_r * equilibrium.psin_r / equilibrium.F
            + equilibrium.Kn * equilibrium.psin_rr / equilibrium.F
            - equilibrium.Kn * equilibrium.psin_r * expected_f_r / equilibrium.F**2
        )
        assert np.allclose(equilibrium.jpara, expected_jpara, atol=1.0e-12, rtol=1.0e-12)

        expected_jphi = -equilibrium.alpha1 / (MU0 * equilibrium.geometry.R) * (
            equilibrium.FFn_psin[:, None] + equilibrium.geometry.R**2 * equilibrium.Pn_psin[:, None]
        )
        assert np.allclose(equilibrium.jphi, expected_jphi, atol=1.0e-12, rtol=1.0e-12)

        resampled = equilibrium.resample(
            target_grid=Grid(
                Nr=33,
                Nt=equilibrium.grid.Nt,
                scheme="uniform",
                L_max=equilibrium.grid.L_max,
                M_max=equilibrium.grid.M_max,
            )
        )
        resampled_psin_rr = resampled.grid.corrected_linear_derivative(resampled.psin_r)
        assert np.allclose(resampled.psin_rr, resampled_psin_rr, atol=1.0e-12, rtol=1.0e-12)
    finally:
        benchmark.BENCHMARK_REPEAT_COUNT = original_repeat


def test_benchmark_report_includes_source_profile_diagnostics():
    benchmark_path = Path(__file__).with_name("benchmark.py")
    spec = importlib.util.spec_from_file_location("veqpy_tests_benchmark", benchmark_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load benchmark module from {benchmark_path}")
    benchmark = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = benchmark
    spec.loader.exec_module(benchmark)

    original_repeat = benchmark.BENCHMARK_REPEAT_COUNT
    original_artifact_dir = benchmark._artifact_dir
    original_plot = benchmark.PLOT
    report_dir = Path(__file__).resolve().parent / ".benchmark-report-test"
    report_dir.mkdir(parents=True, exist_ok=True)
    try:
        benchmark.BENCHMARK_REPEAT_COUNT = 1
        benchmark.PLOT = False
        benchmark._artifact_dir = lambda: report_dir
        reference = benchmark._solve_reference()
        row = benchmark._benchmark_case_result(benchmark.BenchmarkCaseSpec("PQ", "psin", "Ip_beta", "uniform"), reference)
        benchmark._write_report(reference, [row], plot_failures=None)

        report = (report_dir / "benchmark_compare.txt").read_text(encoding="utf-8")
        assert "psi_r / FF_psi / mu0P_psi diagnostics" in report
        assert "psi_r_rms" in report
        assert "FF_psi_rms" in report
        assert "mu0P_rms" in report
        assert "Largest psi_r relative RMS error ranking" in report
        assert "Largest FF_psi relative RMS error ranking" in report
        assert "Largest mu0P_psi relative RMS error ranking" in report
        assert "Most oscillatory psi_r ranking" in report
        assert "Most oscillatory FF_psi ranking" in report
        assert "Most oscillatory mu0P_psi ranking" in report
        assert row.case_name in report
    finally:
        benchmark.BENCHMARK_REPEAT_COUNT = original_repeat
        benchmark._artifact_dir = original_artifact_dir
        benchmark.PLOT = original_plot


@pytest.mark.parametrize(
    ("coordinate", "input_kind"),
    [("rho", "uniform"), ("rho", "grid"), ("psin", "grid")],
)
def test_pj1_nonuniform_routes_benchmark_jpara_is_finite(coordinate, input_kind):
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
            spec_case = benchmark.BenchmarkCaseSpec("PJ1", coordinate, constraint, input_kind)
            row = benchmark._benchmark_case_result(spec_case, reference)
            jpara_head = np.asarray(row.equilibrium.jpara[:12], dtype=np.float64)
            assert np.all(np.isfinite(jpara_head)), (spec_case.case_name, jpara_head.tolist())
    finally:
        benchmark.BENCHMARK_REPEAT_COUNT = original_repeat


@pytest.mark.parametrize("input_kind", ["uniform", "grid"])
def test_pj1_rho_routes_benchmark_ffn_r_is_axis_monotone(input_kind):
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
            spec_case = benchmark.BenchmarkCaseSpec("PJ1", "rho", constraint, input_kind)
            row = benchmark._benchmark_case_result(spec_case, reference)
            ffn_head = np.asarray(row.equilibrium.FFn_r[:12], dtype=np.float64)
            head_signs = np.sign(np.diff(ffn_head))
            head_signs = head_signs[head_signs != 0.0]
            head_turns = int(np.sum(head_signs[1:] * head_signs[:-1] < 0.0)) if head_signs.size >= 2 else 0
            assert head_turns <= 1, (spec_case.case_name, ffn_head.tolist())
            assert float(np.max(ffn_head[:3])) <= 1.0e-3, (spec_case.case_name, ffn_head.tolist())
    finally:
        benchmark.BENCHMARK_REPEAT_COUNT = original_repeat

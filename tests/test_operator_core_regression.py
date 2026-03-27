from pathlib import Path

import numpy as np
import pytest

from veqpy.model import Boundary, Equilibrium
from veqpy.model.grid import Grid
from veqpy.operator import Operator
from veqpy.operator.layout import (
    build_active_profile_metadata,
    build_fourier_profile_names,
    build_profile_index,
    build_profile_layout,
    build_profile_names,
    build_shape_profile_names,
)
from veqpy.operator.operator_case import OperatorCase
from veqpy.solver import Solver, SolverConfig

TEST_SOURCE_SAMPLE_COUNT = 21


def test_operator_case_normalizes_boundary_offset_families():
    default_case = OperatorCase(
        name="PF",
        coordinate="rho",
        nodes="uniform",
        profile_coeffs={},
        boundary=Boundary(a=1.0, R0=1.7, Z0=0.1, B0=3.0),
        heat_input=np.zeros(TEST_SOURCE_SAMPLE_COUNT),
        current_input=np.zeros(TEST_SOURCE_SAMPLE_COUNT),
    )
    normalized_case = OperatorCase(
        name="PF",
        coordinate="rho",
        nodes="uniform",
        profile_coeffs={},
        boundary=Boundary(
            a=1.0,
            R0=1.7,
            Z0=0.1,
            B0=3.0,
            c_offsets=np.array([0.03, 0.14, -0.02]),
            s_offsets=np.array([9.9, -0.05, 0.07, -0.01]),
        ),
        heat_input=np.zeros(TEST_SOURCE_SAMPLE_COUNT),
        current_input=np.zeros(TEST_SOURCE_SAMPLE_COUNT),
    )
    short_case = OperatorCase(
        name="PF",
        coordinate="rho",
        nodes="uniform",
        profile_coeffs={},
        boundary=Boundary(
            a=1.0,
            R0=1.7,
            Z0=0.1,
            B0=3.0,
            c_offsets=np.array([0.03]),
            s_offsets=np.array([-9.0, -0.05]),
        ),
        heat_input=np.zeros(TEST_SOURCE_SAMPLE_COUNT),
        current_input=np.zeros(TEST_SOURCE_SAMPLE_COUNT),
    )

    assert np.allclose(default_case.c_offsets, [0.0])
    assert np.allclose(default_case.s_offsets, [0.0])
    assert np.allclose(normalized_case.c_offsets, [0.03, 0.14, -0.02])
    assert np.allclose(normalized_case.s_offsets, [0.0, -0.05, 0.07, -0.01])
    assert np.allclose(short_case.c_offsets, [0.03])
    assert np.allclose(short_case.s_offsets, [0.0, -0.05])
    assert not hasattr(normalized_case, "c0a")
    assert not hasattr(normalized_case, "s1a")


def test_operator_case_copy_detaches_boundary_offsets():
    case = OperatorCase(
        name="PF",
        coordinate="rho",
        nodes="uniform",
        profile_coeffs={},
        boundary=Boundary(
            a=1.0,
            R0=1.7,
            Z0=0.1,
            B0=3.0,
            c_offsets=np.array([0.03, 0.14, -0.02]),
            s_offsets=np.array([0.0, -0.05, 0.07, -0.01]),
        ),
        heat_input=np.zeros(TEST_SOURCE_SAMPLE_COUNT),
        current_input=np.zeros(TEST_SOURCE_SAMPLE_COUNT),
    )

    copied = case.copy()
    copied.c_offsets[1] = 9.0
    copied.s_offsets[2] = 8.0

    assert case.c_offsets[1] == pytest.approx(0.14)
    assert case.s_offsets[2] == pytest.approx(0.07)


def test_operator_case_defaults_name_and_coordinate_and_copies_them():
    case = OperatorCase(
        name="PF",
        coordinate="rho",
        nodes="uniform",
        profile_coeffs={},
        boundary=Boundary(a=1.0, R0=1.7, Z0=0.1, B0=3.0),
        heat_input=np.zeros(TEST_SOURCE_SAMPLE_COUNT),
        current_input=np.zeros(TEST_SOURCE_SAMPLE_COUNT),
    )

    copied = case.copy()

    assert case.name == "PF"
    assert case.coordinate == "rho"
    assert case.nodes == "uniform"
    assert copied.name == "PF"
    assert copied.coordinate == "rho"
    assert copied.nodes == "uniform"


def test_operator_case_rejects_non_uniform_nodes():
    with pytest.raises(ValueError, match="nodes must be one of"):
        OperatorCase(
            name="PF",
            profile_coeffs={},
            boundary=Boundary(a=1.0, R0=1.7, Z0=0.1, B0=3.0),
            heat_input=np.zeros(8),
            current_input=np.zeros(8),
            coordinate="psin",
            nodes="operator",
        )


def test_operator_caches_source_remap_cache_by_coordinate_and_input_length():
    grid = Grid(Nr=8, Nt=8, scheme="uniform")
    profile_coeffs = {"psin": [0.0, 1.0], "F": [1.0], "h": [0.0], "k": [0.0]}
    rho_case = OperatorCase(
        name="PF",
        profile_coeffs=profile_coeffs,
        boundary=Boundary(a=1.0, R0=1.7, Z0=0.1, B0=3.0),
        heat_input=np.linspace(0.0, 1.0, TEST_SOURCE_SAMPLE_COUNT),
        current_input=np.linspace(1.0, 0.0, TEST_SOURCE_SAMPLE_COUNT),
        coordinate="rho",
        nodes="uniform",
    )
    operator = Operator(grid=grid, case=rho_case)
    rho_matrix = operator.source_fixed_remap_matrix.copy()

    assert operator.source_coordinate == "rho"
    assert operator.source_n_src == TEST_SOURCE_SAMPLE_COUNT
    assert operator.source_fixed_remap_matrix.shape == (grid.Nr, TEST_SOURCE_SAMPLE_COUNT)

    psin_case = OperatorCase(
        name="PF",
        profile_coeffs=profile_coeffs,
        boundary=Boundary(a=1.0, R0=1.7, Z0=0.1, B0=3.0),
        heat_input=np.linspace(0.0, 1.0, TEST_SOURCE_SAMPLE_COUNT),
        current_input=np.linspace(1.0, 0.0, TEST_SOURCE_SAMPLE_COUNT),
        coordinate="psin",
        nodes="uniform",
    )
    operator.replace_case(psin_case)

    assert operator.source_coordinate == "psin"
    assert operator.source_n_src == TEST_SOURCE_SAMPLE_COUNT
    assert operator.source_fixed_remap_matrix.shape == (0, 0)
    assert rho_matrix.shape == (grid.Nr, TEST_SOURCE_SAMPLE_COUNT)


def test_operator_accepts_fixed_external_rho_source_sample_count():
    grid = Grid(Nr=8, Nt=8, scheme="uniform")
    case = OperatorCase(
        name="PF",
        profile_coeffs={"psin": [0.0, 1.0], "F": [1.0], "h": [0.0], "k": [0.0]},
        boundary=Boundary(a=1.0, R0=1.7, Z0=0.1, B0=3.0),
        heat_input=np.linspace(-1.0, 0.0, TEST_SOURCE_SAMPLE_COUNT),
        current_input=np.linspace(0.5, 0.0, TEST_SOURCE_SAMPLE_COUNT),
        coordinate="rho",
        nodes="uniform",
    )

    operator = Operator(grid=grid, case=case)

    assert operator.source_coordinate == "rho"
    assert operator.source_n_src == TEST_SOURCE_SAMPLE_COUNT
    assert operator.source_fixed_remap_matrix.shape == (grid.Nr, TEST_SOURCE_SAMPLE_COUNT)
    assert operator.materialized_heat_input.shape == (grid.Nr,)
    assert operator.materialized_current_input.shape == (grid.Nr,)
    assert np.all(np.isfinite(operator.materialized_heat_input))
    assert np.all(np.isfinite(operator.materialized_current_input))


def test_layout_builders_keep_dynamic_fourier_ordering_stable():
    assert build_fourier_profile_names(4) == ("c0", "c1", "c2", "c3", "c4", "s1", "s2", "s3", "s4")
    assert build_shape_profile_names(3) == ("h", "v", "k", "c0", "c1", "c2", "c3", "s1", "s2", "s3")
    assert build_profile_names(2) == ("psin", "F", "h", "v", "k", "c0", "c1", "c2", "s1", "s2")

    profile_names = build_profile_names(3)
    profile_index = build_profile_index(profile_names)
    profile_coeffs = {
        "psin": [1.0, 2.0],
        "h": [0.1],
        "c0": [0.0],
        "c3": [0.2, 0.3],
        "s2": [0.4],
    }

    profile_L, coeff_index, order_offsets = build_profile_layout(
        profile_coeffs,
        profile_names=profile_names,
    )
    active_mask, active_ids = build_active_profile_metadata(profile_L, profile_names=profile_names)

    assert profile_L[profile_index["psin"]] == 1
    assert profile_L[profile_index["h"]] == 0
    assert profile_L[profile_index["c3"]] == 1
    assert profile_L[profile_index["s2"]] == 0
    assert profile_L[profile_index["c2"]] == -1
    assert order_offsets.shape == (3,)
    assert coeff_index[profile_index["psin"], 0] == 0
    assert coeff_index[profile_index["psin"], 1] == 1
    assert coeff_index[profile_index["h"], 0] == 2
    assert coeff_index[profile_index["c0"], 0] == 3
    assert coeff_index[profile_index["c3"], 0] == 4
    assert coeff_index[profile_index["s2"], 0] == 5
    assert coeff_index[profile_index["c3"], 1] == 6
    assert np.array_equal(active_ids, np.flatnonzero(active_mask))


def test_operator_supports_high_kmax_with_inactive_high_orders():
    profile_coeffs = {
        "psin": [0.0, 1.0],
        "F": [1.0],
        "h": [0.0],
        "v": None,
        "k": [0.0],
        "c0": None,
        "c1": None,
        "c2": None,
        "c3": None,
        "c4": None,
        "s1": [0.0],
        "s2": None,
        "s3": None,
        "s4": None,
    }
    case = OperatorCase(
        name="PF",
        coordinate="rho",
        nodes="uniform",
        profile_coeffs=profile_coeffs,
        boundary=Boundary(
            a=1.0,
            R0=1.7,
            Z0=0.0,
            B0=3.0,
            c_offsets=np.zeros(5),
            s_offsets=np.zeros(5),
        ),
        heat_input=np.zeros(TEST_SOURCE_SAMPLE_COUNT),
        current_input=np.zeros(TEST_SOURCE_SAMPLE_COUNT),
    )

    operator = Operator(grid=Grid(Nr=8, Nt=8, scheme="uniform", K_max=4), case=case)

    assert operator.profile_names[-1] == "s4"
    assert operator.profile_index["c4"] > operator.profile_index["c1"]
    assert "c4" in operator.profiles_by_name
    assert "s4" in operator.profiles_by_name
    assert not hasattr(operator, "c0_profile")
    assert not hasattr(operator, "s1_profile")


def test_operator_effective_orders_follow_active_profiles_and_fixed_offsets():
    inactive_profile_coeffs = {name: None for name in build_profile_names(4)}
    inactive_profile_coeffs.update(
        {
            "psin": [0.0, 1.0],
            "F": [1.0],
            "h": [0.0],
            "k": [0.0],
            "s1": [0.0],
        }
    )
    inactive_case = OperatorCase(
        name="PF",
        coordinate="rho",
        nodes="uniform",
        profile_coeffs=inactive_profile_coeffs,
        boundary=Boundary(
            a=1.0,
            R0=1.7,
            Z0=0.0,
            B0=3.0,
            c_offsets=np.zeros(5),
            s_offsets=np.zeros(5),
        ),
        heat_input=np.zeros(TEST_SOURCE_SAMPLE_COUNT),
        current_input=np.zeros(TEST_SOURCE_SAMPLE_COUNT),
    )
    inactive_operator = Operator(
        grid=Grid(Nr=8, Nt=8, scheme="uniform", K_max=4),
        case=inactive_case,
    )

    fixed_offset_profile_coeffs = {name: None for name in build_profile_names(4)}
    fixed_offset_profile_coeffs.update(
        {
            "psin": [0.0, 1.0],
            "F": [1.0],
            "h": [0.0],
        }
    )
    fixed_offset_case = OperatorCase(
        name="PF",
        coordinate="rho",
        nodes="uniform",
        profile_coeffs=fixed_offset_profile_coeffs,
        boundary=Boundary(
            a=1.0,
            R0=1.7,
            Z0=0.0,
            B0=3.0,
            c_offsets=np.array([0.0, 0.0, 0.0, 0.05, 0.0]),
            s_offsets=np.array([0.0, 0.0, 0.0, 0.0, -0.04]),
        ),
        heat_input=np.zeros(TEST_SOURCE_SAMPLE_COUNT),
        current_input=np.zeros(TEST_SOURCE_SAMPLE_COUNT),
    )
    fixed_offset_operator = Operator(
        grid=Grid(Nr=8, Nt=8, scheme="uniform", K_max=4),
        case=fixed_offset_case,
    )

    assert inactive_operator.c_effective_order == 0
    assert inactive_operator.s_effective_order == 1
    assert fixed_offset_operator.c_effective_order == 3
    assert fixed_offset_operator.s_effective_order == 4


def test_demo_pf_uniform_rho_inputs_match_cached_reference():
    ref_path = Path(__file__).resolve().parent / "demo" / "demo-1-ref.json"
    if not ref_path.exists():
        ref_path = Path(__file__).resolve().parent / "demo" / "demo-1.json"
    reference = Equilibrium.load(str(ref_path))
    grid = Grid(Nr=12, Nt=12, scheme="legendre")
    rho_uniform = np.linspace(0.0, 1.0, TEST_SOURCE_SAMPLE_COUNT)

    beta0 = 0.75
    psin = rho_uniform * rho_uniform
    psin_r = 2.0 * rho_uniform
    alpha_p, alpha_f = 5.0, 3.32
    exp_ap, exp_af = np.exp(alpha_p), np.exp(alpha_f)
    den_p = 1.0 + exp_ap * (alpha_p - 1.0)
    den_f = 1.0 + exp_af * (alpha_f - 1.0)
    current_input = (1.0 - beta0) * alpha_f * (np.exp(alpha_f * psin) - exp_af) / den_f * psin_r
    heat_input = beta0 * alpha_p * (np.exp(alpha_p * psin) - exp_ap) / den_p * psin_r

    case = OperatorCase(
        name="PF",
        profile_coeffs={
            "h": [0.0] * 3,
            "k": [0.0] * 3,
            "s1": [0.0] * 3,
        },
        boundary=Boundary(
            a=1.05 / 1.85,
            R0=1.05,
            Z0=0.0,
            B0=3.0,
            ka=2.2,
            s_offsets=np.array([0.0, float(np.arcsin(0.5))]),
        ),
        heat_input=heat_input,
        current_input=current_input,
        coordinate="rho",
        nodes="uniform",
        Ip=3.0e6,
    )
    operator = Operator(grid=grid, case=case)
    solver = Solver(
        operator=operator,
        config=SolverConfig(method="hybr", enable_verbose=False, enable_warmstart=False, enable_history=False),
    )

    solver.solve(enable_verbose=False, enable_history=False)
    equilibrium = solver.build_equilibrium()

    assert np.allclose(equilibrium.alpha1, reference.alpha1, rtol=1e-2, atol=1e-4)
    assert np.allclose(equilibrium.alpha2, reference.alpha2, rtol=1e-2, atol=1e-4)
    assert np.allclose(equilibrium.psin_r, reference.psin_r, rtol=2e-2, atol=2e-3)
    assert np.allclose(equilibrium.psin_rr, reference.psin_rr, rtol=3e-2, atol=5e-3)
    assert np.allclose(equilibrium.FFn_r, reference.FFn_r, rtol=3e-2, atol=5e-3)
    assert np.allclose(equilibrium.Pn_r, reference.Pn_r, rtol=3e-2, atol=5e-3)

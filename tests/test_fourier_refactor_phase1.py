import numpy as np
import pytest

from veqpy.model import Boundary
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


def test_grid_builds_fourier_tables_and_rho_powers():
    grid = Grid(Nr=8, Nt=16, scheme="uniform", K_max=5)

    assert grid.cos_ktheta.shape == (6, 16)
    assert grid.sin_ktheta.shape == (6, 16)
    assert grid.k_cos_ktheta.shape == (6, 16)
    assert grid.k_sin_ktheta.shape == (6, 16)
    assert grid.k2_cos_ktheta.shape == (6, 16)
    assert grid.k2_sin_ktheta.shape == (6, 16)
    assert grid.rho_powers.shape == (7, 8)

    theta = grid.theta
    assert np.allclose(grid.cos_ktheta[3], np.cos(3.0 * theta))
    assert np.allclose(grid.sin_ktheta[4], np.sin(4.0 * theta))
    assert np.allclose(grid.k_cos_ktheta[5], 5.0 * np.cos(5.0 * theta))
    assert np.allclose(grid.k_sin_ktheta[2], 2.0 * np.sin(2.0 * theta))
    assert np.allclose(grid.k2_cos_ktheta[3], 9.0 * np.cos(3.0 * theta))
    assert np.allclose(grid.k2_sin_ktheta[4], 16.0 * np.sin(4.0 * theta))
    assert np.allclose(grid.rho_powers[0], 1.0)
    assert np.allclose(grid.rho_powers[1], grid.rho)
    assert np.allclose(grid.rho_powers[2], grid.rho**2)
    assert np.allclose(grid.rho_powers[6], grid.rho**6)
    assert not hasattr(grid, "rho2")
    assert not hasattr(grid, "cos_theta")
    assert not hasattr(grid, "sin_theta")
    assert not hasattr(grid, "cos_2theta")
    assert not hasattr(grid, "sin_2theta")


def test_grid_requires_kmax_at_least_two_during_transition():
    with pytest.raises(ValueError, match="K_max"):
        Grid(Nr=8, Nt=16, scheme="uniform", K_max=1)


def test_operator_case_defaults_missing_offsets_to_zero_family_arrays():
    case = OperatorCase(
        profile_coeffs={},
        boundary=Boundary(a=1.0, R0=1.7, Z0=0.1, B0=3.0),
        heat_input=np.zeros(4),
        current_input=np.zeros(4),
    )

    assert np.allclose(case.c_offsets, [0.0])
    assert np.allclose(case.s_offsets, [0.0])


def test_operator_case_normalizes_family_arrays_and_forces_s0_zero():
    case = OperatorCase(
        profile_coeffs={},
        boundary=Boundary(
            a=1.0,
            R0=1.7,
            Z0=0.1,
            B0=3.0,
            c_offsets=np.array([0.03, 0.14, -0.02]),
            s_offsets=np.array([9.9, -0.05, 0.07, -0.01]),
        ),
        heat_input=np.zeros(4),
        current_input=np.zeros(4),
    )

    assert np.allclose(case.c_offsets, [0.03, 0.14, -0.02])
    assert np.allclose(case.s_offsets, [0.0, -0.05, 0.07, -0.01])
    assert not hasattr(case, "c0a")
    assert not hasattr(case, "c1a")
    assert not hasattr(case, "s1a")
    assert not hasattr(case, "s2a")


def test_boundary_keeps_short_family_arrays_without_padding():
    case = OperatorCase(
        profile_coeffs={},
        boundary=Boundary(
            a=1.0,
            R0=1.7,
            Z0=0.1,
            B0=3.0,
            c_offsets=np.array([0.03]),
            s_offsets=np.array([-9.0, -0.05]),
        ),
        heat_input=np.zeros(4),
        current_input=np.zeros(4),
    )

    assert np.allclose(case.c_offsets, [0.03])
    assert np.allclose(case.s_offsets, [0.0, -0.05])


def test_operator_case_copy_keeps_family_arrays_independent():
    case = OperatorCase(
        profile_coeffs={},
        boundary=Boundary(
            a=1.0,
            R0=1.7,
            Z0=0.1,
            B0=3.0,
            c_offsets=np.array([0.03, 0.14, -0.02]),
            s_offsets=np.array([0.0, -0.05, 0.07, -0.01]),
        ),
        heat_input=np.zeros(4),
        current_input=np.zeros(4),
    )

    copied = case.copy()
    copied.c_offsets[1] = 9.0
    copied.s_offsets[2] = 8.0

    assert case.c_offsets[1] == pytest.approx(0.14)
    assert case.s_offsets[2] == pytest.approx(0.07)


def test_layout_builders_expand_fourier_names_from_kmax():
    assert build_fourier_profile_names(4) == ("c0", "c1", "c2", "c3", "c4", "s1", "s2", "s3", "s4")
    assert build_shape_profile_names(3) == ("h", "v", "k", "c0", "c1", "c2", "c3", "s1", "s2", "s3")
    assert build_profile_names(2) == ("psin", "F", "h", "v", "k", "c0", "c1", "c2", "s1", "s2")


def test_layout_accepts_dynamic_profile_names_and_keeps_ordering_stable():
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


def test_operator_accepts_kmax_larger_than_legacy_when_only_low_order_profiles_are_active():
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
        profile_coeffs=profile_coeffs,
        boundary=Boundary(
            a=1.0,
            R0=1.7,
            Z0=0.0,
            B0=3.0,
            c_offsets=np.zeros(5),
            s_offsets=np.zeros(5),
        ),
        heat_input=np.zeros(8),
        current_input=np.zeros(8),
    )
    grid = Grid(Nr=8, Nt=8, scheme="uniform", K_max=4)

    operator = Operator(name="PF", derivative="rho", grid=grid, case=case)

    assert operator.profile_names[-1] == "s4"
    assert operator.profile_index["c4"] > operator.profile_index["c1"]
    assert "c4" in operator.profiles_by_name
    assert "s4" in operator.profiles_by_name
    assert not hasattr(operator, "c0_profile")
    assert not hasattr(operator, "s1_profile")


def test_operator_trims_effective_fourier_order_when_high_orders_are_zero_and_inactive():
    profile_coeffs = {name: None for name in build_profile_names(4)}
    profile_coeffs.update(
        {
            "psin": [0.0, 1.0],
            "F": [1.0],
            "h": [0.0],
            "k": [0.0],
            "s1": [0.0],
        }
    )
    case = OperatorCase(
        profile_coeffs=profile_coeffs,
        boundary=Boundary(
            a=1.0,
            R0=1.7,
            Z0=0.0,
            B0=3.0,
            c_offsets=np.zeros(5),
            s_offsets=np.zeros(5),
        ),
        heat_input=np.zeros(8),
        current_input=np.zeros(8),
    )
    operator = Operator(name="PF", derivative="rho", grid=Grid(Nr=8, Nt=8, scheme="uniform", K_max=4), case=case)

    assert operator.c_effective_order == 0
    assert operator.s_effective_order == 1


def test_operator_keeps_fixed_nonzero_high_order_offsets_in_effective_order():
    profile_coeffs = {name: None for name in build_profile_names(4)}
    profile_coeffs.update(
        {
            "psin": [0.0, 1.0],
            "F": [1.0],
            "h": [0.0],
        }
    )
    case = OperatorCase(
        profile_coeffs=profile_coeffs,
        boundary=Boundary(
            a=1.0,
            R0=1.7,
            Z0=0.0,
            B0=3.0,
            c_offsets=np.array([0.0, 0.0, 0.0, 0.05, 0.0]),
            s_offsets=np.array([0.0, 0.0, 0.0, 0.0, -0.04]),
        ),
        heat_input=np.zeros(8),
        current_input=np.zeros(8),
    )
    operator = Operator(name="PF", derivative="rho", grid=Grid(Nr=8, Nt=8, scheme="uniform", K_max=4), case=case)

    assert operator.c_effective_order == 3
    assert operator.s_effective_order == 4

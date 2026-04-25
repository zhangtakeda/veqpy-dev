import numpy as np
import pytest

from veqpy.model.boundary import Boundary
from veqpy.model.grid import Grid
from veqpy.operator.layout import build_profile_names
from veqpy.operator.operator import Operator
from veqpy.operator.operator_case import OperatorCase
from veqpy.orchestration import resolve_fourier_power


def _build_high_order_case() -> tuple[Grid, OperatorCase]:
    grid = Grid(Nr=8, Nt=16, scheme="uniform", M_max=4)
    profile_coeffs = {name: None for name in build_profile_names(grid.M_max)}
    profile_coeffs.update(
        {
            "psin": [0.0, 1.0],
            "F": [1.0],
            "h": [0.0],
            "c3": [0.05],
            "s4": [-0.03],
        }
    )
    return grid, OperatorCase(
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
        heat_input=np.zeros(21),
        current_input=np.zeros(21),
    )


def test_resolve_fourier_power_handles_strong_and_capped_rules():
    assert resolve_fourier_power(0, None) == 0
    assert resolve_fourier_power(1, None) == 1
    assert resolve_fourier_power(3, None) == 3
    assert resolve_fourier_power(3, 2) == 2
    assert resolve_fourier_power(4, 2) == 2
    assert resolve_fourier_power(5, 4) == 4


@pytest.mark.parametrize("invalid_K_max", [0, -1])
def test_resolve_fourier_power_rejects_invalid_K_max(invalid_K_max):
    with pytest.raises(ValueError, match="K_max"):
        resolve_fourier_power(3, invalid_K_max)


def test_operator_maps_fourier_profile_powers_from_instance_K_max():
    grid, case = _build_high_order_case()

    strong = Operator(grid=grid, case=case)
    capped = Operator(grid=grid, case=case, K_max=2)

    assert strong.profiles_by_name["c1"].power == 1
    assert strong.profiles_by_name["c2"].power == 2
    assert strong.profiles_by_name["c3"].power == 3
    assert strong.profiles_by_name["s4"].power == 4

    assert capped.profiles_by_name["c1"].power == 1
    assert capped.profiles_by_name["c2"].power == 2
    assert capped.profiles_by_name["c3"].power == 2
    assert capped.profiles_by_name["s4"].power == 2


def test_operator_K_max_is_instance_stable_across_replace_case():
    grid, case = _build_high_order_case()
    operator = Operator(grid=grid, case=case)

    operator.replace_case(case.copy())

    assert operator.K_max is None
    assert operator.profiles_by_name["c3"].power == 3
    assert operator.profiles_by_name["s4"].power == 4
    assert Operator(grid=grid, case=case, K_max=2).profiles_by_name["c3"].power == 2


def test_operator_rejects_invalid_instance_K_max():
    grid, case = _build_high_order_case()

    with pytest.raises(ValueError, match="K_max"):
        Operator(grid=grid, case=case, K_max=0)


def test_capped_K_max_changes_materialized_profile_and_geometry_for_high_order_modes():
    grid, case = _build_high_order_case()
    strong = Operator(grid=grid, case=case)
    capped = Operator(grid=grid, case=case, K_max=2)
    x = strong.encode_initial_state()

    strong.stage_a_profile(x)
    strong.stage_b_geometry()
    capped.stage_a_profile(x)
    capped.stage_b_geometry()

    assert not np.allclose(strong.profiles_by_name["c3"].u, capped.profiles_by_name["c3"].u)
    assert not np.allclose(strong.profiles_by_name["s4"].u, capped.profiles_by_name["s4"].u)
    assert not np.allclose(strong.geometry_surface_workspace[1], capped.geometry_surface_workspace[1])


def test_capped_K_max_splits_residual_harmonic_order_from_radial_power():
    grid, case = _build_high_order_case()
    operator = Operator(grid=grid, case=case, K_max=2)
    layout = operator.residual_binding_layout
    active_names = layout.active_profile_names

    c3_slot = active_names.index("c3")
    s4_slot = active_names.index("s4")

    assert layout.active_residual_block_orders[c3_slot] == 3
    assert layout.active_residual_block_orders[s4_slot] == 4
    assert layout.active_residual_block_radial_powers[c3_slot] == 2
    assert layout.active_residual_block_radial_powers[s4_slot] == 2

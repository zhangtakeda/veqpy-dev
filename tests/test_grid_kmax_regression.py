import numpy as np
import pytest

from veqpy.model import Boundary, Grid
from veqpy.operator import Operator, OperatorCase, build_profile_names


def _build_case(grid: Grid) -> OperatorCase:
    profile_coeffs = {name: None for name in build_profile_names(grid.M_max)}
    profile_coeffs.update(
        {
            "psin": [0.0, 1.0],
            "F": [1.0],
            "h": [0.0],
            "c4": [0.05],
            "s5": [-0.03],
        }
    )
    return OperatorCase(
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


def test_grid_owns_fourier_radial_power_cap() -> None:
    grid = Grid(Nr=8, Nt=16, scheme="uniform", M_max=5, K_max=2)

    assert grid.K_max == 2
    np.testing.assert_array_equal(grid.K_values, np.array([0, 1, 2, 2, 2, 2]))
    assert grid.rho_powers.shape == (4, 8)
    np.testing.assert_allclose(grid.rho_powers[3], grid.rho**3)
    assert grid.resolve_fourier_power(5) == 2
    assert grid.resolve_fourier_power(7) == 2


def test_grid_kmax_drives_operator_profile_and_residual_metadata() -> None:
    grid = Grid(Nr=8, Nt=16, scheme="uniform", M_max=5, K_max=2)
    operator = Operator(grid=grid, case=_build_case(grid))

    assert not hasattr(operator, "K_max")
    assert operator.profiles_by_name["c4"].power == 2
    assert operator.profiles_by_name["s5"].power == 2

    powers_by_name = dict(
        zip(
            operator.residual_binding_layout.active_profile_names,
            operator.residual_binding_layout.active_residual_block_radial_powers,
            strict=True,
        )
    )
    assert powers_by_name["c4"] == 2
    assert powers_by_name["s5"] == 2


def test_grid_rejects_invalid_kmax() -> None:
    with pytest.raises(ValueError, match="K_max"):
        Grid(Nr=8, Nt=16, scheme="uniform", M_max=5, K_max=0)

import numpy as np

from veqpy.engine.numba_geometry import update_geometry as numba_update_geometry
from veqpy.engine.numpy_geometry import update_geometry as numpy_update_geometry
from veqpy.model import Boundary, Geometry, Grid
from veqpy.operator import Operator
from veqpy.operator.layout import build_profile_names
from veqpy.operator.operator_case import OperatorCase


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
    c_fields = np.zeros((grid.K_max + 1, 3, grid.Nr), dtype=np.float64)
    s_fields = np.zeros((grid.K_max + 1, 3, grid.Nr), dtype=np.float64)
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


def test_geometry_update_uses_high_order_fourier_family_terms():
    grid = Grid(Nr=8, Nt=16, scheme="uniform", K_max=4)
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


def test_numpy_and_numba_geometry_match_with_high_order_fourier_terms():
    grid = Grid(Nr=8, Nt=16, scheme="uniform", K_max=4)
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
        grid.K_max,
        grid.K_max,
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
        grid.K_max,
        grid.K_max,
    )

    for numpy_arr, numba_arr in zip(numpy_outputs, numba_outputs, strict=True):
        assert np.allclose(numpy_arr, numba_arr, atol=1e-11, rtol=1e-11)


def test_operator_stage_b_geometry_propagates_high_order_offset_profiles():
    grid = Grid(Nr=8, Nt=16, scheme="uniform", K_max=4)
    profile_coeffs = {name: None for name in build_profile_names(grid.K_max)}
    profile_coeffs["psin"] = [0.0]
    case = OperatorCase(
        profile_coeffs=profile_coeffs,
        boundary=Boundary(
            a=1.1,
            R0=1.7,
            Z0=0.2,
            B0=3.0,
            c_offsets=np.array([0.03, 0.0, -0.02, 0.05, 0.0]),
            s_offsets=np.array([0.0, 0.01, 0.0, 0.0, -0.04]),
        ),
        heat_input=np.zeros(grid.Nr),
        current_input=np.zeros(grid.Nr),
    )
    operator = Operator(name="PF", derivative="rho", grid=grid, case=case)

    operator.stage_b_geometry()

    expected_tb, _, _ = _expected_tb(grid, operator.c_family_fields, operator.s_family_fields)

    assert np.max(np.abs(operator.c_family_fields[3, 0])) > 0.0
    assert np.max(np.abs(operator.s_family_fields[4, 0])) > 0.0
    assert np.allclose(operator.geometry.tb, expected_tb)

from pathlib import Path

import numpy as np
import pytest

from veqpy.engine.numba_residual import bind_residual_runner as bind_numba_residual_runner
from veqpy.engine.numpy_residual import bind_residual_runner as bind_numpy_residual_runner
from veqpy.model.grid import Grid
from veqpy.operator import Operator
from veqpy.operator.layout import build_profile_names
from veqpy.operator.operator_case import OperatorCase


def test_numpy_and_numba_residual_runners_match_for_high_order_fourier_blocks():
    grid = Grid(Nr=8, Nt=16, scheme="uniform", K_max=4)
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


def test_operator_runs_with_active_high_order_fourier_profiles():
    grid = Grid(Nr=8, Nt=16, scheme="uniform", K_max=4)
    coeffs_by_name = {name: None for name in build_profile_names(grid.K_max)}
    coeffs_by_name.update(
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
        coeffs_by_name=coeffs_by_name,
        a=1.1,
        R0=1.7,
        Z0=0.2,
        B0=3.0,
        heat_input=np.zeros(grid.Nr),
        current_input=np.zeros(grid.Nr),
        c_offsets=np.zeros(grid.K_max + 1),
        s_offsets=np.zeros(grid.K_max + 1),
    )
    operator = Operator(name="PF", derivative="rho", grid=grid, case=case)
    rng = np.random.default_rng(1)
    operator.G[:] = rng.normal(size=operator.G.shape)
    operator.psin_R[:] = rng.normal(size=operator.psin_R.shape)
    operator.psin_Z[:] = rng.normal(size=operator.psin_Z.shape)
    operator.geometry.tb_fields[7] = rng.normal(size=operator.geometry.sin_tb.shape)

    residual = operator._assemble_residual()

    assert residual.shape == (operator.x_size,)
    assert np.all(np.isfinite(residual))


def test_build_equilibrium_preserves_high_order_shape_profiles():
    grid = Grid(Nr=8, Nt=16, scheme="uniform", K_max=4)
    coeffs_by_name = {name: None for name in build_profile_names(grid.K_max)}
    coeffs_by_name.update(
        {
            "psin": [0.0, 1.0],
            "F": [1.0],
            "h": [0.0],
            "c3": [0.05],
            "s4": [-0.03],
        }
    )
    case = OperatorCase(
        coeffs_by_name=coeffs_by_name,
        a=1.1,
        R0=1.7,
        Z0=0.2,
        B0=3.0,
        heat_input=np.zeros(grid.Nr),
        current_input=np.zeros(grid.Nr),
        c_offsets=np.zeros(grid.K_max + 1),
        s_offsets=np.zeros(grid.K_max + 1),
    )
    operator = Operator(name="PF", derivative="rho", grid=grid, case=case)
    equilibrium = operator.build_equilibrium(operator.encode_initial_state())

    assert "c3" in equilibrium.shape_profile_names
    assert "s4" in equilibrium.shape_profile_names
    assert equilibrium.profiles_by_name["c3"] is equilibrium.c3_profile
    assert equilibrium.profiles_by_name["s4"] is equilibrium.s4_profile
    assert equilibrium.geometry.tb.shape == (grid.Nr, grid.Nt)

    resampled = equilibrium.resample(target_grid=Grid(Nr=10, Nt=12, scheme="uniform", K_max=4))
    assert "c3" in resampled.shape_profile_names
    assert "s4" in resampled.shape_profile_names

    outpath = Path("E:/Dev/veqpy-dev/tests/.tmp-equilibrium-k4.json")
    try:
        equilibrium.write(str(outpath))
        loaded = type(equilibrium).load(str(outpath))
        assert loaded.shape_profile_names == equilibrium.shape_profile_names
        assert np.allclose(loaded.profiles_by_name["c3"].u, equilibrium.profiles_by_name["c3"].u)
        assert np.allclose(loaded.profiles_by_name["s4"].u, equilibrium.profiles_by_name["s4"].u)
    finally:
        outpath.unlink(missing_ok=True)

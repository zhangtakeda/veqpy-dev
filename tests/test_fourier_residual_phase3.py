from pathlib import Path

import numpy as np
import orjson
import pytest

from veqpy.engine.numba_residual import bind_residual_runner as bind_numba_residual_runner
from veqpy.engine.numpy_residual import bind_residual_runner as bind_numpy_residual_runner
from veqpy.model import Boundary
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
    profile_coeffs = {name: None for name in build_profile_names(grid.K_max)}
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
        profile_coeffs=profile_coeffs,
        boundary=Boundary(
            a=1.1,
            R0=1.7,
            Z0=0.2,
            B0=3.0,
            c_offsets=np.zeros(grid.K_max + 1),
            s_offsets=np.zeros(grid.K_max + 1),
        ),
        heat_input=np.zeros(grid.Nr),
        current_input=np.zeros(grid.Nr),
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
    profile_coeffs = {name: None for name in build_profile_names(grid.K_max)}
    profile_coeffs.update(
        {
            "psin": [0.0, 1.0],
            "F": [1.0],
            "h": [0.0],
            "c3": [0.05],
            "s4": [-0.03],
        }
    )
    case = OperatorCase(
        profile_coeffs=profile_coeffs,
        boundary=Boundary(
            a=1.1,
            R0=1.7,
            Z0=0.2,
            B0=3.0,
            c_offsets=np.zeros(grid.K_max + 1),
            s_offsets=np.zeros(grid.K_max + 1),
        ),
        heat_input=np.zeros(grid.Nr),
        current_input=np.zeros(grid.Nr),
    )
    operator = Operator(name="PF", derivative="rho", grid=grid, case=case)
    equilibrium = operator.build_equilibrium(operator.encode_initial_state())

    assert set(equilibrium.active_profiles) == {"k", "c3", "s4"}
    assert equilibrium.active_profiles["c3"] is equilibrium.c3_profile
    assert equilibrium.active_profiles["s4"] is equilibrium.s4_profile
    assert equilibrium.geometry.tb.shape == (grid.Nr, grid.Nt)

    resampled = equilibrium.resample(target_grid=Grid(Nr=10, Nt=12, scheme="uniform", K_max=4))
    assert set(resampled.active_profiles) == {"k", "c3", "s4"}

    outpath = Path("E:/Dev/veqpy-dev/tests/.tmp-equilibrium-k4.json")
    try:
        equilibrium.write(str(outpath))
        payload = orjson.loads(outpath.read_bytes())["Equilibrium"]
        assert set(payload["active_profiles"]) == {"k", "c3", "s4"}
        loaded = type(equilibrium).load(str(outpath))
        assert set(loaded.active_profiles) == set(equilibrium.active_profiles)
        assert np.allclose(loaded.active_profiles["c3"].u, equilibrium.active_profiles["c3"].u)
        assert np.allclose(loaded.active_profiles["s4"].u, equilibrium.active_profiles["s4"].u)
    finally:
        outpath.unlink(missing_ok=True)


def test_equilibrium_load_rejects_legacy_full_shape_profile_payload():
    grid = Grid(Nr=8, Nt=16, scheme="uniform", K_max=4)
    profile_coeffs = {name: None for name in build_profile_names(grid.K_max)}
    profile_coeffs.update(
        {
            "psin": [0.0, 1.0],
            "F": [1.0],
            "h": [0.0],
            "c3": [0.05],
            "s4": [-0.03],
        }
    )
    case = OperatorCase(
        profile_coeffs=profile_coeffs,
        boundary=Boundary(
            a=1.1,
            R0=1.7,
            Z0=0.2,
            B0=3.0,
            c_offsets=np.zeros(grid.K_max + 1),
            s_offsets=np.zeros(grid.K_max + 1),
        ),
        heat_input=np.zeros(grid.Nr),
        current_input=np.zeros(grid.Nr),
    )
    operator = Operator(name="PF", derivative="rho", grid=grid, case=case)
    equilibrium = operator.build_equilibrium(operator.encode_initial_state())

    legacy_payload = {
        "Equilibrium": {
            "R0": equilibrium.R0,
            "Z0": equilibrium.Z0,
            "B0": equilibrium.B0,
            "a": equilibrium.a,
            "grid": {"Grid": {"Nr": grid.Nr, "Nt": grid.Nt, "scheme": grid.scheme, "L_max": grid.L_max, "K_max": grid.K_max}},
            "active_profiles": ["h", "k", "c3", "s4"],
            "shape_profile_names": list(operator.shape_profile_names),
            "shape_profiles": [],
            "FFn_r": equilibrium.FFn_r.tolist(),
            "Pn_r": equilibrium.Pn_r.tolist(),
            "psin_r": equilibrium.psin_r.tolist(),
            "psin_rr": equilibrium.psin_rr.tolist(),
            "alpha1": equilibrium.alpha1,
            "alpha2": equilibrium.alpha2,
        }
    }
    outpath = Path("E:/Dev/veqpy-dev/tests/.tmp-equilibrium-legacy.json")
    try:
        legacy_payload["Equilibrium"]["shape_profiles"] = [
            {"Profile": {"scale": profile.scale, "power": profile.power, "envelope_power": profile.envelope_power, "offset": profile.offset, "coeff": None if profile.coeff is None else profile.coeff.tolist()}}
            for name, profile in ((name, operator.profiles_by_name[name]) for name in operator.shape_profile_names)
        ]
        outpath.write_bytes(orjson.dumps(legacy_payload, option=orjson.OPT_SERIALIZE_NUMPY))
        with pytest.raises(TypeError):
            type(equilibrium).load(str(outpath))
    finally:
        outpath.unlink(missing_ok=True)


def test_equilibrium_compare_reports_only_h_v_k_shape_errors():
    grid = Grid(Nr=8, Nt=16, scheme="uniform", K_max=4)
    profile_coeffs = {name: None for name in build_profile_names(grid.K_max)}
    profile_coeffs.update(
        {
            "psin": [0.0, 1.0],
            "F": [1.0],
            "h": [0.0],
            "v": [0.0],
            "k": [0.0],
            "s1": [0.02],
        }
    )
    case_ref = OperatorCase(
        profile_coeffs=profile_coeffs,
        boundary=Boundary(
            a=1.1,
            R0=1.7,
            Z0=0.2,
            B0=3.0,
            c_offsets=np.zeros(grid.K_max + 1),
            s_offsets=np.zeros(grid.K_max + 1),
        ),
        heat_input=np.zeros(grid.Nr),
        current_input=np.zeros(grid.Nr),
    )
    case_cur = case_ref.copy()
    case_cur.profile_coeffs["v"] = [0.03]
    case_cur.profile_coeffs["s1"] = [0.08]

    ref_operator = Operator(name="PF", derivative="rho", grid=grid, case=case_ref)
    ref_eq = ref_operator.build_equilibrium(ref_operator.encode_initial_state())
    cur_operator = Operator(name="PF", derivative="rho", grid=grid, case=case_cur)
    cur_eq = cur_operator.build_equilibrium(cur_operator.encode_initial_state())

    errors = ref_eq.compare(cur_eq)

    assert "rel_v_max" in errors
    assert "rel_k_max" in errors
    assert "rel_h_max" not in errors
    assert "rel_s1_max" not in errors

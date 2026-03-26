from pathlib import Path

import numpy as np
import orjson
import pytest

from veqpy.model import Boundary
from veqpy.model.equilibrium import Equilibrium
from veqpy.model.geqdsk import Geqdsk
from veqpy.model.grid import Grid
from veqpy.operator import Operator
from veqpy.operator.layout import build_profile_names
from veqpy.operator.operator_case import OperatorCase

GEQDSK_PATH = Path("tests/fitting/geqdsk.txt")


def _build_boundary(*, R0, Z0, a, ka, c_offsets, s_offsets, n=721):
    theta = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    theta_bar = theta + c_offsets[0]
    for order in range(1, len(c_offsets)):
        theta_bar += c_offsets[order] * np.cos(order * theta)
    for order in range(1, len(s_offsets)):
        theta_bar += s_offsets[order] * np.sin(order * theta)
    R = R0 + a * np.cos(theta_bar)
    Z = Z0 - a * ka * np.sin(theta)
    return np.column_stack((R, Z))


def _max_bidirectional_distance(points_a, points_b):
    diff = points_a[:, None, :] - points_b[None, :, :]
    distances = np.sqrt(np.sum(diff * diff, axis=2))
    return max(distances.min(axis=1).max(), distances.min(axis=0).max())


def _build_high_order_equilibrium() -> tuple[Equilibrium, Operator]:
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
    return equilibrium, operator


def _make_geqdsk_with_boundary() -> Geqdsk:
    geqdsk = Geqdsk()
    geqdsk.boundary = _build_boundary(
        R0=1.72,
        Z0=-0.08,
        a=0.43,
        ka=1.68,
        c_offsets=np.array([0.01, 0.08]),
        s_offsets=np.array([0.0, -0.01, 0.01]),
    )
    return geqdsk


def test_grid_precomputes_fourier_tables_and_rho_powers():
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


def test_boundary_from_geqdsk_recovers_synthetic_boundary():
    truth = {
        "R0": 1.72,
        "Z0": -0.08,
        "a": 0.43,
        "ka": 1.68,
        "c_offsets": np.array([0.01, 0.08]),
        "s_offsets": np.array([0.0, -0.01, 0.01]),
    }

    geqdsk = Geqdsk()
    geqdsk.boundary = _build_boundary(**truth)
    geqdsk.R0 = truth["R0"]
    geqdsk.Z0 = truth["Z0"]

    boundary = Boundary.from_geqdsk(geqdsk, M=1, N=2)
    fitted_boundary = _build_boundary(
        R0=boundary.R0,
        Z0=boundary.Z0,
        a=boundary.a,
        ka=boundary.ka,
        c_offsets=boundary.c_offsets,
        s_offsets=boundary.s_offsets,
    )

    assert boundary.R0 == pytest.approx(truth["R0"])
    assert boundary.Z0 == pytest.approx(truth["Z0"])
    assert boundary.a == pytest.approx(truth["a"])
    assert boundary.ka == pytest.approx(truth["ka"], abs=1.0e-2)
    assert np.sign(boundary.c_offsets[1]) == np.sign(truth["c_offsets"][1])
    assert np.sign(boundary.s_offsets[1]) == np.sign(truth["s_offsets"][1])
    assert np.sign(boundary.s_offsets[2]) == np.sign(truth["s_offsets"][2])
    assert _max_bidirectional_distance(geqdsk.boundary, fitted_boundary) < 1.5e-2


def test_boundary_from_geqdsk_auto_selects_minimal_orders():
    truth = {
        "R0": 1.72,
        "Z0": -0.08,
        "a": 0.43,
        "ka": 1.68,
        "c_offsets": np.array([0.01, 0.08, -0.02]),
        "s_offsets": np.array([0.0, -0.01, 0.01, 0.015]),
    }

    geqdsk = Geqdsk()
    geqdsk.boundary = _build_boundary(**truth)

    boundary = Boundary.from_geqdsk(geqdsk)
    fitted_boundary = _build_boundary(
        R0=boundary.R0,
        Z0=boundary.Z0,
        a=boundary.a,
        ka=boundary.ka,
        c_offsets=boundary.c_offsets,
        s_offsets=boundary.s_offsets,
    )

    rms = float(
        np.sqrt(
            np.mean(
                np.concatenate(
                    (
                        geqdsk.boundary[:, 0] - fitted_boundary[:, 0],
                        geqdsk.boundary[:, 1] - fitted_boundary[:, 1],
                    )
                )
                ** 2
            )
        )
    )

    assert len(boundary.c_offsets) == 3
    assert len(boundary.s_offsets) == 4
    assert rms < 1.0e-2


@pytest.mark.parametrize(
    ("build_source", "kwargs", "expected"),
    [
        (lambda: Geqdsk(), {"M": 1, "N": 2}, "Boundary is empty"),
        (lambda: _make_geqdsk_with_boundary(), {"M": 2}, "provided together or both omitted"),
        (lambda: _make_geqdsk_with_boundary(), {"maxtol": 0.0}, "maxtol must be positive"),
        (lambda: _make_geqdsk_with_boundary(), {"M": 11, "N": 2}, "must be <= 10"),
    ],
)
def test_boundary_from_geqdsk_rejects_invalid_inputs(build_source, kwargs, expected):
    with pytest.raises(ValueError, match=expected):
        Boundary.from_geqdsk(build_source(), **kwargs)


def test_boundary_from_geqdsk_supports_instance_and_path_sources():
    geqdsk = Geqdsk(str(GEQDSK_PATH))

    boundary_from_instance = Boundary.from_geqdsk(geqdsk, M=1, N=2, maxtol=1.0e-1)
    boundary_from_path = Boundary.from_geqdsk(GEQDSK_PATH, M=1, N=2, maxtol=1.0e-1)

    assert boundary_from_instance.B0 > 0.0
    assert boundary_from_instance.a > 0.0
    assert boundary_from_instance.ka > 1.0
    assert boundary_from_path.B0 > 0.0
    assert boundary_from_path.a > 0.0
    assert boundary_from_path.ka > 1.0


def test_equilibrium_roundtrips_canonical_active_profiles():
    equilibrium, _ = _build_high_order_equilibrium()

    assert set(equilibrium.active_profiles) == {"k", "c3", "s4"}
    assert equilibrium.geometry.tb.shape == (equilibrium.grid.Nr, equilibrium.grid.Nt)

    resampled = equilibrium.resample(target_grid=Grid(Nr=10, Nt=12, scheme="uniform", K_max=4))
    assert set(resampled.active_profiles) == {"k", "c3", "s4"}

    outpath = Path("tests/.tmp-equilibrium-k4.json")
    try:
        equilibrium.write(str(outpath))
        payload = orjson.loads(outpath.read_bytes())["Equilibrium"]
        loaded = type(equilibrium).load(str(outpath))

        assert set(payload["active_profiles"]) == {"k", "c3", "s4"}
        assert set(loaded.active_profiles) == set(equilibrium.active_profiles)
        assert np.allclose(loaded.active_profiles["c3"].u, equilibrium.active_profiles["c3"].u)
        assert np.allclose(loaded.active_profiles["s4"].u, equilibrium.active_profiles["s4"].u)
    finally:
        outpath.unlink(missing_ok=True)


def test_equilibrium_load_rejects_legacy_shape_payload():
    equilibrium, operator = _build_high_order_equilibrium()
    grid = equilibrium.grid
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
    legacy_payload["Equilibrium"]["shape_profiles"] = [
        {
            "Profile": {
                "scale": profile.scale,
                "power": profile.power,
                "envelope_power": profile.envelope_power,
                "offset": profile.offset,
                "coeff": None if profile.coeff is None else profile.coeff.tolist(),
            }
        }
        for name, profile in ((name, operator.profiles_by_name[name]) for name in operator.shape_profile_names)
    ]

    outpath = Path("tests/.tmp-equilibrium-legacy.json")
    try:
        outpath.write_bytes(orjson.dumps(legacy_payload, option=orjson.OPT_SERIALIZE_NUMPY))
        with pytest.raises(TypeError):
            type(equilibrium).load(str(outpath))
    finally:
        outpath.unlink(missing_ok=True)


def test_equilibrium_compare_reports_only_primary_shape_errors():
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
    cur_operator = Operator(name="PF", derivative="rho", grid=grid, case=case_cur)

    errors = ref_operator.build_equilibrium(ref_operator.encode_initial_state()).compare(
        cur_operator.build_equilibrium(cur_operator.encode_initial_state())
    )

    assert "rel_v_max" in errors
    assert "rel_k_max" in errors
    assert "rel_h_max" not in errors
    assert "rel_s1_max" not in errors

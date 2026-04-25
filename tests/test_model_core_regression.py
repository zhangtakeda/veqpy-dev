from pathlib import Path

import numpy as np
import orjson
import pytest

from veqpy.model import Boundary, Equilibrium, Geqdsk, Grid, Reactive
from veqpy.operator import Operator, OperatorCase, build_profile_names

GEQDSK_PATH = Path("tests/EFIT.geqdsk")
TEST_SOURCE_SAMPLE_COUNT = 21

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
    operator = Operator(grid=grid, case=case)
    equilibrium = operator.build_equilibrium(operator.encode_initial_state())
    return equilibrium, operator


def test_equilibrium_plot_handles_2d_axis_extrapolation():
    equilibrium, _ = _build_high_order_equilibrium()
    outdir = Path("tests") / "_tmp_plot_regression"
    outdir.mkdir(parents=True, exist_ok=True)
    outpath = outdir / "equilibrium-summary.png"

    fig = equilibrium.plot(outpath=outpath)

    assert outpath.exists()
    assert fig is not None


def _build_operator_case(*, mode="PF", coordinate="rho", nodes="uniform") -> OperatorCase:
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
    case = OperatorCase(
        route=mode,
        coordinate=coordinate,
        nodes=nodes,
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
    return grid, case


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


def test_operator_case_autoscales_legacy_unscaled_heat_input():
    _, case = _build_operator_case(mode="PP", coordinate="rho", nodes="uniform")
    source = np.full(TEST_SOURCE_SAMPLE_COUNT, 1.0e5)

    with pytest.warns(RuntimeWarning, match="canonical OperatorCase inputs are mu0-scaled"):
        probe = OperatorCase(
            route=case.route,
            coordinate=case.coordinate,
            nodes=case.nodes,
            profile_coeffs=case.profile_coeffs,
            boundary=case.boundary,
            heat_input=source,
            current_input=case.current_input,
        )

    assert np.allclose(probe.heat_input, source * (4.0e-7 * np.pi))


def test_operator_case_autoscales_legacy_unscaled_current_input():
    _, case = _build_operator_case(mode="PI", coordinate="rho", nodes="uniform")
    source = np.full(TEST_SOURCE_SAMPLE_COUNT, 1.0e5)

    with pytest.warns(RuntimeWarning, match="canonical OperatorCase inputs are mu0-scaled"):
        probe = OperatorCase(
            route=case.route,
            coordinate=case.coordinate,
            nodes=case.nodes,
            profile_coeffs=case.profile_coeffs,
            boundary=case.boundary,
            heat_input=case.heat_input,
            current_input=source,
        )

    assert np.allclose(probe.current_input, source * (4.0e-7 * np.pi))


def test_operator_case_autoscales_legacy_unscaled_ip_constraint():
    _, case = _build_operator_case(mode="PI", coordinate="rho", nodes="uniform")

    with pytest.warns(RuntimeWarning, match="canonical OperatorCase inputs are mu0-scaled"):
        probe = OperatorCase(
            route=case.route,
            coordinate=case.coordinate,
            nodes=case.nodes,
            profile_coeffs=case.profile_coeffs,
            boundary=case.boundary,
            heat_input=case.heat_input,
            current_input=np.ones(TEST_SOURCE_SAMPLE_COUNT),
            Ip=3.0e6,
        )

    assert probe.Ip == pytest.approx(3.0e6 * (4.0e-7 * np.pi))


def test_operator_case_skips_mu0_current_check_for_pq_driver():
    _, case = _build_operator_case(mode="PQ", coordinate="rho", nodes="uniform")
    source = np.full(TEST_SOURCE_SAMPLE_COUNT, 1.0e5)

    with pytest.warns(RuntimeWarning, match="canonical OperatorCase inputs are mu0-scaled"):
        probe = OperatorCase(
            route=case.route,
            coordinate=case.coordinate,
            nodes=case.nodes,
            profile_coeffs=case.profile_coeffs,
            boundary=case.boundary,
            heat_input=source,
            current_input=source,
        )

    assert np.allclose(probe.current_input, source)
    assert np.allclose(probe.heat_input, source * (4.0e-7 * np.pi))


def test_grid_precomputes_fourier_tables_and_rho_powers():
    grid = Grid(Nr=8, Nt=16, scheme="uniform", M_max=5)

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
        (lambda: _make_geqdsk_with_boundary(), {"M": 21, "N": 2}, "must be <= 20"),
    ],
)
def test_boundary_from_geqdsk_rejects_invalid_inputs(build_source, kwargs, expected):
    with pytest.raises(ValueError, match=expected):
        Boundary.from_geqdsk(build_source(), **kwargs)


def test_boundary_from_geqdsk_requires_instance_source():
    geqdsk = Geqdsk(str(GEQDSK_PATH))

    boundary_from_instance = Boundary.from_geqdsk(geqdsk, M=1, N=2, maxtol=1.0e-1)
    with pytest.raises(TypeError, match="geqdsk must be Geqdsk"):
        Boundary.from_geqdsk(GEQDSK_PATH, M=1, N=2, maxtol=1.0e-1)

    assert boundary_from_instance.B0 > 0.0
    assert boundary_from_instance.a > 0.0
    assert boundary_from_instance.ka > 1.0


def test_equilibrium_roundtrips_canonical_shape_profiles():
    equilibrium, _ = _build_high_order_equilibrium()
    expected_profiles = set(equilibrium.shape_profiles)

    assert {"k", "c3", "s4"}.issubset(expected_profiles)
    assert equilibrium.geometry.tb.shape == (equilibrium.grid.Nr, equilibrium.grid.Nt)

    target_grid = Grid(Nr=10, Nt=12, scheme="uniform", M_max=4)
    resampled = equilibrium.resample(target_grid)
    resampled_positional = equilibrium.resample(target_grid)
    assert set(resampled.shape_profiles) == expected_profiles
    assert np.allclose(resampled.rho, target_grid.rho)
    assert np.allclose(resampled_positional.rho, target_grid.rho)
    assert np.allclose(resampled.psin_r, resampled_positional.psin_r)

    outpath = Path("tests/.tmp-equilibrium-k4.json")
    try:
        equilibrium.write(str(outpath))
        payload = orjson.loads(outpath.read_bytes())["Equilibrium"]
        loaded = type(equilibrium).load(str(outpath))

        assert set(payload["shape_profiles"]) == expected_profiles
        assert set(loaded.shape_profiles) == expected_profiles
        assert np.allclose(loaded.shape_profiles["c3"].u, equilibrium.shape_profiles["c3"].u)
        assert np.allclose(loaded.shape_profiles["s4"].u, equilibrium.shape_profiles["s4"].u)
    finally:
        outpath.unlink(missing_ok=True)


def test_equilibrium_declares_reactive_roots_explicitly():
    assert Equilibrium.root_properties == {
        "R0",
        "Z0",
        "B0",
        "a",
        "grid",
        "shape_profiles",
        "FFn_psin",
        "Pn_psin",
        "psin",
        "psin_r",
        "psin_rr",
        "alpha1",
        "alpha2",
    }


def test_equilibrium_to_geqdsk_exports_uniform_psin_profiles_and_marks_exterior():
    equilibrium, _ = _build_high_order_equilibrium()
    export_equilibrium = Equilibrium(
        R0=equilibrium.R0,
        Z0=equilibrium.Z0,
        B0=equilibrium.B0,
        a=equilibrium.a,
        grid=equilibrium.grid,
        shape_profiles=equilibrium.shape_profiles,
        FFn_psin=equilibrium.FFn_psin,
        Pn_psin=equilibrium.Pn_psin,
        psin=equilibrium.psin,
        psin_r=equilibrium.psin_r,
        psin_rr=equilibrium.psin_rr,
        alpha1=1.25,
        alpha2=0.75,
    )
    boundary_R = np.asarray(export_equilibrium.geometry.R[-1], dtype=np.float64)
    boundary_Z = np.asarray(export_equilibrium.geometry.Z[-1], dtype=np.float64)
    psi_axis = 2.5
    psi_scale = float(export_equilibrium.alpha2)

    geqdsk = export_equilibrium.to_geqdsk(
        R_range=(float(np.min(boundary_R)) - 0.2, float(np.max(boundary_R)) + 0.2),
        NR=65,
        Z_range=(float(np.min(boundary_Z)) - 0.2, float(np.max(boundary_Z)) + 0.2),
        NZ=81,
        psi_axis=psi_axis,
    )

    psin_uniform = np.linspace(0.0, 1.0, geqdsk.NR, dtype=np.float64)
    expected_q = np.interp(psin_uniform, export_equilibrium.psin, export_equilibrium.q)

    assert isinstance(geqdsk, Geqdsk)
    assert geqdsk.NR == 65
    assert geqdsk.NZ == 81
    assert geqdsk.psi_axis == pytest.approx(psi_axis)
    assert geqdsk.psi_bound == pytest.approx(psi_axis + psi_scale)
    assert geqdsk.psi.shape == (65, 81)
    assert geqdsk.q.shape == (65,)
    assert np.array_equal(np.isfinite(geqdsk.q), np.isfinite(expected_q))
    assert np.array_equal(np.isinf(geqdsk.q), np.isinf(expected_q))
    finite_q = np.isfinite(expected_q)
    assert np.allclose(geqdsk.q[finite_q], expected_q[finite_q])
    psi_min = min(psi_axis, psi_axis + psi_scale)
    psi_max = max(psi_axis, psi_axis + psi_scale)
    assert np.min(geqdsk.psi) >= psi_min - 1.0e-12
    assert np.max(geqdsk.psi) <= psi_max + 1.0e-12
    assert geqdsk.psi[0, 0] == pytest.approx(psi_axis + psi_scale)
    assert np.any((geqdsk.psi >= psi_min) & (geqdsk.psi <= psi_max))


def test_geqdsk_io_uses_standard_psirz_file_order():
    psi = np.array(
        [
            [11.0, 12.0],
            [21.0, 22.0],
            [31.0, 32.0],
        ],
        dtype=np.float64,
    )
    geqdsk = Geqdsk(
        header="io-order",
        NR=3,
        NZ=2,
        R0=1.0,
        Z0=0.0,
        Rmin=0.5,
        Rmax=1.5,
        Zmin=-0.25,
        Zmax=0.25,
        Bt0=2.0,
        Raxis=1.0,
        Zaxis=0.0,
        Ip=1.0,
        psi_axis=0.0,
        psi_bound=1.0,
        F=np.array([1.0, 2.0, 3.0], dtype=np.float64),
        P=np.array([4.0, 5.0, 6.0], dtype=np.float64),
        FF_psi=np.array([7.0, 8.0, 9.0], dtype=np.float64),
        P_psi=np.array([10.0, 11.0, 12.0], dtype=np.float64),
        q=np.array([13.0, 14.0, 15.0], dtype=np.float64),
        psi=psi,
    )

    outpath = Path("tests/.tmp-geqdsk-io-order.geqdsk")
    try:
        geqdsk.write(str(outpath))
        loaded = Geqdsk(str(outpath))

        payload = outpath.read_text(encoding="utf-8").splitlines()
        header_tail = np.fromstring(payload[4], sep=" ")
        flat_values = np.fromstring(" ".join(payload[5:]), sep=" ")
        psi_start = 4 * geqdsk.NR
        psi_stop = psi_start + geqdsk.NR * geqdsk.NZ
        psi_file = flat_values[psi_start:psi_stop]

        assert np.allclose(loaded.psi, psi)
        assert np.allclose(header_tail, np.array([0.0, 0.0, 1.0, 0.0, 0.0], dtype=np.float64))
        assert np.allclose(psi_file, psi.T.reshape(-1))
    finally:
        outpath.unlink(missing_ok=True)


def test_equilibrium_to_geqdsk_rejects_zero_alpha2_for_physical_psi_export():
    equilibrium, _ = _build_high_order_equilibrium()
    boundary_R = np.asarray(equilibrium.geometry.R[-1], dtype=np.float64)
    boundary_Z = np.asarray(equilibrium.geometry.Z[-1], dtype=np.float64)

    with pytest.raises(ValueError, match="alpha2 is zero"):
        equilibrium.to_geqdsk(
            R_range=(float(np.min(boundary_R)) - 0.2, float(np.max(boundary_R)) + 0.2),
            Z_range=(float(np.min(boundary_Z)) - 0.2, float(np.max(boundary_Z)) + 0.2),
            NR=65,
            NZ=81,
        )


def test_reactive_requires_explicit_root_properties():
    with pytest.raises(TypeError, match="must define root_properties explicitly"):

        class _BadReactive(Reactive):
            @property
            def value(self):
                return 1


def test_equilibrium_exposes_gs_operator_residual_terms():
    x = None
    _, operator = _build_high_order_equilibrium()
    x = operator.encode_initial_state()
    operator.stage_a_profile(x)
    operator.stage_b_geometry()
    operator.stage_c_source()
    operator.stage_d_residual()
    equilibrium = operator.build_equilibrium(x)

    geometry = equilibrium.geometry
    expected_gn1 = geometry.JdivR * (equilibrium.FFn_psin[:, None] + geometry.R**2 * equilibrium.Pn_psin[:, None])
    expected_gn2 = (
        geometry.gttdivJR * equilibrium.psin_rr[:, None]
        + (geometry.gttdivJR_r - geometry.grtdivJR_t) * equilibrium.psin_r[:, None]
    )

    assert np.allclose(equilibrium.Gn1, expected_gn1)
    assert np.allclose(equilibrium.Gn2, expected_gn2)
    assert np.allclose(equilibrium.G, operator.residual_surface_workspace[0])
    assert np.allclose(equilibrium.G, equilibrium.alpha1 * equilibrium.Gn1 + equilibrium.alpha2 * equilibrium.Gn2)


def test_resampled_equilibrium_preserves_profile_shapes_on_target_grid():
    profile_equilibrium, _ = _build_high_order_equilibrium()
    target_grid = Grid(Nr=12, Nt=24, scheme="uniform", M_max=4)

    resampled = profile_equilibrium.resample(target_grid)

    assert resampled.grid is target_grid
    assert resampled.psin.shape == target_grid.rho.shape
    assert resampled.psin_r.shape == target_grid.rho.shape
    assert resampled.FFn_psin.shape == target_grid.rho.shape
    assert resampled.Pn_psin.shape == target_grid.rho.shape
    assert np.all(np.isfinite(resampled.psin))
    assert np.all(np.isfinite(resampled.FFn_psin))


def test_equilibrium_diagnostics_use_grid_corrected_calculus(monkeypatch):
    calls = {"integrate": 0, "corrected_even_derivative": 0, "corrected_linear_derivative": 0}
    original_integrate = Grid.integrate
    original_corrected_even_derivative = Grid.corrected_even_derivative
    original_corrected_linear_derivative = Grid.corrected_linear_derivative

    def _track_integrate(self, *args, **kwargs):
        calls["integrate"] += 1
        return original_integrate(self, *args, **kwargs)

    def _track_corrected_even_derivative(self, *args, **kwargs):
        calls["corrected_even_derivative"] += 1
        return original_corrected_even_derivative(self, *args, **kwargs)

    def _track_corrected_linear_derivative(self, *args, **kwargs):
        calls["corrected_linear_derivative"] += 1
        return original_corrected_linear_derivative(self, *args, **kwargs)

    monkeypatch.setattr(Grid, "integrate", _track_integrate)
    monkeypatch.setattr(Grid, "corrected_even_derivative", _track_corrected_even_derivative)
    monkeypatch.setattr(Grid, "corrected_linear_derivative", _track_corrected_linear_derivative)

    equilibrium, _ = _build_high_order_equilibrium()

    assert np.all(np.isfinite(equilibrium.F2))
    assert np.all(np.isfinite(equilibrium.P))
    assert equilibrium.s.shape == equilibrium.rho.shape
    assert equilibrium.jpara.shape == equilibrium.rho.shape
    assert np.all(np.isfinite(equilibrium.Psi))
    assert np.all(np.isfinite(equilibrium.Phi))
    equilibrium.resample(Grid(Nr=12, Nt=24, scheme="uniform", M_max=4))
    assert calls["integrate"] >= 2
    assert calls["corrected_even_derivative"] >= 1
    assert calls["corrected_linear_derivative"] >= 1


def test_pj2_psin_route_uses_f_profile_parameterization():
    grid, case = _build_operator_case(mode="PJ2", coordinate="psin", nodes="uniform")
    case.profile_coeffs["psin"] = None
    operator = Operator(grid=grid, case=case)
    x = operator.encode_initial_state()
    operator.stage_a_profile(x)

    latent_H = 1.0 + grid.y * grid.y * case.profile_coeffs["F"][0]
    latent_H_r = -4.0 * grid.rho * grid.y * case.profile_coeffs["F"][0]
    latent_H_rr = (-4.0 + 12.0 * grid.rho * grid.rho) * case.profile_coeffs["F"][0]
    scale = case.R0 * case.B0
    expected_F = scale * np.sqrt(latent_H)
    expected_F_r = scale * 0.5 * latent_H_r / np.sqrt(latent_H)
    expected_F_rr = scale * (
        0.5 * latent_H_rr / np.sqrt(latent_H) - 0.25 * latent_H_r * latent_H_r / np.power(latent_H, 1.5)
    )

    assert np.allclose(operator.F_profile.u, expected_F)
    assert np.allclose(operator.F_profile.u_r, expected_F_r)
    assert np.allclose(operator.F_profile.u_rr, expected_F_rr)


def test_pq_psin_route_evaluates_residual_without_active_psin_profile():
    grid, case = _build_operator_case(mode="PQ", coordinate="psin", nodes="uniform")
    case.profile_coeffs["psin"] = None
    case.heat_input = np.linspace(1.0, 2.0, case.heat_input.shape[0])
    case.current_input = np.linspace(1.0, 2.0, case.current_input.shape[0])
    operator = Operator(grid=grid, case=case)
    equilibrium = operator.build_equilibrium(operator.encode_initial_state())

    assert equilibrium.grid is grid
    assert equilibrium.F.shape == grid.rho.shape
    assert np.all(np.isfinite(equilibrium.F))


def test_equilibrium_load_rejects_legacy_shape_payload():
    equilibrium, operator = _build_high_order_equilibrium()
    grid = equilibrium.grid
    legacy_payload = {
        "Equilibrium": {
            "R0": equilibrium.R0,
            "Z0": equilibrium.Z0,
            "B0": equilibrium.B0,
            "a": equilibrium.a,
            "grid": {
                "Grid": {"Nr": grid.Nr, "Nt": grid.Nt, "scheme": grid.scheme, "L_max": grid.L_max, "M_max": grid.M_max}
            },
            "shape_profiles": [],
            "shape_profile_names": list(operator.shape_profile_names),
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
    grid = Grid(Nr=8, Nt=16, scheme="uniform", M_max=4)
    profile_coeffs = {name: None for name in build_profile_names(grid.M_max)}
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
    case_cur = case_ref.copy()
    case_cur.profile_coeffs["v"] = [0.03]
    case_cur.profile_coeffs["s1"] = [0.08]

    ref_operator = Operator(grid=grid, case=case_ref)
    cur_operator = Operator(grid=grid, case=case_cur)

    errors = ref_operator.build_equilibrium(ref_operator.encode_initial_state()).compare(
        cur_operator.build_equilibrium(cur_operator.encode_initial_state())
    )

    assert "rel_h_max" in errors
    assert "rel_k_max" in errors
    assert "rel_s1_max" in errors
    assert "rel_v_max" not in errors

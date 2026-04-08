from pathlib import Path

import numpy as np
import orjson
import pytest

from veqpy.model import Boundary
from veqpy.model import equilibrium as equilibrium_module
from veqpy.model.equilibrium import Equilibrium
from veqpy.model.geqdsk import Geqdsk
from veqpy.model.grid import Grid
from veqpy.model.reactive import Reactive
from veqpy.operator import (
    ExecutionState,
    FieldRuntimeState,
    Operator,
    ResidualBindingLayout,
    RuntimeLayout,
    SourceRuntimeState,
    StaticLayout,
)
from veqpy.operator import layout as layout_module
from veqpy.operator.codec import encode_packed_state
from veqpy.operator.layout import build_profile_layout, build_profile_names
from veqpy.operator.operator_case import OperatorCase

GEQDSK_PATH = Path("tests/fitting/geqdsk.txt")
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
        (lambda: _make_geqdsk_with_boundary(), {"M": 11, "N": 2}, "must be <= 10"),
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
    assert np.allclose(equilibrium.G, operator.G)
    assert np.allclose(equilibrium.G, equilibrium.alpha1 * equilibrium.Gn1 + equilibrium.alpha2 * equilibrium.Gn2)


def test_resampled_equilibrium_uses_linear_profile_reconstruction(monkeypatch):
    profile_equilibrium, _ = _build_high_order_equilibrium()
    sentinel = np.linspace(0.0, 1.0, 12, dtype=np.float64)
    called = {"value": False}

    def _fake_resample_profile_linear(rho_src, y_src, rho_eval, **kwargs):
        called["value"] = True
        return sentinel

    monkeypatch.setattr(equilibrium_module, "_resample_profile_linear", _fake_resample_profile_linear)

    resampled = equilibrium_module._build_resampled_equilibrium(
        profile_equilibrium,
        grid=Grid(Nr=12, Nt=24, scheme="uniform", M_max=4),
    )

    assert called["value"] is True
    assert np.allclose(resampled.psin_r, sentinel)
    assert np.allclose(resampled.psin, sentinel)
    assert np.allclose(resampled.FFn_psin, sentinel)
    assert np.allclose(resampled.Pn_psin, sentinel)


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


def test_operator_exposes_explicit_layout_and_execution_layers():
    grid, case = _build_operator_case()
    operator = Operator(grid=grid, case=case)

    assert isinstance(operator.static_layout, StaticLayout)
    assert isinstance(operator.residual_binding_layout, ResidualBindingLayout)
    assert isinstance(operator.runtime_layout, RuntimeLayout)
    assert isinstance(operator.field_runtime_state, FieldRuntimeState)
    assert isinstance(operator.execution_state, ExecutionState)
    assert isinstance(operator.source_runtime_state, SourceRuntimeState)

    assert operator.static_layout.rho is grid.rho
    assert operator.static_layout.T_fields is grid.T_fields
    assert operator.profile_L.shape[0] == len(operator.profile_names)
    assert operator.coeff_index.shape[0] == len(operator.profile_names)
    assert operator.packed_residual.shape[0] == operator.x_size
    assert operator.residual_binding_layout.active_profile_names == tuple(
        operator.profile_names[int(p)] for p in operator.active_profile_ids
    )
    assert operator.runtime_layout.geometry is operator.geometry
    assert operator.runtime_layout.root_fields is operator.root_fields
    assert operator.runtime_layout.packed_residual is operator.packed_residual
    assert operator.field_runtime_state.root_fields is operator.root_fields
    assert operator.field_runtime_state.residual_fields is operator.residual_fields
    assert operator.execution_state.fused_alpha_state.shape == (2,)
    assert callable(operator.execution_state.fused_residual_runner)
    assert operator.runtime_layout.source_psin_query is operator.source_runtime_state.psin_query
    assert operator.runtime_layout.materialized_heat_input is operator.source_runtime_state.materialized_heat_input
    assert operator.active_u_fields.base is operator.active_profile_slab
    assert operator.c_family_fields.base is operator.family_field_slab
    assert operator.source_runtime_state.psin_query.base is operator.source_vector_slab
    assert operator.geometry.tb_fields.base is operator.geometry_surface_slab
    assert operator.geometry.S_r.base is operator.geometry_radial_slab

def test_operator_replace_case_preserves_static_layout_and_refreshes_runtime_identity():
    grid, case = _build_operator_case()
    operator = Operator(grid=grid, case=case)
    static_layout = operator.static_layout

    next_case = case.copy()
    next_case.beta = 0.25
    next_case.profile_coeffs["c3"] = [0.02]

    operator.replace_case(next_case)

    assert operator.static_layout is static_layout
    assert operator.case.coordinate == next_case.coordinate
    assert operator.case.nodes == next_case.nodes
    assert operator.source_plan.coordinate == next_case.coordinate
    assert operator.source_plan.nodes == next_case.nodes
    assert operator.runtime_layout.geometry is operator.geometry


def test_source_plan_captures_fixed_point_route_metadata():
    grid, case = _build_operator_case(mode="PQ", coordinate="psin", nodes="uniform")
    case.profile_coeffs["psin"] = None
    operator = Operator(grid=grid, case=case)

    assert operator.source_plan.strategy == "fixed_point_psin"
    assert operator.source_plan.n_src == TEST_SOURCE_SAMPLE_COUNT
    assert operator.source_plan.use_projected_finalize is True
    assert operator.source_plan.projection_domain == "sqrt_psin"
    assert operator.source_plan.projection_domain_code == 1
    assert operator.source_plan.heat_projection_degree == 7
    assert operator.source_plan.current_projection_degree == 10
    assert operator.source_plan.endpoint_policy_code == 2
    assert operator.source_plan.is_fixed_point_psin is True


@pytest.mark.parametrize(("mode", "heat_degree", "current_degree"), [("PI", 7, 8), ("PJ1", 7, 8)])
def test_source_plan_captures_profile_owned_projection_policy(mode, heat_degree, current_degree):
    grid, case = _build_operator_case(mode=mode, coordinate="psin", nodes="uniform")
    operator = Operator(grid=grid, case=case)

    assert operator.source_plan.strategy == "profile_owned_psin"
    assert operator.source_plan.has_projection_policy is True
    assert operator.source_plan.projection_domain == "psin"
    assert operator.source_plan.heat_projection_degree == heat_degree
    assert operator.source_plan.current_projection_degree == current_degree
    assert operator.source_plan.endpoint_policy_code == 0
    assert operator.source_plan.use_projected_finalize is True


@pytest.mark.parametrize("mode", ["PI", "PJ1"])
def test_source_plan_uses_both_endpoint_policy_when_ip_constraint_is_active(mode):
    grid, case = _build_operator_case(mode=mode, coordinate="psin", nodes="uniform")
    case.Ip = 3.0e6
    operator = Operator(grid=grid, case=case)

    assert operator.source_plan.has_projection_policy is True
    assert operator.source_plan.endpoint_policy_code == 2


def test_build_profile_names_respects_module_family_order(monkeypatch):
    monkeypatch.setattr(layout_module, "PACKED_PROFILE_FAMILY_ORDER", ("psin", "F", "k", "h", "v", "c0", "s", "c"))
    assert layout_module.build_profile_names(2) == (
        "psin",
        "F",
        "k",
        "h",
        "v",
        "c0",
        "s1",
        "s2",
        "c1",
        "c2",
    )


def test_build_profile_layout_can_group_shape_coeffs_by_profile(monkeypatch):
    monkeypatch.setattr(
        layout_module,
        "PACKED_PROFILE_FAMILY_ORDER",
        ("h", "v", "k", "c0", "c", "s", "psin", "F"),
    )
    monkeypatch.setattr(layout_module, "INTERLEAVE_SHAPE_COEFFS_BY_ORDER", False)
    profile_names = layout_module.build_profile_names(1)
    profile_coeffs = {name: None for name in profile_names}
    profile_coeffs.update(
        {
            "psin": [0.0, 1.0],
            "F": [1.0, 2.0],
            "h": [10.0, 11.0],
            "v": [20.0, 21.0],
            "k": [30.0, 31.0],
        }
    )
    profile_L, coeff_index, _ = build_profile_layout(profile_coeffs, profile_names=profile_names)
    x = encode_packed_state(profile_coeffs, profile_L, coeff_index, profile_names=profile_names)
    assert x.tolist()[:10] == [10.0, 11.0, 20.0, 21.0, 30.0, 31.0, 0.0, 1.0, 1.0, 2.0]


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

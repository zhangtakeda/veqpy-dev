from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from veqpy.model.boundary import Boundary
from veqpy.model.equilibrium import Equilibrium
from veqpy.model.grid import Grid
from veqpy.operator import Operator, OperatorCase


def test_operator_callable_and_snapshot_contract(tmp_path: Path) -> None:
    psin = np.linspace(0.0, 1.0, 8, dtype=np.float64)
    case = OperatorCase(
        route="PF",
        coordinate="psin",
        profile_coeffs={"psin": 3, "h": 2, "k": 2, "s1": 2},
        boundary=Boundary(a=1.0, R0=3.0, Z0=0.0, B0=2.0, ka=1.2),
        heat_input=1.0 - psin,
        current_input=psin,
        Ip=1.0,
    )
    operator = Operator(grid=Grid(Nr=6, Nt=8, L_max=4, M_max=2), case=case)

    x = operator.encode_initial_state()
    residual = operator(x)
    residual_into = np.empty_like(residual)
    operator.residual_var_into(x, residual_into)
    collocation = operator.residual_collocation(x)
    equilibrium = operator.build_equilibrium(x)

    assert residual.shape == x.shape
    np.testing.assert_allclose(residual_into, residual)
    expected_collocation_size = operator.plan.grid_workspace.Nr * operator.plan.grid_workspace.Nt
    assert collocation.shape == (expected_collocation_size,)
    expected_collocation = (
        operator.residual_workspace.collocation_sqrt_weights[:, None]
        * operator.geometry_workspace.surface_fields[1]
        / operator.geometry_workspace.surface_fields[4]
        * operator.residual_workspace.surface_fields[0]
    ).ravel()
    np.testing.assert_allclose(collocation, expected_collocation)
    assert equilibrium.psin.shape == (operator.plan.grid_workspace.Nr,)
    assert equilibrium.shape_profiles
    assert {"h", "v", "k", "s1"}.issubset(equilibrium.shape_profiles)
    for profile in equilibrium.shape_profiles.values():
        for runtime_attr in ("u_fields", "rp_fields", "env_fields", "T", "T_r", "T_rr"):
            assert not hasattr(profile, runtime_attr)
    assert equilibrium.geometry.R.shape == (
        operator.plan.grid_workspace.Nr,
        operator.plan.grid_workspace.Nt,
    )
    np.testing.assert_allclose(
        equilibrium.geometry.S_r,
        operator.geometry_workspace.radial_fields[0],
    )
    np.testing.assert_allclose(
        equilibrium.geometry.R,
        operator.geometry_workspace.surface_fields[1],
    )
    snapshot_R = equilibrium.geometry.R.copy()
    operator.profile_workspace.fields_for("h").fill(123.0)
    np.testing.assert_allclose(equilibrium.geometry.R, snapshot_R)

    path = tmp_path / "equilibrium.json"
    equilibrium.write(str(path))
    payload = json.loads(path.read_text())["Equilibrium"]
    assert "shape_profiles" in payload
    assert "geometry" not in payload
    loaded = Equilibrium.load(str(path))
    assert loaded.shape_profiles
    assert set(loaded.shape_profiles) == set(equilibrium.shape_profiles)
    np.testing.assert_allclose(loaded.geometry.R, equilibrium.geometry.R)


def test_equilibrium_geometry_is_derived_or_legacy_fallback() -> None:
    grid = Grid(Nr=6, Nt=8, L_max=4, M_max=2)
    zeros = np.zeros(grid.Nr, dtype=np.float64)
    surface_zeros = np.zeros((grid.Nr, grid.Nt), dtype=np.float64)
    geometry = {
        "S_r": zeros,
        "V_r": zeros,
        "Kn": zeros,
        "Kn_r": zeros,
        "Ln_r": zeros,
        "tb_fields": np.zeros((8, grid.Nr, grid.Nt), dtype=np.float64),
        "R_fields": np.zeros((6, grid.Nr, grid.Nt), dtype=np.float64),
        "Z_fields": np.zeros((6, grid.Nr, grid.Nt), dtype=np.float64),
        "J_fields": np.zeros((8, grid.Nr, grid.Nt), dtype=np.float64),
        "g_fields": np.zeros((7, grid.Nr, grid.Nt), dtype=np.float64),
    }

    equilibrium = Equilibrium(
        R0=3.0,
        Z0=0.0,
        B0=2.0,
        a=1.0,
        grid=grid,
        psin=zeros,
        FFn_psin=zeros,
        Pn_psin=zeros,
        psin_r=zeros,
        psin_rr=zeros,
    )
    with pytest.raises(RuntimeError, match="shape_profiles or legacy geometry"):
        _ = equilibrium.geometry

    legacy_equilibrium = Equilibrium(
        R0=3.0,
        Z0=0.0,
        B0=2.0,
        a=1.0,
        grid=grid,
        geometry=geometry,
        psin=zeros,
        FFn_psin=zeros,
        Pn_psin=zeros,
        psin_r=zeros,
        psin_rr=zeros,
    )
    np.testing.assert_allclose(legacy_equilibrium.geometry.R, surface_zeros)


def test_profile_workspace_owns_profile_fields() -> None:
    psin = np.linspace(0.0, 1.0, 8, dtype=np.float64)
    case = OperatorCase(
        route="PF",
        coordinate="psin",
        profile_coeffs={"psin": 3, "h": 2, "k": 2, "s1": 2},
        boundary=Boundary(a=1.0, R0=3.0, Z0=0.0, B0=2.0, ka=1.2),
        heat_input=1.0 - psin,
        current_input=psin,
        Ip=1.0,
    )
    operator = Operator(grid=Grid(Nr=6, Nt=8, L_max=4, M_max=2), case=case)
    profile_workspace = operator.profile_workspace

    assert not hasattr(profile_workspace, "active_u_fields")
    assert not hasattr(operator.geometry_workspace, "h_fields")
    assert not hasattr(operator.source_workspace, "f_fields")
    np.testing.assert_array_equal(profile_workspace.active_profile_ids, operator.active_profile_ids)

    for profile_id, name in enumerate(operator.profile_names):
        profile = operator.profiles_by_name[name]
        assert profile is operator.profiles_by_name[name]
        for runtime_attr in ("u_fields", "rp_fields", "env_fields", "T", "T_r", "T_rr"):
            assert not hasattr(profile, runtime_attr)
        assert profile_workspace.profile_fields[profile_id].shape == (
            3,
            operator.plan.grid_workspace.Nr,
        )
        assert profile_workspace.profile_rp_fields[profile_id].shape == (
            3,
            operator.plan.grid_workspace.Nr,
        )
        assert profile_workspace.profile_env_fields[profile_id].shape == (
            3,
            operator.plan.grid_workspace.Nr,
        )

    assert profile_workspace.has_fields_for("h")
    assert profile_workspace.has_fields_for("v")
    assert profile_workspace.has_fields_for("k")
    assert profile_workspace.has_fields_for("F")
    assert profile_workspace.has_fields_for("psin")
    assert not profile_workspace.profile_rp_fields.flags.writeable
    assert not profile_workspace.profile_env_fields.flags.writeable


def test_equilibrium_resample_uses_shape_profile_snapshot_not_field_interpolation() -> None:
    psin = np.linspace(0.0, 1.0, 9, dtype=np.float64)
    case = OperatorCase(
        route="PF",
        coordinate="psin",
        profile_coeffs={
            "psin": [0.0, 0.02],
            "h": [0.04, -0.01],
            "k": [0.03, 0.02],
            "c0": [0.02, -0.01],
            "c1": [0.03, 0.01],
            "s1": [0.04, -0.02],
            "s2": [0.02, 0.01],
        },
        boundary=Boundary(
            a=1.0,
            R0=3.0,
            Z0=0.0,
            B0=2.0,
            ka=1.2,
            c_offsets=np.array([0.0, 0.08, -0.03], dtype=np.float64),
            s_offsets=np.array([0.0, 0.05, 0.02], dtype=np.float64),
        ),
        heat_input=1.0 - 0.2 * psin,
        current_input=0.5 + psin,
        Ip=1.0,
    )
    low_grid = Grid(Nr=7, Nt=10, L_max=4, M_max=2)
    high_grid = Grid(Nr=19, Nt=28, L_max=4, M_max=2)

    low_operator = Operator(grid=low_grid, case=case)
    low_equilibrium = low_operator.build_equilibrium(low_operator.encode_initial_state())
    resampled = low_equilibrium.resample(grid=high_grid)

    high_operator = Operator(grid=high_grid, case=case)
    expected = high_operator.build_equilibrium(high_operator.encode_initial_state())

    np.testing.assert_allclose(resampled.geometry.R, expected.geometry.R, atol=1.0e-12)
    np.testing.assert_allclose(resampled.geometry.Z, expected.geometry.Z, atol=1.0e-12)


def test_pj2_uses_profile_workspace_for_source_profile_inputs() -> None:
    psin = np.linspace(0.0, 1.0, 8, dtype=np.float64)
    case = OperatorCase(
        route="PJ2",
        coordinate="psin",
        profile_coeffs={"F": 3, "h": 2, "k": 2, "s1": 2},
        boundary=Boundary(a=1.0, R0=3.0, Z0=0.0, B0=2.0, ka=1.2),
        heat_input=1.0 - psin,
        current_input=psin,
        Ip=1.0,
    )
    operator = Operator(grid=Grid(Nr=6, Nt=8, L_max=4, M_max=2), case=case)

    residual = operator.residual_var(operator.encode_initial_state())

    assert residual.shape == (operator.x_size,)
    assert np.all(np.isfinite(residual))

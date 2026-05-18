from __future__ import annotations

import numpy as np

from veqpy.model.boundary import Boundary
from veqpy.model.grid import Grid
from veqpy.operator import Operator, OperatorCase


def test_operator_callable_and_snapshot_contract() -> None:
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
    expected_collocation_size = (
        2 * operator.plan.grid_workspace.Nr * operator.plan.grid_workspace.Nt
    )
    assert collocation.shape == (expected_collocation_size,)
    assert equilibrium.psin.shape == (operator.plan.grid_workspace.Nr,)
    assert set(equilibrium.shape_profiles) == set(operator.plan.shape_profile_names)


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
        assert np.shares_memory(profile.u_fields, profile_workspace.profile_fields[profile_id])
        assert np.shares_memory(profile.rp_fields, profile_workspace.profile_rp_fields[profile_id])
        assert np.shares_memory(
            profile.env_fields, profile_workspace.profile_env_fields[profile_id]
        )

    assert np.shares_memory(profile_workspace.fields_for("h"), operator.h_profile.u_fields)
    assert np.shares_memory(profile_workspace.fields_for("v"), operator.v_profile.u_fields)
    assert np.shares_memory(profile_workspace.fields_for("k"), operator.k_profile.u_fields)
    assert np.shares_memory(profile_workspace.fields_for("F"), operator.F_profile.u_fields)
    assert np.shares_memory(profile_workspace.fields_for("psin"), operator.psin_profile.u_fields)


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

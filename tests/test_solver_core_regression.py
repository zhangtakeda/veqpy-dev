import numpy as np
import pytest
import warnings

from veqpy.model import Boundary, Grid
from veqpy.operator import Operator, OperatorCase
from veqpy.solver import Solver, SolverConfig


def _pf_reference_profiles(rho: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    beta0 = 0.75
    psin = rho**2
    psin_r = 2.0 * rho

    alpha_p, alpha_f = 5.0, 3.32
    exp_ap, exp_af = np.exp(alpha_p), np.exp(alpha_f)
    den_p, den_f = 1.0 + exp_ap * (alpha_p - 1.0), 1.0 + exp_af * (alpha_f - 1.0)

    current_input = (1.0 - beta0) * alpha_f * (np.exp(alpha_f * psin) - exp_af) / den_f * psin_r
    heat_input = beta0 * alpha_p * (np.exp(alpha_p * psin) - exp_ap) / den_p * psin_r
    return current_input, heat_input


def _make_solver(
    *,
    enable_history: bool = True,
    enable_warmstart: bool = False,
    enable_fallback: bool = True,
    fallback_methods: tuple[str, ...] = ("root-lm", "trf"),
) -> Solver:
    coeffs = {
        "h": [0.0] * 3,
        "v": None,
        "k": [0.0] * 3,
        "c0": None,
        "c1": None,
        "s1": [0.0] * 3,
        "s2": None,
    }
    boundary = Boundary(
        a=1.05 / 1.85,
        R0=1.05,
        Z0=0.0,
        B0=3.0,
        ka=2.2,
        s_offsets=np.array([0.0, float(np.arcsin(0.5))]),
    )
    grid = Grid(Nr=12, Nt=12, scheme="legendre")
    current_input, heat_input = _pf_reference_profiles(grid.rho)
    case = OperatorCase(
        profile_coeffs=coeffs,
        boundary=boundary,
        heat_input=heat_input,
        current_input=current_input,
        Ip=3.0e6,
    )
    operator = Operator(grid=grid, case=case, name="PF", derivative="rho")
    config = SolverConfig(
        method="hybr",
        enable_verbose=False,
        enable_history=enable_history,
        enable_warmstart=enable_warmstart,
        enable_fallback=enable_fallback,
        fallback_methods=fallback_methods,
    )
    return Solver(operator=operator, config=config)


def test_solver_records_history_and_rebuilds_equilibrium():
    solver = _make_solver(enable_history=True, enable_warmstart=False)

    x = solver.solve(enable_verbose=False)
    coeffs = solver.build_coeffs(include_none=False)
    equilibrium = solver.build_equilibrium()

    assert solver.result is not None
    assert solver.result.success
    assert solver.result.x.shape == x.shape
    assert len(solver.history) == 1
    assert set(coeffs) >= {"h", "k", "s1"}
    assert equilibrium.Ip == pytest.approx(solver.operator.case.Ip)
    assert equilibrium.geometry.tb.shape == (solver.operator.grid.Nr, solver.operator.grid.Nt)


def test_solver_reset_clear_and_history_flags_behave_consistently():
    solver = _make_solver(enable_history=True, enable_warmstart=False)

    solver.solve(enable_verbose=False)
    assert len(solver.history) == 1
    assert not np.allclose(solver.x0, 0.0)

    solver.clear()
    assert solver.history == []

    solver.reset()
    assert np.allclose(solver.x0, 0.0)

    solver.solve(enable_verbose=False, enable_history=False)
    assert solver.history == []


def test_solver_build_equilibrium_history_restores_runtime_state():
    solver = _make_solver(enable_history=True, enable_warmstart=False)

    solver.solve(enable_verbose=False)
    case_variant = solver.operator.case.copy()
    case_variant.profile_coeffs["h"] = [0.02, 0.0, 0.0]
    solver.replace_case(case_variant)
    solver.solve(enable_verbose=False)

    saved_case_h = list(solver.operator.case.profile_coeffs["h"])
    saved_x0 = solver.x0.copy()
    saved_success = solver.result.success

    equilibria = solver.build_equilibrium_history()

    assert len(solver.history) == 2
    assert len(equilibria) == 2
    assert solver.operator.case.profile_coeffs["h"] == saved_case_h
    assert np.allclose(solver.x0, saved_x0)
    assert solver.result is not None
    assert solver.result.success == saved_success


def test_solver_accepts_warmstart_x0_when_attempt_raises(monkeypatch: pytest.MonkeyPatch):
    solver = _make_solver(enable_history=False, enable_warmstart=True)
    x_guess = solver.x0.copy()
    residual = np.zeros_like(x_guess)
    residual[0] = 1.0e-8

    def fake_solve_opt_problem(self, x_guess_arg, *, solve_config):
        raise ZeroDivisionError("division by zero")

    monkeypatch.setattr(Solver, "_solve_opt_problem", fake_solve_opt_problem)
    monkeypatch.setattr(Solver, "_safe_residual_norm", lambda self, x: (float(np.linalg.norm(residual)), None))

    result, error = solver._try_solve_attempt(x_guess, solve_config=solver.config)

    assert error is None
    assert result is not None
    assert result[1] is True
    assert np.allclose(result[0], x_guess)
    assert result[6] == pytest.approx(np.linalg.norm(residual))
    assert "accepted by x0 residual" in result[2]


def test_solver_fallback_order_is_primary_then_root_lm_then_trf(monkeypatch: pytest.MonkeyPatch):
    solver = _make_solver(enable_history=False, enable_warmstart=False)
    call_methods: list[str] = []

    def fake_try_solve_attempt(self, x_guess, *, solve_config):
        call_methods.append(solve_config.method)
        if solve_config.method == "trf":
            return (
                (
                    np.asarray(x_guess, dtype=np.float64).copy(),
                    True,
                    "trf converged",
                    7,
                    0,
                    0,
                    1.0e-8,
                ),
                None,
            )
        return (
            (
                np.asarray(x_guess, dtype=np.float64).copy(),
                False,
                f"{solve_config.method} failed",
                1,
                0,
                0,
                1.0,
            ),
            RuntimeError(f"{solve_config.method} failed"),
        )

    monkeypatch.setattr(Solver, "_try_solve_attempt", fake_try_solve_attempt)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = solver._solve_with_fallbacks(solver.x0.copy(), solve_config=solver.config)

    assert call_methods == ["hybr", "root-lm", "trf"]
    assert result[1] is True
    assert result[2].endswith("selected method=least_squares/trf")
    assert len(caught) == 2
    assert "Retrying with 'root/root-lm'" in str(caught[0].message)
    assert "Retrying with 'least_squares/trf'" in str(caught[1].message)


def test_solver_can_disable_fallback_attempts(monkeypatch: pytest.MonkeyPatch):
    solver = _make_solver(enable_history=False, enable_warmstart=False, enable_fallback=False)
    call_methods: list[str] = []

    def fake_try_solve_attempt(self, x_guess, *, solve_config):
        call_methods.append(solve_config.method)
        return (
            (
                np.asarray(x_guess, dtype=np.float64).copy(),
                False,
                f"{solve_config.method} failed",
                1,
                0,
                0,
                1.0,
            ),
            RuntimeError(f"{solve_config.method} failed"),
        )

    monkeypatch.setattr(Solver, "_try_solve_attempt", fake_try_solve_attempt)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = solver._solve_with_fallbacks(solver.x0.copy(), solve_config=solver.config)

    assert call_methods == ["hybr"]
    assert result[1] is False
    assert "selected method=root/hybr" in result[2]
    assert caught == []


def test_solver_uses_custom_fallback_order_without_duplicates(monkeypatch: pytest.MonkeyPatch):
    solver = _make_solver(
        enable_history=False,
        enable_warmstart=False,
        fallback_methods=("trf", "root-lm", "trf"),
    )
    call_methods: list[str] = []

    def fake_try_solve_attempt(self, x_guess, *, solve_config):
        call_methods.append(solve_config.method)
        if solve_config.method == "root-lm":
            return (
                (
                    np.asarray(x_guess, dtype=np.float64).copy(),
                    True,
                    "root-lm converged",
                    3,
                    0,
                    0,
                    1.0e-9,
                ),
                None,
            )
        return (
            (
                np.asarray(x_guess, dtype=np.float64).copy(),
                False,
                f"{solve_config.method} failed",
                1,
                0,
                0,
                1.0,
            ),
            RuntimeError(f"{solve_config.method} failed"),
        )

    monkeypatch.setattr(Solver, "_try_solve_attempt", fake_try_solve_attempt)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = solver._solve_with_fallbacks(solver.x0.copy(), solve_config=solver.config)

    assert call_methods == ["hybr", "trf", "root-lm"]
    assert result[1] is True
    assert result[2].endswith("selected method=root/root-lm")
    assert len(caught) == 2


def test_solver_solve_allows_fallback_overrides(monkeypatch: pytest.MonkeyPatch):
    solver = _make_solver(enable_history=False, enable_warmstart=False)
    captured_configs: list[SolverConfig] = []

    def fake_solve_with_fallbacks(self, x_guess, *, solve_config):
        captured_configs.append(solve_config)
        return (
            np.asarray(x_guess, dtype=np.float64).copy(),
            True,
            "ok",
            0,
            0,
            0,
            0.0,
        )

    monkeypatch.setattr(Solver, "_solve_with_fallbacks", fake_solve_with_fallbacks)

    solver.solve(enable_fallback=False, fallback_methods=("trf",), enable_verbose=False)

    assert len(captured_configs) == 1
    assert captured_configs[0].enable_fallback is False
    assert captured_configs[0].fallback_methods == ("trf",)

import warnings
from types import SimpleNamespace

import numpy as np
import pytest

from veqpy.operator.operator import Operator
from veqpy.solver import Solver, SolverConfig
from veqpy.solver.solver_config import (
    DEFAULT_COLLOCATION_FALLBACK_METHODS,
    DEFAULT_COLLOCATION_METHOD,
    DEFAULT_VARIATIONAL_FALLBACK_METHODS,
    DEFAULT_VARIATIONAL_METHOD,
    SUPPORTED_METHODS,
)


class _DummyCase:
    def __init__(self, *, Ip=np.nan):
        self.Ip = Ip

    def copy(self):
        return _DummyCase(Ip=self.Ip)


class _DummyOperator:
    def __init__(self, *, x_size: int = 3, case: _DummyCase | None = None):
        self.x_size = x_size
        self.case = _DummyCase() if case is None else case
        self.source_state_invalidations = 0
        self.active_lengths = np.full(x_size, 1, dtype=np.int64)
        self.variational_calls = 0
        self.collocation_calls = 0

    def encode_initial_state(self) -> np.ndarray:
        return np.zeros(self.x_size, dtype=np.float64)

    def coerce_x(self, x: np.ndarray) -> np.ndarray:
        arr = np.asarray(x, dtype=np.float64)
        if arr.shape != (self.x_size,):
            raise ValueError(f"Expected x to have shape ({self.x_size},), got {arr.shape}")
        return arr

    def __call__(self, x: np.ndarray) -> np.ndarray:
        self.variational_calls += 1
        return self.coerce_x(x)

    def residual_collocation(self, x: np.ndarray) -> np.ndarray:
        self.collocation_calls += 1
        x_eval = self.coerce_x(x)
        return np.concatenate((x_eval, x_eval + 1.0))

    def invalidate_source_state(self) -> None:
        self.source_state_invalidations += 1


def _attempt_result(
    x: np.ndarray,
    *,
    success: bool,
    message: str,
    residual_norm: float,
    function_evaluations: int = 1,
) -> tuple[np.ndarray, bool, str, int, int, int, float]:
    return (
        np.asarray(x, dtype=np.float64).copy(),
        success,
        message,
        function_evaluations,
        0,
        function_evaluations,
        residual_norm,
    )


def _install_fake_attempts(monkeypatch, solver: Solver, scripted_attempts):
    calls: list[tuple[str, np.ndarray]] = []
    scripted = iter(scripted_attempts)

    def fake_solve_opt_problem(x_guess: np.ndarray, *, solve_config: SolverConfig):
        calls.append((solve_config.method, np.asarray(x_guess, dtype=np.float64).copy()))
        outcome = next(scripted)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    monkeypatch.setattr(solver, "_solve_opt_problem", fake_solve_opt_problem)
    return calls


def test_operator_collocation_residual_uses_radial_quadrature_weights(monkeypatch):
    operator = object.__new__(Operator)
    operator.grid = SimpleNamespace(Nr=2, Nt=3, weights=np.array([0.25, 0.75], dtype=np.float64))
    operator.residual_surface_workspace = np.asarray(
        [[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]],
        dtype=np.float64,
    )

    monkeypatch.setattr(Operator, "coerce_x", lambda self, x: np.asarray(x, dtype=np.float64))
    monkeypatch.setattr(Operator, "stage_a_profile", lambda self, x: None)
    monkeypatch.setattr(Operator, "stage_b_geometry", lambda self: None)
    monkeypatch.setattr(Operator, "stage_c_source", lambda self: None)
    monkeypatch.setattr(Operator, "_update_residual_surface_workspace", lambda self: None)

    residual = operator.residual_collocation(np.zeros(1, dtype=np.float64))
    sqrt_weights = np.sqrt(np.array([[0.25], [0.75]], dtype=np.float64) / 3.0)
    expected = np.ravel(sqrt_weights * operator.residual_surface_workspace[0])

    assert np.allclose(residual, expected)


def test_solver_config_removes_homotopy_fields():
    config = SolverConfig()

    assert not hasattr(config, "enable_homotopy")
    assert not hasattr(config, "homotopy_truncation_tol")
    assert not hasattr(config, "homotopy_truncation_patience")
    assert "homotopy" not in str(config)


def test_solver_public_api_omits_unwired_max_residuals_and_diagnostics():
    config = SolverConfig()
    solver = Solver(operator=_DummyOperator(), config=config)
    solver.solve(enable_history=False, enable_warmstart=False, enable_fallback=False)

    assert not hasattr(config, "atol")
    assert not hasattr(config, "rtol")
    assert not hasattr(config, "root_maxfev")
    assert not hasattr(config, "root_maxiter")
    assert not hasattr(solver.result, "residual_norm_initial")
    assert "atol" not in str(config)
    assert "rtol" not in str(config)
    assert "root_maxfev" not in str(config)
    assert "root_maxiter" not in str(config)


def test_solver_config_uses_whole_word_control_names():
    config = SolverConfig(max_residual=2.5e-7, max_evaluations=1234)

    assert config.max_residual == 2.5e-7
    assert config.max_evaluations == 1234
    assert "max_residual" in str(config)
    assert "max_evaluations" in str(config)


def test_solver_config_supports_collocation_residual_form():
    config = SolverConfig(residual_form="collocation")

    assert config.method == DEFAULT_COLLOCATION_METHOD
    assert config.residual_form == "collocation"
    assert config.fallback_methods == DEFAULT_COLLOCATION_FALLBACK_METHODS
    assert "residual_form: collocation" in str(config)


def test_solver_config_uses_variational_root_default_with_single_least_squares_fallback():
    config = SolverConfig()

    assert config.method == DEFAULT_VARIATIONAL_METHOD
    assert config.residual_form == "variational"
    assert config.fallback_methods == DEFAULT_VARIATIONAL_FALLBACK_METHODS


def test_solver_config_rejects_unknown_residual_form():
    with pytest.raises(ValueError, match="Unsupported residual form"):
        SolverConfig(residual_form="pointwise")


def test_solver_rejects_collocation_with_root_method():
    solver = Solver(operator=_DummyOperator(), config=SolverConfig(method="hybr", residual_form="collocation"))

    with pytest.raises(ValueError, match="requires least_squares method"):
        solver.solve(enable_history=False)


def test_solver_rejects_root_fallbacks_for_collocation():
    solver = Solver(
        operator=_DummyOperator(),
        config=SolverConfig(method="trf", residual_form="collocation", fallback_methods=("hybr",)),
    )

    with pytest.raises(ValueError, match="unsupported fallback"):
        solver.solve(enable_history=False)


def test_solver_routes_collocation_residual_to_least_squares(monkeypatch):
    operator = _DummyOperator()
    solver = Solver(
        operator=operator,
        config=SolverConfig(method="trf", residual_form="collocation", enable_fallback=False),
    )
    observed = {}

    def fake_registered_method(fun, x0, **kwargs):
        probe = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        observed["probe_residual"] = np.asarray(fun(probe), dtype=np.float64)
        observed["method_kwarg"] = kwargs.get("method")
        return SimpleNamespace(
            x=np.zeros_like(x0),
            fun=np.zeros(6, dtype=np.float64),
            success=True,
            message="fake collocation solve",
            nfev=1,
            njev=0,
            nit=1,
        )

    monkeypatch.setitem(SUPPORTED_METHODS, "trf", fake_registered_method)

    solver.solve(enable_history=False)

    assert observed["method_kwarg"] is None
    assert np.allclose(observed["probe_residual"], [1.0, 2.0, 3.0, 2.0, 3.0, 4.0])
    assert operator.collocation_calls == 1
    assert operator.variational_calls == 0


def test_solver_accepts_collocation_least_squares_success_without_strict_variational_residual(monkeypatch):
    solver = Solver(
        operator=_DummyOperator(),
        config=SolverConfig(method="trf", residual_form="collocation", enable_fallback=False),
    )

    def fake_registered_method(fun, x0, **kwargs):
        fun(np.zeros_like(x0))
        return SimpleNamespace(
            x=np.zeros_like(x0),
            fun=np.ones(6, dtype=np.float64),
            success=True,
            message="fake ftol termination",
            nfev=1,
            njev=0,
            nit=1,
        )

    monkeypatch.setitem(SUPPORTED_METHODS, "trf", fake_registered_method)

    solver.solve(enable_history=False)

    assert solver.result.success is True
    assert solver.result.residual_norm_final > solver.config.max_residual
    assert "rejected by residual" not in solver.result.message


def test_solver_per_solve_collocation_override_uses_collocation_defaults(monkeypatch):
    solver = Solver(operator=_DummyOperator())
    calls = _install_fake_attempts(
        monkeypatch,
        solver,
        [
            _attempt_result(np.zeros(3, dtype=np.float64), success=True, message="trf converged", residual_norm=0.0),
        ],
    )

    solver.solve(residual_form="collocation", enable_history=False)

    assert [method for method, _ in calls] == ["trf"]
    assert np.allclose(calls[0][1], np.zeros(3, dtype=np.float64))


def test_solver_explicit_guess_retries_same_method_after_reset(monkeypatch):
    solver = Solver(operator=_DummyOperator())
    warm_guess = np.array([1.0, -2.0, 0.5], dtype=np.float64)
    cold_guess = np.zeros_like(warm_guess)
    calls = _install_fake_attempts(
        monkeypatch,
        solver,
        [
            _attempt_result(warm_guess, success=False, message="warm attempt failed", residual_norm=1.0),
            _attempt_result(cold_guess, success=True, message="reset attempt converged", residual_norm=0.0),
        ],
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        x_final = solver.solve(
            x0=warm_guess,
            enable_history=False,
            fallback_methods=("lm", "trf"),
        )

    assert [method for method, _ in calls] == ["hybr", "hybr"]
    assert np.allclose(calls[0][1], warm_guess)
    assert np.allclose(calls[1][1], cold_guess)
    assert np.allclose(x_final, cold_guess)
    assert np.allclose(solver.result.x0, warm_guess)
    assert solver.result.success is True


def test_solver_warmstart_failure_falls_back_from_cold_reset_state(monkeypatch):
    solver = Solver(operator=_DummyOperator())
    warm_guess = np.array([0.25, 0.0, -0.75], dtype=np.float64)
    cold_guess = np.zeros_like(warm_guess)
    solver.x0 = warm_guess.copy()
    calls = _install_fake_attempts(
        monkeypatch,
        solver,
        [
            _attempt_result(warm_guess, success=False, message="warm attempt failed", residual_norm=5.0),
            _attempt_result(cold_guess, success=False, message="reset attempt failed", residual_norm=4.0),
            _attempt_result(cold_guess, success=False, message="lm failed", residual_norm=3.0),
            _attempt_result(
                np.array([3.0, 2.0, 1.0], dtype=np.float64),
                success=True,
                message="trf converged",
                residual_norm=0.0,
            ),
        ],
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        x_final = solver.solve(
            enable_history=False,
            fallback_methods=("lm", "trf"),
        )

    assert [method for method, _ in calls] == ["hybr", "hybr", "lm", "trf"]
    assert np.allclose(calls[0][1], warm_guess)
    for _, guess in calls[1:]:
        assert np.allclose(guess, cold_guess)
    assert np.allclose(x_final, [3.0, 2.0, 1.0])
    assert "selected method=least_squares/trf [cold-fallback]" in solver.result.message
    assert solver.result.success is True


def test_solver_cold_start_skips_reset_retry(monkeypatch):
    solver = Solver(operator=_DummyOperator())
    cold_guess = np.zeros(3, dtype=np.float64)
    calls = _install_fake_attempts(
        monkeypatch,
        solver,
        [
            _attempt_result(cold_guess, success=False, message="cold attempt failed", residual_norm=2.0),
            _attempt_result(cold_guess, success=True, message="lm converged", residual_norm=0.0),
        ],
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        solver.solve(enable_history=False, fallback_methods=("lm",))

    assert [method for method, _ in calls] == ["hybr", "lm"]
    for _, guess in calls:
        assert np.allclose(guess, cold_guess)


def test_solver_config_rejects_root_lm():
    try:
        SolverConfig(method="root-lm")
    except ValueError as exc:
        assert "Unsupported solver method" in str(exc)
        assert "root-lm" in str(exc)
    else:
        raise AssertionError("root-lm should no longer be supported")


def test_solver_config_rejects_krylov():
    try:
        SolverConfig(method="krylov")
    except ValueError as exc:
        assert "Unsupported solver method" in str(exc)
        assert "krylov" in str(exc)
    else:
        raise AssertionError("krylov should no longer be supported")


def test_solver_config_rejects_broyden1():
    try:
        SolverConfig(method="broyden1")
    except ValueError as exc:
        assert "Unsupported solver method" in str(exc)
        assert "broyden1" in str(exc)
    else:
        raise AssertionError("broyden1 should no longer be supported")


def test_solver_config_rejects_broyden2():
    try:
        SolverConfig(method="broyden2")
    except ValueError as exc:
        assert "Unsupported solver method" in str(exc)
        assert "broyden2" in str(exc)
    else:
        raise AssertionError("broyden2 should no longer be supported")


def test_solver_config_rejects_dogbox():
    try:
        SolverConfig(method="dogbox")
    except ValueError as exc:
        assert "Unsupported solver method" in str(exc)
        assert "dogbox" in str(exc)
    else:
        raise AssertionError("dogbox should no longer be supported")

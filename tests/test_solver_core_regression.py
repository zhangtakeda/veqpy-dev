import warnings

import numpy as np

from veqpy.solver import Solver, SolverConfig


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

    def encode_initial_state(self) -> np.ndarray:
        return np.zeros(self.x_size, dtype=np.float64)

    def coerce_x(self, x: np.ndarray) -> np.ndarray:
        arr = np.asarray(x, dtype=np.float64)
        if arr.shape != (self.x_size,):
            raise ValueError(f"Expected x to have shape ({self.x_size},), got {arr.shape}")
        return arr

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.coerce_x(x)

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

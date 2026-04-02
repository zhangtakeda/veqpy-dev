import warnings

import numpy as np

from veqpy.solver import Solver, SolverConfig


class _DummyCase:
    def copy(self):
        return _DummyCase()


class _DummyOperator:
    def __init__(self, *, x_size: int = 3):
        self.x_size = x_size
        self.case = _DummyCase()
        self.source_state_invalidations = 0

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
    nfev: int = 1,
) -> tuple[np.ndarray, bool, str, int, int, int, float]:
    return (np.asarray(x, dtype=np.float64).copy(), success, message, nfev, 0, nfev, residual_norm)


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
            fallback_methods=("root-lm", "trf"),
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
            _attempt_result(cold_guess, success=False, message="root-lm failed", residual_norm=3.0),
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
            fallback_methods=("root-lm", "trf"),
        )

    assert [method for method, _ in calls] == ["hybr", "hybr", "root-lm", "trf"]
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
            _attempt_result(cold_guess, success=True, message="root-lm converged", residual_norm=0.0),
        ],
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        solver.solve(enable_history=False, fallback_methods=("root-lm",))

    assert [method for method, _ in calls] == ["hybr", "root-lm"]
    for _, guess in calls:
        assert np.allclose(guess, cold_guess)

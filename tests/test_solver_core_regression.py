import warnings
from types import SimpleNamespace

import numpy as np

import veqpy.solver.solver as solver_module
from veqpy.solver.solver import Solver
from veqpy.solver.solver_config import SolverConfig


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


class _ScaledResidualOperator(_DummyOperator):
    def __init__(self):
        super().__init__(x_size=5)
        self.active_lengths = np.array([2, 3], dtype=np.int64)
        self.call_count = 0

    def __call__(self, x: np.ndarray) -> np.ndarray:
        del x
        self.call_count += 1
        return np.array([10.0, 30.0, 4.0, 5.0, 6.0], dtype=np.float64)


def test_hybr_uses_block_scaled_residual_with_unit_factor(monkeypatch):
    solver = Solver(
        operator=_ScaledResidualOperator(),
        config=SolverConfig(
            method="hybr",
            enable_fallback=False,
            enable_warmstart=False,
            enable_history=False,
        ),
    )
    captured: dict[str, object] = {}

    def fake_root(fun, x0, *, method, tol, options):
        captured["fun0"] = np.asarray(fun(np.asarray(x0, dtype=np.float64)), dtype=np.float64)
        captured["method"] = method
        captured["tol"] = tol
        captured["options"] = dict(options)
        return SimpleNamespace(
            x=np.asarray(x0, dtype=np.float64).copy(),
            success=False,
            message="mock failure",
            nfev=1,
            njev=0,
            nit=1,
            fun=np.asarray(captured["fun0"], dtype=np.float64),
        )

    monkeypatch.setattr(solver_module, "root", fake_root)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        solver.solve(enable_fallback=False, enable_history=False, enable_warmstart=False)

    expected = np.array(
        [
            10.0 / np.sqrt((10.0**2 + 30.0**2) / 2.0),
            30.0 / np.sqrt((10.0**2 + 30.0**2) / 2.0),
            4.0 / np.sqrt((4.0**2 + 5.0**2 + 6.0**2) / 3.0),
            5.0 / np.sqrt((4.0**2 + 5.0**2 + 6.0**2) / 3.0),
            6.0 / np.sqrt((4.0**2 + 5.0**2 + 6.0**2) / 3.0),
        ],
        dtype=np.float64,
    )
    assert captured["method"] == "hybr"
    assert captured["tol"] == solver.config.max_residual
    assert np.allclose(captured["fun0"], expected)
    assert captured["options"]["maxfev"] == solver.config.max_evaluations
    assert captured["options"]["factor"] == 1.0
    assert solver.operator.call_count == 1


def test_hybr_scaled_root_reports_raw_residual_norm(monkeypatch):
    solver = Solver(
        operator=_ScaledResidualOperator(),
        config=SolverConfig(
            method="hybr",
            enable_fallback=False,
            enable_warmstart=False,
            enable_history=False,
        ),
    )

    def fake_root(fun, x0, *, method, tol, options):
        scaled_fun = np.asarray(fun(np.asarray(x0, dtype=np.float64)), dtype=np.float64)
        return SimpleNamespace(
            x=np.asarray(x0, dtype=np.float64).copy(),
            success=True,
            message="mock success",
            nfev=1,
            njev=0,
            nit=1,
            fun=scaled_fun,
        )

    monkeypatch.setattr(solver_module, "root", fake_root)

    x_opt, success, _, _, _, _, residual_norm = solver._solve_opt_problem(
        np.zeros(5, dtype=np.float64),
        solve_config=solver.config,
    )

    expected_raw_norm = float(np.linalg.norm(np.array([10.0, 30.0, 4.0, 5.0, 6.0], dtype=np.float64)))
    assert success is True
    assert np.allclose(x_opt, 0.0)
    assert residual_norm == expected_raw_norm


def test_lm_uses_asinh_block_scaled_residual(monkeypatch):
    solver = Solver(
        operator=_ScaledResidualOperator(),
        config=SolverConfig(
            method="lm",
            enable_fallback=False,
            enable_warmstart=False,
            enable_history=False,
        ),
    )
    captured: dict[str, object] = {}

    def fake_least_squares(fun, x0, **kwargs):
        captured["fun0"] = np.asarray(fun(np.asarray(x0, dtype=np.float64)), dtype=np.float64)
        captured["kwargs"] = dict(kwargs)
        return SimpleNamespace(
            x=np.asarray(x0, dtype=np.float64).copy(),
            success=False,
            message="mock failure",
            nfev=1,
            njev=0,
            nit=1,
            fun=np.asarray(captured["fun0"], dtype=np.float64),
        )

    monkeypatch.setattr(solver_module, "least_squares", fake_least_squares)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        solver.solve(enable_fallback=False, enable_history=False, enable_warmstart=False)

    expected_linear = np.array(
        [
            10.0 / np.sqrt((10.0**2 + 30.0**2) / 2.0),
            30.0 / np.sqrt((10.0**2 + 30.0**2) / 2.0),
            4.0 / np.sqrt((4.0**2 + 5.0**2 + 6.0**2) / 3.0),
            5.0 / np.sqrt((4.0**2 + 5.0**2 + 6.0**2) / 3.0),
            6.0 / np.sqrt((4.0**2 + 5.0**2 + 6.0**2) / 3.0),
        ],
        dtype=np.float64,
    )
    assert np.allclose(captured["fun0"], np.arcsinh(expected_linear))
    assert captured["kwargs"]["method"] == "lm"
    assert captured["kwargs"]["ftol"] == solver.config.max_residual
    assert captured["kwargs"]["xtol"] == solver.config.max_residual
    assert captured["kwargs"]["gtol"] == solver.config.max_residual
    assert captured["kwargs"]["max_nfev"] == solver.config.max_evaluations
    assert captured["kwargs"]["x_scale"] == 1.0


def test_lm_requires_raw_residual_acceptance(monkeypatch):
    solver = Solver(
        operator=_ScaledResidualOperator(),
        config=SolverConfig(
            method="lm",
            enable_fallback=False,
            enable_warmstart=False,
            enable_history=False,
        ),
    )

    def fake_least_squares(fun, x0, **kwargs):
        transformed = np.asarray(fun(np.asarray(x0, dtype=np.float64)), dtype=np.float64)
        return SimpleNamespace(
            x=np.asarray(x0, dtype=np.float64).copy(),
            success=True,
            message="mock success",
            nfev=1,
            njev=0,
            nit=1,
            fun=transformed,
        )

    monkeypatch.setattr(solver_module, "least_squares", fake_least_squares)

    x_opt, success, message, _, _, _, residual_norm = solver._solve_opt_problem(
        np.zeros(5, dtype=np.float64),
        solve_config=solver.config,
    )

    expected_raw_norm = float(np.linalg.norm(np.array([10.0, 30.0, 4.0, 5.0, 6.0], dtype=np.float64)))
    assert np.allclose(x_opt, 0.0)
    assert success is False
    assert residual_norm == expected_raw_norm
    assert "rejected by residual" in message


def test_trf_robust_loss_depends_on_residual_scale_not_ip(monkeypatch):
    operator = _ScaledResidualOperator()
    operator.case = _DummyCase(Ip=1.0)
    solver = Solver(
        operator=operator,
        config=SolverConfig(
            method="trf",
            enable_fallback=False,
            enable_warmstart=False,
            enable_history=False,
        ),
    )
    captured: dict[str, object] = {}

    def fake_least_squares(fun, x0, **kwargs):
        captured["fun0"] = np.asarray(fun(np.asarray(x0, dtype=np.float64)), dtype=np.float64)
        captured["kwargs"] = dict(kwargs)
        return SimpleNamespace(
            x=np.asarray(x0, dtype=np.float64).copy(),
            success=False,
            message="mock failure",
            nfev=1,
            njev=0,
            nit=1,
            fun=np.asarray(captured["fun0"], dtype=np.float64),
        )

    monkeypatch.setattr(solver_module, "least_squares", fake_least_squares)
    monkeypatch.setattr(solver_module, "_trf_robust_block_rms_threshold", lambda: 10.0)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        solver.solve(enable_fallback=False, enable_history=False, enable_warmstart=False)

    assert captured["kwargs"]["loss"] == "cauchy"
    assert np.isclose(
        captured["kwargs"]["f_scale"],
        np.linalg.norm(np.array([10.0, 30.0, 4.0, 5.0, 6.0], dtype=np.float64)) / np.sqrt(5.0),
    )


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

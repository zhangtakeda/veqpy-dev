from __future__ import annotations

import numpy as np

from veqpy.solver import Solver, SolverConfig


class _PlanOnlyOperator:
    def coerce_x(self, x: np.ndarray) -> np.ndarray:
        return np.asarray(x, dtype=np.float64)


class _ResidualOnlyOperator:
    def coerce_x(self, x: np.ndarray) -> np.ndarray:
        return np.asarray(x, dtype=np.float64)

    def residual_var(self, x: np.ndarray) -> np.ndarray:
        x_eval = self.coerce_x(x)
        return np.asarray([x_eval[0] - 1.0, 2.0 * x_eval[1]], dtype=np.float64)

    def residual_collocation(self, x: np.ndarray) -> np.ndarray:
        x_eval = self.coerce_x(x)
        return np.asarray(
            [3.0 * x_eval[0], x_eval[1] - 2.0, x_eval[0] + x_eval[1]],
            dtype=np.float64,
        )


def test_fallback_attempts_reuse_initial_guess_without_reset() -> None:
    solver = Solver.__new__(Solver)
    solver.operator = _PlanOnlyOperator()
    x_initial = np.asarray([1.0, -2.0, 0.5], dtype=np.float64)
    config = SolverConfig(
        method="hybr",
        enable_fallback=True,
        fallback_methods=("lm", "trf"),
    )

    plans = solver._build_attempt_plans(
        x_initial,
        solve_config=config,
        residual_kind="variational",
        x0_was_provided=False,
    )

    labels = [label for label, _, _ in plans]
    assert labels == [
        "root/hybr [warm-start]",
        "least_squares/lm [warm-fallback]",
        "least_squares/trf [warm-fallback]",
    ]
    for _, guess, _ in plans:
        np.testing.assert_array_equal(guess, x_initial)
        assert guess is not x_initial


def test_collocation_weight_selects_polish_residual_kind() -> None:
    solver = Solver.__new__(Solver)

    assert solver._collocation_residual_kind(SolverConfig(collocation_weight=0.0)) == "variational"
    assert (
        solver._collocation_residual_kind(SolverConfig(collocation_weight=0.5))
        == "blended_collocation"
    )
    assert solver._collocation_residual_kind(SolverConfig(collocation_weight=1.0)) == "collocation"


def test_blended_collocation_residual_stacks_rms_weighted_blocks() -> None:
    solver = Solver.__new__(Solver)
    solver.operator = _ResidualOnlyOperator()
    solver.config = SolverConfig()
    config = SolverConfig(collocation_weight=0.25, max_residual=1.0e-9)
    x_reference = np.asarray([1.0, 1.0], dtype=np.float64)
    x_eval = np.asarray([2.0, -1.0], dtype=np.float64)

    residual = solver._residual_function_for(
        "blended_collocation",
        solve_config=config,
        x_reference=x_reference,
    )(x_eval)

    assert residual.shape == (5,)
    assert np.all(np.isfinite(residual))
    np.testing.assert_allclose(
        residual[:2],
        np.sqrt(0.75) * (x_eval - x_reference) / np.sqrt(2.0),
    )
    np.testing.assert_allclose(
        residual[2:],
        np.sqrt(0.25) * solver.operator.residual_collocation(x_eval)
        / (np.sqrt(3.0) * np.sqrt(np.mean(solver.operator.residual_collocation(x_reference) ** 2))),
    )


def test_collocation_weight_validates_unit_interval() -> None:
    for value in (-1.0e-12, 1.0 + 1.0e-12, float("inf")):
        try:
            SolverConfig(collocation_weight=value)
        except ValueError as exc:
            assert "collocation_weight" in str(exc)
        else:
            raise AssertionError(f"collocation_weight={value!r} should be invalid")

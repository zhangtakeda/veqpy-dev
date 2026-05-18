from __future__ import annotations

import numpy as np

from veqpy.solver import Solver, SolverConfig


class _PlanOnlyOperator:
    def coerce_x(self, x: np.ndarray) -> np.ndarray:
        return np.asarray(x, dtype=np.float64)


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


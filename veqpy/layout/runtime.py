"""
Executable runtime layouts.

Layout objects describe fixed stage structure and own the bound callables that execute
that structure.  Workspace objects own memory; layouts execute against already-bound
workspace arrays captured by their callables.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Self

import numpy as np


@dataclass(slots=True)
class ProfileLayout:
    """Executable profile stage layout."""

    run_stage: Callable[[np.ndarray], None]
    run_postprocess: Callable[[], None]

    def run(self, x: np.ndarray) -> None:
        self.run_stage(x)
        self.run_postprocess()


@dataclass(slots=True)
class GeometryLayout:
    """Executable geometry stage layout."""

    run_stage: Callable[[], None]

    def run(self) -> None:
        self.run_stage()


@dataclass(slots=True)
class SourceLayout:
    """Executable source stage layout."""

    run_eval: Callable
    run_stage: Callable[[], tuple[float, float]]

    def run(self) -> tuple[float, float]:
        return self.run_stage()


@dataclass(slots=True)
class ResidualLayout:
    """Executable residual stage layout."""

    run_full_into: Callable[[np.ndarray], None]
    run_full: Callable[[], np.ndarray]
    run_fused_into: Callable[[np.ndarray, np.ndarray], None]
    run_fused: Callable[[np.ndarray], np.ndarray]

    def run_into(self, out: np.ndarray) -> None:
        self.run_full_into(out)

    def run(self) -> np.ndarray:
        return self.run_full()

    def run_fused_residual_into(self, x: np.ndarray, out: np.ndarray) -> None:
        self.run_fused_into(x, out)

    def run_fused_residual(self, x: np.ndarray) -> np.ndarray:
        return self.run_fused(x)


@dataclass(slots=True)
class OperatorLayout:
    """Executable operator layout composed from fixed stage layouts."""

    profile: ProfileLayout
    geometry: GeometryLayout
    source: SourceLayout
    residual: ResidualLayout
    run_collocation_runner_into: Callable[[np.ndarray, np.ndarray], None]

    @classmethod
    def empty(cls, x_size: int) -> Self:
        """Create a no-op layout placeholder before runtime arrays are bound."""

        def residual_full() -> np.ndarray:
            return np.zeros(x_size, dtype=np.float64)

        def fused_residual(x: np.ndarray) -> np.ndarray:
            del x
            return np.zeros(x_size, dtype=np.float64)

        return cls.from_callables(
            profile_stage_runner=lambda x: None,
            profile_postprocess_runner=lambda: None,
            geometry_stage_runner=lambda: None,
            source_eval_runner=lambda *args: (0.0, 0.0),
            source_stage_runner=lambda: (0.0, 0.0),
            residual_full_stage_runner_into=lambda out: out.fill(0.0),
            residual_full_stage_runner=residual_full,
            fused_residual_runner_into=lambda x_eval, out: out.fill(0.0),
            fused_residual_runner=fused_residual,
            collocation_runner_into=lambda x_eval, out: out.fill(0.0),
        )

    @classmethod
    def from_callables(
        cls,
        *,
        profile_stage_runner: Callable[[np.ndarray], None],
        profile_postprocess_runner: Callable[[], None],
        geometry_stage_runner: Callable[[], None],
        source_eval_runner: Callable,
        source_stage_runner: Callable[[], tuple[float, float]],
        residual_full_stage_runner_into: Callable[[np.ndarray], None],
        residual_full_stage_runner: Callable[[], np.ndarray],
        fused_residual_runner_into: Callable[[np.ndarray, np.ndarray], None],
        fused_residual_runner: Callable[[np.ndarray], np.ndarray],
        collocation_runner_into: Callable[[np.ndarray, np.ndarray], None],
    ) -> Self:
        return cls(
            profile=ProfileLayout(
                run_stage=profile_stage_runner,
                run_postprocess=profile_postprocess_runner,
            ),
            geometry=GeometryLayout(run_stage=geometry_stage_runner),
            source=SourceLayout(
                run_eval=source_eval_runner,
                run_stage=source_stage_runner,
            ),
            residual=ResidualLayout(
                run_full_into=residual_full_stage_runner_into,
                run_full=residual_full_stage_runner,
                run_fused_into=fused_residual_runner_into,
                run_fused=fused_residual_runner,
            ),
            run_collocation_runner_into=collocation_runner_into,
        )

    def run_profile(self, x: np.ndarray) -> None:
        self.profile.run(x)

    def run_geometry(self) -> None:
        self.geometry.run()

    def run_source(self) -> tuple[float, float]:
        alpha1, alpha2 = self.source.run()
        return float(alpha1), float(alpha2)

    def run_residual_into(self, out: np.ndarray) -> None:
        self.residual.run_into(out)

    def run_fused_residual_into(self, x: np.ndarray, out: np.ndarray) -> None:
        self.residual.run_fused_residual_into(x, out)

    def run_collocation_into(self, x: np.ndarray, out: np.ndarray) -> None:
        self.run_collocation_runner_into(x, out)

"""Compare demo solves across residual forms and registered solver methods.

The script writes two two-row figures:

- row 1: variational residual
- row 2: collocation residual
- columns: methods registered in ``SUPPORTED_METHODS``

Each successful panel renders the same flux-surface view used as panel (a) in the
minimal demo, with the subplot title showing solve time and final solver
residual averaged over repeated solves after warmup. Unsupported
residual/method pairs are marked in-place.

The second figure renders the force-balance residual field ``G`` on the solve
grid used during the nonlinear solve.

Run:

    .venv/bin/python tests/demo_residual_form_methods.py

Note: the first run may be slower due to Numba JIT compilation.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from demo import (
    MU0,
    SOURCE_SAMPLE_COUNT,
    build_demo_boundary,
    build_surface_from_psin,
    compute_rz_limits,
    pf_reference_profiles,
    plot_equilibrium_surfaces,
    style_surface_axis,
)

from veqpy.model import Grid
from veqpy.operator import Operator, OperatorCase
from veqpy.solver import Solver, SolverConfig
from veqpy.solver.solver_config import LEAST_SQUARES_METHODS, SUPPORTED_METHODS

RESIDUAL_FORMS = ("variational", "collocation")
METHODS = tuple(SUPPORTED_METHODS)
WARMUP_RUNS = 3
REPEAT_RUNS = 5


@dataclass(frozen=True, slots=True)
class SolvePanel:
    residual_form: str
    method: str
    success: bool
    elapsed_ms_mean: float | None = None
    elapsed_ms_std: float | None = None
    residual_norm_mean: float | None = None
    residual_norm_std: float | None = None
    function_evaluations_mean: float | None = None
    function_evaluations_std: float | None = None
    successful_runs: int = 0
    measured_runs: int = 0
    plot_equilibrium: object | None = None
    solve_equilibrium: object | None = None
    error: str | None = None


def ensure_output_dir() -> Path:
    outdir = Path(__file__).resolve().parent / "demo"
    outdir.mkdir(parents=True, exist_ok=True)
    return outdir


def build_demo_case() -> OperatorCase:
    psin = np.linspace(0.0, 1.0, SOURCE_SAMPLE_COUNT, dtype=np.float64)
    current_input, heat_input = pf_reference_profiles(psin)
    return OperatorCase(
        route="PF",
        coordinate="psin",
        profile_coeffs={
            "psin": 5,
            "h": 3,
            "k": 5,
            "s1": 3,
        },
        boundary=build_demo_boundary(),
        heat_input=heat_input,
        current_input=current_input,
        Ip=MU0 * 3.0e6,
    )


def is_supported_pair(*, residual_form: str, method: str) -> bool:
    if residual_form == "collocation":
        return method in LEAST_SQUARES_METHODS
    return method in SUPPORTED_METHODS


def solve_demo_panel(*, residual_form: str, method: str, plot_grid: Grid) -> SolvePanel:
    if not is_supported_pair(residual_form=residual_form, method=method):
        return SolvePanel(
            residual_form=residual_form,
            method=method,
            success=False,
            error="unsupported",
        )

    try:
        solve_grid = Grid(Nr=16, Nt=16, scheme="legendre")
        elapsed_ms: list[float] = []
        residual_norms: list[float] = []
        function_evaluations: list[float] = []
        successes: list[bool] = []
        plot_equilibrium = None
        solve_equilibrium = None
        for run_index in range(WARMUP_RUNS + REPEAT_RUNS):
            solver = Solver(
                operator=Operator(grid=solve_grid, case=build_demo_case()),
                config=SolverConfig(
                    method=method,
                    residual_form=residual_form,
                    enable_fallback=False,
                    enable_warmstart=False,
                    enable_verbose=False,
                    enable_history=False,
                ),
            )
            solver.solve()
            result = solver.result
            if result is None:
                raise RuntimeError("solver.result is unavailable after solve")
            if run_index >= WARMUP_RUNS:
                elapsed_ms.append(float(result.elapsed / 1000.0))
                residual_norms.append(float(result.residual_norm_final))
                function_evaluations.append(float(result.function_evaluations))
                successes.append(bool(result.success))
                solve_equilibrium = solver.build_equilibrium()
                plot_equilibrium = solve_equilibrium.resample(grid=plot_grid)
        elapsed_arr = np.asarray(elapsed_ms, dtype=np.float64)
        residual_arr = np.asarray(residual_norms, dtype=np.float64)
        nfev_arr = np.asarray(function_evaluations, dtype=np.float64)
        return SolvePanel(
            residual_form=residual_form,
            method=method,
            success=bool(successes and all(successes)),
            elapsed_ms_mean=float(np.mean(elapsed_arr)),
            elapsed_ms_std=float(np.std(elapsed_arr)),
            residual_norm_mean=float(np.mean(residual_arr)),
            residual_norm_std=float(np.std(residual_arr)),
            function_evaluations_mean=float(np.mean(nfev_arr)),
            function_evaluations_std=float(np.std(nfev_arr)),
            successful_runs=int(np.count_nonzero(successes)),
            measured_runs=len(successes),
            plot_equilibrium=plot_equilibrium,
            solve_equilibrium=solve_equilibrium,
        )
    except Exception as exc:
        return SolvePanel(
            residual_form=residual_form,
            method=method,
            success=False,
            error=f"{type(exc).__name__}: {exc}",
        )


def collect_panels() -> list[SolvePanel]:
    plot_grid = Grid(Nr=128, Nt=256, scheme="uniform", L_max=20, M_max=20)
    return [
        solve_demo_panel(residual_form=residual_form, method=method, plot_grid=plot_grid)
        for residual_form in RESIDUAL_FORMS
        for method in METHODS
    ]


def compute_shared_limits(panels: list[SolvePanel]) -> tuple[tuple[float, float], tuple[float, float]]:
    curves = [
        build_surface_from_psin(panel.plot_equilibrium, 1.0)
        for panel in panels
        if panel.plot_equilibrium is not None
    ]
    if not curves:
        return ((0.0, 1.0), (-0.5, 0.5))
    return compute_rz_limits(curves)


def panel_title(panel: SolvePanel) -> str:
    label = f"{panel.method}"
    if panel.elapsed_ms_mean is not None and panel.residual_norm_mean is not None:
        status = "ok" if panel.success else "not converged"
        return (
            f"{label}\n"
            f"{panel.elapsed_ms_mean:.1f}±{panel.elapsed_ms_std:.1f} ms, "
            f"nfev={panel.function_evaluations_mean:.1f}±{panel.function_evaluations_std:.1f}\n"
            f"res={panel.residual_norm_mean:.2e}, "
            f"{status} ({panel.successful_runs}/{panel.measured_runs})"
        )
    if panel.error == "unsupported":
        return f"{label}\nunsupported for collocation"
    return f"{label}\nfailed\n{panel.error}"


def render_surface_figure(panels: list[SolvePanel], figure_path: Path) -> None:
    panel_by_key = {(panel.residual_form, panel.method): panel for panel in panels}
    rz_limits = compute_shared_limits(panels)

    fig, axes = plt.subplots(
        len(RESIDUAL_FORMS),
        len(METHODS),
        figsize=(4.2 * len(METHODS), 7.8),
        squeeze=False,
        constrained_layout=True,
    )
    for row, residual_form in enumerate(RESIDUAL_FORMS):
        for col, method in enumerate(METHODS):
            ax = axes[row, col]
            panel = panel_by_key[(residual_form, method)]
            if panel.plot_equilibrium is not None:
                plot_equilibrium_surfaces(ax, panel.plot_equilibrium)
                style_surface_axis(ax, title=panel_title(panel), rz_limits=rz_limits)
            else:
                ax.set_title(panel_title(panel))
                ax.set_axis_off()
            if col == 0:
                ax.text(
                    -0.18,
                    0.5,
                    residual_form,
                    transform=ax.transAxes,
                    rotation=90,
                    va="center",
                    ha="center",
                    fontsize=13,
                    fontweight="bold",
                )

    fig.suptitle("veqpy demo: residual form × solver method", fontsize=16)
    fig.savefig(figure_path, dpi=220)
    plt.close(fig)


def _build_residual_plot_arrays(equilibrium) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    R = np.asarray(equilibrium.geometry.R, dtype=np.float64)
    Z = np.asarray(equilibrium.geometry.Z, dtype=np.float64)
    G = np.asarray(equilibrium.G, dtype=np.float64)
    return (
        np.hstack([R, R[:, :1]]),
        np.hstack([Z, Z[:, :1]]),
        np.hstack([G, G[:, :1]]),
    )


def compute_shared_residual_scale(panels: list[SolvePanel]) -> float:
    finite_abs_chunks = [
        np.abs(np.asarray(panel.solve_equilibrium.G, dtype=np.float64)[np.isfinite(panel.solve_equilibrium.G)])
        for panel in panels
        if panel.solve_equilibrium is not None
    ]
    if not finite_abs_chunks:
        return 1.0
    finite_abs = np.concatenate(finite_abs_chunks)
    if finite_abs.size == 0:
        return 1.0
    vmax = float(np.quantile(finite_abs, 0.99))
    if not np.isfinite(vmax) or vmax <= 0.0:
        return 1.0
    return vmax


def style_residual_axis(
    ax: plt.Axes,
    *,
    title: str,
    rz_limits: tuple[tuple[float, float], tuple[float, float]],
) -> None:
    ax.set_title(title)
    ax.set_xlabel("R [m]")
    ax.set_ylabel("Z [m]")
    ax.set_xlim(*rz_limits[0])
    ax.set_ylim(*rz_limits[1])
    ax.set_aspect("equal", adjustable="box")
    ax.tick_params(direction="in", top=True, right=True)


def render_residual_figure(panels: list[SolvePanel], figure_path: Path) -> None:
    panel_by_key = {(panel.residual_form, panel.method): panel for panel in panels}
    rz_limits = compute_shared_limits(panels)
    vmax = compute_shared_residual_scale(panels)
    norm = mcolors.TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
    cmap = plt.get_cmap("coolwarm")

    fig, axes = plt.subplots(
        len(RESIDUAL_FORMS),
        len(METHODS),
        figsize=(4.2 * len(METHODS), 7.8),
        squeeze=False,
        constrained_layout=True,
    )
    mappable = None
    for row, residual_form in enumerate(RESIDUAL_FORMS):
        for col, method in enumerate(METHODS):
            ax = axes[row, col]
            panel = panel_by_key[(residual_form, method)]
            if panel.solve_equilibrium is not None:
                R_plot, Z_plot, G_plot = _build_residual_plot_arrays(panel.solve_equilibrium)
                mappable = ax.contourf(
                    R_plot,
                    Z_plot,
                    G_plot,
                    levels=np.linspace(-vmax, vmax, 129),
                    cmap=cmap,
                    norm=norm,
                )
                style_residual_axis(ax, title=panel_title(panel), rz_limits=rz_limits)
            else:
                ax.set_title(panel_title(panel))
                ax.set_axis_off()
            if col == 0:
                ax.text(
                    -0.18,
                    0.5,
                    residual_form,
                    transform=ax.transAxes,
                    rotation=90,
                    va="center",
                    ha="center",
                    fontsize=13,
                    fontweight="bold",
                )

    if mappable is not None:
        cbar = fig.colorbar(mappable, ax=axes, location="right", shrink=0.95, pad=0.02)
        cbar.set_label(r"$G$")
    fig.suptitle("veqpy demo: force-balance residual G on solve grid", fontsize=16)
    fig.savefig(figure_path, dpi=220)
    plt.close(fig)


def print_summary(panels: list[SolvePanel], surface_figure_path: Path, residual_figure_path: Path) -> None:
    print(f"Saved surface figure : {surface_figure_path}")
    print(f"Saved residual figure: {residual_figure_path}")
    print(f"Warmup runs: {WARMUP_RUNS}; measured repeats: {REPEAT_RUNS}")
    print(
        "residual_form | method | success | successful_runs | elapsed_ms_mean | elapsed_ms_std | "
        "nfev_mean | nfev_std | residual_norm_mean | residual_norm_std | error"
    )
    for panel in panels:
        elapsed = "nan" if panel.elapsed_ms_mean is None else f"{panel.elapsed_ms_mean:.3f}"
        elapsed_std = "nan" if panel.elapsed_ms_std is None else f"{panel.elapsed_ms_std:.3f}"
        nfev = "nan" if panel.function_evaluations_mean is None else f"{panel.function_evaluations_mean:.3f}"
        nfev_std = "nan" if panel.function_evaluations_std is None else f"{panel.function_evaluations_std:.3f}"
        residual = "nan" if panel.residual_norm_mean is None else f"{panel.residual_norm_mean:.6e}"
        residual_std = "nan" if panel.residual_norm_std is None else f"{panel.residual_norm_std:.6e}"
        error = "" if panel.error is None else panel.error
        print(
            f"{panel.residual_form} | {panel.method} | {panel.success} | "
            f"{panel.successful_runs}/{panel.measured_runs} | {elapsed} | {elapsed_std} | "
            f"{nfev} | {nfev_std} | {residual} | {residual_std} | {error}"
        )


def main() -> None:
    outdir = ensure_output_dir()
    surface_figure_path = outdir / "demo_residual_form_methods.png"
    residual_figure_path = outdir / "demo_residual_form_methods_residual.png"
    panels = collect_panels()
    render_surface_figure(panels, surface_figure_path)
    render_residual_figure(panels, residual_figure_path)
    print_summary(panels, surface_figure_path, residual_figure_path)


if __name__ == "__main__":
    main()

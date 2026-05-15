"""Downstream heat-transport geometry check.

This script is a deliberately small downstream-operator test.  It compares
the Table-5 reduced VEQ configurations against a target curve.  The transport problem
is not intended as predictive modelling; it only propagates geometry
differences through a fixed 1-D steady heat-diffusion operator.
"""

from __future__ import annotations

import importlib.util
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.config import DOUBLE_COLUMN_WIDTH, SAVE_DPI, SAVE_TRANSPARENT, apply_plot_style  # noqa: E402
from scripts.config import (  # noqa: E402
    AXIS_LABEL_FONT_SIZE,
    LEGEND_FONT_SIZE,
    PLOT_LABEL_RIGHT,
    PLOT_LABEL_TOP,
    PLOT_TICK_BOTTOM,
    PLOT_TICK_DIRECTION,
    PLOT_TICK_LEFT,
    PLOT_TICK_RIGHT,
    PLOT_TICK_TOP,
    TICK_LABEL_FONT_SIZE,
    TITLE_FONT_SIZE,
    scaled_font_size,
)

MU0 = 4.0e-7 * np.pi

OUT_PNG = REPO_ROOT / "figures" / "09.png"
OUT_PDF = REPO_ROOT / "figures" / "09.pdf"

CASE_KEYS = ("solovev", "chease", "efit")
CASE_LABELS = {
    "solovev": "D-shape",
    "chease": "H-mode",
    "efit": "X-point",
}
REFERENCE_LABELS = {
    "solovev": "Analytic",
    "chease": "GEQDSK",
    "efit": "GEQDSK",
}
CASE_SIGNATURES = {
    "solovev": (
        {"psin": 1, "h": 1, "k": 1, "s1": 1},
        {"psin": 2, "h": 2, "k": 2, "s1": 2},
        {"psin": 2, "h": 2, "k": 2, "s1": 2, "s2": 2},
    ),
    "chease": (
        {"psin": 4, "h": 4, "k": 4, "v": 4, "c0": 4, "s1": 4, "c1": 2, "s2": 2},
        {"psin": 5, "h": 5, "k": 5, "v": 5, "c0": 5, "s1": 5, "c1": 3, "s2": 3, "c2": 1, "s3": 1},
        {"psin": 5, "h": 7, "k": 7, "v": 3, "c0": 3, "s1": 5, "c1": 1, "s2": 5, "c2": 1, "s3": 3},
    ),
    "efit": (
        {"psin": 3, "h": 3, "k": 3, "v": 3, "c0": 2, "s1": 2},
        {"psin": 4, "h": 4, "k": 4, "v": 4, "c0": 4, "s1": 4, "c1": 3, "s2": 3, "c2": 1, "s3": 1},
        {
            "psin": 2,
            "h": 8,
            "k": 5,
            "v": 9,
            "c0": 5,
            "s1": 5,
            "c1": 5,
            "s2": 5,
            "c2": 5,
            "s3": 5,
            "c3": 4,
            "s4": 4,
            "c4": 4,
            "s5": 3,
            "c5": 1,
            "s6": 4,
        },
    ),
}
LEVEL_LABELS = ("Low", "Medium", "High")
CASE_LINE_COLORS = {
    "solovev": ("#101010", "#777777", "#74a9cf", "#1f77b4", "#08306b"),
    "chease": ("#101010", "#777777", "#fdb863", "#ff7f0e", "#7f2704"),
    "efit": ("#101010", "#777777", "#74c476", "#2ca02c", "#00441b"),
}
LOW_LINESTYLE = (0, (5, 1.6, 1.2, 1.6, 1.2, 1.6))
LEVEL_LINESTYLES = {
    "Ref": ":",
    "Low": LOW_LINESTYLE,
    "Medium": "--",
    "High": "-",
}
LEGEND_LABEL_SPACING = 0.15
RADIAL_LINE_WIDTH = 1.4
EXTERNAL_RADIAL_LINE_WIDTH = 0.75 * RADIAL_LINE_WIDTH
EXTERNAL_RADIAL_MARKER_SIZE = 3.0 * RADIAL_LINE_WIDTH
EXTERNAL_RADIAL_ALPHA = 1.0
EXTERNAL_RADIAL_MARKER_COUNT = 10
COLUMN_TITLES = (
    r"$\mathbf{(a)}$ Temperature",
    r"$\mathbf{(b)}$ $T$ error",
    r"$\mathbf{(c)}$ $V'$ error",
    r"$\mathbf{(d)}$ $V'\langle|\nabla\hat{\psi}|^2\rangle$ error",
)

EVAL_POINTS = 128
ANALYTIC_THETA_POINTS = 384
VEQ_RHO_POINTS = 257
VEQ_THETA_POINTS = 512
X_FLOOR = 1.0e-4


@dataclass(frozen=True)
class TransportGeometry:
    label: str
    x: np.ndarray
    vprime: np.ndarray
    metric_weight: np.ndarray
    q: np.ndarray
    ip: float
    b0: float
    params: int | None = None
    elapsed_ms: float | None = None
    table5_time_ms: float | None = None


@dataclass(frozen=True)
class TransportResult:
    label: str
    geometry: TransportGeometry
    source: np.ndarray
    temperature: np.ndarray
    thermal_energy: float
    beta_proxy: float
    q95: float
    ip: float


def load_script(stem: str, filename: str):
    path = REPO_ROOT / "scripts" / filename
    spec = importlib.util.spec_from_file_location(stem, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[stem] = module
    spec.loader.exec_module(module)
    return module


def cumtrapz(y: np.ndarray, x: np.ndarray) -> np.ndarray:
    out = np.zeros_like(y, dtype=np.float64)
    out[1:] = np.cumsum(0.5 * (y[1:] + y[:-1]) * np.diff(x))
    return out


def reverse_integral_to_edge(y: np.ndarray, x: np.ndarray) -> np.ndarray:
    return cumtrapz(y[::-1], x[::-1])[::-1] * -1.0


def interp_unique(x_src: np.ndarray, y_src: np.ndarray, x_eval: np.ndarray) -> np.ndarray:
    x_arr = np.asarray(x_src, dtype=np.float64)
    y_arr = np.asarray(y_src, dtype=np.float64)
    order = np.argsort(x_arr, kind="mergesort")
    x_sorted = x_arr[order]
    y_sorted = y_arr[order]
    x_unique, unique_idx = np.unique(x_sorted, return_index=True)
    y_unique = y_sorted[unique_idx]
    return np.interp(x_eval, x_unique, y_unique, left=float(y_unique[0]), right=float(y_unique[-1]))


def rel_rms(reference: np.ndarray, current: np.ndarray, x: np.ndarray) -> float:
    ref_all = np.asarray(reference, dtype=np.float64)
    cur_all = np.asarray(current, dtype=np.float64)
    x_all = np.asarray(x, dtype=np.float64)
    n = min(ref_all.size, cur_all.size, x_all.size)
    if n == 0:
        return float("nan")
    ref = ref_all[:n]
    cur = cur_all[:n]
    mask = np.isfinite(x_all[:n]) & np.isfinite(ref) & np.isfinite(cur)
    if not np.any(mask):
        return float("nan")
    ref = ref[mask]
    cur = cur[mask]
    scale = max(float(np.max(np.abs(ref))), 1.0e-14)
    return float(np.sqrt(np.mean((cur - ref) ** 2)) / scale)


def rel_abs(reference: float, current: float) -> float:
    return float(abs(current - reference) / max(abs(reference), 1.0e-14))


def format_tex_sci(value: float) -> str:
    if not np.isfinite(value):
        return "--"
    if value == 0.0:
        return "$0$"
    exponent = int(np.floor(np.log10(abs(value))))
    mantissa = value / (10.0**exponent)
    return rf"${mantissa:.3f}\times 10^{{{exponent}}}$"


def analytic_line_integrals(fig07, x_eval: np.ndarray, *, label: str) -> TransportGeometry:
    cfg = fig07.REFERENCE_CONFIG
    solution = fig07.build_exact_solution(cfg)
    x_surface = np.maximum(np.asarray(x_eval, dtype=np.float64), X_FLOOR)
    vprime = np.empty_like(x_surface)
    metric_weight = np.empty_like(x_surface)
    q = fig07.q_profile(x_surface, cfg)

    for i, level in enumerate(x_surface):
        curve = fig07.analytic_surface_points_direct(
            cfg,
            level=float(level),
            n_theta=ANALYTIC_THETA_POINTS,
        )
        nxt = np.roll(curve, -1, axis=0)
        mid = 0.5 * (curve + nxt)
        ds = np.sqrt(np.sum((nxt - curve) ** 2, axis=1))
        grad_R, grad_Z = solution.grad_psi_raw(mid[:, 0], mid[:, 1])
        grad = np.maximum(np.sqrt(grad_R * grad_R + grad_Z * grad_Z), 1.0e-14)
        R_mid = np.maximum(mid[:, 0], 1.0e-14)
        vprime[i] = 2.0 * np.pi * float(np.sum(R_mid * ds / grad))
        metric_weight[i] = 2.0 * np.pi * float(np.sum(R_mid * grad * ds))

    return TransportGeometry(
        label=label,
        x=np.asarray(x_eval, dtype=np.float64),
        vprime=vprime,
        metric_weight=metric_weight,
        q=q,
        ip=fig07.estimate_total_current(cfg),
        b0=float(cfg.toroidal_field),
        params=None,
        elapsed_ms=None,
    )


def veq_transport_geometry(
    fig08,
    equilibrium,
    x_eval: np.ndarray,
    *,
    label: str,
    params: int | None,
    elapsed_ms: float | None,
):
    grid = fig08._load_veqpy_components()["Grid"](
        Nr=VEQ_RHO_POINTS,
        Nt=VEQ_THETA_POINTS,
        quadrature_scheme="uniform",
        L_max=int(equilibrium.grid.L_max),
        M_max=int(equilibrium.grid.M_max),
    )
    eq = equilibrium.resample(grid=grid)
    geom = eq.geometry
    psin = np.asarray(eq.psin, dtype=np.float64)
    psin_r = np.asarray(eq.psin_r, dtype=np.float64)
    vprime_rho = np.asarray(eq.V_r, dtype=np.float64) / np.maximum(psin_r, 1.0e-14)
    # M = V' <|grad psin|^2> = 2*pi*psin_r*int R*gtt/J dtheta.
    r_gtt_over_j = (
        np.asarray(geom.gttdivJR, dtype=np.float64) * np.asarray(geom.R, dtype=np.float64) ** 2
    )
    metric_rho = (
        2.0
        * np.pi
        * psin_r
        * np.asarray(eq.grid.quadrature(r_gtt_over_j, axis=1), dtype=np.float64)
    )

    return TransportGeometry(
        label=label,
        x=np.asarray(x_eval, dtype=np.float64),
        vprime=interp_unique(psin, vprime_rho, x_eval),
        metric_weight=interp_unique(psin, metric_rho, x_eval),
        q=interp_unique(psin, np.asarray(eq.q, dtype=np.float64), x_eval),
        ip=float(eq.Ip),
        b0=float(eq.B0),
        params=None if params is None else int(params),
        elapsed_ms=None if elapsed_ms is None else float(elapsed_ms),
        table5_time_ms=None,
    )


def solve_heat_profile(geometry: TransportGeometry, source: np.ndarray) -> TransportResult:
    x = geometry.x
    rhs = geometry.vprime * source
    cumulative_power = cumtrapz(rhs, x)
    diffusivity_weight = np.maximum(geometry.metric_weight, 1.0e-14)
    temperature = reverse_integral_to_edge(cumulative_power / diffusivity_weight, x)
    thermal_energy = 1.5 * float(np.trapezoid(geometry.vprime * temperature, x))
    volume = max(float(np.trapezoid(geometry.vprime, x)), 1.0e-14)
    beta_proxy = float(
        2.0 * MU0 * np.trapezoid(geometry.vprime * temperature, x) / (volume * geometry.b0**2)
    )
    q95 = float(np.interp(0.95, geometry.x, geometry.q))
    return TransportResult(
        label=geometry.label,
        geometry=geometry,
        source=source,
        temperature=temperature,
        thermal_energy=thermal_energy,
        beta_proxy=beta_proxy,
        q95=q95,
        ip=float(geometry.ip),
    )


def solve_signature_equilibrium(
    fig08, benchmark, case_key: str, boundary, geqdsk, signature: dict[str, int]
):
    grid = benchmark.Grid(
        Nr=32,
        Nt=32,
        quadrature_scheme="legendre",
        L_max=int(benchmark.REFERENCE_GRID.L_max),
        M_max=int(benchmark.REFERENCE_GRID.M_max),
    )
    _, max_lengths = fig08.get_case_length_bounds(case_key)
    raw_profile_coeffs = fig08.make_profile_coeffs(signature, max_lengths=max_lengths)
    profile_coeffs = {
        name: values for name, values in raw_profile_coeffs.items() if values is not None
    }
    case = fig08.build_solver_case(
        boundary,
        geqdsk,
        profile_coeffs=profile_coeffs,
    )
    solver_config = fig08.get_case_solver_config(benchmark, case_key)
    solver = benchmark.Solver(operator=benchmark.Operator(grid, case), config=solver_config)
    solver.solve(
        method=fig08.CASE_SOLVER_METHODS[case_key],
        max_residual=float(getattr(solver_config, "max_residual", 1.0e-6)),
        max_evaluations=int(
            getattr(solver_config, "max_evaluations", fig08.REFERENCE_SOLVER_MAXFEV)
        ),
        enable_verbose=False,
        enable_history=False,
        enable_warmstart=False,
        enable_fallback=bool(getattr(solver_config, "enable_fallback", False)),
    )
    if solver.result is None:
        raise RuntimeError(f"{case_key} solve did not produce a result for signature {signature}")
    return (
        solver.build_equilibrium(),
        float(solver.result.elapsed) / 1000.0,
        int(np.size(solver.result.x)),
    )


def make_source(x: np.ndarray, reference_vprime: np.ndarray) -> np.ndarray:
    raw = np.exp(-(((x - 0.18) / 0.33) ** 2)) * (1.0 - 0.15 * x)
    raw = np.maximum(raw, 0.0)
    norm = max(float(np.trapezoid(reference_vprime * raw, x)), 1.0e-14)
    return raw / norm


def latex_error_table(case_results: list[tuple[str, list[TransportResult]]]) -> str:
    lines = [
        r"\begin{tabular}{l r r r r r}",
        r"\hline",
        (
            r"Case & \(\mathrm{RMS}(\delta_T)\) & \(\mathrm{RMS}(\delta_{V'})\) & "
            r"\(\mathrm{RMS}(\delta_{V'\langle|\nabla\hat{\psi}|^2\rangle})\) & "
            r"\(\Delta_W\) & \(\Delta_{\beta_t}\) \\"
        ),
        r"\hline",
    ]
    for _case_key, results in case_results:
        reference = results[0]
        x = reference.geometry.x
        for result in results[1:]:
            geom = result.geometry
            lines.append(
                f"{result.label} & "
                f"{format_tex_sci(rel_rms(reference.temperature, result.temperature, x))} & "
                f"{format_tex_sci(rel_rms(reference.geometry.vprime, geom.vprime, x))} & "
                f"{format_tex_sci(rel_rms(reference.geometry.metric_weight, geom.metric_weight, x))} & "
                f"{format_tex_sci(rel_abs(reference.thermal_energy, result.thermal_energy))} & "
                f"{format_tex_sci(rel_abs(reference.beta_proxy, result.beta_proxy))} \\\\"
            )
    lines.extend([r"\hline", r"\end{tabular}"])
    return "\n".join(lines)


def style_axis(ax: plt.Axes) -> None:
    ax.title.set_fontsize(scaled_font_size(TITLE_FONT_SIZE))
    ax.xaxis.label.set_fontsize(scaled_font_size(AXIS_LABEL_FONT_SIZE))
    ax.yaxis.label.set_fontsize(scaled_font_size(AXIS_LABEL_FONT_SIZE))
    ax.tick_params(
        direction=PLOT_TICK_DIRECTION,
        top=PLOT_TICK_TOP,
        right=PLOT_TICK_RIGHT,
        bottom=PLOT_TICK_BOTTOM,
        left=PLOT_TICK_LEFT,
        labeltop=PLOT_LABEL_TOP,
        labelright=PLOT_LABEL_RIGHT,
        labelsize=scaled_font_size(TICK_LABEL_FONT_SIZE),
    )
    ax.grid(True, linestyle=":", linewidth=0.6, alpha=0.45)


def level_line_style(
    case_key: str, level_label: str
) -> tuple[str | tuple[int, tuple[float, ...]], str]:
    colors = CASE_LINE_COLORS[case_key]
    color_by_label = {
        "Ref": colors[-1],
        "Low": colors[-4],
        "Medium": colors[-3],
        "High": colors[-2],
    }
    return LEVEL_LINESTYLES[level_label], color_by_label[level_label]


def level_display_label(level_label: str, result: TransportResult) -> str:
    params = result.geometry.params
    if params is None:
        return level_label
    return f"{level_label} ({params:d})"


def marker_indices(length: int, count: int = EXTERNAL_RADIAL_MARKER_COUNT) -> list[int]:
    n = int(length)
    if n <= 0:
        return []
    if n <= int(count):
        return list(range(n))
    return np.unique(np.linspace(0, n - 1, int(count), dtype=int)).tolist()


def relative_profile_error(current: np.ndarray, reference: np.ndarray) -> np.ndarray:
    reference_arr = np.asarray(reference, dtype=np.float64)
    scale = max(float(np.nanmax(np.abs(reference_arr))), 1.0e-14)
    return np.abs(np.asarray(current, dtype=np.float64) - reference_arr) / scale


def plot_relative_error_family(
    ax: plt.Axes,
    *,
    x: np.ndarray,
    reference_profile: np.ndarray,
    level_results: list[tuple[str, TransportResult]],
    profile_getter,
    case_key: str,
    show_legend: bool = False,
    legend_loc: str = "upper left",
) -> None:
    for level_label, result in reversed(level_results):
        linestyle, color = level_line_style(case_key, level_label)
        rel = relative_profile_error(profile_getter(result), reference_profile)
        ax.semilogy(
            x,
            rel,
            color=color,
            linestyle=linestyle,
            linewidth=RADIAL_LINE_WIDTH,
            label=level_display_label(level_label, result),
        )
    ax.set_ylim(1.0e-8, 1.0e0)
    if show_legend:
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(
            handles[::-1],
            labels[::-1],
            loc=legend_loc,
            frameon=False,
            fontsize=scaled_font_size(LEGEND_FONT_SIZE),
            labelspacing=LEGEND_LABEL_SPACING,
        )


def plot_results(
    case_results: list[tuple[str, list[TransportResult]]], output_png: Path, output_pdf: Path
) -> None:
    apply_plot_style()
    fig, axes = plt.subplots(
        len(case_results),
        4,
        figsize=(DOUBLE_COLUMN_WIDTH, 6.15),
        constrained_layout=True,
        sharex="col",
    )
    axes = np.atleast_2d(axes)

    for row_idx, (case_key, results) in enumerate(case_results):
        reference = results[0]
        x = reference.geometry.x
        ax_t, ax_terr, ax_vprime, ax_metric = axes[row_idx]
        case_label = CASE_LABELS[case_key]
        show_xlabel = row_idx == len(case_results) - 1

        ax_t.plot(
            x,
            reference.temperature,
            label=reference.label,
            color="black",
            linestyle="-",
            linewidth=EXTERNAL_RADIAL_LINE_WIDTH,
            alpha=EXTERNAL_RADIAL_ALPHA,
            marker="o",
            markersize=EXTERNAL_RADIAL_MARKER_SIZE,
            markevery=marker_indices(len(x)),
            markerfacecolor="black",
            markeredgecolor="black",
        )
        level_results = list(zip(LEVEL_LABELS, results[1:], strict=True))
        for level_label, result in reversed(level_results):
            linestyle, color = level_line_style(case_key, level_label)
            ax_t.plot(
                x,
                result.temperature,
                label=level_display_label(level_label, result),
                color=color,
                linestyle=linestyle,
                linewidth=RADIAL_LINE_WIDTH,
            )
        ax_t.set_xlabel(r"$\hat{\psi}$" if show_xlabel else "")
        ax_t.set_ylabel(f"{case_label}\n" + r"$T$ [arb.]")
        if row_idx == 0:
            ax_t.set_title(COLUMN_TITLES[0])
        handles, labels = ax_t.get_legend_handles_labels()
        ax_t.legend(
            handles[::-1],
            labels[::-1],
            frameon=False,
            fontsize=scaled_font_size(LEGEND_FONT_SIZE),
            labelspacing=LEGEND_LABEL_SPACING,
        )

        plot_relative_error_family(
            ax_terr,
            x=x,
            reference_profile=reference.temperature,
            level_results=level_results,
            profile_getter=lambda result: result.temperature,
            case_key=case_key,
            show_legend=True,
            legend_loc="upper left",
        )
        ax_terr.set_xlabel(r"$\hat{\psi}$" if show_xlabel else "")
        ax_terr.set_ylabel("rel. error")
        if row_idx == 0:
            ax_terr.set_title(COLUMN_TITLES[1])

        plot_relative_error_family(
            ax_vprime,
            x=x,
            reference_profile=reference.geometry.vprime,
            level_results=level_results,
            profile_getter=lambda result: result.geometry.vprime,
            case_key=case_key,
        )
        ax_vprime.set_xlabel(r"$\hat{\psi}$" if show_xlabel else "")
        ax_vprime.set_ylabel("rel. error")
        if row_idx == 0:
            ax_vprime.set_title(COLUMN_TITLES[2])

        plot_relative_error_family(
            ax_metric,
            x=x,
            reference_profile=reference.geometry.metric_weight,
            level_results=level_results,
            profile_getter=lambda result: result.geometry.metric_weight,
            case_key=case_key,
        )
        ax_metric.set_xlabel(r"$\hat{\psi}$" if show_xlabel else "")
        ax_metric.set_ylabel("rel. error")
        if row_idx == 0:
            ax_metric.set_title(COLUMN_TITLES[3])

        style_axis(ax_t)
        style_axis(ax_terr)
        style_axis(ax_vprime)
        style_axis(ax_metric)

    fig.savefig(output_png, dpi=SAVE_DPI, transparent=SAVE_TRANSPARENT)
    fig.savefig(output_pdf, transparent=SAVE_TRANSPARENT)
    plt.close(fig)


def prepare_case_reference(
    fig07, fig08, benchmark, case_key: str, x_eval: np.ndarray
) -> tuple[object, object, TransportGeometry]:
    gfile_path = REPO_ROOT / fig08.CASE_REFERENCE_GFILES[case_key]
    if case_key == "solovev" and not gfile_path.exists():
        fig07.write_solovev_reference_gfile(str(gfile_path), cfg=fig07.REFERENCE_CONFIG)
    geqdsk = fig08.read_geqdsk(gfile_path)
    boundary, _ = fig08.build_boundary(
        geqdsk,
        fit_m=fig08.CASE_BOUNDARY_FIT_M[case_key],
        fit_n=fig08.CASE_BOUNDARY_FIT_N[case_key],
    )
    if case_key == "solovev":
        reference_geometry = analytic_line_integrals(
            fig07, x_eval, label=REFERENCE_LABELS[case_key]
        )
    else:
        reference_case = fig08.build_reference_case(benchmark, case_key)
        reference_geometry = veq_transport_geometry(
            fig08,
            reference_case.equilibrium,
            x_eval,
            label=REFERENCE_LABELS[case_key],
            params=None,
            elapsed_ms=None,
        )
    return geqdsk, boundary, reference_geometry


def solve_case_results(
    fig07, fig08, benchmark, case_key: str, x_eval: np.ndarray
) -> list[TransportResult]:
    geqdsk, boundary, reference_geometry = prepare_case_reference(
        fig07, fig08, benchmark, case_key, x_eval
    )
    veq_geometries = []
    for signature in CASE_SIGNATURES[case_key]:
        eq, elapsed_ms, params = solve_signature_equilibrium(
            fig08, benchmark, case_key, boundary, geqdsk, signature
        )
        veq_geometries.append(
            veq_transport_geometry(
                fig08,
                eq,
                x_eval,
                label=f"{CASE_LABELS[case_key]}({params:d})",
                params=params,
                elapsed_ms=elapsed_ms,
            )
        )

    source = make_source(x_eval, reference_geometry.vprime)
    return [solve_heat_profile(reference_geometry, source)] + [
        solve_heat_profile(geometry, source) for geometry in veq_geometries
    ]


def main() -> None:
    fig07 = load_script("veqpy_zhang2026_fig07", "06-high-order-reconstructions.py")
    fig08 = load_script("veqpy_zhang2026_fig08", "07-pareto-analysis.py")

    benchmark = fig08.load_benchmark("numba")

    x_eval = np.linspace(0.0, 1.0, EVAL_POINTS, dtype=np.float64)
    x_eval[0] = X_FLOOR

    case_results = [
        (case_key, solve_case_results(fig07, fig08, benchmark, case_key, x_eval))
        for case_key in CASE_KEYS
    ]
    plot_results(case_results, OUT_PNG, OUT_PDF)

    print(f"wrote {OUT_PNG.relative_to(REPO_ROOT)}")
    print(f"wrote {OUT_PDF.relative_to(REPO_ROOT)}")
    print(latex_error_table(case_results))


if __name__ == "__main__":
    main()

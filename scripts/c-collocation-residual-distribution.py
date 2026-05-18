"""Point-collocation least-squares residual diagnostic for Figure c.

This script is a diagnostic counterpart to ``scripts/09.py``.  It keeps the
same case construction, boundary fits, selected active signatures and default
32x32 Legendre solve grid used by the Figure 9 residual diagnostic, but after
obtaining the standard weak-form VEQ solution it uses the built-in veqpy
``enable_collocation`` workflow: a normal variational solve followed by a
warm-started DESC-style collocation polish, which minimizes quadrature-weighted
pointwise force-balance component samples on the solve grid.

The objective is the core veqpy collocation residual itself, with no externally
appended weak-form closure.
The reference files, tenth-order boundary fits and default 32x32 solve grid are
the same settings used by the Figure 7/9 workflow:

    min_x || [W^{1/2}G psi_R, W^{1/2}G psi_Z](x) ||_2,

where the fields are sampled on the same radial/poloidal quadrature grid that Figure 9
uses for diagnostic maps.  Route preprocessing, source scaling, boundary data
and active families are otherwise unchanged, so the comparison isolates the
change from projected weak residuals to pointwise residual least squares.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import ticker

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.config import SAVE_DPI as CONFIG_SAVE_DPI, SAVE_TRANSPARENT as CONFIG_SAVE_TRANSPARENT  # noqa: E402

SAVE_PNG_PATH = str(REPO_ROOT / "figures" / "c.png")
SAVE_PDF_PATH = str(REPO_ROOT / "figures" / "c.pdf")
SAVE_COMPACT_PNG_PATH = str(REPO_ROOT / "figures" / "c-1.png")
DEFAULT_CACHE_PATH = str(REPO_ROOT / "data" / "c-collocation-cache.npz")
CACHE_VERSION = 6


def load_fig09_helpers():
    helper_path = Path(__file__).with_name("08-residual-distribution.py")
    spec = importlib.util.spec_from_file_location("veqpy_fig09_helpers", helper_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load helper script: {helper_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


fig09 = load_fig09_helpers()

CASE_KEYS = fig09.CASE_KEYS
CASE_LABELS = fig09.CASE_LABELS
CONFIG_LABELS = fig09.CONFIG_LABELS
CONFIG_CMAP = fig09.CONFIG_CMAP
LOG_RESIDUAL_FLOOR = fig09.LOG_RESIDUAL_FLOOR
LOG_RESIDUAL_CEIL = fig09.LOG_RESIDUAL_CEIL
SAVE_DPI = CONFIG_SAVE_DPI
SAVE_TRANSPARENT = CONFIG_SAVE_TRANSPARENT


@dataclass(frozen=True)
class CollocationSample:
    weak: fig09.ResidualSample
    collocation: fig09.ResidualSample
    initial_weighted_rms: float
    final_weighted_rms: float
    cost_initial: float
    cost_final: float
    nfev: int
    njev: int
    status: int
    success: bool
    message: str
    elapsed_ms: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run point-collocation least-squares residual minimization for the "
            "same configurations used by Figure 9 and plot Figure c."
        )
    )
    parser.add_argument("--backend", default="numba")
    parser.add_argument("--case", choices=(*CASE_KEYS, "all"), default="all")
    parser.add_argument("--solve-nr", type=int, default=32)
    parser.add_argument("--solve-nt", type=int, default=32)
    parser.add_argument("--weak-repeat-count", type=int, default=10)
    parser.add_argument("--initial-solve-timeout-s", type=float, default=30.0)
    parser.add_argument("--collocation-method", choices=("lm", "trf"), default="trf")
    parser.add_argument("--collocation-repeat-count", type=int, default=10)
    parser.add_argument("--max-residual", type=float, default=1.0e-8)
    parser.add_argument("--max-nfev", type=int, default=40)
    parser.add_argument("--ftol", type=float, default=1.0e-8, help=argparse.SUPPRESS)
    parser.add_argument("--xtol", type=float, default=1.0e-8, help=argparse.SUPPRESS)
    parser.add_argument("--gtol", type=float, default=1.0e-8, help=argparse.SUPPRESS)
    parser.add_argument(
        "--scale-source",
        choices=("weak", "collocation"),
        default="collocation",
        help="Use weak or collocation samples to set the case-local color scale for Figure c.",
    )
    parser.add_argument("--cache-path", default=DEFAULT_CACHE_PATH)
    parser.add_argument("--refresh-cache", action="store_true")
    parser.add_argument("--save-png", default=SAVE_PNG_PATH)
    parser.add_argument("--save-pdf", default=SAVE_PDF_PATH)
    parser.add_argument(
        "--save-compact-png",
        default=SAVE_COMPACT_PNG_PATH,
        help="Also write the compact 3x3 Low/Medium/High-vs-target collocation shape comparison.",
    )
    return parser.parse_args()


def quadrature_sqrt_weights(grid) -> np.ndarray:
    radial = np.asarray(grid.weights, dtype=np.float64)
    if radial.ndim != 1 or radial.size != int(grid.Nr):
        raise ValueError("Expected one-dimensional radial quadrature weights")
    # The constant trapezoidal factor in theta does not affect the minimizer;
    # including 1/Nt keeps the objective close to an average RMS scale.
    return np.sqrt(radial[:, None] / max(int(grid.Nt), 1))


def weighted_pointwise_vector(G: np.ndarray, sqrt_weights: np.ndarray, *, scale: float) -> np.ndarray:
    values = np.asarray(G, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] != sqrt_weights.shape[0] or sqrt_weights.shape[1] != 1:
        raise ValueError(
            f"Expected G with shape (Nr, Nt) and weights with shape (Nr, 1), got {values.shape} and {sqrt_weights.shape}"
        )
    weighted = sqrt_weights * values
    return np.ravel(weighted / max(float(scale), 1.0e-30))


def weighted_rms(G: np.ndarray, sqrt_weights: np.ndarray) -> float:
    vec = weighted_pointwise_vector(G, sqrt_weights, scale=1.0)
    finite = vec[np.isfinite(vec)]
    if finite.size == 0:
        return float("nan")
    return float(np.sqrt(np.mean(finite * finite)))


def make_residual_sample(
    *,
    case_key: str,
    config_label: str,
    signature: dict[str, int],
    parameter_count: int,
    elapsed_ms: float,
    packed_residual_norm: float,
    equilibrium,
) -> fig09.ResidualSample:
    G = np.asarray(equilibrium.G, dtype=np.float64)
    radial_rms = np.sqrt(np.nanmean(G * G, axis=1))
    return fig09.ResidualSample(
        case_key=case_key,
        config_label=config_label,
        signature=dict(signature),
        parameter_count=int(parameter_count),
        elapsed_ms=float(elapsed_ms),
        solver_residual_norm=float(packed_residual_norm),
        rho=np.asarray(equilibrium.rho, dtype=np.float64),
        psin=np.asarray(equilibrium.psin, dtype=np.float64),
        R=np.asarray(equilibrium.geometry.R, dtype=np.float64),
        Z=np.asarray(equilibrium.geometry.Z, dtype=np.float64),
        G=G,
        radial_rms=radial_rms,
    )


def solve_variational_with_average_timing(
    fig08,
    benchmark,
    case,
    grid,
    repeat_count: int,
    *,
    method: str,
    solver_config,
    initial_solve_timeout_s: float,
):
    solver = benchmark.Solver(
        operator=benchmark.Operator(grid, case),
        config=solver_config,
    )
    solve_kwargs = {
        "method": str(method),
        "max_residual": float(getattr(solver_config, "max_residual", 1.0e-6)),
        "max_evaluations": int(getattr(solver_config, "max_evaluations", fig08.REFERENCE_SOLVER_MAXFEV)),
        "enable_verbose": False,
        "enable_history": False,
        "enable_warmstart": False,
        "enable_fallback": bool(getattr(solver_config, "enable_fallback", False)),
    }

    probe_started = time.perf_counter()
    solver.solve(**solve_kwargs)
    probe_elapsed_ms = (time.perf_counter() - probe_started) * 1000.0
    if solver.result is not None:
        probe_elapsed_ms = max(probe_elapsed_ms, float(solver.result.elapsed) / 1000.0)
    if probe_elapsed_ms > float(initial_solve_timeout_s) * 1000.0:
        raise fig08.InitialSolveTimeoutError(
            elapsed_ms=probe_elapsed_ms,
            timeout_s=initial_solve_timeout_s,
        )

    elapsed_values: list[float] = []
    final_result = None
    for _ in range(max(int(repeat_count), 1)):
        solver.solve(**solve_kwargs)
        final_result = solver.result
        elapsed_values.append(float(final_result.elapsed) / 1000.0)
    if final_result is None:
        raise RuntimeError("No variational solver result returned")

    return final_result, solver.build_equilibrium(), float(np.mean(elapsed_values))


def solve_collocation_with_average_timing(
    benchmark,
    operator,
    args: argparse.Namespace,
    *,
    variational_method: str,
    solver_config,
):
    solver = benchmark.Solver(operator=operator, config=benchmark.CONFIG)
    solve_kwargs = {
        "method": str(variational_method),
        "max_residual": float(getattr(solver_config, "max_residual", 1.0e-6)),
        "max_evaluations": int(getattr(solver_config, "max_evaluations", 1000)),
        "enable_collocation": True,
        "collocation_method": str(args.collocation_method),
        "collocation_max_residual": float(args.max_residual),
        "collocation_max_evaluations": int(args.max_nfev),
        "enable_fallback": bool(getattr(solver_config, "enable_fallback", False)),
        "enable_history": False,
        "enable_verbose": False,
        "enable_warmstart": False,
    }

    solver.solve(**solve_kwargs)
    if solver.result is None:
        raise RuntimeError("Built-in variational-plus-collocation warm-up solve did not produce a result")

    elapsed_values: list[float] = []
    final_result = None
    for _ in range(max(int(args.collocation_repeat_count), 1)):
        solver.solve(**solve_kwargs)
        final_result = solver.result
        elapsed_values.append(float(final_result.elapsed) / 1000.0)
    if final_result is None:
        raise RuntimeError("Built-in variational-plus-collocation solve did not produce a result")

    return final_result, solver.build_equilibrium(), float(np.mean(elapsed_values))


def collocation_objective(operator, x: np.ndarray) -> np.ndarray:
    return np.asarray(operator.residual_collocation(x), dtype=np.float64)


def result_count(result, solver_name: str, optimize_name: str) -> int:
    value = getattr(result, solver_name, getattr(result, optimize_name, 0))
    if value is None:
        return 0
    return int(value)


def solve_collocation_sample(
    fig08,
    benchmark,
    reference,
    *,
    case_key: str,
    config_label: str,
    signature: dict[str, int],
    args: argparse.Namespace,
) -> CollocationSample:
    grid = benchmark.Grid(
        Nr=int(args.solve_nr),
        Nt=int(args.solve_nt),
        quadrature_scheme="legendre",
        L_max=int(benchmark.REFERENCE_GRID.L_max),
        M_max=int(benchmark.REFERENCE_GRID.M_max),
    )
    case = fig08.build_pf_case(benchmark, reference, grid, signature)

    weak_result, weak_equilibrium, weak_elapsed_ms = solve_variational_with_average_timing(
        fig08,
        benchmark,
        case,
        grid,
        int(args.weak_repeat_count),
        method=fig08.CASE_SOLVER_METHODS[case_key],
        solver_config=benchmark.CONFIG,
        initial_solve_timeout_s=float(args.initial_solve_timeout_s),
    )
    operator = benchmark.Operator(grid, case)
    weak_x = operator.coerce_x(weak_result.x).copy()
    weak_equilibrium = operator.build_equilibrium(weak_x)
    weak_packed = np.asarray(operator(weak_x), dtype=np.float64)
    initial_collocation_residual = np.asarray(collocation_objective(operator, weak_x), dtype=np.float64)
    collocation_result, collocation_equilibrium, elapsed_ms = solve_collocation_with_average_timing(
        benchmark,
        operator,
        args,
        variational_method=fig08.CASE_SOLVER_METHODS[case_key],
        solver_config=benchmark.CONFIG,
    )

    x_final = operator.coerce_x(collocation_result.x)
    final_packed = np.asarray(operator(x_final), dtype=np.float64)
    final_collocation_residual = np.asarray(collocation_objective(operator, x_final), dtype=np.float64)
    initial_weighted_rms = float(np.sqrt(np.mean(initial_collocation_residual * initial_collocation_residual)))
    final_weighted_rms = float(np.sqrt(np.mean(final_collocation_residual * final_collocation_residual)))

    weak_sample = make_residual_sample(
        case_key=case_key,
        config_label=config_label,
        signature=signature,
        parameter_count=weak_x.size,
        elapsed_ms=weak_elapsed_ms,
        packed_residual_norm=float(np.linalg.norm(weak_packed)),
        equilibrium=weak_equilibrium,
    )
    collocation_sample = make_residual_sample(
        case_key=case_key,
        config_label=config_label,
        signature=signature,
        parameter_count=x_final.size,
        elapsed_ms=elapsed_ms,
        packed_residual_norm=float(np.linalg.norm(final_packed)),
        equilibrium=collocation_equilibrium,
    )
    return CollocationSample(
        weak=weak_sample,
        collocation=collocation_sample,
        initial_weighted_rms=float(initial_weighted_rms),
        final_weighted_rms=float(final_weighted_rms),
        cost_initial=float(0.5 * np.dot(initial_collocation_residual, initial_collocation_residual)),
        cost_final=float(0.5 * np.dot(final_collocation_residual, final_collocation_residual)),
        nfev=result_count(collocation_result, "function_evaluations", "nfev"),
        njev=result_count(collocation_result, "jacobian_evaluations", "njev"),
        status=0,
        success=bool(collocation_result.success),
        message=str(collocation_result.message),
        elapsed_ms=float(elapsed_ms),
    )


def solve_case_samples(
    fig08, benchmark, *, case_key: str, signatures: list[dict[str, int]], args
) -> list[CollocationSample]:
    reference = fig08.build_reference_case(benchmark, case_key)
    samples: list[CollocationSample] = []
    for config_label, signature in zip(CONFIG_LABELS, signatures, strict=True):
        print(f"[collocation] {CASE_LABELS[case_key]} {config_label} params={sum(signature.values())}", flush=True)
        samples.append(
            solve_collocation_sample(
                fig08,
                benchmark,
                reference,
                case_key=case_key,
                config_label=config_label,
                signature=signature,
                args=args,
            )
        )
    return samples


def cache_signature(*, case_keys, args, signatures_by_case) -> dict[str, object]:
    return {
        "cache_version": CACHE_VERSION,
        "case_keys": list(case_keys),
        "backend": str(args.backend),
        "solve_nr": int(args.solve_nr),
        "solve_nt": int(args.solve_nt),
        "weak_repeat_count": int(args.weak_repeat_count),
        "collocation_method": str(args.collocation_method),
        "collocation_repeat_count": int(args.collocation_repeat_count),
        "max_residual": float(args.max_residual),
        "max_nfev": int(args.max_nfev),
        "config_labels": list(CONFIG_LABELS),
        "selected_signatures": {
            key: [fig09.normalize_signature(sig) for sig in signatures_by_case[key]] for key in case_keys
        },
    }


def _sample_arrays(prefix: str, idx: int, sample: fig09.ResidualSample, arrays: dict[str, np.ndarray]) -> None:
    arrays[f"{prefix}_rho_{idx}"] = np.asarray(sample.rho, dtype=np.float64)
    arrays[f"{prefix}_psin_{idx}"] = np.asarray(sample.psin, dtype=np.float64)
    arrays[f"{prefix}_R_{idx}"] = np.asarray(sample.R, dtype=np.float64)
    arrays[f"{prefix}_Z_{idx}"] = np.asarray(sample.Z, dtype=np.float64)
    arrays[f"{prefix}_G_{idx}"] = np.asarray(sample.G, dtype=np.float64)
    arrays[f"{prefix}_radial_rms_{idx}"] = np.asarray(sample.radial_rms, dtype=np.float64)


def save_cache(path, case_samples, *, case_keys, args, signatures_by_case) -> None:
    entries = []
    arrays: dict[str, np.ndarray] = {}
    idx = 0
    for case_key in case_keys:
        for item in case_samples[case_key]:
            entries.append(
                {
                    "idx": idx,
                    "case_key": item.collocation.case_key,
                    "config_label": item.collocation.config_label,
                    "signature": item.collocation.signature,
                    "parameter_count": item.collocation.parameter_count,
                    "weak_elapsed_ms": item.weak.elapsed_ms,
                    "collocation_elapsed_ms": item.collocation.elapsed_ms,
                    "total_elapsed_ms": item.collocation.elapsed_ms,
                    "weak_packed_residual_norm": item.weak.solver_residual_norm,
                    "collocation_packed_residual_norm": item.collocation.solver_residual_norm,
                    "initial_weighted_rms": item.initial_weighted_rms,
                    "final_weighted_rms": item.final_weighted_rms,
                    "cost_initial": item.cost_initial,
                    "cost_final": item.cost_final,
                    "nfev": item.nfev,
                    "njev": item.njev,
                    "status": item.status,
                    "success": item.success,
                    "message": item.message,
                }
            )
            _sample_arrays("weak", idx, item.weak, arrays)
            _sample_arrays("collocation", idx, item.collocation, arrays)
            idx += 1
    payload = {
        **cache_signature(case_keys=case_keys, args=args, signatures_by_case=signatures_by_case),
        "samples": entries,
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, metadata=np.asarray(json.dumps(payload)), **arrays)


def _load_sample(prefix: str, idx: int, entry: dict[str, object], data) -> fig09.ResidualSample:
    return fig09.ResidualSample(
        case_key=str(entry["case_key"]),
        config_label=str(entry["config_label"]),
        signature={str(k): int(v) for k, v in dict(entry["signature"]).items()},
        parameter_count=int(entry["parameter_count"]),
        elapsed_ms=float(entry[f"{prefix}_elapsed_ms"]),
        solver_residual_norm=float(entry[f"{prefix}_packed_residual_norm"]),
        rho=np.asarray(data[f"{prefix}_rho_{idx}"], dtype=np.float64),
        psin=np.asarray(data[f"{prefix}_psin_{idx}"], dtype=np.float64),
        R=np.asarray(data[f"{prefix}_R_{idx}"], dtype=np.float64),
        Z=np.asarray(data[f"{prefix}_Z_{idx}"], dtype=np.float64),
        G=np.asarray(data[f"{prefix}_G_{idx}"], dtype=np.float64),
        radial_rms=np.asarray(data[f"{prefix}_radial_rms_{idx}"], dtype=np.float64),
    )


def load_cache(path, *, case_keys, args, signatures_by_case) -> dict[str, list[CollocationSample]] | None:
    if not os.path.exists(path):
        return None
    try:
        with np.load(path, allow_pickle=False) as data:
            payload = json.loads(str(data["metadata"].item()))
            expected = cache_signature(case_keys=case_keys, args=args, signatures_by_case=signatures_by_case)
            for key, value in expected.items():
                if payload.get(key) != value:
                    return None
            out: dict[str, list[CollocationSample]] = {key: [] for key in case_keys}
            for entry in payload.get("samples", []):
                idx = int(entry["idx"])
                case_key = str(entry["case_key"])
                if case_key not in out:
                    continue
                weak = _load_sample("weak", idx, entry, data)
                collocation = _load_sample("collocation", idx, entry, data)
                out[case_key].append(
                    CollocationSample(
                        weak=weak,
                        collocation=collocation,
                        initial_weighted_rms=float(entry["initial_weighted_rms"]),
                        final_weighted_rms=float(entry["final_weighted_rms"]),
                        cost_initial=float(entry["cost_initial"]),
                        cost_final=float(entry["cost_final"]),
                        nfev=int(entry["nfev"]),
                        njev=int(entry["njev"]),
                        status=int(entry["status"]),
                        success=bool(entry["success"]),
                        message=str(entry["message"]),
                        elapsed_ms=float(entry["collocation_elapsed_ms"]),
                    )
                )
            if any(len(out[key]) != len(CONFIG_LABELS) for key in case_keys):
                return None
            return out
    except Exception:
        return None


def residual_scale(samples: list[fig09.ResidualSample]) -> float:
    return fig09.residual_scale(samples)


def build_figure(
    collocation_by_case: dict[str, list[CollocationSample]],
    *,
    external_by_case: dict[str, fig09.ResidualSample],
    scale_source: str,
) -> plt.Figure:
    case_keys = list(collocation_by_case)
    figure_height = min(max(fig09.ROW_HEIGHT * len(case_keys), fig09.ROW_HEIGHT), fig09.FIGURE_MAX_HEIGHT)
    fig = plt.figure(figsize=(fig09.FIGURE_WIDTH, figure_height))
    grid = fig.add_gridspec(
        nrows=len(case_keys),
        ncols=len(fig09.FIGURE_GRID_WIDTH_RATIOS),
        width_ratios=fig09.FIGURE_GRID_WIDTH_RATIOS,
        left=fig09.FIGURE_LEFT,
        right=fig09.FIGURE_RIGHT,
        bottom=fig09.FIGURE_BOTTOM,
        top=fig09.FIGURE_TOP,
        wspace=fig09.FIGURE_GRID_WSPACE,
        hspace=fig09.FIGURE_GRID_HSPACE,
    )
    mappable = None
    heatmap_rows: list[list[plt.Axes]] = []
    radial_axes: list[plt.Axes] = []
    for row, case_key in enumerate(case_keys):
        items = collocation_by_case[case_key]
        external_sample = external_by_case[case_key]
        samples = [*(item.collocation for item in items), external_sample]
        scale_samples = [
            *(item.weak if scale_source == "weak" else item.collocation for item in items),
            external_sample,
        ]
        scale = residual_scale(scale_samples)
        xlim, ylim = fig09.rz_limits(samples)
        heatmap_axes = []
        for col, (grid_col, sample) in enumerate(zip(fig09.HEATMAP_GRID_COLS, samples, strict=True)):
            ax = fig.add_subplot(grid[row, grid_col])
            heatmap_axes.append(ax)
            mappable = fig09.plot_heatmap_panel(ax, sample, scale=scale, xlim=xlim, ylim=ylim)
            ax.set_xticks(fig09.HEATMAP_X_TICKS[case_key])
            ax.xaxis.set_major_formatter(ticker.FormatStrFormatter("%.1f"))
            ax.set_anchor("C")
            if col == 0:
                ax.set_ylabel(f"{CASE_LABELS[case_key]}\nZ [m]", fontsize=fig09.scaled_font_size(fig09.AXIS_LABEL_SIZE))
            else:
                ax.set_yticklabels([])
            if row == len(case_keys) - 1:
                ax.set_xlabel("R [m]", fontsize=fig09.scaled_font_size(fig09.AXIS_LABEL_SIZE))
        heatmap_rows.append(heatmap_axes)
        radial_ax = fig.add_subplot(grid[row, fig09.RADIAL_GRID_COL])
        radial_axes.append(radial_ax)
        fig09.plot_radial_panel(
            radial_ax,
            samples,
            scale=scale,
            case_key=case_key,
            show_xlabel=row == len(case_keys) - 1,
            legend_loc="upper left" if row == 0 else "lower right",
        )
        if row == 0:
            radial_ax.set_title(
                r"$\bf{(b)}$ $\varepsilon_{\mathcal{G}}(\rho)$",
                pad=fig09.PANEL_TITLE_PAD,
                fontsize=fig09.scaled_font_size(fig09.TITLE_FONT_SIZE),
                fontweight="normal",
            )
    max_heatmap_right = fig09.align_heatmap_columns(fig, heatmap_rows)
    if heatmap_rows:
        fig09.add_centered_panel_title(
            fig,
            heatmap_rows[0][0],
            heatmap_rows[0][-1],
            r"$\bf{(a)}$ Collocation-LS residual distribution",
            pad=fig09.PANEL_TITLE_PAD,
        )
    radial_left = min(max_heatmap_right + fig09.RADIAL_LEFT_PAD, fig09.FIGURE_RIGHT - 0.25)
    for radial_ax, heatmap_axes in zip(radial_axes, heatmap_rows, strict=True):
        row_bboxes = [ax.get_position().frozen() for ax in heatmap_axes]
        y0 = min(float(bbox.y0) for bbox in row_bboxes)
        y1 = max(float(bbox.y1) for bbox in row_bboxes)
        radial_ax.set_position([radial_left, y0, fig09.FIGURE_RIGHT - radial_left, y1 - y0])
    if mappable is not None:
        cax = fig.add_subplot(grid[:, fig09.COLORBAR_GRID_COL])
        cbar_bbox = cax.get_position()
        cbar_width = min(fig09.COLORBAR_MAX_WIDTH, cbar_bbox.width)
        cbar_height = fig09.COLORBAR_HEIGHT_FRACTION * cbar_bbox.height
        cbar_y = cbar_bbox.y0 + 0.5 * (cbar_bbox.height - cbar_height)
        cbar_x = max_heatmap_right + fig09.COLORBAR_LEFT_PAD
        cax.set_position([cbar_x, cbar_y, cbar_width, cbar_height])
        cbar = fig.colorbar(mappable, cax=cax)
        cbar_ticks = np.arange(LOG_RESIDUAL_FLOOR, LOG_RESIDUAL_CEIL + 0.5, 1.0)
        cbar.set_ticks(cbar_ticks)
        cbar.set_ticklabels([rf"$10^{{{int(tick)}}}$" for tick in cbar_ticks])
        cbar.ax.yaxis.set_ticks_position("right")
        cbar.ax.yaxis.set_label_position("right")
        cbar.ax.set_title(r"$\tilde{\mathcal{G}}$", fontsize=fig09.scaled_font_size(fig09.AXIS_LABEL_SIZE), pad=6.0)
        cbar.ax.tick_params(which="both", direction="out", labelsize=fig09.scaled_font_size(fig09.TICK_LABEL_SIZE))
    return fig


def compact_collocation_samples(
    items: list[CollocationSample],
    external_sample: fig09.ResidualSample,
) -> list[fig09.ResidualSample]:
    by_label = {item.collocation.config_label: item for item in items}
    selected_items = [by_label[label] for label in fig09.COMPACT_CONFIG_LABELS]
    return [*(item.collocation for item in selected_items), external_sample]


def build_compact_figure(
    collocation_by_case: dict[str, list[CollocationSample]],
    *,
    external_by_case: dict[str, fig09.ResidualSample],
    scale_source: str,
    fig08=None,
) -> plt.Figure:
    _ = scale_source
    compact_case_samples = {
        case_key: compact_collocation_samples(items, external_by_case[case_key])
        for case_key, items in collocation_by_case.items()
    }
    return fig09.build_compact_figure(compact_case_samples, fig08=fig08)


def _format_tex_number(value: object, *, precision: int = 3) -> str:
    return fig09._format_tex_number(value, precision=precision)


def collocation_rows(case_samples: dict[str, list[CollocationSample]]) -> list[dict[str, object]]:
    rows = []

    def ratio(numerator: float, denominator: float) -> float:
        return (
            numerator / denominator
            if np.isfinite(numerator) and np.isfinite(denominator) and denominator > 0.0
            else float("nan")
        )

    for case_key, items in case_samples.items():
        for item in items:
            var_row = fig09.residual_norm_row(item.weak)
            coll_row = fig09.residual_norm_row(item.collocation)
            rows.append(
                {
                    "case_params": coll_row["case_params"],
                    "ratio_rms_all": ratio(float(coll_row["G_rms"]), float(var_row["G_rms"])),
                    "ratio_rms_interior": ratio(
                        float(coll_row["G_rms_interior"]),
                        float(var_row["G_rms_interior"]),
                    ),
                    "ratio_rms_edge": ratio(float(coll_row["G_rms_edge"]), float(var_row["G_rms_edge"])),
                    "ratio_max": ratio(float(coll_row["G_max"]), float(var_row["G_max"])),
                    "ratio_solve_time": ratio(float(item.collocation.elapsed_ms), float(item.weak.elapsed_ms)),
                    "nfev": item.nfev,
                    "success": item.success,
                }
            )
    return rows


def build_collocation_latex_table(rows: list[dict[str, object]]) -> str:
    indent = "              "
    header = [
        "Case",
        r"$r_{\mathrm{RMS,all}}^{\mathrm{coll/var}}$",
        r"$r_{\mathrm{RMS},<0.8}^{\mathrm{coll/var}}$",
        r"$r_{\mathrm{RMS},\geq0.8}^{\mathrm{coll/var}}$",
        r"$r_{|\mathcal{G}|_{\max}}^{\mathrm{coll/var}}$",
        r"$r_{t_{\mathrm{solve}}}^{\mathrm{coll/var}}$",
    ]
    table_rows = []
    for row in rows:

        def fmt_ratio(key: str) -> str:
            value = float(row[key])
            return f"${value:.3f}$" if np.isfinite(value) else "--"

        table_rows.append(
            [
                str(row["case_params"]),
                fmt_ratio("ratio_rms_all"),
                fmt_ratio("ratio_rms_interior"),
                fmt_ratio("ratio_rms_edge"),
                fmt_ratio("ratio_max"),
                fmt_ratio("ratio_solve_time"),
            ]
        )
    widths = [max(len(row[i]) for row in [header, *table_rows]) for i in range(len(header))]

    def fmt(row):
        return " & ".join(cell.ljust(widths[i]) for i, cell in enumerate(row)) + r" \\"

    body = "\n".join(
        indent + line for line in [r"\hline", fmt(header), r"\hline", *(fmt(r) for r in table_rows), r"\hline"]
    )
    return "\n".join(
        [
            r"\begin{table}[htbp]",
            r"       \caption{DESC-style point-collocation force-balance residual and solve-time ratios for the VEQ rows of Table~\ref{tab:residual-diagnostics}. Each residual entry is collocation divided by variational VEQ; the solve-time entry is the variational-plus-warm-start-collocation time divided by the variational time.}",
            r"       \centering",
            r"       \begin{tabular}{l c c c c c}",
            body,
            r"       \end{tabular}",
            r"       \label{tab:collocation-residual-diagnostics}",
            r"\end{table}",
            "",
        ]
    )


def main() -> None:
    args = parse_args()
    fig09.apply_plot_style()
    selected_cases = CASE_KEYS if args.case == "all" else (args.case,)
    fig08 = fig09.load_fig08_helpers()
    signatures_by_case = fig09.selected_signature_map(fig08, selected_cases)

    case_samples = None
    if not args.refresh_cache:
        case_samples = load_cache(
            args.cache_path, case_keys=selected_cases, args=args, signatures_by_case=signatures_by_case
        )
    benchmark = fig08.load_benchmark(args.backend)
    if case_samples is None:
        case_samples = {
            case_key: solve_case_samples(
                fig08, benchmark, case_key=case_key, signatures=signatures_by_case[case_key], args=args
            )
            for case_key in selected_cases
        }
        save_cache(
            args.cache_path, case_samples, case_keys=selected_cases, args=args, signatures_by_case=signatures_by_case
        )

    external_by_case = {
        case_key: fig09.external_reference_residual_sample(
            fig08,
            fig08.build_reference_case(benchmark, case_key),
            case_key=case_key,
        )
        for case_key in selected_cases
    }

    Path(args.save_png).parent.mkdir(parents=True, exist_ok=True)
    Path(args.save_pdf).parent.mkdir(parents=True, exist_ok=True)
    fig = build_figure(case_samples, external_by_case=external_by_case, scale_source=args.scale_source)
    fig.savefig(args.save_png, dpi=SAVE_DPI, transparent=SAVE_TRANSPARENT)
    fig.savefig(args.save_pdf, dpi=SAVE_DPI, transparent=SAVE_TRANSPARENT)
    plt.close(fig)
    compact_fig = build_compact_figure(
        case_samples,
        external_by_case=external_by_case,
        scale_source=args.scale_source,
        fig08=fig08,
    )
    Path(args.save_compact_png).parent.mkdir(parents=True, exist_ok=True)
    compact_fig.savefig(args.save_compact_png, dpi=SAVE_DPI, transparent=SAVE_TRANSPARENT)
    plt.close(compact_fig)
    table = build_collocation_latex_table(collocation_rows(case_samples))
    print(table)


if __name__ == "__main__":
    main()

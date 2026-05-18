import argparse
import importlib.util
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors, ticker
from scipy.interpolate import RegularGridInterpolator

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.config import (
    AXIS_LABEL_SIZE,
    FONT_SCALE,
    LEGEND_FONT_SIZE,
    PLOT_BASE_FONT_SIZE,
    PLOT_FONT_FAMILY,
    PLOT_MATH_FONTSET,
    SAVE_DPI,
    SAVE_TRANSPARENT,
    TICK_LABEL_SIZE,
    TITLE_FONT_SIZE,
    WIDE_DOUBLE_COLUMN_WIDTH,
    scaled_font_size,
)

SAVE_PNG_PATH = str(REPO_ROOT / "figures" / "08.png")
SAVE_PDF_PATH = str(REPO_ROOT / "figures" / "08.pdf")
SAVE_COMPACT_PNG_PATH = str(REPO_ROOT / "figures" / "08-1.png")
DEFAULT_CACHE_PATH = str(REPO_ROOT / "data" / "09-residual-cache.npz")
RESIDUAL_CACHE_VERSION = 10

CASE_KEYS = ("solovev", "chease", "efit")
CASE_LABELS = {
    "solovev": "D-shape",
    "chease": "H-mode",
    "efit": "X-point",
}
CASE_COLORS = {
    "solovev": "#1f77b4",
    "chease": "#ff7f0e",
    "efit": "#2ca02c",
}
CASE_LINE_COLORS = {
    "solovev": ("#111111", "#777777", "#74a9cf", "#1f77b4", "#08306b"),
    "chease": ("#111111", "#777777", "#fdb863", "#ff7f0e", "#7f2704"),
    "efit": ("#111111", "#777777", "#74c476", "#2ca02c", "#00441b"),
}
CONFIG_LABELS = ("Low", "Medium", "High", "Ref")
EXTERNAL_REFERENCE_LABELS = {
    "solovev": "Analytic",
    "chease": "GEQDSK",
    "efit": "GEQDSK",
}
CONFIG_CMAP = "magma"
LOG_RESIDUAL_FLOOR = -5.0
LOG_RESIDUAL_CEIL = 0.0

FIGURE_WIDTH = WIDE_DOUBLE_COLUMN_WIDTH
ROW_HEIGHT = 2.45
FIGURE_MAX_HEIGHT = 5
FIGURE_LEFT = 0.06
FIGURE_RIGHT = 0.96
FIGURE_BOTTOM = 0.08
FIGURE_TOP = 0.90
FIGURE_GRID_WSPACE = -0.05
FIGURE_GRID_HSPACE = 0.2
FIGURE_GRID_WIDTH_RATIOS = (0.6, 0.6, 0.6, 0.6, 0.6, -0.18, 0.05, 0.42, 1.25)
HEATMAP_GRID_COLS = (0, 1, 2, 3, 4)
COLORBAR_GRID_COL = 6
RADIAL_GRID_COL = 8
PANEL_TITLE_PAD = 6
HEATMAP_COLUMN_GAP = 0.018
COLORBAR_MAX_WIDTH = 0.012
COLORBAR_HEIGHT_FRACTION = 0.64
COLORBAR_LEFT_PAD = 0.030
RADIAL_LEFT_PAD = 0.160
HEATMAP_LEVEL_COUNT = 129
HEATMAP_X_TICK_BINS = 2
HEATMAP_Y_TICK_BINS = 4
HEATMAP_X_TICKS = {
    "solovev": (4.0, 8.0),
    "chease": (0.5, 1.5),
    "efit": (1.0, 2.0),
}
BOUNDARY_LINE_WIDTH = 0.8
RADIAL_LINE_WIDTH = 1.4
EXTERNAL_RADIAL_LINE_WIDTH = 0.75 * RADIAL_LINE_WIDTH
EXTERNAL_RADIAL_MARKER_SIZE = 3.0 * RADIAL_LINE_WIDTH
EXTERNAL_RADIAL_MARKER_COUNT = 15
EXTERNAL_RADIAL_ALPHA = 1.0
RADIAL_YMIN = 1.0e-8
RADIAL_YMAX = 100.0
GRID_ALPHA = 0.25
GRID_LINE_WIDTH = 0.5
GRID_LINESTYLE = "-"
COMPACT_COLUMN_LABELS = ("Low", "Medium", "High")
COMPACT_CONFIG_LABELS = ("Low", "Medium", "High")
COMPACT_FIGURE_WIDTH = 7.2
COMPACT_ROW_HEIGHT = 2.1
COMPACT_LEFT = 0.11
COMPACT_RIGHT = 0.96
COMPACT_BOTTOM = 0.08
COMPACT_TOP = 0.93
COMPACT_WSPACE = 0.08
COMPACT_HSPACE = 0.18
SHAPE_SURFACE_LEVELS = (0.2, 0.4, 0.6, 0.8, 1.0)
SHAPE_TARGET_LINESTYLE = (0, (4.0, 2.0))
SHAPE_LINE_WIDTH = 1.15
SHAPE_TARGET_LINE_WIDTH = 0.95


@dataclass(frozen=True)
class ResidualSample:
    case_key: str
    config_label: str
    signature: dict[str, int]
    parameter_count: int | None
    elapsed_ms: float
    solver_residual_norm: float
    rho: np.ndarray
    psin: np.ndarray
    R: np.ndarray
    Z: np.ndarray
    G: np.ndarray
    radial_rms: np.ndarray


def apply_plot_style(font_family: str = PLOT_FONT_FAMILY) -> None:
    plt.rcParams.update(
        {
            "font.family": font_family,
            "font.size": PLOT_BASE_FONT_SIZE * FONT_SCALE,
            "mathtext.fontset": PLOT_MATH_FONTSET,
            "axes.titlesize": TITLE_FONT_SIZE * FONT_SCALE,
            "axes.labelsize": AXIS_LABEL_SIZE * FONT_SCALE,
            "xtick.labelsize": TICK_LABEL_SIZE * FONT_SCALE,
            "ytick.labelsize": TICK_LABEL_SIZE * FONT_SCALE,
            "legend.fontsize": LEGEND_FONT_SIZE * FONT_SCALE,
            "axes.grid": False,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.top": True,
            "ytick.right": True,
        }
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot weak-form GS residual spatial distributions for increasing "
            "MXH/Fourier-Chebyshev representation capacity."
        )
    )
    parser.add_argument("--backend", default="numba")
    parser.add_argument("--case", choices=(*CASE_KEYS, "all"), default="all")
    parser.add_argument("--solve-nr", type=int, default=32)
    parser.add_argument("--solve-nt", type=int, default=32)
    parser.add_argument("--repeat-count", type=int, default=1)
    parser.add_argument("--initial-solve-timeout-s", type=float, default=30.0)
    parser.add_argument("--cache-path", default=DEFAULT_CACHE_PATH)
    parser.add_argument(
        "--refresh-cache",
        action="store_true",
        help="Ignore cached residual samples and recompute before plotting.",
    )
    parser.add_argument("--save-png", default=SAVE_PNG_PATH)
    parser.add_argument("--save-pdf", default=SAVE_PDF_PATH)
    parser.add_argument(
        "--save-compact-png",
        default=SAVE_COMPACT_PNG_PATH,
        help="Also write the compact 3x3 Low/Medium/High-vs-target shape comparison.",
    )
    return parser.parse_args()


def load_fig08_helpers():
    helper_path = Path(__file__).with_name("07-pareto-analysis.py")
    spec = importlib.util.spec_from_file_location("veqpy_fig08_helpers", helper_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load helper script: {helper_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def active_fourier_order(signature: dict[str, int]) -> int:
    order = 0
    for name, length in signature.items():
        if int(length) <= 0:
            continue
        if len(name) >= 2 and name[0] in {"c", "s"} and name[1:].isdigit():
            order = max(order, int(name[1:]))
    return order


def active_radial_length(signature: dict[str, int]) -> int:
    return max((int(length) for length in signature.values() if int(length) > 0), default=0)


def format_config_title(sample: ResidualSample) -> str:
    return ""


def sample_legend_label(sample: ResidualSample) -> str:
    if sample.parameter_count is None:
        return str(sample.config_label)
    return f"{sample.config_label} ({int(sample.parameter_count)})"


def marker_indices(length: int, count: int = EXTERNAL_RADIAL_MARKER_COUNT) -> list[int]:
    n = int(length)
    if n <= 0:
        return []
    if n <= int(count):
        return list(range(n))
    return np.unique(np.linspace(0, n - 1, int(count), dtype=int)).tolist()


def normalize_signature(signature: dict[str, int]) -> dict[str, int]:
    return {str(name): int(length) for name, length in sorted(signature.items()) if int(length) > 0}


def selected_signature_map(fig08, case_keys: tuple[str, ...]) -> dict[str, list[dict[str, int]]]:
    json_stems = [
        str(fig08.DEFAULT_JSON_STEM),
        str(REPO_ROOT / "data" / "pareto"),
    ]
    samples_by_case = None
    for json_stem in dict.fromkeys(json_stems):
        samples_by_case, _frontiers = fig08.load_plot_data_bundle(json_stem, fig08.DEFAULT_SWEEP_MODE)
        if samples_by_case is not None:
            break
    if samples_by_case is None:
        raise FileNotFoundError(
            "Figure 09 now uses the standard configurations selected by scripts/07-pareto-analysis.py; "
            f"run scripts/07-pareto-analysis.py first to generate {REPO_ROOT / 'data' / ('pareto_' + fig08.DEFAULT_SWEEP_MODE + '_*.json')}."
        )

    signatures_by_case: dict[str, list[dict[str, int]]] = {case_key: [] for case_key in case_keys}
    for row_case_key, _threshold, sample in fig08.fastest_standard_config_rows(samples_by_case):
        if row_case_key not in signatures_by_case:
            continue
        if sample is None:
            raise RuntimeError(f"No Figure 08 standard configuration found for {row_case_key!r}.")
        signatures_by_case[row_case_key].append(normalize_signature(sample.signature))

    standard_count = len(CONFIG_LABELS) - 1
    for case_key, signatures in signatures_by_case.items():
        if len(signatures) != standard_count:
            raise RuntimeError(
                f"Expected {standard_count} Figure 08 standard configurations for {case_key!r}, got {len(signatures)}."
            )
        signatures.append(normalize_signature(fig08.get_max_lengths(case_key)))
    return signatures_by_case


def solve_residual_sample(
    fig08,
    benchmark,
    reference,
    *,
    case_key: str,
    config_label: str,
    signature: dict[str, int],
    solve_nr: int,
    solve_nt: int,
    repeat_count: int,
    initial_solve_timeout_s: float,
) -> ResidualSample:
    solve_grid = benchmark.Grid(
        Nr=int(solve_nr),
        Nt=int(solve_nt),
        quadrature_scheme="legendre",
        L_max=int(benchmark.REFERENCE_GRID.L_max),
        M_max=int(benchmark.REFERENCE_GRID.M_max),
    )
    case = fig08.build_pf_case(benchmark, reference, solve_grid, signature)
    result, equilibrium, elapsed_ms = fig08.solve_with_timing(
        benchmark,
        case,
        solve_grid,
        int(repeat_count),
        method=fig08.CASE_SOLVER_METHODS[case_key],
        solver_config=benchmark.CONFIG,
        initial_solve_timeout_s=float(initial_solve_timeout_s),
    )
    # Figure 9 is meant to show the residual on the actual residual/solve
    # quadrature grid, not on a post-solve presentation grid.  This keeps the
    # heatmaps and radial RMS curves tied to the same nodes that enter the weak
    # residual projections.
    surface_equilibrium = equilibrium
    G = np.asarray(surface_equilibrium.G, dtype=np.float64)
    radial_rms = np.sqrt(np.nanmean(G * G, axis=1))
    return ResidualSample(
        case_key=case_key,
        config_label=config_label,
        signature=dict(signature),
        parameter_count=int(np.size(result.x)),
        elapsed_ms=float(elapsed_ms),
        solver_residual_norm=float(result.residual_norm_final),
        rho=np.asarray(surface_equilibrium.rho, dtype=np.float64),
        psin=np.asarray(surface_equilibrium.psin, dtype=np.float64),
        R=np.asarray(surface_equilibrium.geometry.R, dtype=np.float64),
        Z=np.asarray(surface_equilibrium.geometry.Z, dtype=np.float64),
        G=G,
        radial_rms=radial_rms,
    )


def geqdsk_standard_balance_grid(fig08, case_key: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate the standard R-Z GS balance on the exported GEQDSK grid.

    The returned field is

        Delta*psi + mu0 R^2 p'(psi) + F F'(psi)

    on the second-order finite-difference interior of the rectangular grid.
    It is not yet multiplied by the flux-coordinate factor J/R.
    """

    geqdsk = fig08.read_geqdsk(fig08.CASE_REFERENCE_GFILES[case_key])
    psi = np.asarray(geqdsk.psi, dtype=np.float64)
    if psi.shape != (int(geqdsk.NR), int(geqdsk.NZ)):
        raise ValueError(f"Unexpected GEQDSK psi shape for {case_key}: {psi.shape}")
    if int(geqdsk.NR) < 3 or int(geqdsk.NZ) < 3:
        empty = np.asarray([], dtype=np.float64)
        return empty, empty, np.empty((0, 0), dtype=np.float64)

    R = np.linspace(float(geqdsk.Rmin), float(geqdsk.Rmax), int(geqdsk.NR), dtype=np.float64)
    Z = np.linspace(float(geqdsk.Zmin), float(geqdsk.Zmax), int(geqdsk.NZ), dtype=np.float64)
    dR = float(R[1] - R[0])
    dZ = float(Z[1] - Z[0])
    if dR == 0.0 or dZ == 0.0:
        empty = np.asarray([], dtype=np.float64)
        return empty, empty, np.empty((0, 0), dtype=np.float64)

    psi_center = psi[1:-1, 1:-1]
    psi_R = (psi[2:, 1:-1] - psi[:-2, 1:-1]) / (2.0 * dR)
    psi_RR = (psi[2:, 1:-1] - 2.0 * psi_center + psi[:-2, 1:-1]) / (dR * dR)
    psi_ZZ = (psi[1:-1, 2:] - 2.0 * psi_center + psi[1:-1, :-2]) / (dZ * dZ)
    R_center = R[1:-1, None]

    psi_span = float(geqdsk.psi_bound) - float(geqdsk.psi_axis)
    if psi_span == 0.0:
        return R[1:-1], Z[1:-1], np.full_like(psi_center, np.nan, dtype=np.float64)
    psin = (psi_center - float(geqdsk.psi_axis)) / psi_span
    source_axis = np.linspace(0.0, 1.0, len(geqdsk.P_psi), dtype=np.float64)
    p_psi = np.interp(psin.ravel(), source_axis, np.asarray(geqdsk.P_psi, dtype=np.float64)).reshape(psin.shape)
    ff_psi = np.interp(psin.ravel(), source_axis, np.asarray(geqdsk.FF_psi, dtype=np.float64)).reshape(psin.shape)

    gs_balance = psi_RR - psi_R / R_center + psi_ZZ + fig08.MU0 * R_center**2 * p_psi + ff_psi
    return R[1:-1], Z[1:-1], gs_balance


def external_reference_residual_sample(fig08, reference, *, case_key: str) -> ResidualSample:
    """Build the left-column external residual sample used in Figure 9.

    For the analytic D-shape file this is the finite-difference residual of the
    generated analytic exchange grid.  For the GEQDSK-based cases, the standard
    R-Z balance residual is evaluated on the exported grid, interpolated to the
    same flux-coordinate nodes used by the VEQ diagnostics, and multiplied by
    the mapped J/R factor so that the reported quantity is the same
    coordinate-form residual density mathcal G as in the VEQ rows.
    """

    equilibrium = reference.equilibrium
    R_axis, Z_axis, gs_balance = geqdsk_standard_balance_grid(fig08, case_key)
    if R_axis.size == 0 or Z_axis.size == 0:
        G = np.full_like(np.asarray(equilibrium.geometry.R, dtype=np.float64), np.nan)
    else:
        interpolator = RegularGridInterpolator(
            (R_axis, Z_axis),
            gs_balance,
            bounds_error=False,
            fill_value=np.nan,
        )
        points = np.column_stack(
            (
                np.asarray(equilibrium.geometry.R, dtype=np.float64).ravel(),
                np.asarray(equilibrium.geometry.Z, dtype=np.float64).ravel(),
            )
        )
        gs_on_flux_grid = interpolator(points).reshape(np.asarray(equilibrium.geometry.R).shape)
        G = np.asarray(equilibrium.geometry.JdivR, dtype=np.float64) * gs_on_flux_grid
    radial_rms = np.sqrt(np.nanmean(G * G, axis=1))
    return ResidualSample(
        case_key=case_key,
        config_label=EXTERNAL_REFERENCE_LABELS[case_key],
        signature={},
        parameter_count=None,
        elapsed_ms=float("nan"),
        solver_residual_norm=float("nan"),
        rho=np.asarray(equilibrium.rho, dtype=np.float64),
        psin=np.asarray(equilibrium.psin, dtype=np.float64),
        R=np.asarray(equilibrium.geometry.R, dtype=np.float64),
        Z=np.asarray(equilibrium.geometry.Z, dtype=np.float64),
        G=G,
        radial_rms=radial_rms,
    )


def solve_case_samples(
    fig08,
    benchmark,
    *,
    case_key: str,
    signatures: list[dict[str, int]],
    args: argparse.Namespace,
) -> list[ResidualSample]:
    reference = fig08.build_reference_case(benchmark, case_key)
    samples: list[ResidualSample] = []
    for config_label, signature in zip(CONFIG_LABELS, signatures, strict=True):
        samples.append(
            solve_residual_sample(
                fig08,
                benchmark,
                reference,
                case_key=case_key,
                config_label=config_label,
                signature=signature,
                solve_nr=args.solve_nr,
                solve_nt=args.solve_nt,
                repeat_count=args.repeat_count,
                initial_solve_timeout_s=args.initial_solve_timeout_s,
            )
        )
    samples.append(external_reference_residual_sample(fig08, reference, case_key=case_key))
    return samples


def cache_signature(
    *,
    case_keys: tuple[str, ...],
    solve_nr: int,
    solve_nt: int,
    backend: str,
    signatures_by_case: dict[str, list[dict[str, int]]],
) -> dict[str, object]:
    return {
        "cache_version": RESIDUAL_CACHE_VERSION,
        "case_keys": list(case_keys),
        "solve_nr": int(solve_nr),
        "solve_nt": int(solve_nt),
        "backend": str(backend),
        "quantity": "local_gs_residual_on_solve_grid",
        "signature_source": "fig08_fastest_standard_frontier",
        "config_labels": [*CONFIG_LABELS, "External"],
        "selected_signatures": {
            case_key: [normalize_signature(signature) for signature in signatures_by_case[case_key]]
            for case_key in case_keys
        },
    }


def save_case_samples_cache(
    path: str | os.PathLike[str],
    case_samples: dict[str, list[ResidualSample]],
    *,
    case_keys: tuple[str, ...],
    solve_nr: int,
    solve_nt: int,
    backend: str,
    signatures_by_case: dict[str, list[dict[str, int]]],
) -> None:
    entries: list[dict[str, object]] = []
    arrays: dict[str, np.ndarray] = {}
    idx = 0
    for case_key in case_keys:
        for sample in case_samples.get(case_key, []):
            entries.append(
                {
                    "idx": idx,
                    "case_key": sample.case_key,
                    "config_label": sample.config_label,
                    "signature": sample.signature,
                    "parameter_count": None if sample.parameter_count is None else int(sample.parameter_count),
                    "elapsed_ms": float(sample.elapsed_ms),
                    "solver_residual_norm": float(sample.solver_residual_norm),
                }
            )
            arrays[f"rho_{idx}"] = np.asarray(sample.rho, dtype=np.float64)
            arrays[f"psin_{idx}"] = np.asarray(sample.psin, dtype=np.float64)
            arrays[f"R_{idx}"] = np.asarray(sample.R, dtype=np.float64)
            arrays[f"Z_{idx}"] = np.asarray(sample.Z, dtype=np.float64)
            arrays[f"G_{idx}"] = np.asarray(sample.G, dtype=np.float64)
            arrays[f"radial_rms_{idx}"] = np.asarray(sample.radial_rms, dtype=np.float64)
            idx += 1

    payload = {
        **cache_signature(
            case_keys=case_keys,
            solve_nr=solve_nr,
            solve_nt=solve_nt,
            backend=backend,
            signatures_by_case=signatures_by_case,
        ),
        "samples": entries,
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, metadata=np.asarray(json.dumps(payload)), **arrays)


def load_case_samples_cache(
    path: str | os.PathLike[str],
    *,
    case_keys: tuple[str, ...],
    solve_nr: int,
    solve_nt: int,
    backend: str,
    signatures_by_case: dict[str, list[dict[str, int]]],
) -> dict[str, list[ResidualSample]] | None:
    if not os.path.exists(path):
        return None
    try:
        with np.load(path, allow_pickle=False) as data:
            raw_metadata = data["metadata"]
            payload = json.loads(str(raw_metadata.item() if raw_metadata.shape == () else raw_metadata))
            expected = cache_signature(
                case_keys=case_keys,
                solve_nr=solve_nr,
                solve_nt=solve_nt,
                backend=backend,
                signatures_by_case=signatures_by_case,
            )
            for key, expected_value in expected.items():
                if payload.get(key) != expected_value:
                    return None
            entries = payload.get("samples")
            if not isinstance(entries, list):
                return None
            case_samples: dict[str, list[ResidualSample]] = {case_key: [] for case_key in case_keys}
            for entry in entries:
                if not isinstance(entry, dict):
                    return None
                idx = int(entry["idx"])
                case_key = str(entry["case_key"])
                if case_key not in case_samples:
                    continue
                sample = ResidualSample(
                    case_key=case_key,
                    config_label=str(entry["config_label"]),
                    signature={str(name): int(value) for name, value in dict(entry["signature"]).items()},
                    parameter_count=None if entry.get("parameter_count") is None else int(entry["parameter_count"]),
                    elapsed_ms=float(entry["elapsed_ms"]),
                    solver_residual_norm=float(entry["solver_residual_norm"]),
                    rho=np.asarray(data[f"rho_{idx}"], dtype=np.float64),
                    psin=np.asarray(data[f"psin_{idx}"], dtype=np.float64),
                    R=np.asarray(data[f"R_{idx}"], dtype=np.float64),
                    Z=np.asarray(data[f"Z_{idx}"], dtype=np.float64),
                    G=np.asarray(data[f"G_{idx}"], dtype=np.float64),
                    radial_rms=np.asarray(data[f"radial_rms_{idx}"], dtype=np.float64),
                )
                case_samples[case_key].append(sample)
            if any(len(case_samples[case_key]) != len(CONFIG_LABELS) + 1 for case_key in case_keys):
                return None
            return case_samples
    except (OSError, KeyError, ValueError, json.JSONDecodeError):
        return None


def residual_scale(samples: list[ResidualSample]) -> float:
    values = np.concatenate([np.ravel(np.abs(sample.G[np.isfinite(sample.G)])) for sample in samples])
    if values.size == 0:
        return 1.0
    scale = float(np.nanmax(values))
    return max(scale, 1.0e-30)


def log_residual_field(sample: ResidualSample, *, scale: float) -> np.ndarray:
    normalized = np.abs(sample.G) / max(float(scale), 1.0e-30)
    with np.errstate(divide="ignore", invalid="ignore"):
        log_field = np.log10(normalized + 10.0 ** (LOG_RESIDUAL_FLOOR - 1.0))
    return np.clip(log_field, LOG_RESIDUAL_FLOOR, LOG_RESIDUAL_CEIL)


def _finite_abs_values(values: np.ndarray) -> np.ndarray:
    finite = np.asarray(values, dtype=np.float64)
    finite = np.abs(finite[np.isfinite(finite)])
    return finite


def _rms(values: np.ndarray) -> float:
    finite = _finite_abs_values(values)
    if finite.size == 0:
        return float("nan")
    return float(np.sqrt(np.mean(finite * finite)))


def _max(values: np.ndarray) -> float:
    finite = _finite_abs_values(values)
    if finite.size == 0:
        return float("nan")
    return float(np.max(finite))


def _region_rms(sample: ResidualSample, *, lower: float | None = None, upper: float | None = None) -> float:
    psin = np.asarray(sample.psin, dtype=np.float64)
    values = np.asarray(sample.G, dtype=np.float64)
    if psin.ndim != 1 or values.ndim == 0 or values.shape[0] != psin.shape[0]:
        return float("nan")
    mask = np.ones(psin.shape, dtype=bool)
    if lower is not None:
        mask &= psin >= float(lower)
    if upper is not None:
        mask &= psin < float(upper)
    if not np.any(mask):
        return float("nan")
    return _rms(values[mask, ...])


def residual_norm_row(sample: ResidualSample) -> dict[str, object]:
    residual_rms = _rms(sample.G)
    residual_max = _max(sample.G)
    residual_rms_interior = _region_rms(sample, upper=0.8)
    residual_rms_edge = _region_rms(sample, lower=0.8)
    return {
        "case_key": sample.case_key,
        "case": CASE_LABELS[sample.case_key],
        "case_params": (
            f"{CASE_LABELS[sample.case_key]} {sample.config_label}"
            if sample.parameter_count is None
            else f"{CASE_LABELS[sample.case_key]} ({int(sample.parameter_count)})"
        ),
        "config": sample.config_label,
        "params": None if sample.parameter_count is None else int(sample.parameter_count),
        # The packed weak/projected residual is the nonlinear system that VEQ
        # actually solves.  We report the final packed norm directly instead
        # of normalizing by the zero-coefficient initial residual, because some
        # strongly shaped cases have a singular or near-singular zero geometry
        # that makes the initial residual scale non-physical.
        "epsilon_proj": float(sample.solver_residual_norm),
        "G_rms": residual_rms,
        "G_rms_interior": residual_rms_interior,
        "G_rms_edge": residual_rms_edge,
        "G_max": residual_max,
        "solver_residual_norm": float(sample.solver_residual_norm),
    }


def residual_norm_rows(case_samples: dict[str, list[ResidualSample]]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for case_key in case_samples:
        rows.extend(residual_norm_row(sample) for sample in case_samples[case_key])
    return rows


def _format_scientific(value: object) -> str:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return ""
    if not np.isfinite(numeric):
        return "--"
    return f"{numeric:.3e}"


def _format_tex_number(value: object, *, precision: int = 3) -> str:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return "--"
    if not np.isfinite(numeric):
        return "--"
    if numeric == 0.0:
        return f"${numeric:.{precision}f}$"
    magnitude = abs(numeric)
    if 1.0e-2 <= magnitude < 1.0e3:
        return f"${numeric:.{precision}f}$"
    exponent = int(np.floor(np.log10(magnitude)))
    mantissa = numeric / (10.0**exponent)
    return rf"${mantissa:.{precision}f}\times 10^{{{exponent}}}$"


def build_residual_norm_latex_table(rows: list[dict[str, object]]) -> str:
    indent = "              "
    header = [
        "Case",
        r"$\epsilon_{\mathrm{proj}}$",
        r"$\mathrm{RMS}_{\mathrm{all}}(\mathcal{G})$",
        r"$\mathrm{RMS}_{<0.8}(\mathcal{G})$",
        r"$\mathrm{RMS}_{\geq0.8}(\mathcal{G})$",
        r"$|\mathcal{G}|_{\max}$",
    ]
    table_rows: list[list[str]] = []
    for row in rows:
        table_rows.append(
            [
                str(row["case_params"]),
                _format_tex_number(row["epsilon_proj"]),
                _format_tex_number(row["G_rms"]),
                _format_tex_number(row["G_rms_interior"]),
                _format_tex_number(row["G_rms_edge"]),
                _format_tex_number(row["G_max"]),
            ]
        )
    column_widths = [
        max(len(row[column_index]) for row in [header, *table_rows]) for column_index in range(len(header))
    ]

    def format_row(row: list[str]) -> str:
        return " & ".join(cell.ljust(column_widths[index]) for index, cell in enumerate(row)) + r" \\"

    return "\n".join(
        indent + line
        for line in [
            r"\hline",
            format_row(header),
            r"\hline",
            *(format_row(row) for row in table_rows),
            r"\hline",
        ]
    )


def print_residual_norm_latex_table(rows: list[dict[str, object]]) -> None:
    print(build_residual_norm_latex_table(rows))


def periodic_surface_arrays(sample: ResidualSample, values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return (
        np.hstack([sample.R, sample.R[:, :1]]),
        np.hstack([sample.Z, sample.Z[:, :1]]),
        np.hstack([values, values[:, :1]]),
    )


def rz_limits(
    samples: list[ResidualSample],
) -> tuple[tuple[float, float], tuple[float, float]]:
    R_all = np.concatenate([np.ravel(sample.R) for sample in samples])
    Z_all = np.concatenate([np.ravel(sample.Z) for sample in samples])
    rmin, rmax = float(np.nanmin(R_all)), float(np.nanmax(R_all))
    zmin, zmax = float(np.nanmin(Z_all)), float(np.nanmax(Z_all))
    rpad = max(0.06 * (rmax - rmin), 1.0e-6)
    zpad = max(0.06 * (zmax - zmin), 1.0e-6)
    return (rmin - rpad, rmax + rpad), (zmin - zpad, zmax + zpad)


def style_rz_axis(ax: plt.Axes, *, xlim: tuple[float, float], ylim: tuple[float, float]) -> None:
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.tick_params(labelsize=scaled_font_size(TICK_LABEL_SIZE))
    ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=HEATMAP_X_TICK_BINS))
    ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=HEATMAP_Y_TICK_BINS))


def plot_heatmap_panel(
    ax: plt.Axes,
    sample: ResidualSample,
    *,
    scale: float,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
) -> matplotlib.contour.QuadContourSet:
    log_field = log_residual_field(sample, scale=scale)
    R_plot, Z_plot, G_plot = periodic_surface_arrays(sample, log_field)
    levels = np.linspace(LOG_RESIDUAL_FLOOR, LOG_RESIDUAL_CEIL, HEATMAP_LEVEL_COUNT)
    contour = ax.contourf(
        R_plot,
        Z_plot,
        G_plot,
        levels=levels,
        cmap=CONFIG_CMAP,
        norm=colors.Normalize(vmin=LOG_RESIDUAL_FLOOR, vmax=LOG_RESIDUAL_CEIL),
        extend="min",
    )
    ax.plot(
        np.r_[sample.R[-1], sample.R[-1, 0]],
        np.r_[sample.Z[-1], sample.Z[-1, 0]],
        color="white",
        lw=BOUNDARY_LINE_WIDTH,
        alpha=0.95,
    )
    ax.set_title(
        format_config_title(sample),
        fontsize=scaled_font_size(TITLE_FONT_SIZE),
        fontweight="normal",
    )
    style_rz_axis(ax, xlim=xlim, ylim=ylim)
    return contour


def add_centered_panel_title(
    fig: plt.Figure,
    left_ax: plt.Axes,
    right_ax: plt.Axes,
    title: str,
    *,
    pad: float,
) -> None:
    left_bbox = left_ax.get_position()
    right_bbox = right_ax.get_position()
    center_x = 0.5 * (left_bbox.x0 + right_bbox.x1)
    y = max(left_bbox.y1, right_bbox.y1) + (pad / 72.0) / fig.get_figheight()
    fig.text(
        center_x,
        y,
        title,
        ha="center",
        va="bottom",
        fontsize=scaled_font_size(TITLE_FONT_SIZE),
    )


def align_heatmap_columns(
    fig: plt.Figure,
    heatmap_rows: list[list[plt.Axes]],
    *,
    gap: float = HEATMAP_COLUMN_GAP,
) -> float:
    """Align the four heatmap columns while preserving equal-aspect panel boxes."""
    if not heatmap_rows:
        return 0.0
    bboxes_by_row = [[ax.get_position().frozen() for ax in row] for row in heatmap_rows]
    ncols = len(bboxes_by_row[0])
    column_widths = [max(float(row[col].width) for row in bboxes_by_row) for col in range(ncols)]
    x0 = min(float(row[0].x0) for row in bboxes_by_row)
    centers: list[float] = []
    x = x0
    for width in column_widths:
        centers.append(x + 0.5 * width)
        x += width + float(gap)

    for axes, row_bboxes in zip(heatmap_rows, bboxes_by_row, strict=True):
        for col, (ax, bbox) in enumerate(zip(axes, row_bboxes, strict=True)):
            ax.set_position([centers[col] - 0.5 * bbox.width, bbox.y0, bbox.width, bbox.height])
    return centers[-1] + 0.5 * column_widths[-1]


def plot_radial_panel(
    ax: plt.Axes,
    samples: list[ResidualSample],
    *,
    scale: float,
    case_key: str,
    show_xlabel: bool,
    legend_loc: str = "upper left",
) -> None:
    style_by_label = {
        "Ref": ("-", CASE_LINE_COLORS[case_key][-1], RADIAL_LINE_WIDTH, 1.0, "x", 1.15 * EXTERNAL_RADIAL_MARKER_SIZE),
        "High": ("-", CASE_LINE_COLORS[case_key][-2], RADIAL_LINE_WIDTH, 1.0, None, 0.0),
        "Medium": ("--", CASE_LINE_COLORS[case_key][-3], RADIAL_LINE_WIDTH, 1.0, None, 0.0),
        "Low": (
            (0, (5, 1.6, 1.2, 1.6, 1.2, 1.6)),
            CASE_LINE_COLORS[case_key][-4],
            RADIAL_LINE_WIDTH,
            1.0,
            None,
            0.0,
        ),
        "Analytic": (
            "-",
            "#000000",
            EXTERNAL_RADIAL_LINE_WIDTH,
            EXTERNAL_RADIAL_ALPHA,
            "o",
            EXTERNAL_RADIAL_MARKER_SIZE,
        ),
        "GEQDSK": ("-", "#000000", EXTERNAL_RADIAL_LINE_WIDTH, EXTERNAL_RADIAL_ALPHA, "o", EXTERNAL_RADIAL_MARKER_SIZE),
    }
    zorder_by_label = {
        "Analytic": 1.0,
        "GEQDSK": 1.0,
        "Ref": 1.5,
        "Low": 2.0,
        "Medium": 2.5,
        "High": 3.0,
    }
    for sample in reversed(samples):
        _ = scale
        linestyle, line_color, line_width, line_alpha, marker, marker_size = style_by_label.get(
            sample.config_label,
            ("-", "#111111", RADIAL_LINE_WIDTH, 1.0, None, 0.0),
        )
        y = sample.radial_rms
        ax.semilogy(
            sample.rho,
            np.maximum(y, 1.0e-12),
            color=line_color,
            ls=linestyle,
            lw=line_width,
            alpha=line_alpha,
            marker=marker,
            markersize=marker_size,
            markevery=marker_indices(len(sample.rho)) if marker is not None else None,
            markerfacecolor=line_color,
            markeredgecolor=line_color,
            markeredgewidth=0.9 if marker == "x" else 0.6,
            zorder=zorder_by_label.get(sample.config_label, 2.0),
            label=sample_legend_label(sample),
        )
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(RADIAL_YMIN, RADIAL_YMAX)
    ax.yaxis.set_major_locator(ticker.LogLocator(base=10.0))
    ax.yaxis.set_major_formatter(ticker.LogFormatterMathtext(base=10.0))
    ax.yaxis.set_minor_locator(ticker.NullLocator())
    ax.set_ylabel(r"$\varepsilon_{\mathcal{G}}(\rho)$", fontsize=scaled_font_size(AXIS_LABEL_SIZE))
    if show_xlabel:
        ax.set_xlabel(r"$\rho$", fontsize=scaled_font_size(AXIS_LABEL_SIZE))
    else:
        ax.set_xlabel("")
    ax.grid(True, alpha=GRID_ALPHA, linewidth=GRID_LINE_WIDTH, linestyle=GRID_LINESTYLE)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(
        handles[::-1],
        labels[::-1],
        loc=legend_loc,
        ncol=3,
        frameon=False,
        fontsize=scaled_font_size(LEGEND_FONT_SIZE),
        handlelength=1.7,
        columnspacing=0.7,
        labelspacing=0.2,
    )
    ax.tick_params(labelsize=scaled_font_size(TICK_LABEL_SIZE), labelbottom=show_xlabel)


def build_figure(case_samples: dict[str, list[ResidualSample]]) -> plt.Figure:
    case_keys = list(case_samples)
    figure_height = min(max(ROW_HEIGHT * len(case_keys), ROW_HEIGHT), FIGURE_MAX_HEIGHT)
    fig = plt.figure(figsize=(FIGURE_WIDTH, figure_height))
    grid = fig.add_gridspec(
        nrows=len(case_keys),
        ncols=len(FIGURE_GRID_WIDTH_RATIOS),
        width_ratios=FIGURE_GRID_WIDTH_RATIOS,
        left=FIGURE_LEFT,
        right=FIGURE_RIGHT,
        bottom=FIGURE_BOTTOM,
        top=FIGURE_TOP,
        wspace=FIGURE_GRID_WSPACE,
        hspace=FIGURE_GRID_HSPACE,
    )
    mappable = None
    max_heatmap_right = 0.0
    heatmap_rows: list[list[plt.Axes]] = []
    radial_axes: list[plt.Axes] = []
    for row, case_key in enumerate(case_keys):
        samples = case_samples[case_key]
        scale = residual_scale(samples)
        xlim, ylim = rz_limits(samples)
        heatmap_anchors = ("C", "C", "C", "C", "C")
        heatmap_axes = []
        for col, (grid_col, sample) in enumerate(zip(HEATMAP_GRID_COLS, samples, strict=True)):
            ax = fig.add_subplot(grid[row, grid_col])
            heatmap_axes.append(ax)
            mappable = plot_heatmap_panel(ax, sample, scale=scale, xlim=xlim, ylim=ylim)
            ax.set_xticks(HEATMAP_X_TICKS[case_key])
            ax.xaxis.set_major_formatter(ticker.FormatStrFormatter("%.1f"))
            ax.set_anchor(heatmap_anchors[col])
            if col == 0:
                ax.set_ylabel(
                    f"{CASE_LABELS[case_key]}\nZ [m]",
                    fontsize=scaled_font_size(AXIS_LABEL_SIZE),
                )
            else:
                ax.set_yticklabels([])
            if row == len(case_keys) - 1:
                ax.set_xlabel("R [m]", fontsize=scaled_font_size(AXIS_LABEL_SIZE))
            else:
                ax.set_xlabel("")
        heatmap_rows.append(heatmap_axes)
        radial_ax = fig.add_subplot(grid[row, RADIAL_GRID_COL])
        radial_axes.append(radial_ax)
        plot_radial_panel(
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
                pad=PANEL_TITLE_PAD,
                fontsize=scaled_font_size(TITLE_FONT_SIZE),
                fontweight="normal",
            )

    max_heatmap_right = align_heatmap_columns(fig, heatmap_rows)
    if heatmap_rows:
        add_centered_panel_title(
            fig,
            heatmap_rows[0][0],
            heatmap_rows[0][-1],
            r"$\bf{(a)}$ Residual distribution",
            pad=PANEL_TITLE_PAD,
        )

    radial_left = min(max_heatmap_right + RADIAL_LEFT_PAD, FIGURE_RIGHT - 0.25)
    for radial_ax, heatmap_axes in zip(radial_axes, heatmap_rows, strict=True):
        row_bboxes = [ax.get_position().frozen() for ax in heatmap_axes]
        y0 = min(float(bbox.y0) for bbox in row_bboxes)
        y1 = max(float(bbox.y1) for bbox in row_bboxes)
        radial_ax.set_position([radial_left, y0, FIGURE_RIGHT - radial_left, y1 - y0])

    if mappable is not None:
        cax = fig.add_subplot(grid[:, COLORBAR_GRID_COL])
        cbar_bbox = cax.get_position()
        cbar_width = min(COLORBAR_MAX_WIDTH, cbar_bbox.width)
        cbar_height = COLORBAR_HEIGHT_FRACTION * cbar_bbox.height
        cbar_y = cbar_bbox.y0 + 0.5 * (cbar_bbox.height - cbar_height)
        cbar_x = max_heatmap_right + COLORBAR_LEFT_PAD
        cax.set_position([cbar_x, cbar_y, cbar_width, cbar_height])
        cbar = fig.colorbar(
            mappable,
            cax=cax,
        )
        cbar_ticks = np.arange(LOG_RESIDUAL_FLOOR, LOG_RESIDUAL_CEIL + 0.5, 1.0)
        cbar.set_ticks(cbar_ticks)
        cbar.set_ticklabels([rf"$10^{{{int(tick)}}}$" for tick in cbar_ticks])
        cbar.ax.yaxis.set_ticks_position("right")
        cbar.ax.yaxis.set_label_position("right")
        cbar.set_label("")

        # latex 波浪线
        cbar.ax.set_title(r"$\tilde{\mathcal{G}}$", fontsize=scaled_font_size(AXIS_LABEL_SIZE), pad=6.0)
        cbar.ax.tick_params(which="both", direction="out", labelsize=scaled_font_size(TICK_LABEL_SIZE))
    return fig


def compact_case_samples(samples: list[ResidualSample]) -> list[ResidualSample]:
    by_label = {sample.config_label: sample for sample in samples}
    return [by_label[label] for label in COMPACT_CONFIG_LABELS]


def target_sample(samples: list[ResidualSample]) -> ResidualSample:
    return next((sample for sample in samples if sample.parameter_count is None), samples[-1])


def surface_at_level(sample: ResidualSample, level: float) -> tuple[np.ndarray, np.ndarray]:
    # Shape overlays compare like-for-like normalized radial locations.  The
    # solved psin profile is itself an active unknown, so nearest-psin lookup
    # can select different radial grid rows across configurations and create a
    # spurious apparent shape mismatch.  The Legendre rho grid is common across
    # all residual samples, including the target/reference sample.
    rho = np.asarray(sample.rho, dtype=np.float64)
    if rho.ndim != 1 or rho.size == 0:
        return np.asarray([], dtype=np.float64), np.asarray([], dtype=np.float64)
    idx = int(np.nanargmin(np.abs(rho - float(level))))
    return np.r_[sample.R[idx], sample.R[idx, 0]], np.r_[sample.Z[idx], sample.Z[idx, 0]]


def plot_shape_comparison_panel(
    ax: plt.Axes,
    sample: ResidualSample,
    target: ResidualSample,
    *,
    case_key: str,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
    show_legend: bool,
) -> None:
    color_by_label = {
        "Low": CASE_LINE_COLORS[case_key][1],
        "Medium": CASE_LINE_COLORS[case_key][2],
        "High": CASE_LINE_COLORS[case_key][3],
        "Ref": CASE_LINE_COLORS[case_key][4],
    }
    line_color = color_by_label.get(sample.config_label, CASE_COLORS[case_key])
    sample_psin = np.asarray(sample.psin, dtype=np.float64)
    for level in SHAPE_SURFACE_LEVELS:
        target_r, target_z = surface_at_level(target, level)
        sample_idx = int(np.nanargmin(np.abs(sample_psin - float(level))))
        sample_r = np.r_[sample.R[sample_idx], sample.R[sample_idx, 0]]
        sample_z = np.r_[sample.Z[sample_idx], sample.Z[sample_idx, 0]]
        if target_r.size:
            ax.plot(
                target_r,
                target_z,
                color="#111111",
                lw=SHAPE_TARGET_LINE_WIDTH,
                ls=SHAPE_TARGET_LINESTYLE,
                alpha=0.85 if level == 1.0 else 0.45,
                label="Target" if show_legend and level == SHAPE_SURFACE_LEVELS[-1] else None,
            )
        if sample_r.size:
            ax.plot(
                sample_r,
                sample_z,
                color=line_color,
                lw=SHAPE_LINE_WIDTH,
                alpha=1.0 if level == 1.0 else 0.65,
                label=sample_legend_label(sample) if show_legend and level == SHAPE_SURFACE_LEVELS[-1] else None,
            )
    style_rz_axis(ax, xlim=xlim, ylim=ylim)
    ax.xaxis.set_major_formatter(ticker.FormatStrFormatter("%.1f"))
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.1f"))
    if show_legend:
        ax.legend(
            loc="upper left",
            frameon=False,
            fontsize=scaled_font_size(LEGEND_FONT_SIZE),
            handlelength=1.8,
            labelspacing=0.2,
        )


def _point_in_polygon(poly: np.ndarray, R: float, Z: float) -> bool:
    """Ray-casting test: whether (R, Z) lies inside a closed polygon."""
    pts = np.asarray(poly, dtype=np.float64)
    n = pts.shape[0]
    inside = False
    j = n - 1
    for i in range(n):
        yi, zi = pts[i, 1], pts[i, 0]
        yj, zj = pts[j, 1], pts[j, 0]
        if (zi > Z) != (zj > Z):
            intersect = (yj - yi) * (Z - zi) / (zj - zi) + yi
            if R < intersect:
                inside = not inside
        j = i
    return inside


def _select_contour(candidates: list[np.ndarray], *, axis_center: tuple[float, float]) -> np.ndarray | None:
    """Pick the longest closed contour segment that encloses the magnetic axis."""
    selected = None
    selected_length = -1
    for curve in candidates:
        arr = np.asarray(curve, dtype=np.float64)
        if arr.shape[0] < 8:
            continue
        if not _point_in_polygon(arr, axis_center[0], axis_center[1]):
            continue
        if arr.shape[0] > selected_length:
            selected = arr.copy()
            selected_length = arr.shape[0]
    if selected is not None:
        return selected
    if candidates:
        return max((np.asarray(c, dtype=np.float64) for c in candidates), key=len)
    return None


def _contour_flux_surfaces(geqdsk, levels: tuple[float, ...]) -> dict[float, np.ndarray]:
    """Extract flux-surface (R,Z) contours from a GEQDSK psi grid at given psin levels."""
    psi = np.asarray(geqdsk.psi, dtype=np.float64)
    psi_span = float(geqdsk.psi_bound - geqdsk.psi_axis)
    psin_grid = (psi.T - float(geqdsk.psi_axis)) / psi_span
    R = np.linspace(float(geqdsk.Rmin), float(geqdsk.Rmax), int(geqdsk.NR), dtype=np.float64)
    Z = np.linspace(float(geqdsk.Zmin), float(geqdsk.Zmax), int(geqdsk.NZ), dtype=np.float64)
    axis_center = (float(geqdsk.Raxis), float(geqdsk.Zaxis))
    surfaces: dict[float, np.ndarray] = {}
    subsurf = [lv for lv in levels if lv < 1.0 - 1e-12]
    if subsurf:
        fig, ax = plt.subplots()
        contour = ax.contour(R, Z, psin_grid, levels=subsurf)
        plt.close(fig)
        for idx, level in enumerate(subsurf):
            seg = _select_contour(contour.allsegs[idx], axis_center=axis_center)
            if seg is not None:
                surfaces[level] = seg
    if any(abs(lv - 1.0) <= 1e-12 for lv in levels):
        surfaces[1.0] = np.asarray(geqdsk.boundary, dtype=np.float64)
    return surfaces


def _resample_contour_sequential(points: np.ndarray, n_theta: int) -> np.ndarray:
    """Resample a closed contour to *n_theta* points by cumulative chord length."""
    pts = np.asarray(points, dtype=np.float64)
    if pts.shape[0] < 2:
        return np.full((n_theta, 2), np.nan, dtype=np.float64)
    closed = np.allclose(pts[0], pts[-1], rtol=1e-12, atol=1e-12)
    loop = pts if closed else np.vstack([pts, pts[:1]])
    d = np.sqrt(np.sum(np.diff(loop, axis=0) ** 2, axis=1))
    s = np.concatenate([[0.0], np.cumsum(d)])
    total = s[-1]
    if total < 1e-30:
        return np.full((n_theta, 2), np.nan, dtype=np.float64)
    s_uniform = np.linspace(0.0, total, int(n_theta), endpoint=False, dtype=np.float64)
    out = np.empty((n_theta, 2), dtype=np.float64)
    out[:, 0] = np.interp(s_uniform, s, loop[:, 0])
    out[:, 1] = np.interp(s_uniform, s, loop[:, 1])
    return out


def build_true_target_sample(fig08, case_key: str, *, n_theta: int = 256) -> ResidualSample:
    """Build a ResidualSample whose R,Z come from direct GEQDSK psi contouring."""
    geqdsk = fig08.read_geqdsk(fig08.CASE_REFERENCE_GFILES[case_key])
    surfaces = _contour_flux_surfaces(geqdsk, SHAPE_SURFACE_LEVELS)
    n_levels = len(SHAPE_SURFACE_LEVELS)
    R = np.full((n_levels, n_theta), np.nan, dtype=np.float64)
    Z = np.full((n_levels, n_theta), np.nan, dtype=np.float64)
    for idx, level in enumerate(SHAPE_SURFACE_LEVELS):
        contour = surfaces.get(level)
        if contour is not None:
            resampled = _resample_contour_sequential(contour, n_theta)
            R[idx, :] = resampled[:, 0]
            Z[idx, :] = resampled[:, 1]
    return ResidualSample(
        case_key=case_key,
        config_label=EXTERNAL_REFERENCE_LABELS[case_key],
        signature={},
        parameter_count=None,
        elapsed_ms=float("nan"),
        solver_residual_norm=float("nan"),
        rho=np.array(SHAPE_SURFACE_LEVELS, dtype=np.float64),
        psin=np.array(SHAPE_SURFACE_LEVELS, dtype=np.float64),
        R=R,
        Z=Z,
        G=np.full((n_levels, n_theta), np.nan, dtype=np.float64),
        radial_rms=np.full(n_levels, np.nan, dtype=np.float64),
    )


def build_compact_figure(case_samples: dict[str, list[ResidualSample]], fig08=None) -> plt.Figure:
    case_keys = list(case_samples)
    fig = plt.figure(figsize=(COMPACT_FIGURE_WIDTH, COMPACT_ROW_HEIGHT * len(case_keys)))
    grid = fig.add_gridspec(
        nrows=len(case_keys),
        ncols=3,
        left=COMPACT_LEFT,
        right=COMPACT_RIGHT,
        bottom=COMPACT_BOTTOM,
        top=COMPACT_TOP,
        wspace=COMPACT_WSPACE,
        hspace=COMPACT_HSPACE,
    )
    for row, case_key in enumerate(case_keys):
        samples = compact_case_samples(case_samples[case_key])
        target = target_sample(case_samples[case_key])
        if fig08 is not None:
            try:
                target = build_true_target_sample(fig08, case_key)
            except Exception:
                pass
        xlim, ylim = rz_limits([*samples, target])
        for col, sample in enumerate(samples):
            ax = fig.add_subplot(grid[row, col])
            plot_shape_comparison_panel(
                ax,
                sample,
                target,
                case_key=case_key,
                xlim=xlim,
                ylim=ylim,
                show_legend=False,
            )
            ax.set_xticks(HEATMAP_X_TICKS[case_key])
            ax.set_anchor("C")
            ax.set_title(
                COMPACT_COLUMN_LABELS[col] if row == 0 else "",
                fontsize=scaled_font_size(TITLE_FONT_SIZE),
                fontweight="normal",
            )
            if col == 0:
                ax.set_ylabel(f"{CASE_LABELS[case_key]}\nZ [m]", fontsize=scaled_font_size(AXIS_LABEL_SIZE))
            else:
                ax.set_yticklabels([])
            if row == len(case_keys) - 1:
                ax.set_xlabel("R [m]", fontsize=scaled_font_size(AXIS_LABEL_SIZE))
            else:
                ax.set_xlabel("")
    return fig


def main() -> None:
    args = parse_args()
    apply_plot_style()
    selected_cases = CASE_KEYS if args.case == "all" else (args.case,)
    fig08 = load_fig08_helpers()
    signatures_by_case = selected_signature_map(fig08, selected_cases)

    case_samples = None
    if not args.refresh_cache:
        case_samples = load_case_samples_cache(
            args.cache_path,
            case_keys=selected_cases,
            solve_nr=args.solve_nr,
            solve_nt=args.solve_nt,
            backend=args.backend,
            signatures_by_case=signatures_by_case,
        )
        if case_samples is not None:
            pass

    if case_samples is None:
        benchmark = fig08.load_benchmark(args.backend)
        case_samples = {
            case_key: solve_case_samples(
                fig08,
                benchmark,
                case_key=case_key,
                signatures=signatures_by_case[case_key],
                args=args,
            )
            for case_key in selected_cases
        }
        save_case_samples_cache(
            args.cache_path,
            case_samples,
            case_keys=selected_cases,
            solve_nr=args.solve_nr,
            solve_nt=args.solve_nt,
            backend=args.backend,
            signatures_by_case=signatures_by_case,
        )

    Path(args.save_png).parent.mkdir(parents=True, exist_ok=True)
    Path(args.save_pdf).parent.mkdir(parents=True, exist_ok=True)
    fig = build_figure(case_samples)
    fig.savefig(args.save_png, dpi=SAVE_DPI, transparent=SAVE_TRANSPARENT)
    fig.savefig(args.save_pdf, dpi=SAVE_DPI, transparent=SAVE_TRANSPARENT)
    plt.close(fig)
    compact_fig = build_compact_figure(case_samples, fig08=fig08)
    Path(args.save_compact_png).parent.mkdir(parents=True, exist_ok=True)
    compact_fig.savefig(args.save_compact_png, dpi=SAVE_DPI, transparent=SAVE_TRANSPARENT)
    plt.close(compact_fig)
    rows = residual_norm_rows(case_samples)
    print_residual_norm_latex_table(rows)


if __name__ == "__main__":
    main()

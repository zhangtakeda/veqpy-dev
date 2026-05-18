import importlib.util
import sys
from pathlib import Path
from dataclasses import dataclass

import matplotlib

matplotlib.use("Agg")

import matplotlib.colors as mcolors
import matplotlib.ticker as mticker
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.config import (
    AXIS_LABEL_FONT_SIZE,
    SAVE_DPI,
    SAVE_TRANSPARENT,
    SINGLE_COLUMN_WIDTH,
    TICK_LABEL_FONT_SIZE,
    TITLE_FONT_SIZE,
    apply_plot_style,
    scaled_font_size,
)

FIGURE_SIZE = (SINGLE_COLUMN_WIDTH, 4.5)
FIGURE_NROWS = 2
FIGURE_NCOLS = 2
FIGURE_CONSTRAINED_LAYOUT = True
FIGURE_COLORBAR_WIDTH_RATIO = 0.065
FIGURE_WSPACE = 0.05
FIGURE_HSPACE = 0.04
COLORBAR_HEIGHT_FRACTION = 0.68
COLORBAR_Y0_FRACTION = 0.16
SAVE_PNG_PATH = "figures/04.png"
SAVE_PDF_PATH = "figures/04.pdf"

HEATMAP_TICK_DIRECTION = "out"
TOP_SPINE_VISIBLE = True
RIGHT_SPINE_VISIBLE = True

COLORBAR_TICK_DIRECTION = "out"
COLORBAR_TICK_RIGHT = True
COLORBAR_TICK_LEFT = False

HEATMAP_SHADING = "auto"
HEATMAP_CMAP = "viridis"

NR_LABEL = r"$N_\rho$"
NT_LABEL = r"$N_\theta$"
METRIC_SPECS = (
    ("shape_error", r"$\bf{(a)}$ $E_{\mathrm{coeff}}$"),
    ("ip_rel_error", r"$\bf{(b)}$ $\Delta_{I_p}$"),
    ("beta_rel_error", r"$\bf{(c)}$ $\Delta_{\beta_t}$"),
    ("q95_rel_error", r"$\bf{(d)}$ $\Delta_{q_{95}}$"),
)
Q95_PSIN = 0.95

REFERENCE_NR = 64
REFERENCE_NT = 64
NR_LIST = tuple(range(12, 64, 2))
NT_LIST = tuple(range(12, 64, 2))


@dataclass(frozen=True)
class ReferenceData:
    shape_x: np.ndarray
    target_ip: float
    target_beta_t: float
    target_q95: float
    l_max: int
    m_max: int


@dataclass(frozen=True)
class HeatmapRow:
    nr: int
    nt: int
    shape_error: float
    ip_rel_error: float
    beta_rel_error: float
    q95_rel_error: float
    nfev: int
    nit: int
    residual_norm_final: float


def load_demo_module():
    demo_path = "scripts/03-demo-equilibrium.py"
    spec = importlib.util.spec_from_file_location("veqpy_demo_reference_case", demo_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load demo script from {demo_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def build_demo_case(demo):
    if hasattr(demo, "build_operator_case"):
        source_case = demo.build_operator_case()
    elif hasattr(demo, "HIGHER_CASE"):
        source_case = demo.HIGHER_CASE
    else:
        raise AttributeError("Reference module does not provide build_operator_case() or HIGHER_CASE")
    return demo.OperatorCase(
        route=source_case.route,
        coordinate=source_case.coordinate,
        nodes=source_case.nodes,
        profile_coeffs={
            name: None if values is None else list(values) for name, values in source_case.profile_coeffs.items()
        },
        boundary=source_case.boundary,
        heat_input=np.asarray(source_case.heat_input, dtype=np.float64).copy(),
        current_input=np.asarray(source_case.current_input, dtype=np.float64).copy(),
        Ip=None,
        beta=None,
    )


def extract_shape_x(solver) -> np.ndarray:
    coeffs = solver.build_coeffs(include_none=False)
    shape_values: list[float] = []
    max_len = max((len(values) for values in coeffs.values() if values is not None), default=0)
    shape_names = []
    for name in coeffs:
        if name in {"psin", "h", "k", "v"} or name.startswith("c") or name.startswith("s"):
            shape_names.append(name)
    for idx in range(max_len):
        for name in shape_names:
            values = coeffs.get(name)
            if values is None or idx >= len(values):
                continue
            shape_values.append(float(values[idx]))
    return np.asarray(shape_values, dtype=np.float64)


def shape_error(reference_shape_x: np.ndarray, current_shape_x: np.ndarray) -> float:
    n = min(reference_shape_x.shape[0], current_shape_x.shape[0])
    if n == 0:
        return 0.0
    diff = current_shape_x[:n] - reference_shape_x[:n]
    return float(np.sqrt(np.mean(diff * diff)))


def relative_scalar_error(reference: float, current: float) -> float:
    scale = max(abs(float(reference)), 1.0e-12)
    return float(abs(float(current) - float(reference)) / scale)


def q_at_psin(equilibrium, psin_query: float = Q95_PSIN) -> float:
    """Interpolate q at a target normalized flux coordinate."""
    psin = np.asarray(equilibrium.psin, dtype=np.float64)
    q = np.asarray(equilibrium.q, dtype=np.float64)
    if psin.ndim != 1 or q.ndim != 1 or psin.size != q.size:
        raise ValueError(f"Expected 1D psin/q arrays with equal length, got {psin.shape} and {q.shape}")

    order = np.argsort(psin)
    psin_sorted = psin[order]
    q_sorted = q[order]
    psin_unique, unique_idx = np.unique(psin_sorted, return_index=True)
    q_unique = q_sorted[unique_idx]
    psin_clamped = float(np.clip(psin_query, psin_unique[0], psin_unique[-1]))
    return float(np.interp(psin_clamped, psin_unique, q_unique))


def build_reference(demo) -> ReferenceData:
    demo_case = build_demo_case(demo)
    grid = demo.Grid(
        Nr=REFERENCE_NR,
        Nt=REFERENCE_NT,
        quadrature_scheme="legendre",
        L_max=demo.GRID.L_max,
        M_max=demo.GRID.M_max,
    )
    solver = demo.Solver(
        operator=demo.Operator(grid=grid, case=demo_case),
        config=demo.CONFIG,
    )
    solver.solve(enable_verbose=False, enable_history=False, enable_warmstart=False)
    result = solver.result
    if result is None:
        raise RuntimeError("solver.result is unavailable after reference solve")
    equilibrium = solver.build_equilibrium()
    return ReferenceData(
        shape_x=extract_shape_x(solver),
        target_ip=float(equilibrium.Ip),
        target_beta_t=float(equilibrium.beta_t),
        target_q95=q_at_psin(equilibrium),
        l_max=int(grid.L_max),
        m_max=int(grid.M_max),
    )


def solve_case(demo, reference: ReferenceData, nr: int, nt: int) -> HeatmapRow:
    demo_case = build_demo_case(demo)
    grid = demo.Grid(
        Nr=nr,
        Nt=nt,
        quadrature_scheme="legendre",
        L_max=reference.l_max,
        M_max=reference.m_max,
    )
    solver = demo.Solver(
        operator=demo.Operator(grid=grid, case=demo_case),
        config=demo.CONFIG,
    )
    solver.solve(enable_verbose=False, enable_history=False, enable_warmstart=False)
    result = solver.result
    if result is None:
        raise RuntimeError("solver.result is unavailable after solve")
    equilibrium = solver.build_equilibrium()
    current_shape_x = extract_shape_x(solver)
    return HeatmapRow(
        nr=nr,
        nt=nt,
        shape_error=shape_error(reference.shape_x, current_shape_x),
        ip_rel_error=relative_scalar_error(reference.target_ip, float(equilibrium.Ip)),
        beta_rel_error=relative_scalar_error(reference.target_beta_t, float(equilibrium.beta_t)),
        q95_rel_error=relative_scalar_error(reference.target_q95, q_at_psin(equilibrium)),
        nfev=int(getattr(result, "nfev", result.function_evaluations)),
        nit=int(getattr(result, "nit", result.iterations)),
        residual_norm_final=float(result.residual_norm_final),
    )


def run_scan(demo, reference: ReferenceData, nr_list: list[int], nt_list: list[int]) -> list[HeatmapRow]:
    rows: list[HeatmapRow] = []
    total = len(nr_list) * len(nt_list)
    index = 0
    for nt in nt_list:
        for nr in nr_list:
            index += 1
            row = solve_case(demo, reference, nr, nt)
            rows.append(row)
            print(
                f"[{index:03d}/{total:03d}] Nr={nr:>2d}, Nt={nt:>2d}: "
                f"shape={row.shape_error:.3e} | "
                f"Ip={row.ip_rel_error:.3e} | "
                f"beta={row.beta_rel_error:.3e} | "
                f"q95={row.q95_rel_error:.3e}"
            )
    return rows


def build_matrix(rows: list[HeatmapRow], nr_list: list[int], nt_list: list[int], metric_name: str) -> np.ndarray:
    matrix = np.full((len(nt_list), len(nr_list)), np.nan, dtype=np.float64)
    index = {(row.nt, row.nr): row for row in rows}
    for i, nt in enumerate(nt_list):
        for j, nr in enumerate(nr_list):
            matrix[i, j] = float(getattr(index[(nt, nr)], metric_name))
    return matrix


def positive_matrix(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    finite = values[np.isfinite(values) & (values > 0.0)]
    floor = float(np.min(finite)) if finite.size else 1.0e-16
    return np.where(np.isfinite(values), np.maximum(values, floor), np.nan)


def cell_edges(values: list[int]) -> np.ndarray:
    axis = np.asarray(values, dtype=np.float64)
    if axis.size == 1:
        delta = 1.0
        return np.asarray([axis[0] - 0.5 * delta, axis[0] + 0.5 * delta])
    diffs = np.diff(axis)
    edges = np.empty(axis.size + 1, dtype=np.float64)
    edges[1:-1] = 0.5 * (axis[:-1] + axis[1:])
    edges[0] = axis[0] - 0.5 * diffs[0]
    edges[-1] = axis[-1] + 0.5 * diffs[-1]
    return edges


def _style_heatmap_axis(
    ax: plt.Axes,
    *,
    title: str,
    row: int,
    col: int,
    nrows: int,
    ncols: int,
) -> None:
    """Apply shared heatmap-axis styling with ticks only on left/bottom outer edges."""
    show_bottom = True
    show_left = col == 0

    ax.set_title(title, fontsize=scaled_font_size(TITLE_FONT_SIZE), fontweight="normal")
    ax.set_xlabel(NR_LABEL if show_bottom else "", fontsize=scaled_font_size(AXIS_LABEL_FONT_SIZE))
    ax.set_ylabel(NT_LABEL if show_left else "", fontsize=scaled_font_size(AXIS_LABEL_FONT_SIZE))
    ax.tick_params(
        direction=HEATMAP_TICK_DIRECTION,
        top=False,
        right=False,
        bottom=show_bottom,
        left=show_left,
        labeltop=False,
        labelright=False,
        labelbottom=show_bottom,
        labelleft=show_left,
        labelsize=scaled_font_size(TICK_LABEL_FONT_SIZE),
    )
    ax.spines["top"].set_visible(TOP_SPINE_VISIBLE)
    ax.spines["right"].set_visible(RIGHT_SPINE_VISIBLE)


def _style_colorbar(cbar) -> None:
    """Apply shared colorbar tick styling."""
    cbar.minorticks_off()
    cbar.ax.yaxis.set_minor_locator(mticker.NullLocator())
    cbar.ax.set_title("error", fontsize=scaled_font_size(AXIS_LABEL_FONT_SIZE), pad=6.0)
    cbar.ax.tick_params(
        which="both",
        direction=COLORBAR_TICK_DIRECTION,
        right=COLORBAR_TICK_RIGHT,
        left=COLORBAR_TICK_LEFT,
        labelsize=scaled_font_size(TICK_LABEL_FONT_SIZE),
    )


def build_grid_convergence_figure(rows: list[HeatmapRow], nr_list: list[int], nt_list: list[int]) -> plt.Figure:
    apply_plot_style()

    fig = plt.figure(
        figsize=FIGURE_SIZE,
        constrained_layout=FIGURE_CONSTRAINED_LAYOUT,
    )
    gs = fig.add_gridspec(
        FIGURE_NROWS,
        FIGURE_NCOLS + 1,
        width_ratios=[1.0] * FIGURE_NCOLS + [FIGURE_COLORBAR_WIDTH_RATIO],
        wspace=FIGURE_WSPACE,
        hspace=FIGURE_HSPACE,
    )

    axes_grid: list[list[plt.Axes]] = []
    shared_y_ax: plt.Axes | None = None
    for row in range(FIGURE_NROWS):
        row_axes: list[plt.Axes] = []
        for col in range(FIGURE_NCOLS):
            share_y_target = shared_y_ax if shared_y_ax is not None else None
            ax = fig.add_subplot(gs[row, col], sharey=share_y_target)
            if shared_y_ax is None:
                shared_y_ax = ax
            row_axes.append(ax)
        axes_grid.append(row_axes)

    axes = np.asarray(axes_grid, dtype=object)
    axes_flat = axes.ravel()
    cax = fig.add_subplot(gs[:, -1])
    x_edges = cell_edges(nr_list)
    y_edges = cell_edges(nt_list)
    metric_matrices: list[tuple[str, str, np.ndarray]] = []
    finite_groups: list[np.ndarray] = []
    for metric_name, title in METRIC_SPECS:
        values = positive_matrix(build_matrix(rows, nr_list, nt_list, metric_name))
        finite = values[np.isfinite(values)]
        if finite.size == 0:
            raise ValueError(f"No finite values available for {metric_name}")
        metric_matrices.append((metric_name, title, values))
        finite_groups.append(finite)

    all_finite = np.concatenate(finite_groups)
    norm = mcolors.LogNorm(
        vmin=float(np.min(all_finite)),
        vmax=float(np.max(all_finite)),
    )

    mesh = None
    for idx, (ax, (_, title, values)) in enumerate(zip(axes_flat, metric_matrices, strict=True)):
        row = idx // FIGURE_NCOLS
        col = idx % FIGURE_NCOLS
        mesh = ax.pcolormesh(
            x_edges,
            y_edges,
            values,
            shading=HEATMAP_SHADING,
            cmap=HEATMAP_CMAP,
            norm=norm,
        )
        _style_heatmap_axis(
            ax,
            title=title,
            row=row,
            col=col,
            nrows=FIGURE_NROWS,
            ncols=FIGURE_NCOLS,
        )
    if mesh is None:
        raise RuntimeError("No heatmap mesh was created")

    cax.set_axis_off()
    cbar_ax = cax.inset_axes([0.0, COLORBAR_Y0_FRACTION, 1.0, COLORBAR_HEIGHT_FRACTION])
    cbar = fig.colorbar(mesh, cax=cbar_ax)
    _style_colorbar(cbar)

    return fig


def normalized_grid_list(values: tuple[int, ...], *, name: str) -> list[int]:
    grid_list = sorted({int(value) for value in values})
    if not grid_list:
        raise ValueError(f"{name} must be non-empty")
    if any(value <= 0 for value in grid_list):
        raise ValueError(f"{name} must contain only positive integers")
    return grid_list


def main() -> None:
    nr_list = normalized_grid_list(NR_LIST, name="NR_LIST")
    nt_list = normalized_grid_list(NT_LIST, name="NT_LIST")
    demo = load_demo_module()
    reference = build_reference(demo)
    rows = run_scan(demo, reference, nr_list, nt_list)
    fig = build_grid_convergence_figure(rows, nr_list, nt_list)
    fig.savefig(
        SAVE_PNG_PATH,
        dpi=SAVE_DPI,
        transparent=SAVE_TRANSPARENT,
    )
    fig.savefig(
        SAVE_PDF_PATH,
        dpi=SAVE_DPI,
        transparent=SAVE_TRANSPARENT,
    )
    plt.close(fig)
    print(f"saved: {SAVE_PNG_PATH}")
    print(f"saved: {SAVE_PDF_PATH}")


if __name__ == "__main__":
    main()

from pathlib import Path

import matplotlib
import matplotlib.colors as mcolors
import numpy as np
from matplotlib import ticker
from matplotlib.gridspec import GridSpec
from mpl_toolkits.axes_grid1 import make_axes_locatable

from veqpy.model import Boundary, Grid
from veqpy.operator import Operator, OperatorCase
from veqpy.solver import Solver, SolverConfig
import sys

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.config import (
    AXIS_LABEL_FONT_SIZE,
    FIGURE_FACE_COLOR,
    LEGEND_FONT_SIZE,
    PLOT_FONT_FAMILY,
    PLOT_LABEL_RIGHT,
    PLOT_LABEL_TOP,
    PLOT_TICK_BOTTOM,
    PLOT_TICK_DIRECTION,
    PLOT_TICK_LEFT,
    PLOT_TICK_RIGHT,
    PLOT_TICK_TOP,
    SAVE_DPI,
    SAVE_TRANSPARENT,
    SINGLE_COLUMN_WIDTH,
    TICK_LABEL_FONT_SIZE,
    TITLE_FONT_SIZE,
    apply_plot_style,
    scaled_font_size,
)

FIGURE_SIZE = (SINGLE_COLUMN_WIDTH, 4.0)
SAVE_PNG_PATH = "figures/03.png"
SAVE_PDF_PATH = "figures/03.pdf"

GRID_SPEC_NROWS = 3
GRID_SPEC_NCOLS = 2
GRID_SPEC_WIDTH_RATIOS = [1, 1]
GRID_SPEC_HEIGHT_RATIOS = [1, 1, 1]
GRID_SPEC_HSPACE = 0.25
GRID_SPEC_WSPACE = 0.45
GRID_SPEC_TOP = 0.94
GRID_SPEC_BOTTOM = 0.12
GRID_SPEC_LEFT = 0.10
GRID_SPEC_RIGHT = 0.985

GRID_ALPHA = 0.25
GRID_LINESTYLE = "-"
GRID_LINE_WIDTH = 0.5
LINE_WIDTH = 1.5
SUBPLOT_TITLE_FONTSIZE = TITLE_FONT_SIZE
LEGEND_COLUMN_SPACING = 0.6
LEGEND_LABEL_SPACING = 0.1
TOP_SPINE_VISIBLE = True
RIGHT_SPINE_VISIBLE = True
COLORBAR_TICK_DIRECTION = "out"
COLORBAR_TICK_RIGHT = True
COLORBAR_TICK_LEFT = False
COLORBAR_HEIGHT_FRACTION = 0.68
COLORBAR_Y0_FRACTION = 0.16

WARMUP_TIMES = 10
SOURCE_SAMPLE_COUNT = 51

BLACK = "black"
BLUE = mcolors.TABLEAU_COLORS["tab:blue"]
ORANGE = mcolors.TABLEAU_COLORS["tab:orange"]
GREEN = mcolors.TABLEAU_COLORS["tab:green"]
RED = mcolors.TABLEAU_COLORS["tab:red"]
PURPLE = mcolors.TABLEAU_COLORS["tab:purple"]
GRAY = "#9aa0a6"

R_LABEL = r"$R$ [m]"
Z_LABEL = r"$Z$ [m]"
RHO_LABEL = r"$\rho$"
PSIN_LABEL = r"$\hat{\psi}$"
PROFILE_LABEL = "value"
SOURCE_LABEL = "source"
CURRENT_LABEL = "current [MA]"

PANEL_A_TITLE = ""
PANEL_B_TITLE = ""
PANEL_C_TITLE = ""
PANEL_D_TITLE = ""
PANEL_E_TITLE = ""

SURFACE_RAY_COLOR = GRAY
SURFACE_RAY_LINE_WIDTH = 0.8
SURFACE_RAY_ALPHA = 0.55
SURFACE_CURVE_LINE_WIDTH = LINE_WIDTH
SURFACE_COLORBAR_SIZE = "6.5%"
SURFACE_COLORBAR_PAD = 0.1
SURFACE_COLORBAR_NBINS = 2
SURFACE_CMAP_NAME = "inferno"
SURFACE_CMAP_MIN = 0.15
SURFACE_CMAP_MAX = 0.92
SURFACE_AXIS_MARKER = "o"
SURFACE_AXIS_MARKER_COLOR = plt.cm.inferno(SURFACE_CMAP_MIN)
SURFACE_AXIS_MARKER_SIZE = 6
SURFACE_AXIS_MARKER_LINE_WIDTH = 1.2

PSI_LEVEL_COUNT = 128
PSI_COLORBAR_SIZE = "5%"
PSI_COLORBAR_PAD = 0.1
PSI_COLORBAR_NBINS = 5

SOURCE_FF_COLOR = BLUE
SOURCE_PRESSURE_COLOR = ORANGE
SOURCE_FF_STYLE = "-"
SOURCE_PRESSURE_STYLE = "--"
SOURCE_LINE_WIDTH = LINE_WIDTH
SOURCE_TOP_HEADROOM = 0.3
SOURCE_LEGEND_LOC = "upper left"

CURRENT_IP_COLOR = BLACK
CURRENT_IP_STYLE = "--"
CURRENT_IP_LINE_WIDTH = LINE_WIDTH
CURRENT_IP_XMIN = 0.75
CURRENT_IP_XMAX = 1.0
CURRENT_ITOR_COLOR = BLUE
CURRENT_ITOR_STYLE = "-"
CURRENT_JTOR_COLOR = ORANGE
CURRENT_JTOR_STYLE = "--"
CURRENT_JPARA_COLOR = GREEN
CURRENT_JPARA_STYLE = "-."
CURRENT_LINE_WIDTH = LINE_WIDTH
CURRENT_TOP_HEADROOM = 0.35
CURRENT_LEGEND_LOC = "upper left"
CURRENT_LEGEND_NCOLS = 2

SAFETY_Q_COLOR = RED
SAFETY_Q_STYLE = "-"
SAFETY_S_COLOR = PURPLE
SAFETY_S_STYLE = "--"
SAFETY_LINE_WIDTH = LINE_WIDTH
SAFETY_TOP_HEADROOM = 0.2
SAFETY_LEGEND_LOC = "upper left"

GRID = Grid(
    Nr=64,
    Nt=64,
    quadrature_scheme="legendre",
)
SNAPSHOT_GRID = Grid(
    Nr=128,
    Nt=256,
    quadrature_scheme="uniform",
    L_max=GRID.L_max,
    M_max=GRID.M_max,
)
BOUNDARY = Boundary(
    a=1.05 / 1.85,
    R0=1.05,
    Z0=0.0,
    B0=3.0,
    ka=2.2,
    s_offsets=np.array([0.0, float(np.arcsin(0.5))]),
)
CONFIG = SolverConfig(
    method="hybr",
    enable_verbose=False,
    enable_warmstart=False,
)
COEFFS = {
    "psin": [0.0] * 5,
    "h": [0.0] * 3,
    "k": [0.0] * 5,
    "s1": [0.0] * 3,
}


def warmup_solver(solver: Solver) -> None:
    """Run silent solves so output timings exclude first-call startup overhead."""
    for _ in range(WARMUP_TIMES):
        solver.solve(enable_verbose=False, enable_history=False)


def pf_psin_reference_profiles(psin: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return the reference PF source profiles used by this script."""
    beta0 = 0.75
    alpha_p, alpha_f = 5.0, 3.32
    exp_ap = np.exp(alpha_p)
    exp_af = np.exp(alpha_f)
    den_p = 1.0 + exp_ap * (alpha_p - 1.0)
    den_f = 1.0 + exp_af * (alpha_f - 1.0)

    current_input = (1.0 - beta0) * alpha_f * (np.exp(alpha_f * psin) - exp_af) / den_f
    heat_input = beta0 * alpha_p * (np.exp(alpha_p * psin) - exp_ap) / den_p
    return current_input, heat_input


def build_operator_case() -> OperatorCase:
    """Construct the PF operator case rendered in this figure."""
    psin = np.linspace(0.0, 1.0, SOURCE_SAMPLE_COUNT)
    current_input, heat_input = pf_psin_reference_profiles(psin)
    return OperatorCase(
        route="PF",
        profile_coeffs=COEFFS,
        boundary=BOUNDARY,
        heat_input=heat_input,
        current_input=current_input,
        coordinate="psin",
        nodes="uniform",
        Ip=3.0e6,
    )


def solve_reference_equilibrium() -> object:
    """Solve the reference equilibrium and return a plotting-ready snapshot."""
    operator = Operator(grid=GRID, case=build_operator_case())
    solver = Solver(operator=operator, config=CONFIG)
    warmup_solver(solver)
    solver.solve()
    return solver.build_equilibrium().resample(SNAPSHOT_GRID)


def _close_periodic_curve(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim != 1:
        raise ValueError(f"Expected a 1D periodic curve, got {arr.shape}")
    return np.concatenate([arr, arr[:1]])


def _apply_rz_limits(ax: plt.Axes, boundary_data: dict[str, np.ndarray]) -> None:
    R_bnd = np.asarray(boundary_data["R"], dtype=np.float64)
    Z_bnd = np.asarray(boundary_data["Z"], dtype=np.float64)
    R_margin = (R_bnd.max() - R_bnd.min()) * 0.1
    Z_margin = (Z_bnd.max() - Z_bnd.min()) * 0.1
    ax.set_xlim(R_bnd.min() - R_margin, R_bnd.max() + R_margin)
    ax.set_ylim(Z_bnd.min() - Z_margin, Z_bnd.max() + Z_margin)
    ax.set_aspect("equal")


def _get_trunc_inferno() -> mcolors.LinearSegmentedColormap:
    cmap = plt.get_cmap(SURFACE_CMAP_NAME)
    return mcolors.LinearSegmentedColormap.from_list(
        "trunc_inferno",
        cmap(np.linspace(SURFACE_CMAP_MIN, SURFACE_CMAP_MAX, 256)),
    )


def _add_top_headroom(ax: plt.Axes, ratio: float) -> None:
    y0, y1 = ax.get_ylim()
    span = y1 - y0
    if ratio > 0.0:
        ax.set_ylim(y0, y1 + ratio * span)
    else:
        ax.set_ylim(y0 + ratio * span, y1)


def _style_axis(
    ax: plt.Axes,
    *,
    xlabel: str,
    ylabel: str,
    title: str,
    grid: bool = True,
) -> None:
    ax.set_title(title, fontsize=scaled_font_size(SUBPLOT_TITLE_FONTSIZE), fontweight="normal")
    ax.set_xlabel(xlabel, fontsize=scaled_font_size(AXIS_LABEL_FONT_SIZE))
    ax.set_ylabel(ylabel, fontsize=scaled_font_size(AXIS_LABEL_FONT_SIZE))
    if grid:
        ax.grid(True, alpha=GRID_ALPHA, linewidth=GRID_LINE_WIDTH, linestyle=GRID_LINESTYLE)
    else:
        ax.grid(False)
    ax.set_axisbelow(True)
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
    ax.spines["top"].set_visible(TOP_SPINE_VISIBLE)
    ax.spines["right"].set_visible(RIGHT_SPINE_VISIBLE)


def _style_legend(ax: plt.Axes, *, loc: str = "upper left", ncols: int = 1) -> None:
    ax.legend(
        frameon=False,
        loc=loc,
        ncols=ncols,
        fontsize=scaled_font_size(LEGEND_FONT_SIZE),
        columnspacing=LEGEND_COLUMN_SPACING,
        labelspacing=LEGEND_LABEL_SPACING,
    )


def _style_colorbar(cbar, *, label: str) -> None:
    cbar.ax.set_title(label, fontsize=scaled_font_size(AXIS_LABEL_FONT_SIZE), pad=6.0)
    cbar.ax.tick_params(
        which="both",
        direction=COLORBAR_TICK_DIRECTION,
        right=COLORBAR_TICK_RIGHT,
        left=COLORBAR_TICK_LEFT,
        labelsize=scaled_font_size(TICK_LABEL_FONT_SIZE),
    )


def _build_surface_panel_data(equilibrium) -> dict[str, object]:
    R = equilibrium.geometry.R
    Z = equilibrium.geometry.Z
    rho = equilibrium.rho
    Nt = equilibrium.grid.Nt

    sample_rho = np.linspace(0.0, 1.0, 12)
    surfaces = []
    for rho_value in sample_rho:
        idx = int(np.argmin(np.abs(rho - rho_value)))
        if rho[idx] <= 0.0:
            continue
        surfaces.append(
            {
                "rho": float(rho[idx]),
                "R": _close_periodic_curve(R[idx, :]),
                "Z": _close_periodic_curve(Z[idx, :]),
            }
        )

    theta_count = min(max(Nt, 1), 16)
    theta_indices = np.unique(np.linspace(0, Nt - 1, theta_count, dtype=int))
    rays = []
    for theta_idx in theta_indices:
        rays.append(
            {
                "theta_index": int(theta_idx),
                "R": np.asarray(R[:, theta_idx], dtype=np.float64),
                "Z": np.asarray(Z[:, theta_idx], dtype=np.float64),
            }
        )

    return {
        "surfaces": surfaces,
        "rays": rays,
        "axis": {"R": float(R[0, 0]), "Z": float(Z[0, 0])},
        "boundary": {"R": _close_periodic_curve(R[-1, :]), "Z": _close_periodic_curve(Z[-1, :])},
    }


def _build_source_panel_data(equilibrium) -> dict[str, np.ndarray]:
    return {
        "rho": np.asarray(equilibrium.rho, dtype=np.float64),
        "FF_psi": np.asarray(equilibrium.alpha1 * equilibrium.FFn_psin, dtype=np.float64),
        "mu0_P_psi": np.asarray(equilibrium.alpha1 * equilibrium.Pn_psin, dtype=np.float64),
    }


def _build_psi_panel_data(equilibrium) -> dict[str, np.ndarray]:
    psi = np.asarray(equilibrium.psin, dtype=np.float64)
    return {
        "R": np.hstack([equilibrium.geometry.R, equilibrium.geometry.R[:, :1]]),
        "Z": np.hstack([equilibrium.geometry.Z, equilibrium.geometry.Z[:, :1]]),
        "psi": np.repeat(psi[:, None], equilibrium.geometry.R.shape[1] + 1, axis=1),
    }


def _build_current_panel_data(equilibrium) -> dict[str, np.ndarray | float]:
    return {
        "rho": np.asarray(equilibrium.rho, dtype=np.float64),
        "itor": np.asarray(equilibrium.Itor, dtype=np.float64) / 1e6,
        "jtor": np.asarray(equilibrium.jtor, dtype=np.float64) / 1e6,
        "jpara": np.asarray(equilibrium.jpara, dtype=np.float64) / 1e6,
        "Ip": float(equilibrium.Ip) / 1e6,
    }


def _build_safety_panel_data(equilibrium) -> dict[str, np.ndarray]:
    return {
        "rho": np.asarray(equilibrium.rho, dtype=np.float64),
        "q": np.asarray(equilibrium.q, dtype=np.float64),
        "s": np.asarray(equilibrium.s, dtype=np.float64),
    }


def _render_panel_a_surfaces(ax: plt.Axes, fig: plt.Figure, data: dict[str, object]) -> None:
    _style_axis(ax, xlabel=R_LABEL, ylabel=Z_LABEL, title=PANEL_A_TITLE, grid=False)
    for ray in data.get("rays", []):
        ax.plot(
            ray["R"],
            ray["Z"],
            color=SURFACE_RAY_COLOR,
            linewidth=SURFACE_RAY_LINE_WIDTH,
            alpha=SURFACE_RAY_ALPHA,
            zorder=1,
        )

    surfaces = data["surfaces"]
    colors = plt.cm.inferno(np.linspace(0.0, 1.0, max(len(surfaces), 1)) * 0.77 + 0.15)
    for ci, surf in enumerate(surfaces):
        ax.plot(
            surf["R"],
            surf["Z"],
            color=colors[min(ci, len(colors) - 1)],
            linewidth=SURFACE_CURVE_LINE_WIDTH,
            zorder=2,
        )

    axis = data["axis"]
    ax.scatter(
        [axis["R"]],
        [axis["Z"]],
        color=SURFACE_AXIS_MARKER_COLOR,
        marker=SURFACE_AXIS_MARKER,
        s=SURFACE_AXIS_MARKER_SIZE,
        linewidths=SURFACE_AXIS_MARKER_LINE_WIDTH,
        zorder=3,
    )

    _apply_rz_limits(ax, data["boundary"])

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size=SURFACE_COLORBAR_SIZE, pad=SURFACE_COLORBAR_PAD)
    cax.set_axis_off()
    cbar_ax = cax.inset_axes([0.0, COLORBAR_Y0_FRACTION, 1.0, COLORBAR_HEIGHT_FRACTION])
    sm = plt.cm.ScalarMappable(
        cmap=_get_trunc_inferno(), norm=mcolors.Normalize(vmin=0.0, vmax=1.0)
    )
    cbar = fig.colorbar(sm, cax=cbar_ax)
    _style_colorbar(cbar, label=RHO_LABEL)
    cbar.locator = ticker.MaxNLocator(nbins=SURFACE_COLORBAR_NBINS)
    cbar.update_ticks()


def _render_panel_b_psi(
    ax: plt.Axes,
    fig: plt.Figure,
    data: dict[str, np.ndarray],
    boundary: dict[str, np.ndarray],
) -> None:
    _style_axis(ax, xlabel=R_LABEL, ylabel=Z_LABEL, title=PANEL_B_TITLE, grid=False)
    cmap = _get_trunc_inferno()
    vmin = float(np.nanmin(data["psi"]))
    vmax = float(np.nanmax(data["psi"]))
    if not np.isfinite(vmax) or vmax <= vmin:
        vmax = vmin + 1.0
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

    pcm = ax.contourf(
        data["R"],
        data["Z"],
        data["psi"],
        levels=np.linspace(vmin, vmax, PSI_LEVEL_COUNT),
        cmap=cmap,
        norm=norm,
    )
    _apply_rz_limits(ax, boundary)

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size=PSI_COLORBAR_SIZE, pad=PSI_COLORBAR_PAD)
    cax.set_axis_off()
    cbar_ax = cax.inset_axes([0.0, COLORBAR_Y0_FRACTION, 1.0, COLORBAR_HEIGHT_FRACTION])
    cbar = fig.colorbar(pcm, cax=cbar_ax)
    _style_colorbar(cbar, label=PSIN_LABEL)
    cbar.locator = ticker.MaxNLocator(nbins=PSI_COLORBAR_NBINS)
    cbar.update_ticks()


def _render_panel_c_sources(ax: plt.Axes, data: dict[str, np.ndarray]) -> None:
    _style_axis(ax, xlabel="", ylabel=SOURCE_LABEL, title=PANEL_C_TITLE)
    rho = data["rho"]
    ax.plot(
        rho,
        data["FF_psi"],
        SOURCE_FF_STYLE,
        color=SOURCE_FF_COLOR,
        linewidth=SOURCE_LINE_WIDTH,
        label=r"$FF_\psi$",
    )
    ax.plot(
        rho,
        data["mu0_P_psi"],
        SOURCE_PRESSURE_STYLE,
        color=SOURCE_PRESSURE_COLOR,
        linewidth=SOURCE_LINE_WIDTH,
        label=r"$\mu_0 P_\psi$",
    )

    _add_top_headroom(ax, ratio=SOURCE_TOP_HEADROOM)
    _style_legend(ax, loc=SOURCE_LEGEND_LOC)


def _render_panel_d_current(ax: plt.Axes, data: dict[str, np.ndarray | float]) -> None:
    _style_axis(ax, xlabel="", ylabel=CURRENT_LABEL, title=PANEL_D_TITLE)
    rho = data["rho"]
    ax.axhline(
        data["Ip"],
        xmin=CURRENT_IP_XMIN,
        xmax=CURRENT_IP_XMAX,
        color=CURRENT_IP_COLOR,
        linestyle=CURRENT_IP_STYLE,
        linewidth=CURRENT_IP_LINE_WIDTH,
        label=r"$I_p$",
    )
    ax.plot(
        rho,
        data["itor"],
        CURRENT_ITOR_STYLE,
        color=CURRENT_ITOR_COLOR,
        linewidth=CURRENT_LINE_WIDTH,
        label=r"$I_{\mathrm{tor}}$",
    )
    ax.plot(
        rho,
        data["jtor"],
        CURRENT_JTOR_STYLE,
        color=CURRENT_JTOR_COLOR,
        linewidth=CURRENT_LINE_WIDTH,
        label=r"$j_{\mathrm{tor}}$",
    )
    ax.plot(
        rho,
        data["jpara"],
        CURRENT_JPARA_STYLE,
        color=CURRENT_JPARA_COLOR,
        linewidth=CURRENT_LINE_WIDTH,
        label=r"$j_{\parallel}$",
    )

    _add_top_headroom(ax, ratio=CURRENT_TOP_HEADROOM)
    _style_legend(ax, loc=CURRENT_LEGEND_LOC, ncols=CURRENT_LEGEND_NCOLS)


def _render_panel_e_safety(ax: plt.Axes, data: dict[str, np.ndarray]) -> None:
    _style_axis(ax, xlabel=RHO_LABEL, ylabel=PROFILE_LABEL, title=PANEL_E_TITLE)
    rho = data["rho"]
    ax.plot(
        rho,
        data["q"],
        SAFETY_Q_STYLE,
        color=SAFETY_Q_COLOR,
        linewidth=SAFETY_LINE_WIDTH,
        label=r"$q$",
    )
    ax.plot(
        rho,
        data["s"],
        SAFETY_S_STYLE,
        color=SAFETY_S_COLOR,
        linewidth=SAFETY_LINE_WIDTH,
        label=r"$s$",
    )
    _add_top_headroom(ax, ratio=SAFETY_TOP_HEADROOM)
    _style_legend(ax, loc=SAFETY_LEGEND_LOC)


def build_equilibrium_figure(equilibrium, *, font_family: str = PLOT_FONT_FAMILY) -> plt.Figure:
    """Build the standalone 5-panel equilibrium visualization."""
    apply_plot_style(font_family)

    fig = plt.figure(figsize=FIGURE_SIZE)
    gs = GridSpec(
        GRID_SPEC_NROWS,
        GRID_SPEC_NCOLS,
        figure=fig,
        width_ratios=GRID_SPEC_WIDTH_RATIOS,
        height_ratios=GRID_SPEC_HEIGHT_RATIOS,
        hspace=GRID_SPEC_HSPACE,
        wspace=GRID_SPEC_WSPACE,
        top=GRID_SPEC_TOP,
        bottom=GRID_SPEC_BOTTOM,
        left=GRID_SPEC_LEFT,
        right=GRID_SPEC_RIGHT,
    )

    panel_a = _build_surface_panel_data(equilibrium)
    panel_c = _build_source_panel_data(equilibrium)
    panel_d = _build_current_panel_data(equilibrium)
    panel_e = _build_safety_panel_data(equilibrium)

    _render_panel_a_surfaces(fig.add_subplot(gs[:, 0]), fig, panel_a)
    ax_c = fig.add_subplot(gs[0, 1])
    ax_d = fig.add_subplot(gs[1, 1], sharex=ax_c)
    ax_e = fig.add_subplot(gs[2, 1], sharex=ax_c)
    _render_panel_c_sources(ax_c, panel_c)
    _render_panel_d_current(ax_d, panel_d)
    _render_panel_e_safety(ax_e, panel_e)
    for ax in (ax_c, ax_d):
        ax.tick_params(labelbottom=False)
    return fig


def main() -> None:
    outdir = Path(__file__).resolve().parents[1] / "figures"
    outdir.mkdir(parents=True, exist_ok=True)

    equilibrium = solve_reference_equilibrium()
    fig = build_equilibrium_figure(equilibrium)

    png_path = outdir / Path(SAVE_PNG_PATH).name
    pdf_path = outdir / Path(SAVE_PDF_PATH).name
    fig.savefig(
        png_path,
        dpi=SAVE_DPI,
        transparent=SAVE_TRANSPARENT,
        facecolor=FIGURE_FACE_COLOR,
    )
    fig.savefig(
        pdf_path,
        dpi=SAVE_DPI,
        transparent=SAVE_TRANSPARENT,
        facecolor=FIGURE_FACE_COLOR,
    )
    plt.close(fig)

    print(f"saved: {png_path}")
    print(f"saved: {pdf_path}")


if __name__ == "__main__":
    main()

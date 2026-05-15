import os
from dataclasses import dataclass
from functools import lru_cache

import matplotlib
import sys
from pathlib import Path

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from contourpy import contour_generator

from veqpy.model import Boundary, Geqdsk, Grid
from veqpy.model.boundary import _fit_boundary_params
from veqpy.operator import Operator, OperatorCase
from veqpy.solver import Solver, SolverConfig

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.config import (
    AXIS_LABEL_FONT_SIZE,
    DOUBLE_COLUMN_WIDTH,
    LEGEND_FONT_SIZE,
    PLOT_LABEL_RIGHT,
    PLOT_LABEL_TOP,
    PLOT_TICK_BOTTOM,
    PLOT_TICK_DIRECTION,
    PLOT_TICK_LEFT,
    PLOT_TICK_RIGHT,
    PLOT_TICK_TOP,
    SAVE_DPI,
    SAVE_TRANSPARENT,
    TICK_LABEL_FONT_SIZE,
    TITLE_FONT_SIZE,
    apply_plot_style,
    scaled_font_size,
)

MU0 = 4.0e-7 * np.pi

FIGURE_SIZE = (DOUBLE_COLUMN_WIDTH, 6.5)
FIGURE_GRID_NCOLS = 3
FIGURE_GRID_WSPACE = 0.3
PANEL_GRID_NROWS = 5
PANEL_GRID_HEIGHT_RATIOS = (3.5, 0.60, 1.25, 0.20, 1.25)
PANEL_GRID_HSPACE = 0.0
SAVE_GFILE_PATH = "data/SOLOVEV.geqdsk"
SAVE_PNG_PATH = "figures/06.png"
SAVE_PDF_PATH = "figures/06.pdf"

TOP_SPINE_VISIBLE = True
RIGHT_SPINE_VISIBLE = True
GRID_ALPHA = 0.35
GRID_LINESTYLE = ":"
GRID_LINE_WIDTH = 0.8
LOG_Y_FLOOR = 1.0e-8
PSI_ERROR_YMIN = 1.0e-6
PSI_ERROR_YMAX = 5.0

ANALYTIC_LINE_COLOR = "black"
ANALYTIC_LINE_STYLE = "--"
VEQPY_LINE_COLOR = "#d62728"
VEQPY_LINE_STYLE = "-"
SOURCE_FF_COLOR = "#1f77b4"
SOURCE_P_COLOR = "#ff7f0e"
PROFILE_LINE_WIDTH = 1.25
THIRD_ROW_PSI_COLOR = "#9467bd"
THIRD_ROW_SHAPE_COLOR = VEQPY_LINE_COLOR
THIRD_ROW_PSI_STYLE = "-"
THIRD_ROW_SHAPE_STYLE = "-"
THIRD_ROW_PSI_MARKER = "x"
THIRD_ROW_PSI_MARKER_SIZE = 4
THIRD_ROW_PSI_MARKER_EDGE_WIDTH = 1.2
THIRD_ROW_SHAPE_MARKER = "o"
THIRD_ROW_SHAPE_MARKER_SIZE = 3.5
SURFACE_LINE_WIDTH = 1.0
BOUNDARY_SURFACE_LINE_WIDTH = 1.35
REFERENCE_SURFACE_SCALE = 1.5
REFERENCE_AXIS_COLOR = "black"
REFERENCE_AXIS_MARKER = "x"
REFERENCE_AXIS_MARKER_SIZE = 30
VEQPY_AXIS_COLOR = "#d62728"
VEQPY_AXIS_MARKER = "x"
VEQPY_AXIS_MARKER_SIZE = 15
LEGEND_FRAME_ON = False
SURFACE_LEGEND_LOC = "upper right"
SOURCE_LEGEND_LOC = "upper right"
THIRD_ROW_LEGEND_LOC = "upper right"
Q_LEGEND_LOC = "best"
SURFACE_PAD_FRACTION = 0.07
SURFACE_XPAD_EXTRA_FRACTION = 0.12
FIRST_COLUMN_SURFACE_XPAD_EXTRA_FRACTION = 0.18
SOURCE_TOP_HEADROOM = 0.30
SOURCE_BOTTOM_HEADROOM = 0.08
Q_TOP_HEADROOM = 0.20
Q_BOTTOM_HEADROOM = 0.08
GEQDSK_LABEL = "reference"

LEGEND_COLUMN_SPACING = 0.8
LEGEND_LABEL_SPACING = 0.15

SOLVE_NR = 64
SOLVE_NT = 32
SOLOVEV_SOLVE_NR = 64
SOLOVEV_SOLVE_NT = SOLVE_NT
SOLOVEV_SOURCE_NR = 257
PLOT_NR = 128
PLOT_NT = 256
SURFACE_COUNT = 10
PSIN_ERROR_RHO_LEVELS = tuple(np.linspace(0.1, 0.9, 9, dtype=np.float64))
SHAPE_RMS_PSIN_LEVELS = tuple(np.linspace(0.0, 1.0, 11, dtype=np.float64))
BOUNDARY_MAXTOL = 1.0
SOLVER_METHOD = "hybr"
SOLVER_MAXFEV = 2000

SOLOVEV_BOUNDARY_FIT_M = 10
SOLOVEV_BOUNDARY_FIT_N = 10
CHEASE_BOUNDARY_FIT_M = 10
CHEASE_BOUNDARY_FIT_N = 10
EFIT_BOUNDARY_FIT_M = 10
EFIT_BOUNDARY_FIT_N = 10

DEFAULT_PROFILE_COEFFS = {
    "psin": [0.0] * 10,
    "h": [0.0] * 10,
    "k": [0.0] * 10,
    "s1": [0.0] * 10,
    "s2": [0.0] * 5,
    "s3": [0.0] * 5,
    "s4": [0.0] * 5,
    "s5": [0.0] * 5,
    "s6": [0.0] * 5,
    "s7": [0.0] * 5,
    "s8": [0.0] * 5,
    "s9": [0.0] * 5,
    "s10": [0.0] * 5,
}

CHEASE_PROFILE_COEFFS = {
    "psin": [0.0] * 10,
    "h": [0.0] * 10,
    "k": [0.0] * 10,
    "v": [0.0] * 10,
    "c0": [0.0] * 10,
    "c1": [0.0] * 5,
    "c2": [0.0] * 5,
    "c3": [0.0] * 5,
    "c4": [0.0] * 5,
    "c5": [0.0] * 5,
    "c6": [0.0] * 5,
    "c7": [0.0] * 5,
    "c8": [0.0] * 5,
    "c9": [0.0] * 5,
    "s1": [0.0] * 10,
    "s2": [0.0] * 5,
    "s3": [0.0] * 5,
    "s4": [0.0] * 5,
    "s5": [0.0] * 5,
    "s6": [0.0] * 5,
    "s7": [0.0] * 5,
    "s8": [0.0] * 5,
    "s9": [0.0] * 5,
    "s10": [0.0] * 5,
}

EFIT_PROFILE_COEFFS = {
    "psin": [0.0] * 10,
    "h": [0.0] * 10,
    "k": [0.0] * 10,
    "v": [0.0] * 10,
    "c0": [0.0] * 10,
    "c1": [0.0] * 5,
    "c2": [0.0] * 5,
    "c3": [0.0] * 5,
    "c4": [0.0] * 5,
    "c5": [0.0] * 5,
    "c6": [0.0] * 5,
    "c7": [0.0] * 5,
    "c8": [0.0] * 5,
    "c9": [0.0] * 5,
    "s1": [0.0] * 10,
    "s2": [0.0] * 5,
    "s3": [0.0] * 5,
    "s4": [0.0] * 5,
    "s5": [0.0] * 5,
    "s6": [0.0] * 5,
    "s7": [0.0] * 5,
    "s8": [0.0] * 5,
    "s9": [0.0] * 5,
    "s10": [0.0] * 5,
}


@dataclass(frozen=True)
class SolovevReferenceConfig:
    major_radius: float = 6.2
    minor_radius: float = 1.984
    elongation: float = 2.0
    triangularity: float = 0.45
    toroidal_field: float = 5.3
    axis_shift_fraction: float = 0.1
    psi_boundary: float = 1.0
    nr: int = SOLOVEV_SOURCE_NR
    nz: int = SOLOVEV_SOURCE_NR
    boundary_points: int = 512
    contour_nr: int = 700
    contour_nz: int = 700
    q_samples: int = 24
    q_sign: float = -1.0


@dataclass(frozen=True)
class ExactDShapeSolution:
    config: SolovevReferenceConfig
    axis_radius: float
    c0: float
    c_r2: float
    c_log: float
    c_poly4: float
    mu0_pprime_raw: float
    ffprime_raw: float

    @property
    def psi_axis_raw(self) -> float:
        return float(self.psi_raw(self.axis_radius, 0.0))

    @property
    def F_axis(self) -> float:
        return float(self.config.major_radius * self.config.toroidal_field)

    def psi_raw(self, R: np.ndarray | float, Z: np.ndarray | float) -> np.ndarray:
        RR = np.asarray(R, dtype=np.float64)
        ZZ = np.asarray(Z, dtype=np.float64)
        return (
            self.c0
            + self.c_r2 * RR * RR
            + self.c_log * (RR * RR * np.log(RR) - ZZ * ZZ)
            + self.c_poly4 * (RR**4 - 4.0 * RR * RR * ZZ * ZZ)
            - self.mu0_pprime_raw * RR**4 / 8.0
            - self.ffprime_raw * ZZ * ZZ / 2.0
        )

    def grad_psi_raw(
        self, R: np.ndarray | float, Z: np.ndarray | float
    ) -> tuple[np.ndarray, np.ndarray]:
        RR = np.asarray(R, dtype=np.float64)
        ZZ = np.asarray(Z, dtype=np.float64)
        dpsi_dR = (
            2.0 * self.c_r2 * RR
            + self.c_log * (2.0 * RR * np.log(RR) + RR)
            + self.c_poly4 * (4.0 * RR**3 - 8.0 * RR * ZZ * ZZ)
            - self.mu0_pprime_raw * RR**3 / 2.0
        )
        dpsi_dZ = -2.0 * self.c_log * ZZ - 8.0 * self.c_poly4 * RR * RR * ZZ - self.ffprime_raw * ZZ
        return dpsi_dR, dpsi_dZ

    def normalized_psin(self, R: np.ndarray | float, Z: np.ndarray | float) -> np.ndarray:
        return self.psi_axis_raw - self.psi_raw(R, Z)


@dataclass(frozen=True)
class CaseSpec:
    title: str
    reference_label: str
    gfile_path: str
    boundary_fit_m: int
    boundary_fit_n: int
    profile_coeffs: dict[str, list[float]]
    solve_nr: int = SOLVE_NR
    solve_nt: int = SOLVE_NT
    generate_gfile: bool = False


@dataclass(frozen=True)
class CaseResult:
    title: str
    reference_label: str
    case_spec: CaseSpec
    geqdsk: Geqdsk
    equilibrium: object
    plot_equilibrium: object
    reference_surfaces: dict[float, np.ndarray]
    veqpy_surfaces: dict[float, np.ndarray]
    reference_psin_profile: tuple[np.ndarray, np.ndarray]
    veqpy_psin_profile: tuple[np.ndarray, np.ndarray]
    psin_error_profile: tuple[np.ndarray, np.ndarray]
    shape_rms_profile: tuple[np.ndarray, np.ndarray]
    parameter_count: int
    boundary_fit_rms: float
    solver_residual: float
    solve_elapsed_ms: float


REFERENCE_CONFIG = SolovevReferenceConfig()

CASE_SPECS = (
    CaseSpec(
        title=r"$\bf{(a)}$ D-shaped Equilibrium",
        reference_label="Solov'ev",
        gfile_path=SAVE_GFILE_PATH,
        boundary_fit_m=SOLOVEV_BOUNDARY_FIT_M,
        boundary_fit_n=SOLOVEV_BOUNDARY_FIT_N,
        profile_coeffs=DEFAULT_PROFILE_COEFFS,
        solve_nr=SOLOVEV_SOLVE_NR,
        solve_nt=SOLOVEV_SOLVE_NT,
        generate_gfile=True,
    ),
    CaseSpec(
        title=r"$\bf{(b)}$ H-mode Equilibrium",
        reference_label="CHEASE",
        gfile_path="data/CHEASE.geqdsk",
        boundary_fit_m=CHEASE_BOUNDARY_FIT_M,
        boundary_fit_n=CHEASE_BOUNDARY_FIT_N,
        profile_coeffs=CHEASE_PROFILE_COEFFS,
    ),
    CaseSpec(
        title=r"$\bf{(c)}$ X-point Equilibrium",
        reference_label="EFIT",
        gfile_path="data/EFIT.geqdsk",
        boundary_fit_m=EFIT_BOUNDARY_FIT_M,
        boundary_fit_n=EFIT_BOUNDARY_FIT_N,
        profile_coeffs=EFIT_PROFILE_COEFFS,
    ),
)


CASE_DISPLAY_NAMES = {
    "Solov'ev": "D-shape",
    "CHEASE": "H-mode",
    "EFIT": "X-point",
}

CASE_SOURCE_LABELS = {
    "Solov'ev": r"analytic Solov'ev",
    "CHEASE": "CHEASE GEQDSK",
    "EFIT": "EFIT GEQDSK",
}

CASE_BOUNDARY_LABELS = {
    "Solov'ev": "D-shaped",
    "CHEASE": "H-mode",
    "EFIT": "X-point",
}


def profile_parameter_count(profile_coeffs: dict[str, list[float]]) -> int:
    return int(sum(len(values) for values in profile_coeffs.values() if values is not None))


def format_tex_float(value: float, *, precision: int = 2) -> str:
    value = float(value)
    if not np.isfinite(value):
        return "--"
    if value == 0.0:
        return "0"
    exponent = int(np.floor(np.log10(abs(value))))
    if -2 <= exponent <= 3:
        return f"{value:.{precision}f}"
    mantissa = value / (10.0**exponent)
    return rf"{mantissa:.{precision}f}\times 10^{{{exponent}}}"


def max_profile_value(profile: tuple[np.ndarray, np.ndarray]) -> float:
    values = np.asarray(profile[1], dtype=np.float64)
    if values.size == 0:
        return float("nan")
    return float(np.max(np.abs(values)))


def aggregate_surface_rms_error(profile: tuple[np.ndarray, np.ndarray]) -> float:
    """Match the scalar surface-error convention used by scripts/08.py.

    The profile contains the magnetic-axis displacement followed by
    theta-resolved flux-surface radial RMS values over normalized-flux levels.
    The scalar error is the root mean square over those entries, in metres.
    """
    values = np.asarray(profile[1], dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float("nan")
    return float(np.sqrt(np.mean(values * values)))


def format_boundary_fit(case_spec: CaseSpec) -> str:
    return rf"$c/s=({int(case_spec.boundary_fit_m)},{int(case_spec.boundary_fit_n)})$"


def _coeff_length(profile_coeffs: dict[str, list[float]], name: str) -> int:
    values = profile_coeffs.get(name)
    return 0 if values is None else int(len(values))


def format_core_psin_tuple(case_spec: CaseSpec) -> str:
    profile_coeffs = case_spec.profile_coeffs
    values = (
        _coeff_length(profile_coeffs, "h"),
        _coeff_length(profile_coeffs, "v"),
        _coeff_length(profile_coeffs, "k"),
        _coeff_length(profile_coeffs, "psin"),
    )
    return "$(" + ", ".join(str(value) for value in values) + ")$"


def _cs_family_names(profile_coeffs: dict[str, list[float]]) -> list[str]:
    def sort_key(name: str) -> tuple[int, int]:
        if name == "c0":
            return (0, 0)
        prefix = 1 if name.startswith("c") else 2
        suffix = name[1:]
        return (prefix, int(suffix) if suffix.isdigit() else 10**6)

    names = [
        name
        for name in profile_coeffs
        if name == "c0" or (len(name) > 1 and name[0] in {"c", "s"} and name[1:].isdigit())
    ]
    return sorted(names, key=sort_key)


def format_repeated_tuple_tex(values: tuple[int, ...]) -> str:
    if not values:
        return "--"
    unique_values = tuple(dict.fromkeys(int(value) for value in values))
    if len(unique_values) == 1:
        if len(values) == 1:
            tuple_text = f"({unique_values[0]})"
        else:
            tuple_text = rf"({unique_values[0]}^{{\times{len(values)}}})"
    else:
        tuple_text = "(" + ", ".join(str(value) for value in unique_values) + ")"
    return rf"${tuple_text}$"


def format_cs_tuple(case_spec: CaseSpec) -> str:
    names = _cs_family_names(case_spec.profile_coeffs)
    lengths = tuple(_coeff_length(case_spec.profile_coeffs, name) for name in names)
    return format_repeated_tuple_tex(lengths)


def add_top_headroom(ax: plt.Axes, ratio: float) -> None:
    y0, y1 = ax.get_ylim()
    span = y1 - y0
    if ratio > 0.0:
        ax.set_ylim(y0, y1 + ratio * span)
    else:
        ax.set_ylim(y0 + ratio * span, y1)


def add_y_headroom(ax: plt.Axes, *, lower_ratio: float, upper_ratio: float) -> None:
    y0, y1 = ax.get_ylim()
    span = y1 - y0
    ax.set_ylim(y0 - lower_ratio * span, y1 + upper_ratio * span)


def set_symmetric_ylim(ax: plt.Axes, *, headroom_ratio: float = 0.0) -> None:
    y0, y1 = ax.get_ylim()
    bound = max(abs(y0), abs(y1))
    if bound <= 0.0:
        bound = 1.0
    ax.set_ylim(-bound * (1.0 + headroom_ratio), bound * (1.0 + headroom_ratio))


def style_axis(ax: plt.Axes, *, title: str, xlabel: str, ylabel: str) -> None:
    ax.set_title(title, fontsize=scaled_font_size(TITLE_FONT_SIZE), fontweight="normal")
    ax.set_xlabel(xlabel, fontsize=scaled_font_size(AXIS_LABEL_FONT_SIZE))
    ax.set_ylabel(ylabel, fontsize=scaled_font_size(AXIS_LABEL_FONT_SIZE))
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
    ax.grid(True, linestyle=GRID_LINESTYLE, alpha=GRID_ALPHA, linewidth=GRID_LINE_WIDTH)


def close_curve(points: np.ndarray) -> np.ndarray:
    arr = np.asarray(points, dtype=np.float64)
    return np.vstack((arr, arr[:1]))


def polygon_area(points: np.ndarray) -> float:
    curve = close_curve(points)
    x = curve[:, 0]
    y = curve[:, 1]
    return float(0.5 * abs(np.dot(x[:-1], y[1:]) - np.dot(y[:-1], x[1:])))


def compute_rz_limits(
    curves: list[np.ndarray], *, pad_fraction: float = SURFACE_PAD_FRACTION
) -> tuple[tuple[float, float], tuple[float, float]]:
    stacked = np.vstack(
        [np.asarray(curve, dtype=np.float64) for curve in curves if np.asarray(curve).size]
    )
    r_min = float(np.min(stacked[:, 0]))
    r_max = float(np.max(stacked[:, 0]))
    z_min = float(np.min(stacked[:, 1]))
    z_max = float(np.max(stacked[:, 1]))
    r_pad = max((r_max - r_min) * pad_fraction, 1.0e-3)
    z_pad = max((z_max - z_min) * pad_fraction, 1.0e-3)
    return (r_min - r_pad, r_max + r_pad), (z_min - z_pad, z_max + z_pad)


def default_levels_from_count(surface_count: int) -> list[float]:
    return [level / surface_count for level in range(1, surface_count + 1)] + [1.0]


def basis_row(R: float, Z: float) -> np.ndarray:
    return np.asarray(
        (
            1.0,
            R * R,
            R * R * np.log(R) - Z * Z,
            R**4 - 4.0 * R * R * Z * Z,
            -(R**4) / 8.0,
            -(Z * Z) / 2.0,
        ),
        dtype=np.float64,
    )


def basis_dR_row(R: float, Z: float) -> np.ndarray:
    return np.asarray(
        (0.0, 2.0 * R, 2.0 * R * np.log(R) + R, 4.0 * R**3 - 8.0 * R * Z * Z, -(R**3) / 2.0, 0.0),
        dtype=np.float64,
    )


@lru_cache(maxsize=None)
def build_exact_solution(cfg: SolovevReferenceConfig) -> ExactDShapeSolution:
    R0 = float(cfg.major_radius)
    a = float(cfg.minor_radius)
    ka = float(cfg.elongation)
    delta = float(cfg.triangularity)
    axis_radius = R0 + float(cfg.axis_shift_fraction) * a
    R_outer = R0 + a
    R_inner = R0 - a
    R_top = R0 - delta * a
    Z_top = ka * a

    lhs = np.vstack(
        (
            basis_row(R_outer, 0.0),
            basis_row(R_inner, 0.0),
            basis_row(R_top, Z_top),
            basis_dR_row(R_top, Z_top),
            basis_row(axis_radius, 0.0),
            basis_dR_row(axis_radius, 0.0),
        )
    )
    rhs = np.asarray((0.0, 0.0, 0.0, 0.0, 1.0, 0.0), dtype=np.float64)
    coeffs = np.linalg.solve(lhs, rhs)
    return ExactDShapeSolution(
        config=cfg,
        axis_radius=axis_radius,
        c0=float(coeffs[0]),
        c_r2=float(coeffs[1]),
        c_log=float(coeffs[2]),
        c_poly4=float(coeffs[3]),
        mu0_pprime_raw=float(coeffs[4]),
        ffprime_raw=float(coeffs[5]),
    )


@lru_cache(maxsize=None)
def contour_grid(cfg: SolovevReferenceConfig) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rr = np.linspace(
        cfg.major_radius - 1.2 * cfg.minor_radius,
        cfg.major_radius + 1.2 * cfg.minor_radius,
        int(cfg.contour_nr),
        dtype=np.float64,
    )
    zz = np.linspace(
        -1.2 * cfg.elongation * cfg.minor_radius,
        1.2 * cfg.elongation * cfg.minor_radius,
        int(cfg.contour_nz),
        dtype=np.float64,
    )
    RR, ZZ = np.meshgrid(rr, zz, indexing="xy")
    psi_grid = build_exact_solution(cfg).normalized_psin(RR, ZZ)
    return rr, zz, psi_grid


@lru_cache(maxsize=None)
def contour_gen(cfg: SolovevReferenceConfig):
    rr, zz, psi_grid = contour_grid(cfg)
    return contour_generator(x=rr, y=zz, z=psi_grid, name="serial")


def point_in_polygon(points: np.ndarray, R: float, Z: float) -> bool:
    vertices = np.asarray(points, dtype=np.float64)
    inside = False
    j = vertices.shape[0] - 1
    for i in range(vertices.shape[0]):
        xi, yi = vertices[i]
        xj, yj = vertices[j]
        if (yi > Z) != (yj > Z):
            dy = yj - yi
            if abs(dy) < 1.0e-14:
                dy = 1.0e-14 if dy >= 0.0 else -1.0e-14
            x_cross = (xj - xi) * (Z - yi) / dy + xi
            if R < x_cross:
                inside = not inside
        j = i
    return inside


def resample_curve_by_angle(
    points: np.ndarray, *, center: tuple[float, float], count: int
) -> np.ndarray:
    curve = np.asarray(points, dtype=np.float64)
    theta = np.unwrap(np.arctan2(curve[:, 1] - center[1], curve[:, 0] - center[0]))
    order = np.argsort(theta)
    theta_sorted = theta[order]
    curve_sorted = curve[order]
    theta_periodic = np.concatenate((theta_sorted, [theta_sorted[0] + 2.0 * np.pi]))
    R_periodic = np.concatenate((curve_sorted[:, 0], [curve_sorted[0, 0]]))
    Z_periodic = np.concatenate((curve_sorted[:, 1], [curve_sorted[0, 1]]))
    theta_eval = np.linspace(
        theta_sorted[0], theta_sorted[0] + 2.0 * np.pi, int(count), endpoint=False, dtype=np.float64
    )
    return np.column_stack(
        (
            np.interp(theta_eval, theta_periodic, R_periodic),
            np.interp(theta_eval, theta_periodic, Z_periodic),
        )
    )


def sample_curve_at_theta(
    points: np.ndarray, *, center: tuple[float, float], theta_eval: np.ndarray
) -> np.ndarray:
    curve = np.asarray(points, dtype=np.float64)
    theta = np.mod(np.arctan2(curve[:, 1] - center[1], curve[:, 0] - center[0]), 2.0 * np.pi)
    order = np.argsort(theta, kind="mergesort")
    theta_sorted = theta[order]
    curve_sorted = curve[order]
    theta_periodic = np.concatenate((theta_sorted, [theta_sorted[0] + 2.0 * np.pi]))
    R_periodic = np.concatenate((curve_sorted[:, 0], [curve_sorted[0, 0]]))
    Z_periodic = np.concatenate((curve_sorted[:, 1], [curve_sorted[0, 1]]))
    theta_target = np.mod(np.asarray(theta_eval, dtype=np.float64), 2.0 * np.pi)
    return np.column_stack(
        (
            np.interp(theta_target, theta_periodic, R_periodic),
            np.interp(theta_target, theta_periodic, Z_periodic),
        )
    )


def radial_profile_from_surface(
    points: np.ndarray, *, center: tuple[float, float], theta_eval: np.ndarray
) -> np.ndarray:
    sampled = sample_curve_at_theta(points, center=center, theta_eval=theta_eval)
    center_arr = np.asarray(center, dtype=np.float64)
    return np.sqrt(np.sum((sampled - center_arr[None, :]) ** 2, axis=1))


def axis_position_error(
    reference_axis: tuple[float, float], veqpy_axis: tuple[float, float]
) -> float:
    ref_axis = np.asarray(reference_axis, dtype=np.float64)
    vq_axis = np.asarray(veqpy_axis, dtype=np.float64)
    return float(np.linalg.norm(vq_axis - ref_axis))


def interpolate_unique(x: np.ndarray, y: np.ndarray, x_eval: np.ndarray) -> np.ndarray:
    x_arr = np.asarray(x, dtype=np.float64)
    y_arr = np.asarray(y, dtype=np.float64)
    xq = np.asarray(x_eval, dtype=np.float64)
    order = np.argsort(x_arr, kind="mergesort")
    x_sorted = x_arr[order]
    y_sorted = y_arr[order]
    x_unique, unique_idx = np.unique(x_sorted, return_index=True)
    y_unique = y_sorted[unique_idx]
    return np.interp(xq, x_unique, y_unique)


def analytic_surface_points_direct(
    cfg: SolovevReferenceConfig, *, level: float, n_theta: int
) -> np.ndarray:
    solution = build_exact_solution(cfg)
    axis_center = (float(solution.axis_radius), 0.0)
    theta_eval = np.linspace(0.0, 2.0 * np.pi, int(n_theta), endpoint=False, dtype=np.float64)
    boundary = analytic_boundary_points(cfg, level=1.0, n_theta=256)
    max_extent = 1.2 * float(
        np.max(
            np.sqrt((boundary[:, 0] - axis_center[0]) ** 2 + (boundary[:, 1] - axis_center[1]) ** 2)
        )
    )
    points = np.empty((theta_eval.size, 2), dtype=np.float64)

    for idx, theta in enumerate(theta_eval):
        s_grid = np.linspace(0.0, max_extent, 512, dtype=np.float64)
        R_grid = axis_center[0] + s_grid * np.cos(theta)
        Z_grid = axis_center[1] + s_grid * np.sin(theta)
        psin_grid = solution.normalized_psin(R_grid, Z_grid)
        above = np.flatnonzero(psin_grid >= float(level))
        if above.size == 0:
            raise RuntimeError(
                f"Could not bracket analytic surface for level={level:.6f} at theta={theta:.6f}"
            )
        hi_idx = int(above[0])
        if hi_idx == 0:
            s_lo, s_hi = 0.0, float(s_grid[hi_idx])
        else:
            s_lo, s_hi = float(s_grid[hi_idx - 1]), float(s_grid[hi_idx])
        for _ in range(50):
            s_mid = 0.5 * (s_lo + s_hi)
            R_mid = axis_center[0] + s_mid * np.cos(theta)
            Z_mid = axis_center[1] + s_mid * np.sin(theta)
            if float(solution.normalized_psin(R_mid, Z_mid)) >= float(level):
                s_hi = s_mid
            else:
                s_lo = s_mid
        s_final = 0.5 * (s_lo + s_hi)
        points[idx, 0] = axis_center[0] + s_final * np.cos(theta)
        points[idx, 1] = axis_center[1] + s_final * np.sin(theta)
    return points


def analytic_boundary_points(
    cfg: SolovevReferenceConfig, *, level: float = 1.0, n_theta: int | None = None
) -> np.ndarray:
    solution = build_exact_solution(cfg)
    lines = contour_gen(cfg).lines(float(level))
    axis_center = (float(solution.axis_radius), 0.0)
    selected = None
    selected_length = -1
    for line in lines:
        vertices = np.asarray(line, dtype=np.float64)
        if vertices.shape[0] < 16:
            continue
        if not point_in_polygon(vertices, axis_center[0], axis_center[1]):
            continue
        if vertices.shape[0] > selected_length:
            selected = vertices.copy()
            selected_length = vertices.shape[0]
    if selected is None:
        raise RuntimeError(f"No closed contour enclosing the axis for normalized psi={level:.6f}.")
    count = cfg.boundary_points if n_theta is None else int(n_theta)
    return resample_curve_by_angle(selected, center=axis_center, count=max(count, 64))


def analytic_psin(
    R: np.ndarray | float, Z: np.ndarray | float, cfg: SolovevReferenceConfig
) -> np.ndarray:
    return build_exact_solution(cfg).normalized_psin(R, Z)


def ffprim_profile(psin: np.ndarray | float, cfg: SolovevReferenceConfig) -> np.ndarray:
    psi_arr = np.asarray(psin, dtype=np.float64)
    return np.full_like(psi_arr, -build_exact_solution(cfg).ffprime_raw, dtype=np.float64)


def mu0_pprime_profile(psin: np.ndarray | float, cfg: SolovevReferenceConfig) -> np.ndarray:
    psi_arr = np.asarray(psin, dtype=np.float64)
    return np.full_like(psi_arr, -build_exact_solution(cfg).mu0_pprime_raw, dtype=np.float64)


def pprime_profile(psin: np.ndarray | float, cfg: SolovevReferenceConfig) -> np.ndarray:
    return mu0_pprime_profile(psin, cfg) / MU0


def pressure_profile(psin: np.ndarray | float, cfg: SolovevReferenceConfig) -> np.ndarray:
    psi_arr = np.asarray(psin, dtype=np.float64)
    solution = build_exact_solution(cfg)
    return solution.mu0_pprime_raw * (float(cfg.psi_boundary) - psi_arr) / MU0


def fpol_profile(psin: np.ndarray | float, cfg: SolovevReferenceConfig) -> np.ndarray:
    psi_arr = np.asarray(psin, dtype=np.float64)
    solution = build_exact_solution(cfg)
    f_squared = solution.F_axis * solution.F_axis - 2.0 * solution.ffprime_raw * psi_arr
    return np.sqrt(f_squared)


@lru_cache(maxsize=None)
def q_sample_table(cfg: SolovevReferenceConfig) -> tuple[np.ndarray, np.ndarray]:
    solution = build_exact_solution(cfg)
    q_levels = np.concatenate(
        (
            np.asarray([0.05], dtype=np.float64),
            np.linspace(0.1, float(cfg.psi_boundary), max(int(cfg.q_samples), 4)),
        )
    )
    q_levels = np.unique(np.clip(q_levels, 0.05, float(cfg.psi_boundary)))
    q_values = np.empty_like(q_levels)
    for idx, level in enumerate(q_levels):
        curve = analytic_boundary_points(cfg, level=float(level), n_theta=1024)
        closed = np.vstack((curve, curve[:1]))
        ds = np.sqrt(np.diff(closed[:, 0]) ** 2 + np.diff(closed[:, 1]) ** 2)
        dpsi_raw_dR, dpsi_raw_dZ = solution.grad_psi_raw(curve[:, 0], curve[:, 1])
        grad_norm = np.sqrt(dpsi_raw_dR * dpsi_raw_dR + dpsi_raw_dZ * dpsi_raw_dZ)
        f_value = float(fpol_profile(np.asarray([level], dtype=np.float64), cfg)[0])
        q_values[idx] = (
            float(cfg.q_sign)
            * f_value
            * np.sum(ds / np.maximum(curve[:, 0] * grad_norm, 1.0e-12))
            / (2.0 * np.pi)
        )
    return q_levels, q_values


def q_profile(psin: np.ndarray | float, cfg: SolovevReferenceConfig) -> np.ndarray:
    psi_arr = np.asarray(psin, dtype=np.float64)
    q_levels, q_values = q_sample_table(cfg)
    return np.interp(
        np.clip(psi_arr, 0.0, float(cfg.psi_boundary)),
        q_levels,
        q_values,
        left=float(q_values[0]),
        right=float(q_values[-1]),
    )


def estimate_total_current(cfg: SolovevReferenceConfig) -> float:
    boundary = analytic_boundary_points(cfg, level=float(cfg.psi_boundary))
    r_min = float(np.min(boundary[:, 0]))
    r_max = float(np.max(boundary[:, 0]))
    z_min = float(np.min(boundary[:, 1]))
    z_max = float(np.max(boundary[:, 1]))
    rr = np.linspace(r_min, r_max, max(int(cfg.nr), 320), dtype=np.float64)
    zz = np.linspace(z_min, z_max, max(int(cfg.nz), 320), dtype=np.float64)
    RR, ZZ = np.meshgrid(rr, zz, indexing="xy")
    psin = analytic_psin(RR, ZZ, cfg)
    mask = psin <= float(cfg.psi_boundary) + 1.0e-9
    source = ffprim_profile(psin, cfg) + RR * RR * mu0_pprime_profile(psin, cfg)
    jphi = source / (MU0 * np.maximum(RR, 1.0e-9))
    return float(np.trapezoid(np.trapezoid(np.where(mask, jphi, 0.0), zz, axis=0), rr))


def build_solovev_geqdsk(cfg: SolovevReferenceConfig) -> Geqdsk:
    solution = build_exact_solution(cfg)
    boundary = analytic_boundary_points(cfg, level=float(cfg.psi_boundary))
    r_min = float(np.min(boundary[:, 0]))
    r_max = float(np.max(boundary[:, 0]))
    z_min = float(np.min(boundary[:, 1]))
    z_max = float(np.max(boundary[:, 1]))
    r_pad = max((r_max - r_min) * 0.08, 1.0e-3)
    z_pad = max((z_max - z_min) * 0.08, 1.0e-3)

    Rmin = r_min - r_pad
    Rmax = r_max + r_pad
    Zmin = z_min - z_pad
    Zmax = z_max + z_pad
    rr = np.linspace(Rmin, Rmax, int(cfg.nr), dtype=np.float64)
    zz = np.linspace(Zmin, Zmax, int(cfg.nz), dtype=np.float64)
    RR, ZZ = np.meshgrid(rr, zz, indexing="xy")
    psirz = analytic_psin(RR, ZZ, cfg)
    psin_axis = np.linspace(0.0, float(cfg.psi_boundary), int(cfg.nr), dtype=np.float64)

    return Geqdsk(
        header="exact analytic positive-D reference",
        NR=int(cfg.nr),
        NZ=int(cfg.nz),
        R0=float(cfg.major_radius),
        Z0=0.0,
        Rmin=Rmin,
        Rmax=Rmax,
        Zmin=Zmin,
        Zmax=Zmax,
        boundary=np.asarray(boundary, dtype=np.float64),
        limiter=np.empty((0, 2), dtype=np.float64),
        Bt0=float(cfg.toroidal_field),
        Raxis=float(solution.axis_radius),
        Zaxis=0.0,
        Ip=estimate_total_current(cfg),
        psi_axis=0.0,
        psi_bound=float(cfg.psi_boundary),
        F=fpol_profile(psin_axis, cfg),
        P=pressure_profile(psin_axis, cfg),
        FF_psi=ffprim_profile(psin_axis, cfg),
        P_psi=pprime_profile(psin_axis, cfg),
        q=q_profile(psin_axis, cfg),
        psi=np.asarray(psirz.T, dtype=np.float64).copy(),
    )


def write_solovev_reference_gfile(path: str, *, cfg: SolovevReferenceConfig) -> str:
    geqdsk = build_solovev_geqdsk(cfg)
    geqdsk.write_geqdsk(path)
    return path


def read_geqdsk(path: str) -> Geqdsk:
    geqdsk = Geqdsk()
    geqdsk.read_geqdsk(path)
    return geqdsk


def build_boundary(
    geqdsk: Geqdsk, *, fit_m: int, fit_n: int
) -> tuple[Boundary, dict[str, float | np.ndarray]]:
    fit = _fit_boundary_params(
        geqdsk,
        M=fit_m,
        N=fit_n,
        maxtol=BOUNDARY_MAXTOL,
        R0=None,
        Z0=None,
        a=None,
        ka=None,
    )
    normalized = {
        "rms": float(fit["rms"]),
        "a": float(fit["a"]),
        "R0": float(fit["R0"]),
        "Z0": float(fit["Z0"]),
        "ka": float(fit["ka"]),
        "c_offsets": np.asarray(fit["c_offsets"], dtype=np.float64),
        "s_offsets": np.asarray(fit["s_offsets"], dtype=np.float64),
    }
    boundary = Boundary(
        a=normalized["a"],
        R0=normalized["R0"],
        Z0=normalized["Z0"],
        B0=float(geqdsk.Bt0),
        ka=normalized["ka"],
        c_offsets=normalized["c_offsets"],
        s_offsets=normalized["s_offsets"],
    )
    return boundary, normalized


def build_solver_case(
    boundary: Boundary, geqdsk: Geqdsk, *, profile_coeffs: dict[str, list[float]]
) -> OperatorCase:
    return OperatorCase(
        route="PF",
        coordinate="psin",
        nodes="uniform",
        profile_coeffs={name: list(values) for name, values in profile_coeffs.items()},
        boundary=boundary,
        heat_input=MU0 * np.asarray(geqdsk.P_psi, dtype=np.float64),
        current_input=np.asarray(geqdsk.FF_psi, dtype=np.float64),
        Ip=MU0 * float(geqdsk.Ip),
    )


def solve_equilibrium(
    case: OperatorCase,
    *,
    solve_nr: int = SOLVE_NR,
    solve_nt: int = SOLVE_NT,
) -> tuple[Solver, object, object]:
    solve_grid = Grid(Nr=int(solve_nr), Nt=int(solve_nt), quadrature_scheme="legendre")
    plot_grid = Grid(
        Nr=max(PLOT_NR, int(solve_nr)),
        Nt=max(PLOT_NT, int(solve_nt)),
        quadrature_scheme="uniform",
        L_max=solve_grid.L_max,
        M_max=solve_grid.M_max,
    )
    solver = Solver(
        operator=Operator(solve_grid, case),
        config=SolverConfig(
            method=SOLVER_METHOD,
            max_evaluations=SOLVER_MAXFEV,
            enable_warmstart=False,
            enable_fallback=False,
            enable_verbose=False,
            enable_history=False,
        ),
    )
    solver.solve(
        enable_verbose=False, enable_history=False, enable_warmstart=False, enable_fallback=False
    )
    equilibrium = solver.build_equilibrium()
    return solver, equilibrium, equilibrium.resample(grid=plot_grid)


def curve_distance_metrics(points_a: np.ndarray, points_b: np.ndarray) -> dict[str, float]:
    dist = np.sqrt(
        np.sum((np.asarray(points_a)[:, None, :] - np.asarray(points_b)[None, :, :]) ** 2, axis=2)
    )
    nearest_a = dist.min(axis=1)
    nearest_b = dist.min(axis=0)
    return {
        "hausdorff": float(max(nearest_a.max(), nearest_b.max())),
        "rms": float(np.sqrt(0.5 * (np.mean(nearest_a**2) + np.mean(nearest_b**2)))),
    }


def build_surface_from_psin(equilibrium, level: float) -> np.ndarray:
    psin = np.asarray(equilibrium.psin, dtype=np.float64)
    rho = np.asarray(equilibrium.rho, dtype=np.float64)
    order = np.argsort(psin)
    psin_unique, unique_idx = np.unique(psin[order], return_index=True)
    rho_level = float(np.interp(level, psin_unique, rho[order][unique_idx]))
    geometry = equilibrium.geometry
    R = np.array(
        [np.interp(rho_level, rho, geometry.R[:, idx]) for idx in range(equilibrium.grid.Nt)],
        dtype=np.float64,
    )
    Z = np.array(
        [np.interp(rho_level, rho, geometry.Z[:, idx]) for idx in range(equilibrium.grid.Nt)],
        dtype=np.float64,
    )
    return np.column_stack((R, Z))


def select_contour(
    candidates: list[np.ndarray], *, axis_center: tuple[float, float]
) -> np.ndarray | None:
    selected = None
    selected_length = -1
    for curve in candidates:
        arr = np.asarray(curve, dtype=np.float64)
        if arr.shape[0] < 8:
            continue
        if not point_in_polygon(arr, axis_center[0], axis_center[1]):
            continue
        if arr.shape[0] > selected_length:
            selected = arr.copy()
            selected_length = arr.shape[0]
    if selected is not None:
        return selected
    if candidates:
        return max((np.asarray(curve, dtype=np.float64) for curve in candidates), key=len)
    return None


def extract_gfile_surfaces(geqdsk: Geqdsk, levels: list[float]) -> dict[float, np.ndarray]:
    psi_span = float(geqdsk.psi_bound - geqdsk.psi_axis)
    psin_grid = (np.asarray(geqdsk.psi.T, dtype=np.float64) - float(geqdsk.psi_axis)) / psi_span
    R = np.linspace(geqdsk.Rmin, geqdsk.Rmax, geqdsk.NR, dtype=np.float64)
    Z = np.linspace(geqdsk.Zmin, geqdsk.Zmax, geqdsk.NZ, dtype=np.float64)
    axis_center = (float(geqdsk.Raxis), float(geqdsk.Zaxis))
    surfaces: dict[float, np.ndarray] = {}
    contour_levels = [float(level) for level in levels if level < 1.0 - 1.0e-12]
    if contour_levels:
        fig, ax = plt.subplots()
        contour = ax.contour(R, Z, psin_grid, levels=contour_levels)
        plt.close(fig)
        for idx, level in enumerate(contour_levels):
            selected = select_contour(contour.allsegs[idx], axis_center=axis_center)
            if selected is not None:
                surfaces[level] = selected
    if any(abs(level - 1.0) <= 1.0e-12 for level in levels):
        surfaces[1.0] = np.asarray(geqdsk.boundary, dtype=np.float64)
    return surfaces


def collect_surface_metrics(
    reference_surfaces: dict[float, np.ndarray], equilibrium, levels: list[float]
) -> tuple[dict[float, np.ndarray], dict[float, dict[str, float]]]:
    veqpy_surfaces: dict[float, np.ndarray] = {}
    metrics: dict[float, dict[str, float]] = {}
    for level in levels:
        if level not in reference_surfaces:
            continue
        veqpy_surface = build_surface_from_psin(equilibrium, level)
        if (
            np.asarray(reference_surfaces[level], dtype=np.float64).size == 0
            or np.asarray(veqpy_surface, dtype=np.float64).size == 0
        ):
            continue
        veqpy_surfaces[level] = veqpy_surface
        metrics[level] = curve_distance_metrics(reference_surfaces[level], veqpy_surface)
    return veqpy_surfaces, metrics


def build_profile_from_equilibrium(equilibrium) -> tuple[np.ndarray, np.ndarray]:
    geometry = equilibrium.geometry
    psin = np.asarray(equilibrium.psin, dtype=np.float64)
    curves = [
        np.column_stack((geometry.R[idx], geometry.Z[idx])) for idx in range(1, equilibrium.grid.Nr)
    ]
    edge_area = polygon_area(curves[-1])
    rho_geom = [0.0]
    psin_values = [0.0]
    for idx, curve in enumerate(curves, start=1):
        rho_geom.append(float((polygon_area(curve) / edge_area) ** 0.5))
        psin_values.append(float(psin[idx]))
    return np.asarray(rho_geom, dtype=np.float64), np.asarray(psin_values, dtype=np.float64)


def build_profile_from_analytic_levels(
    cfg: SolovevReferenceConfig, psin_values: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    psin = np.asarray(psin_values, dtype=np.float64)
    curves = []
    for level in psin[1:]:
        try:
            curve = analytic_boundary_points(
                cfg, level=float(level), n_theta=max(cfg.boundary_points, 1024)
            )
        except RuntimeError:
            curve = analytic_surface_points_direct(
                cfg, level=float(level), n_theta=max(cfg.boundary_points, 1024)
            )
        curves.append(curve)
    edge_area = polygon_area(curves[-1])
    rho_geom = [0.0]
    profile_psin = [0.0]
    for level, curve in zip(psin[1:], curves, strict=True):
        rho_geom.append(float((polygon_area(curve) / edge_area) ** 0.5))
        profile_psin.append(float(level))
    return np.asarray(rho_geom, dtype=np.float64), np.asarray(profile_psin, dtype=np.float64)


def build_profile_from_gfile_levels(
    geqdsk: Geqdsk, psin_values: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    levels = sorted(
        {
            float(level)
            for level in np.asarray(psin_values, dtype=np.float64)
            if 0.0 < float(level) <= 1.0
        }
        | {1.0}
    )
    surfaces = extract_gfile_surfaces(geqdsk, levels)
    edge_area = polygon_area(surfaces[1.0])
    rho_geom = [0.0]
    profile_psin = [0.0]
    for level in np.asarray(psin_values[1:], dtype=np.float64):
        nearest = min(
            (key for key in surfaces if key > 0.0), key=lambda key: abs(key - float(level))
        )
        rho_geom.append(float((polygon_area(surfaces[nearest]) / edge_area) ** 0.5))
        profile_psin.append(float(level))
    return np.asarray(rho_geom, dtype=np.float64), np.asarray(profile_psin, dtype=np.float64)


def build_shape_rms_profile(
    geqdsk: Geqdsk,
    plot_equilibrium,
    *,
    generate_analytic_reference: bool,
) -> tuple[np.ndarray, np.ndarray]:
    profile_levels = [float(level) for level in SHAPE_RMS_PSIN_LEVELS]
    if not profile_levels:
        return np.asarray([], dtype=np.float64), np.asarray([], dtype=np.float64)

    theta_eval = np.linspace(
        0.0,
        2.0 * np.pi,
        max(int(plot_equilibrium.grid.Nt), 512),
        endpoint=False,
        dtype=np.float64,
    )

    ref_center = (float(geqdsk.Raxis), float(geqdsk.Zaxis))
    veqpy_center = (
        float(plot_equilibrium.geometry.R[0, 0]),
        float(plot_equilibrium.geometry.Z[0, 0]),
    )
    if generate_analytic_reference:
        psin_axis: list[float] = []
        rms_values: list[float] = []
        for level in profile_levels:
            if abs(level) <= 1.0e-12:
                psin_axis.append(level)
                rms_values.append(axis_position_error(ref_center, veqpy_center))
                continue
            try:
                reference_surface = analytic_surface_points_direct(
                    REFERENCE_CONFIG,
                    level=level,
                    n_theta=theta_eval.size,
                )
            except RuntimeError:
                continue
            veqpy_surface = build_surface_from_psin(plot_equilibrium, level)
            ref_r = radial_profile_from_surface(
                reference_surface,
                center=ref_center,
                theta_eval=theta_eval,
            )
            veqpy_r = radial_profile_from_surface(
                veqpy_surface,
                center=veqpy_center,
                theta_eval=theta_eval,
            )
            rms = float(np.sqrt(np.mean((veqpy_r - ref_r) ** 2)))
            psin_axis.append(level)
            rms_values.append(rms)
        return np.asarray(psin_axis, dtype=np.float64), np.asarray(rms_values, dtype=np.float64)

    reference_surfaces = extract_gfile_surfaces(
        geqdsk,
        [level for level in profile_levels if level > 1.0e-12],
    )
    psin_axis: list[float] = []
    rms_values: list[float] = []
    for level in profile_levels:
        if abs(level) <= 1.0e-12:
            psin_axis.append(level)
            rms_values.append(axis_position_error(ref_center, veqpy_center))
            continue
        if level not in reference_surfaces:
            continue
        reference_surface = np.asarray(reference_surfaces[level], dtype=np.float64)
        veqpy_surface = np.asarray(
            build_surface_from_psin(plot_equilibrium, level), dtype=np.float64
        )
        if reference_surface.size == 0 or veqpy_surface.size == 0:
            continue
        ref_r = radial_profile_from_surface(
            reference_surface,
            center=ref_center,
            theta_eval=theta_eval,
        )
        veqpy_r = radial_profile_from_surface(
            veqpy_surface,
            center=veqpy_center,
            theta_eval=theta_eval,
        )
        rms = float(np.sqrt(np.mean((veqpy_r - ref_r) ** 2)))
        psin_axis.append(level)
        rms_values.append(rms)
    return np.asarray(psin_axis, dtype=np.float64), np.asarray(rms_values, dtype=np.float64)


def build_psin_error_profile(
    reference_psin_profile: tuple[np.ndarray, np.ndarray],
    veqpy_psin_profile: tuple[np.ndarray, np.ndarray],
    *,
    rho_samples: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    rho_eval = np.asarray(
        PSIN_ERROR_RHO_LEVELS if rho_samples is None else rho_samples, dtype=np.float64
    )
    if rho_eval.size == 0:
        return np.asarray([], dtype=np.float64), np.asarray([], dtype=np.float64)

    rho_ref, psin_ref = reference_psin_profile
    rho_vq, psin_vq = veqpy_psin_profile
    psin_axis = interpolate_unique(rho_ref, psin_ref, rho_eval)
    ref_psin_samples = interpolate_unique(rho_ref, psin_ref, rho_eval)
    vq_psin_samples = interpolate_unique(rho_vq, psin_vq, rho_eval)
    errors = np.abs(vq_psin_samples - ref_psin_samples)
    return psin_axis, errors


def build_case_result(case_spec: CaseSpec) -> CaseResult:
    if case_spec.generate_gfile:
        write_solovev_reference_gfile(case_spec.gfile_path, cfg=REFERENCE_CONFIG)
        geqdsk = build_solovev_geqdsk(REFERENCE_CONFIG)
    else:
        geqdsk = read_geqdsk(case_spec.gfile_path)

    boundary, fit = build_boundary(
        geqdsk,
        fit_m=case_spec.boundary_fit_m,
        fit_n=case_spec.boundary_fit_n,
    )
    case = build_solver_case(boundary, geqdsk, profile_coeffs=case_spec.profile_coeffs)
    solver, equilibrium, plot_equilibrium = solve_equilibrium(
        case,
        solve_nr=case_spec.solve_nr,
        solve_nt=case_spec.solve_nt,
    )

    levels = default_levels_from_count(SURFACE_COUNT)
    if case_spec.generate_gfile:
        reference_surfaces = {
            float(level): analytic_surface_points_direct(
                REFERENCE_CONFIG,
                level=float(level),
                n_theta=max(int(plot_equilibrium.grid.Nt), 512),
            )
            for level in levels
        }
        reference_psin_profile = build_profile_from_analytic_levels(
            REFERENCE_CONFIG, equilibrium.psin
        )
    else:
        reference_surfaces = extract_gfile_surfaces(geqdsk, levels)
        reference_psin_profile = build_profile_from_gfile_levels(geqdsk, equilibrium.psin)

    veqpy_surfaces, surface_metrics = collect_surface_metrics(
        reference_surfaces, plot_equilibrium, levels
    )
    veqpy_psin_profile = build_profile_from_equilibrium(equilibrium)
    psin_error_profile = build_psin_error_profile(
        reference_psin_profile,
        veqpy_psin_profile,
    )
    shape_rms_profile = build_shape_rms_profile(
        geqdsk,
        plot_equilibrium,
        generate_analytic_reference=case_spec.generate_gfile,
    )

    result = solver.result
    if result is not None:
        print(case_spec.title)
        print(f"boundary fit rms: {float(fit['rms']):.3e}")
        print(f"solver residual: {float(result.residual_norm_final):.3e}")
        print(f"solve elapsed: {float(result.elapsed) / 1000.0:.3f} ms")
        print(f"solved Ip: {float(equilibrium.Ip):.6e}")
        if 1.0 in surface_metrics:
            print(f"boundary rms distance: {surface_metrics[1.0]['rms']:.3e}")

    return CaseResult(
        title=case_spec.title,
        reference_label=case_spec.reference_label,
        case_spec=case_spec,
        geqdsk=geqdsk,
        equilibrium=equilibrium,
        plot_equilibrium=plot_equilibrium,
        reference_surfaces=reference_surfaces,
        veqpy_surfaces=veqpy_surfaces,
        reference_psin_profile=reference_psin_profile,
        veqpy_psin_profile=veqpy_psin_profile,
        psin_error_profile=psin_error_profile,
        shape_rms_profile=shape_rms_profile,
        parameter_count=profile_parameter_count(case_spec.profile_coeffs),
        boundary_fit_rms=float(fit["rms"]),
        solver_residual=float("nan") if result is None else float(result.residual_norm_final),
        solve_elapsed_ms=float("nan") if result is None else float(result.elapsed) / 1000.0,
    )


def format_case_error_metric(case_result: CaseResult) -> str:
    surface_rms_mm = 1.0e3 * aggregate_surface_rms_error(case_result.shape_rms_profile)
    if not np.isfinite(surface_rms_mm):
        return "--"
    return f"{surface_rms_mm:.3f}"


def format_boundary_fit_metric(case_result: CaseResult) -> str:
    boundary_fit_mm = 1.0e3 * float(case_result.boundary_fit_rms)
    if not np.isfinite(boundary_fit_mm):
        return "--"
    return f"{boundary_fit_mm:.3f}"


def format_named_family_tuple(case_spec: CaseSpec, prefix: str, *, include_c0: bool = False) -> str:
    names = []
    if include_c0 and "c0" in case_spec.profile_coeffs:
        names.append("c0")
    indexed = sorted(
        (
            name
            for name in case_spec.profile_coeffs
            if len(name) > 1
            and name.startswith(prefix)
            and name[1:].isdigit()
            and not (include_c0 and name == "c0")
        ),
        key=lambda name: int(name[1:]),
    )
    names.extend(indexed)
    lengths = tuple(_coeff_length(case_spec.profile_coeffs, name) for name in names)
    return format_repeated_tuple_tex(lengths)


def build_case_summary_latex_table(case_results: list[CaseResult]) -> str:
    indent = "              "
    header = [
        "Case",
        "Params",
        r"$E_{\mathrm{tgt}}$ [mm]",
        r"$E_{\mathrm{bdry}}$ [mm]",
        "Core",
        "Cos",
        "Sin",
    ]
    rows: list[list[str]] = []
    for case_result in case_results:
        label = case_result.reference_label
        rows.append(
            [
                CASE_DISPLAY_NAMES.get(label, label),
                str(int(case_result.parameter_count)),
                format_case_error_metric(case_result),
                format_boundary_fit_metric(case_result),
                format_core_psin_tuple(case_result.case_spec),
                format_named_family_tuple(case_result.case_spec, "c", include_c0=True),
                format_named_family_tuple(case_result.case_spec, "s"),
            ]
        )

    column_widths = [
        max(len(row[index]) for row in [header, *rows]) for index in range(len(header))
    ]

    def format_row(row: list[str]) -> str:
        return (
            " & ".join(cell.ljust(column_widths[index]) for index, cell in enumerate(row)) + r" \\"
        )

    return "\n".join(
        indent + line
        for line in [
            r"\hline",
            format_row(header),
            r"\hline",
            *(format_row(row) for row in rows),
            r"\hline",
        ]
    )


def print_case_summary_latex_table(case_results: list[CaseResult]) -> None:
    print(build_case_summary_latex_table(case_results))


def build_compare_figure(case_results: list[CaseResult]) -> plt.Figure:
    apply_plot_style()
    fig = plt.figure(figsize=FIGURE_SIZE)
    outer_grid = fig.add_gridspec(
        1,
        FIGURE_GRID_NCOLS,
        wspace=FIGURE_GRID_WSPACE,
    )

    for col, case_result in enumerate(case_results):
        panel_grid = outer_grid[0, col].subgridspec(
            PANEL_GRID_NROWS,
            1,
            height_ratios=PANEL_GRID_HEIGHT_RATIOS,
            hspace=PANEL_GRID_HSPACE,
        )
        ax_surfaces = fig.add_subplot(panel_grid[0, 0])
        ax_fp = fig.add_subplot(panel_grid[2, 0])
        ax_psin = fig.add_subplot(panel_grid[4, 0], sharex=ax_fp)

        shared_levels = sorted(case_result.reference_surfaces)
        for idx, level in enumerate(shared_levels):
            linewidth = SURFACE_LINE_WIDTH if level < 1.0 - 1.0e-12 else BOUNDARY_SURFACE_LINE_WIDTH
            ax_surfaces.plot(
                close_curve(case_result.reference_surfaces[level])[:, 0],
                close_curve(case_result.reference_surfaces[level])[:, 1],
                linestyle=ANALYTIC_LINE_STYLE,
                color=ANALYTIC_LINE_COLOR,
                linewidth=linewidth * REFERENCE_SURFACE_SCALE,
                label=(case_result.reference_label if idx == 0 else None),
            )
            if level in case_result.veqpy_surfaces:
                ax_surfaces.plot(
                    close_curve(case_result.veqpy_surfaces[level])[:, 0],
                    close_curve(case_result.veqpy_surfaces[level])[:, 1],
                    linestyle=VEQPY_LINE_STYLE,
                    color=VEQPY_LINE_COLOR,
                    linewidth=linewidth,
                    label=("VEQ" if idx == 0 else None),
                )

        style_axis(
            ax_surfaces,
            title=case_result.title,
            xlabel="R [m]",
            ylabel="Z [m]",
        )
        surface_curves = list(case_result.reference_surfaces.values()) + list(
            case_result.veqpy_surfaces.values()
        )
        limits = compute_rz_limits(surface_curves)
        x0, x1 = limits[0]
        xpad_extra_fraction = (
            FIRST_COLUMN_SURFACE_XPAD_EXTRA_FRACTION if col == 0 else SURFACE_XPAD_EXTRA_FRACTION
        )
        xpad_extra = (x1 - x0) * xpad_extra_fraction
        ax_surfaces.set_xlim(x0, x1 + xpad_extra)
        ax_surfaces.set_ylim(*limits[1])
        ax_surfaces.set_aspect("equal")
        ax_surfaces.scatter(
            [float(case_result.geqdsk.Raxis)],
            [float(case_result.geqdsk.Zaxis)],
            color=REFERENCE_AXIS_COLOR,
            marker=REFERENCE_AXIS_MARKER,
            s=REFERENCE_AXIS_MARKER_SIZE,
            zorder=5,
            label="_nolegend_",
        )
        ax_surfaces.scatter(
            [float(case_result.equilibrium.geometry.R[0, 0])],
            [float(case_result.equilibrium.geometry.Z[0, 0])],
            color=VEQPY_AXIS_COLOR,
            marker=VEQPY_AXIS_MARKER,
            s=VEQPY_AXIS_MARKER_SIZE,
            zorder=6,
            label="_nolegend_",
        )
        ax_surfaces.legend(
            loc=SURFACE_LEGEND_LOC,
            fontsize=scaled_font_size(LEGEND_FONT_SIZE),
            frameon=LEGEND_FRAME_ON,
            columnspacing=LEGEND_COLUMN_SPACING,
            labelspacing=LEGEND_LABEL_SPACING,
        )

        psin_axis = np.linspace(0.0, 1.0, case_result.geqdsk.NR, dtype=np.float64)
        ax_fp.plot(
            psin_axis,
            np.asarray(case_result.geqdsk.FF_psi, dtype=np.float64),
            color=SOURCE_FF_COLOR,
            linewidth=PROFILE_LINE_WIDTH,
            label=r"$FF_\psi$",
        )
        ax_fp.plot(
            psin_axis,
            MU0 * np.asarray(case_result.geqdsk.P_psi, dtype=np.float64),
            color=SOURCE_P_COLOR,
            linestyle="--",
            linewidth=PROFILE_LINE_WIDTH,
            label=r"$\mu_0 P_\psi$",
        )
        style_axis(ax_fp, title="", xlabel="", ylabel="value")
        ax_fp.tick_params(labelbottom=False)
        set_symmetric_ylim(
            ax_fp,
            headroom_ratio=max(SOURCE_BOTTOM_HEADROOM, SOURCE_TOP_HEADROOM),
        )
        if col == 0:
            ax_fp.legend(
                loc=SOURCE_LEGEND_LOC,
                fontsize=scaled_font_size(LEGEND_FONT_SIZE),
                frameon=LEGEND_FRAME_ON,
                columnspacing=LEGEND_COLUMN_SPACING,
                labelspacing=LEGEND_LABEL_SPACING,
            )

        psi_psin, psi_error = case_result.psin_error_profile
        shape_psin, shape_rms = case_result.shape_rms_profile
        if shape_psin.size and shape_rms.size:
            ax_psin.semilogy(
                shape_psin,
                np.maximum(shape_rms, LOG_Y_FLOOR),
                color=THIRD_ROW_SHAPE_COLOR,
                linestyle=THIRD_ROW_SHAPE_STYLE,
                linewidth=PROFILE_LINE_WIDTH,
                marker=THIRD_ROW_SHAPE_MARKER,
                markersize=THIRD_ROW_SHAPE_MARKER_SIZE,
                label=r"${R}_{\text{rms}}(\hat{\psi})$ [m]",
            )
        if psi_psin.size and psi_error.size:
            ax_psin.semilogy(
                psi_psin,
                np.maximum(psi_error, LOG_Y_FLOOR),
                color=THIRD_ROW_PSI_COLOR,
                linestyle=THIRD_ROW_PSI_STYLE,
                linewidth=PROFILE_LINE_WIDTH,
                marker=THIRD_ROW_PSI_MARKER,
                markersize=THIRD_ROW_PSI_MARKER_SIZE,
                markeredgewidth=THIRD_ROW_PSI_MARKER_EDGE_WIDTH,
                label=r"$|\Delta\hat{\psi}(\rho)|$",
            )
        style_axis(
            ax_psin,
            title="",
            xlabel=r"$\hat{\psi}$",
            ylabel="error",
        )
        ax_psin.set_ylim(PSI_ERROR_YMIN, PSI_ERROR_YMAX)
        if col == 0:
            ax_psin.legend(
                loc=THIRD_ROW_LEGEND_LOC,
                fontsize=scaled_font_size(LEGEND_FONT_SIZE),
                frameon=LEGEND_FRAME_ON,
                columnspacing=LEGEND_COLUMN_SPACING,
                labelspacing=LEGEND_LABEL_SPACING,
            )

    return fig


def main() -> None:
    os.makedirs("figures", exist_ok=True)
    case_results = [build_case_result(case_spec) for case_spec in CASE_SPECS]
    print_case_summary_latex_table(case_results)
    fig = build_compare_figure(case_results)
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

    print(f"saved: {SAVE_GFILE_PATH}")
    print(f"saved: {SAVE_PNG_PATH}")
    print(f"saved: {SAVE_PDF_PATH}")


if __name__ == "__main__":
    main()

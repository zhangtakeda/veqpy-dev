import copy
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm
from rich import print

from veqpy.engine import validate_route
from veqpy.model import Boundary, Grid
from veqpy.operator import Operator, OperatorCase
from veqpy.operator.codec import decode_packed_blocks
from veqpy.operator.layout import build_shape_profile_names
from veqpy.solver import Solver, SolverConfig

WARMUP_TIMES = 10

GRID_SIZES = (8, 12, 16, 24, 32, 48, 64)
REPORT_PATH = Path(__file__).resolve().parent / "demo" / "grid-shape-error.png"
SOURCE_SAMPLE_COUNT = 21


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
    enable_verbose=True,
    enable_warmstart=False,
)


LOWER_COEFFS = {
    "h": [0.0] * 3,
    "k": [0.0] * 5,
    "s1": [0.0] * 3,
}


HIGHER_COEFFS = {
    "h": [0.0] * 10,
    "k": [0.0] * 10,
    "v": [0.0] * 10,
    "c0": [0.0] * 10,
    "c1": [0.0] * 5,
    "c2": [0.0] * 5,
    "s1": [0.0] * 10,
    "s2": [0.0] * 5,
    "s3": [0.0] * 5,
}


LOWER_GRID = Grid(
    Nr=12,
    Nt=12,
    scheme="legendre",
)

HIGHER_GRID = Grid(
    Nr=32,
    Nt=32,
    scheme="legendre",
)


def warmup_solver(solver: Solver) -> None:
    """Run one silent solve so demo timings exclude first-call JIT overhead."""
    for _ in range(WARMUP_TIMES):
        solver.solve(enable_verbose=False, enable_history=False)


def pf_reference_profiles(rho: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    beta0 = 0.75
    psin = rho**2
    psin_r = 2.0 * rho

    alpha_p, alpha_f = 5.0, 3.32
    exp_ap, exp_af = np.exp(alpha_p), np.exp(alpha_f)
    den_p, den_f = 1.0 + exp_ap * (alpha_p - 1.0), 1.0 + exp_af * (alpha_f - 1.0)

    current_input = (1.0 - beta0) * alpha_f * (np.exp(alpha_f * psin) - exp_af) / den_f * psin_r
    heat_input = beta0 * alpha_p * (np.exp(alpha_p * psin) - exp_ap) / den_p * psin_r
    return current_input, heat_input


def pf_psin_reference_profiles(psin: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    beta0 = 0.75

    alpha_p, alpha_f = 5.0, 3.32
    exp_ap, exp_af = np.exp(alpha_p), np.exp(alpha_f)
    den_p, den_f = 1.0 + exp_ap * (alpha_p - 1.0), 1.0 + exp_af * (alpha_f - 1.0)

    current_input = (1.0 - beta0) * alpha_f * (np.exp(alpha_f * psin) - exp_af) / den_f
    heat_input = beta0 * alpha_p * (np.exp(alpha_p * psin) - exp_ap) / den_p
    return current_input, heat_input


def _normalize_psin_coordinate_samples(psin_samples: np.ndarray) -> tuple[np.ndarray, float]:
    psin = np.asarray(psin_samples, dtype=np.float64).copy()
    if psin.ndim != 1 or psin.size < 2:
        raise ValueError("psin_samples must be a 1D array with at least two points")

    offset = float(psin[0])
    scale = float(psin[-1] - offset)
    if abs(scale) < 1e-12:
        raise ValueError("psin_samples do not span a valid normalized flux interval")

    psin -= offset
    psin /= scale
    np.clip(psin, 0.0, 1.0, out=psin)
    psin[0] = 0.0
    psin[-1] = 1.0
    return psin, scale


def _enforce_psin_endpoint_constraints(
    current_input: np.ndarray,
    heat_input: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    current = np.asarray(current_input, dtype=np.float64).copy()
    heat = np.asarray(heat_input, dtype=np.float64).copy()
    current[-1] = 0.0
    heat[-1] = 0.0
    return current, heat


def _enforce_rho_endpoint_constraints(
    current_input: np.ndarray,
    heat_input: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    current = np.asarray(current_input, dtype=np.float64).copy()
    heat = np.asarray(heat_input, dtype=np.float64).copy()
    current[0] = 0.0
    current[-1] = 0.0
    heat[0] = 0.0
    heat[-1] = 0.0
    return current, heat


def _build_uniform_rho_inputs(n_src: int = SOURCE_SAMPLE_COUNT) -> tuple[np.ndarray, np.ndarray]:
    rho_uniform = np.linspace(0.0, 1.0, n_src)
    current_input, heat_input = pf_reference_profiles(rho_uniform)
    return _enforce_rho_endpoint_constraints(current_input, heat_input)


def _build_uniform_psin_inputs(n_src: int = SOURCE_SAMPLE_COUNT) -> tuple[np.ndarray, np.ndarray]:
    psin_uniform = np.linspace(0.0, 1.0, n_src)
    current_input, heat_input = pf_psin_reference_profiles(psin_uniform)
    return _enforce_psin_endpoint_constraints(current_input, heat_input)


FFn_r, Pn_r = _build_uniform_rho_inputs()


LOWER_CASE = OperatorCase(
    route="PF",
    coordinate="rho",
    nodes="uniform",
    Ip=3.0e6,
    profile_coeffs=LOWER_COEFFS,
    boundary=BOUNDARY,
    heat_input=Pn_r,
    current_input=FFn_r,
)

HIGHER_CASE = OperatorCase(
    route="PF",
    profile_coeffs=HIGHER_COEFFS,
    boundary=BOUNDARY,
    heat_input=Pn_r,
    current_input=FFn_r,
    coordinate="rho",
    nodes="uniform",
    Ip=3.0e6,
)


@dataclass(frozen=True)
class CaseSpec:
    name: str
    profile_coeffs: dict[str, list[float] | None]
    grid_sizes: tuple[int, ...]


@dataclass(frozen=True)
class SolveRecord:
    case_name: str
    nr: int
    nt: int
    success: bool
    message: str
    nfev: int
    shape_labels: tuple[str, ...]
    shape_values: np.ndarray

    @property
    def used_fallback(self) -> bool:
        return "selected method=" in self.message and "fallback" in self.message


CASE_SPECS = (
    CaseSpec(name="lower", profile_coeffs=LOWER_COEFFS, grid_sizes=GRID_SIZES),
    CaseSpec(name="higher", profile_coeffs=HIGHER_COEFFS, grid_sizes=GRID_SIZES),
)


def _prepare_profile_coeffs(
    *,
    route: str,
    coordinate: str,
    nodes: str,
    profile_coeffs: dict[str, list[float] | None],
) -> dict[str, list[float] | None]:
    coeffs = copy.deepcopy(profile_coeffs)
    route_spec = validate_route(route, coordinate, nodes)
    if route_spec.source_strategy == "profile_owned_psin" and coeffs.get("psin") is None:
        coeffs["psin"] = [0.0] * 5
    return coeffs


def _build_case(grid: Grid, profile_coeffs: dict[str, list[float] | None]) -> OperatorCase:
    del grid
    current_input, heat_input = _build_uniform_rho_inputs()
    return OperatorCase(
        route="PF",
        profile_coeffs=_prepare_profile_coeffs(
            route="PF",
            coordinate="rho",
            nodes="uniform",
            profile_coeffs=profile_coeffs,
        ),
        boundary=BOUNDARY,
        heat_input=heat_input,
        current_input=current_input,
        coordinate="rho",
        nodes="uniform",
        Ip=3.0e6,
    )


def _build_psin_case(profile_coeffs: dict[str, list[float] | None]) -> OperatorCase:
    current_input, heat_input = _build_uniform_psin_inputs()
    return OperatorCase(
        route="PF",
        profile_coeffs=_prepare_profile_coeffs(
            route="PF",
            coordinate="psin",
            nodes="uniform",
            profile_coeffs=profile_coeffs,
        ),
        boundary=BOUNDARY,
        heat_input=heat_input,
        current_input=current_input,
        coordinate="psin",
        nodes="uniform",
        Ip=3.0e6,
    )


def _build_rho_case_from_psin_solution(
    equilibrium,
    *,
    profile_coeffs: dict[str, list[float] | None],
    psin_profile_u: np.ndarray,
    n_src: int,
) -> OperatorCase:
    rho_uniform = np.linspace(0.0, 1.0, n_src)
    raw_psin_uniform = np.interp(rho_uniform, equilibrium.rho, np.asarray(psin_profile_u, dtype=np.float64))
    psin_uniform, psin_scale = _normalize_psin_coordinate_samples(raw_psin_uniform)
    psin_r_uniform = np.interp(rho_uniform, equilibrium.rho, np.asarray(equilibrium.psin_r, dtype=np.float64))
    psin_r_uniform = np.asarray(psin_r_uniform, dtype=np.float64) / psin_scale
    psin_r_uniform[0] = 0.0

    current_psin, heat_psin = pf_psin_reference_profiles(psin_uniform)
    current_psin, heat_psin = _enforce_psin_endpoint_constraints(current_psin, heat_psin)
    current_rho = current_psin * psin_r_uniform
    heat_rho = heat_psin * psin_r_uniform
    current_rho, heat_rho = _enforce_rho_endpoint_constraints(current_rho, heat_rho)
    return OperatorCase(
        route="PF",
        profile_coeffs=_prepare_profile_coeffs(
            route="PF",
            coordinate="rho",
            nodes="uniform",
            profile_coeffs=profile_coeffs,
        ),
        boundary=BOUNDARY,
        heat_input=heat_rho,
        current_input=current_rho,
        coordinate="rho",
        nodes="uniform",
        Ip=3.0e6,
    )


def _print_equilibrium_delta(title: str, lhs, rhs) -> None:
    def _max_profile_delta(
        lhs_rho: np.ndarray, lhs_values: np.ndarray, rhs_rho: np.ndarray, rhs_values: np.ndarray
    ) -> float:
        lhs_rho = np.asarray(lhs_rho, dtype=np.float64)
        rhs_rho = np.asarray(rhs_rho, dtype=np.float64)
        lhs_values = np.asarray(lhs_values, dtype=np.float64)
        rhs_values = np.asarray(rhs_values, dtype=np.float64)
        rhs_on_lhs = np.interp(lhs_rho, rhs_rho, rhs_values)
        return float(np.max(np.abs(lhs_values - rhs_on_lhs)))

    print(title)
    print(
        {
            "max_d_psin_r": _max_profile_delta(lhs.rho, lhs.psin_r, rhs.rho, rhs.psin_r),
            "max_d_FFn_r": _max_profile_delta(lhs.rho, lhs.FFn_r, rhs.rho, rhs.FFn_r),
            "max_d_Pn_r": _max_profile_delta(lhs.rho, lhs.Pn_r, rhs.rho, rhs.Pn_r),
            "lhs_alpha": (float(lhs.alpha1), float(lhs.alpha2)),
            "rhs_alpha": (float(rhs.alpha1), float(rhs.alpha2)),
        }
    )


def _extract_shape_parameters(operator: Operator, x: np.ndarray) -> tuple[tuple[str, ...], np.ndarray]:
    blocks = decode_packed_blocks(
        x,
        operator.profile_L,
        operator.coeff_index,
        profile_names=operator.profile_names,
    )
    shape_names = set(build_shape_profile_names(operator.grid.M_max))
    labels: list[str] = []
    values: list[float] = []
    for name, block in zip(operator.profile_names, blocks, strict=True):
        if name not in shape_names or block is None:
            continue
        for order, value in enumerate(np.asarray(block, dtype=np.float64)):
            labels.append(f"{name}[{order}]")
            values.append(float(value))
    return tuple(labels), np.asarray(values, dtype=np.float64)


def solve_case(case_spec: CaseSpec, nr: int, nt: int) -> SolveRecord:
    grid = Grid(Nr=nr, Nt=nt, scheme="legendre")
    case = _build_case(grid, case_spec.profile_coeffs)
    operator = Operator(grid=grid, case=case)
    solver = Solver(operator=operator, config=CONFIG)
    x = solver.solve(enable_verbose=False, enable_history=False)
    labels, values = _extract_shape_parameters(operator, x)
    result = solver.result
    if result is None:
        raise RuntimeError("solver.result is None after solve")
    return SolveRecord(
        case_name=case_spec.name,
        nr=nr,
        nt=nt,
        success=bool(result.success),
        message=str(result.message),
        nfev=int(result.nfev),
        shape_labels=labels,
        shape_values=values,
    )


def _build_metric_arrays(
    case_spec: CaseSpec,
    records: list[SolveRecord],
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    records_by_grid = {(record.nr, record.nt): record for record in records}
    ref = records_by_grid[(64, 64)]
    if not all(record.shape_labels == ref.shape_labels for record in records):
        raise ValueError(f"Inconsistent shape parameter labels for case={case_spec.name}")

    shape = (len(case_spec.grid_sizes), len(case_spec.grid_sizes))
    max_matrix = np.empty(shape, dtype=np.float64)
    mean_matrix = np.empty(shape, dtype=np.float64)
    rms_matrix = np.empty(shape, dtype=np.float64)
    issue_mask = np.zeros(shape, dtype=bool)

    for i, nr in enumerate(case_spec.grid_sizes):
        for j, nt in enumerate(case_spec.grid_sizes):
            record = records_by_grid[(nr, nt)]
            abs_errors = np.abs(record.shape_values - ref.shape_values)
            max_matrix[i, j] = float(np.max(abs_errors)) if abs_errors.size else 0.0
            mean_matrix[i, j] = float(np.mean(abs_errors)) if abs_errors.size else 0.0
            rms_matrix[i, j] = float(np.sqrt(np.mean(abs_errors**2))) if abs_errors.size else 0.0
            issue_mask[i, j] = (not bool(record.success)) or record.used_fallback

    return (
        {
            "Max Abs Error": max_matrix,
            "Mean Abs Error": mean_matrix,
            "RMS Abs Error": rms_matrix,
        },
        issue_mask,
    )


def _plot_metric_panel(
    fig: plt.Figure,
    ax: plt.Axes,
    case_spec: CaseSpec,
    metric: np.ndarray,
    issue_mask: np.ndarray,
    *,
    title: str,
) -> None:
    positive = metric[np.isfinite(metric) & (metric > 0.0)]
    vmin = float(np.min(positive)) if positive.size else 1e-16
    vmax = float(np.max(positive)) if positive.size else 1.0
    if vmax <= vmin:
        vmax = vmin * 10.0
    display = np.maximum(metric, vmin)

    image = ax.imshow(display, cmap="viridis", norm=LogNorm(vmin=vmin, vmax=vmax), aspect="auto")
    ax.set_title(f"{case_spec.name}: {title}")
    ax.set_xlabel("Nt")
    ax.set_ylabel("Nr")
    ax.set_xticks(np.arange(len(case_spec.grid_sizes)))
    ax.set_xticklabels([str(size) for size in case_spec.grid_sizes])
    ax.set_yticks(np.arange(len(case_spec.grid_sizes)))
    ax.set_yticklabels([str(size) for size in case_spec.grid_sizes])

    marked_rows, marked_cols = np.where(issue_mask)
    if marked_rows.size > 0:
        ax.scatter(marked_cols, marked_rows, marker="x", s=90, linewidths=2.0, color="red")

    for edge in np.arange(-0.5, len(case_spec.grid_sizes), 1.0):
        ax.axvline(edge, color="white", linewidth=0.5, alpha=0.35)
        ax.axhline(edge, color="white", linewidth=0.5, alpha=0.35)

    colorbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    colorbar.set_label("abs error")


def generate_grid_shape_error_report() -> Path:
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 3, figsize=(16, 9), constrained_layout=True)

    for row_idx, case_spec in enumerate(CASE_SPECS):
        records = [solve_case(case_spec, nr, nt) for nr in case_spec.grid_sizes for nt in case_spec.grid_sizes]
        metric_arrays, issue_mask = _build_metric_arrays(case_spec, records)
        for col_idx, (title, metric) in enumerate(metric_arrays.items()):
            _plot_metric_panel(
                fig,
                axes[row_idx, col_idx],
                case_spec,
                metric,
                issue_mask,
                title=title,
            )

    fig.savefig(REPORT_PATH, dpi=220, facecolor="white")
    plt.close(fig)
    return REPORT_PATH


if __name__ == "__main__":
    operator_1 = Operator(grid=LOWER_GRID, case=LOWER_CASE)
    operator_2 = Operator(grid=HIGHER_GRID, case=HIGHER_CASE)

    solver_1 = Solver(operator=operator_1, config=CONFIG)
    solver_2 = Solver(operator=operator_2, config=CONFIG)

    warmup_solver(solver_1)
    warmup_solver(solver_2)

    print("\n\n=== Typical Coefficients + Typical Grid ===")
    x_typical = solver_1.solve()
    eq1 = solver_1.build_equilibrium()
    eq1.write("tests/demo/demo-1.json")
    eq1.plot("tests/demo/demo-1.svg")
    eq1.plot("tests/demo/demo-1.png")
    print(eq1)

    print("\n\n=== High Precision Coefficients + High Precision Grid ===")
    x_high_precision = solver_2.solve()
    eq2 = solver_2.build_equilibrium()
    eq2.write("tests/demo/demo-2.json")
    eq2.plot("tests/demo/demo-2.svg")
    eq2.plot("tests/demo/demo-2.png")
    eq2.compare(
        eq1,
        "tests/demo/demo-comparison.svg",
        label_ref="Higher",
        label_other="Lower",
    )
    print(eq2)

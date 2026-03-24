from __future__ import annotations

import os
import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

PLOT = False
SHOW_PROGRESS = True
WARMSTART = False
ASSERT_EXPECTATIONS = False
BACKEND = os.environ.get("VEQPY_BACKEND", "numba")
os.environ["VEQPY_BACKEND"] = BACKEND

BENCHMARK_REPEAT_COUNT = 100
F_ROBUST_COEFF_COUNT = 5
MAX_SHAPE_ERROR = 1e-1

WARM_START_SCALE_MIN = 0.95
WARM_START_SCALE_MAX = 1.05
WARM_START_SEED = 20260324

BENCHMARK_MODES = ("PF", "PP", "PI", "PJ1", "PJ2", "PQ")
BENCHMARK_MODE_CONSTRAINTS = {
    "PF": ("null", "Ip", "beta"),
    "PP": ("Ip_beta", "Ip", "beta", "null"),
    "PI": ("Ip_beta", "Ip", "beta", "null"),
    "PJ1": ("Ip_beta", "Ip", "beta", "null"),
    "PJ2": ("Ip_beta", "Ip", "beta", "null"),
    "PQ": ("Ip_beta", "Ip", "beta", "null"),
}

try:
    from demo import (
        CASE_1 as PF_REFERENCE_CASE,
        CONFIG as PF_REFERENCE_SOLVER_CONFIG,
        COEFFS as PF_REFERENCE_COEFFS,
        pf_reference_profiles,
    )
except ImportError:
    from veqpy.demo import (
        CASE_1 as PF_REFERENCE_CASE,
        CONFIG as PF_REFERENCE_SOLVER_CONFIG,
        COEFFS as PF_REFERENCE_COEFFS,
        pf_reference_profiles,
    )

from veqpy.model import Equilibrium, Grid, resample_equilibrium
from veqpy.operator import PROFILE_INDEX, Operator, OperatorCase, build_profile_layout
from veqpy.solver import Solver, SolverConfig

PF_REFERENCE_GRID = Grid(
    Nr=32,
    Nt=32,
    scheme="legendre",
)

GRID = Grid(Nr=12, Nt=12, scheme="legendre", L_max=PF_REFERENCE_GRID.L_max)
CONFIG = SolverConfig(method=PF_REFERENCE_SOLVER_CONFIG.method)
BENCHMARK_SOLVE_CONFIG = replace(CONFIG, enable_warmstart=False, enable_verbose=False, enable_history=False)

PF_REFERENCE_CASE_KWARGS = {
    "a": PF_REFERENCE_CASE.a,
    "R0": PF_REFERENCE_CASE.R0,
    "Z0": PF_REFERENCE_CASE.Z0,
    "B0": PF_REFERENCE_CASE.B0,
    "ka": PF_REFERENCE_CASE.ka,
    "c0a": PF_REFERENCE_CASE.c0a,
    "c1a": PF_REFERENCE_CASE.c1a,
    "s1a": PF_REFERENCE_CASE.s1a,
    "s2a": PF_REFERENCE_CASE.s2a,
}
PF_REFERENCE_IP = PF_REFERENCE_CASE.Ip
PF_REFERENCE_PROFILE_COEFF_COUNTS = {
    name: (0 if coeffs is None else len(coeffs)) for name, coeffs in PF_REFERENCE_COEFFS.items()
}


def _solve_with_config(solver: Solver, *, x0: np.ndarray | None = None, config: SolverConfig):
    solver.solve(
        x0=x0,
        method=config.method,
        rtol=config.rtol,
        atol=config.atol,
        root_maxiter=config.root_maxiter,
        root_maxfev=config.root_maxfev,
        enable_warmstart=config.enable_warmstart,
        enable_homotopy=config.enable_homotopy,
        enable_verbose=config.enable_verbose,
        enable_history=config.enable_history,
    )
    return solver.result


@dataclass(frozen=True)
class BenchmarkCaseSpec:
    mode: str
    derivative: str
    constraint: str

    @property
    def case_name(self) -> str:
        return f"{self.mode}-{self.derivative}-{self.constraint}"


@dataclass(frozen=True)
class PFReferenceBundle:
    solver: Solver
    result: object
    equilibrium: Equilibrium
    equilibrium_on_grid: Equilibrium
    ref_profiles: dict[str, np.ndarray | float]
    reference_shape_x: np.ndarray


@dataclass(frozen=True)
class BenchmarkCaseResult:
    spec: BenchmarkCaseSpec
    result: object
    equilibrium: Equilibrium
    avg_ms: float
    std_ms: float
    rel_max: float
    rel_var: float
    extra: dict[str, float]
    shape_error: float
    notes: tuple[str, ...]

    @property
    def case_name(self) -> str:
        return self.spec.case_name

    @property
    def success(self) -> bool:
        return bool(self.result.success)

    @property
    def note_text(self) -> str:
        return ", ".join(self.notes) if self.notes else "ok"


def _artifact_root() -> Path:
    root = Path(__file__).resolve().parent.parent / "tests" / "benchmark"
    root.mkdir(parents=True, exist_ok=True)
    return root


def _artifact_dir() -> Path:
    mode = "warm" if WARMSTART else "cold"
    outdir = _artifact_root() / f"{mode}-{BACKEND}"
    outdir.mkdir(parents=True, exist_ok=True)
    return outdir


def _plot_dir() -> Path:
    outdir = _artifact_dir() / "plots"
    outdir.mkdir(parents=True, exist_ok=True)
    return outdir


def make_zero_reference_coeffs() -> dict[str, list[float] | None]:
    coeffs: dict[str, list[float] | None] = {}
    for name, count in PF_REFERENCE_PROFILE_COEFF_COUNTS.items():
        coeffs[name] = None if count == 0 else [0.0] * count
    return coeffs


def make_pf_reference_solver(
    grid: Grid | None = None,
    *,
    config: SolverConfig | None = None,
) -> Solver:
    grid = grid or PF_REFERENCE_GRID
    current_input, heat_input = pf_reference_profiles(grid.rho)
    case = OperatorCase(
        coeffs_by_name=make_zero_reference_coeffs(),
        heat_input=heat_input,
        current_input=current_input,
        Ip=PF_REFERENCE_IP,
        **PF_REFERENCE_CASE_KWARGS,
    )
    operator = Operator(grid=grid, case=case, name="PF", derivative="rho")
    return Solver(operator=operator, config=config or CONFIG)


def solve_pf_reference(
    grid: Grid | None = None,
    *,
    config: SolverConfig | None = None,
) -> tuple[Solver, object, Equilibrium]:
    solve_config = CONFIG if config is None else config
    solver = make_pf_reference_solver(grid, config=solve_config)
    _solve_with_config(solver, config=solve_config)
    return solver, solver.result, solver.build_equilibrium()


def build_pf_reference_profiles(equilibrium: Equilibrium) -> dict[str, np.ndarray | float]:
    psin_r = np.asarray(equilibrium.psin_r, dtype=np.float64).copy()
    psin_r_safe = np.where(np.abs(psin_r) > 1e-14, psin_r, 1e-14)

    psi_r = np.asarray(equilibrium.alpha2 * psin_r, dtype=np.float64)
    psi_r_safe = np.where(np.abs(psi_r) > 1e-14, psi_r, 1e-14)

    FFn_r = np.asarray(equilibrium.FFn_r, dtype=np.float64).copy()
    Pn_r = np.asarray(equilibrium.Pn_r, dtype=np.float64).copy()
    FF_r = np.asarray(equilibrium.FF_r, dtype=np.float64).copy()
    P_r = np.asarray(equilibrium.P_r, dtype=np.float64).copy()
    Itor = np.asarray(equilibrium.Itor, dtype=np.float64).copy()
    jtor = np.asarray(equilibrium.jtor, dtype=np.float64).copy()
    jpara = np.asarray(equilibrium.jpara, dtype=np.float64).copy()
    q = np.asarray(equilibrium.q, dtype=np.float64).copy()
    mu0 = 4e-7 * np.pi

    return {
        "psin_r": psin_r,
        "psi_r": psi_r,
        "FFn_r": FFn_r,
        "Pn_r": Pn_r,
        "FFn_psi": FFn_r / psin_r_safe,
        "Pn_psi": Pn_r / psin_r_safe,
        "FF_r": FF_r,
        "P_r": P_r,
        "FF_psi": FF_r / psi_r_safe,
        "P_psi": P_r / psi_r_safe,
        "Itorn": Itor * mu0,
        "Itor": Itor,
        "jtorn": jtor * mu0,
        "jtor": jtor,
        "jparan": jpara.copy(),
        "jpara": jpara,
        "qn": q * 0.1,
        "q": q,
        "beta_constraint": float(equilibrium.beta_t),
    }


def summarize_pf_reference_run(result, equilibrium: Equilibrium) -> list[tuple[str, str]]:
    return [
        ("success", str(bool(result.success))),
        ("nfev", str(int(result.nfev))),
        ("elapsed", f"{float(result.elapsed) / 1000.0:.3f} ms"),
        ("residual_initial", f"{float(result.residual_norm_initial):.6e}"),
        ("residual_final", f"{float(result.residual_norm_final):.6e}"),
        ("Ip", f"{float(equilibrium.Ip):.6e}"),
        ("beta_t", f"{float(equilibrium.beta_t):.6e}"),
        ("alpha1", f"{float(equilibrium.alpha1):.6e}"),
        ("alpha2", f"{float(equilibrium.alpha2):.6e}"),
    ]


def render_key_values(pairs: Sequence[tuple[str, str]], *, indent: int = 0) -> list[str]:
    if not pairs:
        return []
    key_width = max(len(key) for key, _ in pairs)
    prefix = " " * indent
    return [f"{prefix}{key:<{key_width}} : {value}" for key, value in pairs]


def render_table(
    headers: Sequence[str],
    rows: Sequence[Sequence[str]],
    *,
    aligns: Sequence[str] | None = None,
) -> list[str]:
    if not headers:
        return []

    align_tokens = list(aligns or ["left"] * len(headers))
    if len(align_tokens) != len(headers):
        raise ValueError("aligns must match the number of headers")

    normalized_rows = [[str(cell) for cell in row] for row in rows]
    widths = [len(str(header)) for header in headers]

    for row in normalized_rows:
        if len(row) != len(headers):
            raise ValueError("row width must match header width")
        for index, cell in enumerate(row):
            widths[index] = max(widths[index], len(cell))

    def _format(cell: str, width: int, align: str) -> str:
        if align == "right":
            return f"{cell:>{width}}"
        return f"{cell:<{width}}"

    header_line = " | ".join(
        _format(str(header), widths[index], align_tokens[index]) for index, header in enumerate(headers)
    )
    divider = "-+-".join("-" * width for width in widths)

    lines = [header_line, divider]
    for row in normalized_rows:
        lines.append(" | ".join(_format(cell, widths[index], align_tokens[index]) for index, cell in enumerate(row)))
    return lines


def render_ranked_mapping(
    title: str,
    mapping: Mapping[str, float],
    *,
    indent: int = 2,
    limit: int | None = None,
    value_fmt: str = ".3f",
    suffix: str = "",
) -> list[str]:
    items = sorted(mapping.items(), key=lambda item: item[1], reverse=True)
    if limit is not None:
        items = items[:limit]

    lines = [title]
    prefix = " " * indent
    if not items:
        lines.append(f"{prefix}(none)")
        return lines

    key_width = max(len(name) for name, _ in items)
    for name, value in items:
        lines.append(f"{prefix}{name:<{key_width}} : {value:{value_fmt}}{suffix}")
    return lines


def format_ms(mean_ms: float, std_ms: float | None = None) -> str:
    if std_ms is None:
        return f"{mean_ms:.3f} ms"
    return f"{mean_ms:.3f} +/- {std_ms:.3f} ms"


def format_share(share: float) -> str:
    return f"{100.0 * share:.1f}%"


def _extract_shape_x(coeffs_by_name: dict[str, list[float] | None], x: np.ndarray) -> np.ndarray:
    _, coeff_index, _ = build_profile_layout(coeffs_by_name)
    shape_values: list[float] = []
    shape_names = tuple(name for name in PF_REFERENCE_PROFILE_COEFF_COUNTS)
    for k in range(coeff_index.shape[1]):
        for name in shape_names:
            idx = int(coeff_index[PROFILE_INDEX[name], k])
            if idx >= 0:
                shape_values.append(float(x[idx]))
    return np.asarray(shape_values, dtype=np.float64)


def _pf_reference_profiles() -> PFReferenceBundle:
    solver, result, equilibrium = solve_pf_reference(PF_REFERENCE_GRID, config=CONFIG)
    equilibrium_on_grid = resample_equilibrium(equilibrium, target_grid=GRID)
    reference_shape_x = _extract_shape_x(
        solver.operator.case.coeffs_by_name,
        np.asarray(result.x, dtype=np.float64),
    )
    return PFReferenceBundle(
        solver=solver,
        result=result,
        equilibrium=equilibrium,
        equilibrium_on_grid=equilibrium_on_grid,
        ref_profiles=build_pf_reference_profiles(equilibrium_on_grid),
        reference_shape_x=reference_shape_x,
    )


def _iter_benchmark_specs():
    for mode in BENCHMARK_MODES:
        for derivative in ("rho", "psi"):
            for constraint in BENCHMARK_MODE_CONSTRAINTS[mode]:
                yield BenchmarkCaseSpec(mode=mode, derivative=derivative, constraint=constraint)


def _constraint_route_domains(constraint: str) -> tuple[str, str]:
    if constraint == "Ip_beta":
        return "normalized", "normalized"
    if constraint == "Ip":
        return "normalized", "physical"
    if constraint == "beta":
        return "physical", "normalized"
    return "physical", "physical"


def _pressure_keys_for_derivative(derivative: str) -> tuple[str, str]:
    if derivative == "rho":
        return "Pn_r", "P_r"
    return "Pn_psi", "P_psi"


def _pick_ref_profile(
    ref: dict[str, np.ndarray | float],
    normalized_key: str,
    physical_key: str,
    normalized: bool,
) -> np.ndarray:
    key = normalized_key if normalized else physical_key
    return np.asarray(ref[key], dtype=np.float64)


def _split_benchmark_inputs(init_kwargs: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    heat_key = next(name for name in init_kwargs if name.startswith("P"))
    current_key = next(name for name in init_kwargs if not name.startswith("P"))
    return np.asarray(init_kwargs[heat_key], dtype=np.float64), np.asarray(init_kwargs[current_key], dtype=np.float64)


def _build_mode_init_kwargs(
    mode: str,
    derivative: str,
    constraint: str,
    ref: dict[str, np.ndarray | float],
) -> dict[str, np.ndarray]:
    pressure_keys = _pressure_keys_for_derivative(derivative)

    if mode == "PF":
        use_normalized = constraint in {"Ip", "beta"}
        driver_keys = ("FFn_r", "FF_r") if derivative == "rho" else ("FFn_psi", "FF_psi")
        return {
            driver_keys[0] if use_normalized else driver_keys[1]: _pick_ref_profile(
                ref, driver_keys[0], driver_keys[1], use_normalized
            ),
            pressure_keys[0] if use_normalized else pressure_keys[1]: _pick_ref_profile(
                ref, pressure_keys[0], pressure_keys[1], use_normalized
            ),
        }

    if mode == "PP":
        driver_normalized = constraint in {"Ip_beta", "Ip"}
        pressure_normalized = constraint in {"Ip_beta", "beta"}
        return {
            "psin_r" if driver_normalized else "psi_r": _pick_ref_profile(ref, "psin_r", "psi_r", driver_normalized),
            pressure_keys[0] if pressure_normalized else pressure_keys[1]: _pick_ref_profile(
                ref, pressure_keys[0], pressure_keys[1], pressure_normalized
            ),
        }

    driver_domain, pressure_domain = _constraint_route_domains(constraint)
    driver_keys = {
        "PI": ("Itorn", "Itor"),
        "PJ1": ("jtorn", "jtor"),
        "PJ2": ("jparan", "jpara"),
        "PQ": ("qn", "q"),
    }[mode]
    driver_normalized = driver_domain == "normalized"
    pressure_normalized = pressure_domain == "normalized"
    return {
        driver_keys[0] if driver_normalized else driver_keys[1]: _pick_ref_profile(
            ref, driver_keys[0], driver_keys[1], driver_normalized
        ),
        pressure_keys[0] if pressure_normalized else pressure_keys[1]: _pick_ref_profile(
            ref, pressure_keys[0], pressure_keys[1], pressure_normalized
        ),
    }


def _make_benchmark_solver_case(
    mode: str,
    derivative: str,
    constraint: str,
    ref: dict[str, np.ndarray | float],
) -> OperatorCase:
    init_kwargs = _build_mode_init_kwargs(mode, derivative, constraint, ref)
    heat_input, current_input = _split_benchmark_inputs(init_kwargs)
    coeffs = make_zero_reference_coeffs()
    if mode in {"PJ2", "PQ"}:
        coeffs["F"] = [0.0] * F_ROBUST_COEFF_COUNT

    Ip = PF_REFERENCE_IP if constraint in {"Ip", "Ip_beta"} else None
    beta = float(ref["beta_constraint"]) if constraint in {"beta", "Ip_beta"} else None

    return OperatorCase(
        coeffs_by_name=coeffs,
        heat_input=heat_input,
        current_input=current_input,
        Ip=Ip,
        beta=beta,
        **PF_REFERENCE_CASE_KWARGS,
    )


def _rel_stats_vs_ref(reference: Equilibrium, other: Equilibrium) -> tuple[float, float, dict[str, float]]:
    rho_ref = np.asarray(reference.rho, dtype=np.float64)
    rho_cur = np.asarray(other.rho, dtype=np.float64)

    def _align(y_ref: np.ndarray, y_cur: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if rho_ref.shape == rho_cur.shape and np.allclose(rho_ref, rho_cur):
            return y_ref, y_cur
        return np.interp(rho_cur, rho_ref, y_ref), y_cur

    def _profiles(eq: Equilibrium):
        return {
            "psin_r": np.asarray(eq.psin_r, dtype=np.float64),
            "FF_r": np.asarray(eq.FF_r, dtype=np.float64),
            "MU0P_r": np.asarray(4e-7 * np.pi * eq.P_r, dtype=np.float64),
        }

    ref_profiles = _profiles(reference)
    cur_profiles = _profiles(other)
    rel_all = []
    for key in ("psin_r", "FF_r", "MU0P_r"):
        ref_vec, cur_vec = _align(ref_profiles[key], cur_profiles[key])
        scale = max(float(np.max(np.abs(ref_vec))), 1e-14)
        rel_all.append((cur_vec - ref_vec) / scale)

    rel_vec = np.concatenate(rel_all)
    extra_max: dict[str, float] = {}
    for key in ("q", "Itor", "jtor", "jpara"):
        ref_vec, cur_vec = _align(
            np.asarray(getattr(reference, key), dtype=np.float64),
            np.asarray(getattr(other, key), dtype=np.float64),
        )
        scale = max(float(np.max(np.abs(ref_vec))), 1e-14)
        extra_max[key] = float(np.max(np.abs(cur_vec - ref_vec)) / scale)

    return float(np.max(np.abs(rel_vec))), float(np.var(rel_vec)), extra_max


def _shape_error(reference_x: np.ndarray, current_x: np.ndarray) -> float:
    n = min(reference_x.shape[0], current_x.shape[0])
    return float(
        np.max(np.abs(np.asarray(current_x[:n], dtype=np.float64) - np.asarray(reference_x[:n], dtype=np.float64)))
    )


def _collect_case_notes(
    result,
    equilibrium: Equilibrium,
    rel_max: float,
    rel_var: float,
    extra: dict[str, float],
    *,
    shape_error: float,
) -> tuple[str, ...]:
    notes: list[str] = []
    if not bool(result.success):
        notes.append("solve_did_not_converge")
    if float(result.residual_norm_final) > 1e-2:
        notes.append(f"high_residual={float(result.residual_norm_final):.3e}")
    if shape_error >= MAX_SHAPE_ERROR:
        notes.append(f"shape_error={shape_error:.3e}")

    arrays = (
        equilibrium.psin_r,
        equilibrium.psin_rr,
        equilibrium.FFn_r,
        equilibrium.Pn_r,
        equilibrium.q,
        equilibrium.Itor,
        equilibrium.jtor,
        equilibrium.jpara,
    )
    if not all(np.all(np.isfinite(arr)) for arr in arrays):
        notes.append("non_finite_equilibrium_diagnostics")
    if not np.isfinite(rel_max) or not np.isfinite(rel_var):
        notes.append("non_finite_benchmark_metrics")
    if not all(np.isfinite(value) for value in extra.values()):
        notes.append("non_finite_benchmark_extras")
    return tuple(notes)


def _case_seed(spec: BenchmarkCaseSpec) -> int:
    return WARM_START_SEED + sum((index + 1) * ord(ch) for index, ch in enumerate(spec.case_name))


def _build_warm_start_guess(rng: np.random.Generator, true_x: np.ndarray) -> np.ndarray:
    scale = rng.uniform(WARM_START_SCALE_MIN, WARM_START_SCALE_MAX, size=true_x.shape)
    return np.asarray(true_x, dtype=np.float64) * scale


def _benchmark_case_result(
    spec: BenchmarkCaseSpec,
    bundle: PFReferenceBundle,
) -> BenchmarkCaseResult:
    case = _make_benchmark_solver_case(spec.mode, spec.derivative, spec.constraint, bundle.ref_profiles)
    operator = Operator(grid=GRID, case=case, name=spec.mode, derivative=spec.derivative)
    solver = Solver(operator=operator, config=CONFIG)
    elapsed_ms_samples: list[float] = []
    result = None
    initial_guess = None
    warm_start_rng = None
    warm_start_true_x = None

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="Corrected spectral integration failed; falling back to full integration",
        )
        warnings.filterwarnings("ignore", category=RuntimeWarning)

        if WARMSTART:
            true_result = _solve_with_config(solver, config=BENCHMARK_SOLVE_CONFIG)
            warm_start_true_x = np.asarray(true_result.x, dtype=np.float64)
            initial_guess = warm_start_true_x
            warm_start_rng = np.random.default_rng(_case_seed(spec))

        for _ in range(BENCHMARK_REPEAT_COUNT):
            if warm_start_rng is not None:
                initial_guess = _build_warm_start_guess(warm_start_rng, warm_start_true_x)
            result = _solve_with_config(solver, x0=initial_guess, config=BENCHMARK_SOLVE_CONFIG)
            elapsed_ms_samples.append(float(result.elapsed) / 1000.0)

    if result is None:
        raise RuntimeError("benchmark solve produced no result")

    equilibrium = solver.build_equilibrium()
    rel_max, rel_var, extra = _rel_stats_vs_ref(bundle.equilibrium_on_grid, equilibrium)
    current_shape_x = _extract_shape_x(case.coeffs_by_name, result.x)
    shape_error = _shape_error(bundle.reference_shape_x, current_shape_x)
    notes = _collect_case_notes(result, equilibrium, rel_max, rel_var, extra, shape_error=shape_error)

    return BenchmarkCaseResult(
        spec=spec,
        result=result,
        equilibrium=equilibrium,
        avg_ms=float(np.mean(elapsed_ms_samples)),
        std_ms=float(np.std(elapsed_ms_samples)),
        rel_max=float(rel_max),
        rel_var=float(rel_var),
        extra={key: float(value) for key, value in extra.items()},
        shape_error=shape_error,
        notes=notes,
    )


def _summary_pairs(rows: list[BenchmarkCaseResult]) -> list[tuple[str, str]]:
    failures = [row.case_name for row in rows if not row.success or row.shape_error >= MAX_SHAPE_ERROR]
    slowest = max(rows, key=lambda row: row.avg_ms)
    largest_shape = max(rows, key=lambda row: row.shape_error)
    largest_rel = max(rows, key=lambda row: row.rel_max)
    startup_name = "Warm-start" if WARMSTART else "Cold-start"
    pairs = [
        ("backend", BACKEND),
        ("startup", startup_name),
        ("reference_case", "PF_ref@32x32"),
        ("benchmark_grid", f"{GRID.Nr}x{GRID.Nt} ({GRID.scheme})"),
        ("repeat_count", str(BENCHMARK_REPEAT_COUNT)),
        ("case_count", str(len(rows))),
        ("hard_failures", str(len(failures))),
        ("slowest_case", f"{slowest.case_name} ({slowest.avg_ms:.3f} ms)"),
        ("largest_shape", f"{largest_shape.case_name} ({largest_shape.shape_error:.3e})"),
        ("largest_rel_max", f"{largest_rel.case_name} ({largest_rel.rel_max:.3e})"),
    ]
    if WARMSTART:
        pairs.append(("warm_start_seed", str(WARM_START_SEED)))
        pairs.append(("warm_start_scale", f"[{WARM_START_SCALE_MIN:.2f}, {WARM_START_SCALE_MAX:.2f}]"))
    return pairs


def _solve_summary_rows(rows: list[BenchmarkCaseResult]) -> list[list[str]]:
    table_rows: list[list[str]] = []
    for row in rows:
        table_rows.append(
            [
                row.case_name,
                "ok" if row.success else "fail",
                str(int(row.result.nfev)),
                format_ms(row.avg_ms, row.std_ms),
                f"{float(row.result.residual_norm_final):.3e}",
                f"{row.shape_error:.3e}",
                row.note_text,
            ]
        )
    return table_rows


def _delta_summary_rows(rows: list[BenchmarkCaseResult]) -> list[list[str]]:
    table_rows: list[list[str]] = []
    for row in rows:
        table_rows.append(
            [
                row.case_name,
                f"{row.rel_max:.3e}",
                f"{row.rel_var:.3e}",
                f"{float(row.extra['q']):.3e}",
                f"{float(row.extra['Itor']):.3e}",
                f"{float(row.extra['jtor']):.3e}",
                f"{float(row.extra['jpara']):.3e}",
            ]
        )
    return table_rows


def _write_reference_artifacts(bundle: PFReferenceBundle) -> None:
    artifact_dir = _artifact_dir()
    bundle.equilibrium.plot(outpath=artifact_dir / "pf_reference_summary.png")
    lines = ["PF reference summary", ""]
    lines.extend(render_key_values(summarize_pf_reference_run(bundle.result, bundle.equilibrium)))
    (artifact_dir / "pf_reference_summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_benchmark_report(rows: list[BenchmarkCaseResult]) -> None:
    artifact_dir = _artifact_dir()
    startup_name = "Warm-start" if WARMSTART else "Cold-start"
    lines = [f"{startup_name} 46-case benchmark vs PF reference", ""]
    lines.extend(render_key_values(_summary_pairs(rows)))
    lines.extend(["", "Solve summary"])
    lines.extend(
        render_table(
            ["case", "ok", "nfev", "avg_ms", "resid", "shape", "notes"],
            _solve_summary_rows(rows),
            aligns=("left", "left", "right", "right", "right", "right", "left"),
        )
    )
    lines.extend(["", "Physics deltas"])
    lines.extend(
        render_table(
            ["case", "rel_max", "rel_var", "q", "Itor", "jtor", "jpara"],
            _delta_summary_rows(rows),
            aligns=("left", "right", "right", "right", "right", "right", "right"),
        )
    )
    lines.extend([""])
    lines.extend(
        render_ranked_mapping(
            "Slowest cases",
            {row.case_name: row.avg_ms for row in rows},
            value_fmt=".3f",
            suffix=" ms",
        )
    )
    lines.extend([""])
    lines.extend(
        render_ranked_mapping(
            "Largest shape errors",
            {row.case_name: row.shape_error for row in rows},
            value_fmt=".3e",
        )
    )
    lines.extend([""])
    lines.extend(
        render_ranked_mapping(
            "Largest rel_max deltas",
            {row.case_name: row.rel_max for row in rows},
            value_fmt=".3e",
        )
    )
    (artifact_dir / "benchmark_compare.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_benchmark_notes(rows: list[BenchmarkCaseResult]) -> None:
    artifact_dir = _artifact_dir()
    startup_name = "Warm-start" if WARMSTART else "Cold-start"
    lines = [f"{startup_name} 46-case benchmark notes", ""]
    for row in rows:
        lines.append(f"[{row.case_name}]")
        lines.extend(
            render_key_values(
                [
                    ("status", "ok" if row.success else "failed"),
                    ("notes", row.note_text),
                    ("avg_ms", format_ms(row.avg_ms, row.std_ms)),
                    ("residual_final", f"{float(row.result.residual_norm_final):.3e}"),
                    ("shape_error", f"{row.shape_error:.3e}"),
                    ("rel_max", f"{row.rel_max:.3e}"),
                    ("rel_var", f"{row.rel_var:.3e}"),
                ],
                indent=2,
            )
        )
        lines.append("")
    (artifact_dir / "benchmark_notes.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _assert_benchmark_expectations(rows: list[BenchmarkCaseResult]) -> None:
    failures: list[str] = []
    for row in rows:
        if not row.success:
            failures.append(f"{row.case_name}: solve did not converge")
        if row.shape_error >= MAX_SHAPE_ERROR:
            failures.append(f"{row.case_name}: shape_error={row.shape_error:.3e} >= {MAX_SHAPE_ERROR:.1e}")
    if failures:
        raise AssertionError("\n".join(failures))


def run_full_benchmark(*, show_progress: bool = SHOW_PROGRESS) -> list[BenchmarkCaseResult]:
    bundle = _pf_reference_profiles()
    _write_reference_artifacts(bundle)

    plot_dir = _plot_dir()
    rows: list[BenchmarkCaseResult] = []
    specs = list(_iter_benchmark_specs())
    total_cases = len(specs)

    for index, spec in enumerate(specs, start=1):
        row = _benchmark_case_result(spec, bundle)
        rows.append(row)
        if show_progress:
            startup_key = "warm" if WARMSTART else "cold"
            print(
                f"[{BACKEND}] [{startup_key}] [{index}/{total_cases}] {row.case_name}: "
                f"{row.avg_ms:.3f} +/- {row.std_ms:.3f} ms | "
                f"rel_max={row.rel_max:.3e} | shape={row.shape_error:.3e}"
            )
        if PLOT:
            bundle.equilibrium.compare(
                row.equilibrium,
                outpath=plot_dir / f"{row.case_name}.png",
                label_ref="PF_ref",
                label_other=row.case_name,
            )
            bundle.equilibrium.plot(
                outpath=plot_dir / f"{row.case_name}_summary.png",
            )

    _write_benchmark_report(rows)
    _write_benchmark_notes(rows)
    if ASSERT_EXPECTATIONS:
        _assert_benchmark_expectations(rows)
    return rows


def _run_as_script() -> int:
    run_full_benchmark(show_progress=SHOW_PROGRESS)
    return 0


if __name__ == "__main__":
    raise SystemExit(_run_as_script())

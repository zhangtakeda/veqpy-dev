from __future__ import annotations

import json
import os
import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.interpolate import PchipInterpolator, interp1d

from veqpy.engine import validate_route
from veqpy.model import Boundary, Grid
from veqpy.operator import Operator, OperatorCase, build_profile_index, build_profile_layout, build_profile_names
from veqpy.operator.layout import build_shape_profile_names
from veqpy.solver import Solver, SolverConfig

PLOT = True
SHOW_PROGRESS = True
BACKEND = os.environ.get("VEQPY_BACKEND", "numba")
os.environ["VEQPY_BACKEND"] = BACKEND

REFERENCE_SOURCE_SAMPLE_COUNT = 51
TEST_SOURCE_SAMPLE_COUNT = 51
BENCHMARK_REPEAT_COUNT = 100
SHAPE_MATCH_TOL = 1e-2
REFERENCE_CACHE_VERSION = 1
DIAGNOSTIC_SIGN_CHANGE_WINDOW = 12

REFERENCE_GRID = Grid(Nr=64, Nt=32, scheme="legendre")
TEST_GRID = Grid(Nr=16, Nt=16, scheme="legendre", L_max=REFERENCE_GRID.L_max)
REFERENCE_SUMMARY_GRID = Grid(Nr=64, Nt=128, scheme="uniform", L_max=REFERENCE_GRID.L_max, M_max=REFERENCE_GRID.M_max)
CONFIG = SolverConfig(
    method="hybr",
    enable_verbose=False,
    enable_warmstart=False,
    enable_history=False,
)

BASE_COEFFS = {
    "h": [0.0] * 3,
    "k": [0.0] * 5,
    "s1": [0.0] * 3,
}

PSIN_ROBUST_COEFFS = {
    **BASE_COEFFS,
    "psin": [0.0] * 5,
}

F_ROBUST_COEFFS = {
    **BASE_COEFFS,
    "F": [0.0] * 5,
}

BOUNDARY = Boundary(
    a=1.05 / 1.85,
    R0=1.05,
    Z0=0.0,
    B0=3.0,
    ka=2.2,
    s_offsets=np.array([0.0, float(np.arcsin(0.5))]),
)

REFERENCE_IP = 3.0e6
SHAPE_PROFILE_NAMES = build_shape_profile_names(REFERENCE_GRID.M_max)
BENCHMARK_MODES = ("PF", "PP", "PI", "PJ1", "PJ2", "PQ")
BENCHMARK_INPUT_KINDS = ("uniform", "grid")
BENCHMARK_MODE_CONSTRAINTS = {
    "PF": ("null", "Ip", "beta"),
    "PP": ("Ip_beta", "Ip", "beta", "null"),
    "PI": ("Ip_beta", "Ip", "beta", "null"),
    "PJ1": ("Ip_beta", "Ip", "beta", "null"),
    "PJ2": ("Ip_beta", "Ip", "beta", "null"),
    "PQ": ("Ip_beta", "Ip", "beta", "null"),
}


@dataclass(frozen=True)
class PreparedInterpAxis:
    unique_axis: np.ndarray
    order: np.ndarray
    unique_index: np.ndarray


@dataclass(frozen=True)
class ReferenceBundle:
    result: object
    equilibrium: object
    ref_profiles: dict[str, np.ndarray | float]
    reference_shape_x: np.ndarray
    rho_axis: np.ndarray
    psin_axis: np.ndarray
    rho_interp_axis: PreparedInterpAxis
    psin_interp_axis: PreparedInterpAxis


@dataclass(frozen=True)
class BenchmarkCaseSpec:
    mode: str
    coordinate: str
    constraint: str
    input_kind: str

    @property
    def case_name(self) -> str:
        return f"{self.mode}_{self.coordinate}_{self.input_kind}_{self.constraint}"


@dataclass(frozen=True)
class BenchmarkCaseResult:
    spec: BenchmarkCaseSpec
    result: object
    equilibrium: object
    avg_ms: float
    std_ms: float
    shape_error: float
    psi_r_rel_rms_error: float
    psi_r_rel_max_error: float
    psi_r_head_sign_changes: int
    psi_r_tail_sign_changes: int
    ff_psi_rel_rms_error: float
    ff_psi_rel_max_error: float
    ff_psi_head_sign_changes: int
    ff_psi_tail_sign_changes: int
    mu0_p_psi_rel_rms_error: float
    mu0_p_psi_rel_max_error: float
    mu0_p_psi_head_sign_changes: int
    mu0_p_psi_tail_sign_changes: int

    @property
    def case_name(self) -> str:
        return self.spec.case_name


_UNIFORM_SOURCE_AXIS = np.linspace(0.0, 1.0, TEST_SOURCE_SAMPLE_COUNT, dtype=np.float64)
_UNIFORM_SOURCE_AXIS_SQRT_PSIN = _UNIFORM_SOURCE_AXIS**2
_TEST_GRID_RHO_AXIS = np.asarray(TEST_GRID.rho, dtype=np.float64)
_REFERENCE_SUMMARY_RHO_AXIS = np.asarray(REFERENCE_SUMMARY_GRID.rho, dtype=np.float64)


def _sort_rows_desc(rows: list[BenchmarkCaseResult], key_fn) -> list[BenchmarkCaseResult]:
    return sorted(rows, key=lambda row: (-float(key_fn(row)), row.case_name))


def _render_ranking_section(
    title: str,
    rows: list[BenchmarkCaseResult],
    *,
    columns,
) -> list[str]:
    lines = ["", title, ""]
    header_parts = []
    for align, label, width, _ in columns:
        if align == "left":
            header_parts.append(label.ljust(width))
        else:
            header_parts.append(label.rjust(width))
    header = " | ".join(header_parts)
    lines.append(header)
    lines.append("-" * len(header))
    for index, row in enumerate(rows, start=1):
        value_parts = []
        for align, _, width, formatter in columns:
            value = str(formatter(index, row))
            if align == "left":
                value_parts.append(value.ljust(width))
            else:
                value_parts.append(value.rjust(width))
        lines.append(" | ".join(value_parts))
    return lines


def _artifact_dir() -> Path:
    outdir = Path(__file__).resolve().parent / f"benchmark-{BACKEND}"
    outdir.mkdir(parents=True, exist_ok=True)
    return outdir


def _plot_dir() -> Path:
    outdir = _artifact_dir() / "plots"
    outdir.mkdir(parents=True, exist_ok=True)
    return outdir


def _reference_summary_json_path() -> Path:
    return _artifact_dir() / "reference_summary.json"


def _reference_cache_path() -> Path:
    return _artifact_dir() / "reference_bundle.pkl"


def _render_pairs(pairs: list[tuple[str, str]]) -> list[str]:
    if not pairs:
        return []
    key_width = max(len(key) for key, _ in pairs)
    return [f"{key:<{key_width}} : {value}" for key, value in pairs]


def _as_float64_array(values, *, copy: bool = False) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    if copy:
        return arr.copy()
    return arr


def _extract_shape_x(profile_coeffs: dict[str, list[float] | None], x: np.ndarray) -> np.ndarray:
    profile_names = build_profile_names(REFERENCE_GRID.M_max)
    profile_index = build_profile_index(profile_names)
    _, coeff_index, _ = build_profile_layout(profile_coeffs, profile_names=profile_names)
    shape_values: list[float] = []
    for k in range(coeff_index.shape[1]):
        for name in SHAPE_PROFILE_NAMES:
            idx = int(coeff_index[profile_index[name], k])
            if idx >= 0:
                shape_values.append(float(x[idx]))
    return np.asarray(shape_values, dtype=np.float64)


def _prepare_interp_axis(axis: np.ndarray) -> PreparedInterpAxis:
    axis_f64 = _as_float64_array(axis)
    order = np.argsort(axis_f64)
    axis_sorted = axis_f64[order]
    unique_axis, unique_index = np.unique(axis_sorted, return_index=True)
    return PreparedInterpAxis(unique_axis=unique_axis, order=order, unique_index=unique_index)


def _prepare_interp_values(values: np.ndarray, prepared_axis: PreparedInterpAxis) -> np.ndarray:
    values_f64 = _as_float64_array(values)
    return values_f64[prepared_axis.order][prepared_axis.unique_index]


def _unique_interp(
    axis: np.ndarray | PreparedInterpAxis,
    values: np.ndarray,
    x_new: np.ndarray,
    *,
    kind: str = "cubic",
) -> np.ndarray:
    prepared_axis = axis if isinstance(axis, PreparedInterpAxis) else _prepare_interp_axis(axis)
    unique_axis = prepared_axis.unique_axis
    unique_values = _prepare_interp_values(values, prepared_axis)
    interp_kind = kind if unique_axis.size >= 4 else "linear"
    fn = interp1d(unique_axis, unique_values, kind=interp_kind, fill_value="extrapolate", assume_sorted=True)
    return _as_float64_array(fn(_as_float64_array(x_new)))


def _profile_interp(axis: np.ndarray | PreparedInterpAxis, values: np.ndarray, x_new: np.ndarray) -> np.ndarray:
    prepared_axis = axis if isinstance(axis, PreparedInterpAxis) else _prepare_interp_axis(axis)
    unique_axis = prepared_axis.unique_axis
    unique_values = _prepare_interp_values(values, prepared_axis)
    x_new = _as_float64_array(x_new)
    if unique_axis.size < 2:
        return np.full_like(x_new, float(unique_values[0] if unique_values.size else 0.0), dtype=np.float64)
    if unique_axis.size < 3:
        return np.interp(x_new, unique_axis, unique_values).astype(np.float64, copy=False)
    return _as_float64_array(PchipInterpolator(unique_axis, unique_values, extrapolate=True)(x_new))


def pf_reference_profiles(psin: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    beta0 = 0.75

    alpha_p, alpha_f = 5.0, 3.32
    exp_ap, exp_af = np.exp(alpha_p), np.exp(alpha_f)
    den_p, den_f = 1.0 + exp_ap * (alpha_p - 1.0), 1.0 + exp_af * (alpha_f - 1.0)

    current_input = (1.0 - beta0) * alpha_f * (np.exp(alpha_f * psin) - exp_af) / den_f
    heat_input = beta0 * alpha_p * (np.exp(alpha_p * psin) - exp_ap) / den_p
    return current_input, heat_input


def build_pf_reference_profiles(equilibrium) -> dict[str, np.ndarray | float]:
    psin_r = _as_float64_array(equilibrium.psin_r, copy=True)
    psin_r_safe = np.where(np.abs(psin_r) > 1e-14, psin_r, 1e-14)

    psi_r = _as_float64_array(equilibrium.alpha2 * psin_r)
    psi_r_safe = np.where(np.abs(psi_r) > 1e-14, psi_r, 1e-14)

    FFn_r = _as_float64_array(equilibrium.FFn_r, copy=True)
    Pn_r = _as_float64_array(equilibrium.Pn_r, copy=True)
    FF_r = _as_float64_array(equilibrium.FF_r, copy=True)
    P_r = _as_float64_array(equilibrium.P_r, copy=True)
    Itor = _as_float64_array(equilibrium.Itor, copy=True)
    jtor = _as_float64_array(equilibrium.jtor, copy=True)
    jpara = _as_float64_array(equilibrium.jpara, copy=True)
    q = _as_float64_array(equilibrium.q, copy=True)
    mu0 = 4e-7 * np.pi

    return {
        "psin_r": psin_r,
        "psi_r": psi_r,
        "FFn_r": FFn_r,
        "Pn_r": Pn_r,
        "FFn_psin": FFn_r / psin_r_safe,
        "Pn_psin": Pn_r / psin_r_safe,
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


def _reference_pf_case() -> OperatorCase:
    rho_src = np.linspace(0.0, 1.0, REFERENCE_SOURCE_SAMPLE_COUNT)
    psin_src = rho_src * rho_src
    FFn_psin_src, Pn_psin_src = pf_reference_profiles(psin_src)
    FFn_r_src = FFn_psin_src * (2.0 * rho_src)
    Pn_r_src = Pn_psin_src * (2.0 * rho_src)
    return OperatorCase(
        route="PF",
        coordinate="rho",
        nodes="uniform",
        profile_coeffs=BASE_COEFFS,
        boundary=BOUNDARY,
        heat_input=Pn_r_src,
        current_input=FFn_r_src,
        Ip=REFERENCE_IP,
    )


def _reference_cache_signature() -> dict[str, object]:
    return {
        "version": REFERENCE_CACHE_VERSION,
        "backend": BACKEND,
        "reference_source_sample_count": int(REFERENCE_SOURCE_SAMPLE_COUNT),
        "reference_ip": float(REFERENCE_IP),
        "reference_grid": {
            "Nr": int(REFERENCE_GRID.Nr),
            "Nt": int(REFERENCE_GRID.Nt),
            "scheme": REFERENCE_GRID.scheme,
            "L_max": int(REFERENCE_GRID.L_max),
            "M_max": int(REFERENCE_GRID.M_max),
        },
        "boundary": {
            "a": float(BOUNDARY.a),
            "R0": float(BOUNDARY.R0),
            "Z0": float(BOUNDARY.Z0),
            "B0": float(BOUNDARY.B0),
            "ka": float(BOUNDARY.ka),
            "s_offsets": _as_float64_array(BOUNDARY.s_offsets).tolist(),
        },
        "config": {
            "method": CONFIG.method,
            "rtol": float(CONFIG.rtol),
            "atol": float(CONFIG.atol),
            "root_maxiter": int(CONFIG.root_maxiter),
            "root_maxfev": int(CONFIG.root_maxfev),
        },
    }


def _load_reference_cache() -> ReferenceBundle | None:
    path = _reference_cache_path()
    if not path.exists():
        return None
    try:
        with path.open("rb") as f:
            payload = pickle.load(f)
    except Exception:
        return None

    if not isinstance(payload, dict) or payload.get("signature") != _reference_cache_signature():
        return None

    bundle = payload.get("bundle")
    if not isinstance(bundle, dict):
        return None

    rho_axis = _as_float64_array(bundle["rho_axis"])
    psin_axis = _as_float64_array(bundle["psin_axis"])
    return ReferenceBundle(
        result=bundle["result"],
        equilibrium=bundle["equilibrium"],
        ref_profiles=bundle["ref_profiles"],
        reference_shape_x=_as_float64_array(bundle["reference_shape_x"]),
        rho_axis=rho_axis,
        psin_axis=psin_axis,
        rho_interp_axis=_prepare_interp_axis(rho_axis),
        psin_interp_axis=_prepare_interp_axis(psin_axis),
    )


def _write_reference_cache(reference: ReferenceBundle) -> None:
    path = _reference_cache_path()
    payload = {
        "signature": _reference_cache_signature(),
        "bundle": {
            "result": reference.result,
            "equilibrium": reference.equilibrium,
            "ref_profiles": reference.ref_profiles,
            "reference_shape_x": reference.reference_shape_x,
            "rho_axis": reference.rho_axis,
            "psin_axis": reference.psin_axis,
        },
    }
    tmp_path = path.with_suffix(f"{path.suffix}.tmp")
    with tmp_path.open("wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp_path, path)


def _solve_reference(*, show_progress: bool = False) -> ReferenceBundle:
    cached = _load_reference_cache()
    if cached is not None:
        if show_progress:
            print(f"[{BACKEND}] reference cache hit: {_reference_cache_path().name}")
        return cached

    solver = Solver(operator=Operator(REFERENCE_GRID, _reference_pf_case()), config=CONFIG)
    solver.solve(
        method=CONFIG.method,
        rtol=CONFIG.rtol,
        atol=CONFIG.atol,
        root_maxiter=CONFIG.root_maxiter,
        root_maxfev=CONFIG.root_maxfev,
        enable_verbose=False,
        enable_history=False,
        enable_warmstart=False,
    )
    result = solver.result
    equilibrium = solver.build_equilibrium()
    rho_axis = _as_float64_array(equilibrium.rho)
    psin_axis = _as_float64_array(equilibrium.psin)
    reference = ReferenceBundle(
        result=result,
        equilibrium=equilibrium,
        ref_profiles=build_pf_reference_profiles(equilibrium),
        reference_shape_x=_extract_shape_x(solver.operator.case.profile_coeffs, result.x),
        rho_axis=rho_axis,
        psin_axis=psin_axis,
        rho_interp_axis=_prepare_interp_axis(rho_axis),
        psin_interp_axis=_prepare_interp_axis(psin_axis),
    )
    _write_reference_cache(reference)
    if show_progress:
        print(f"[{BACKEND}] reference cache saved: {_reference_cache_path().name}")
    return reference


def _constraint_route_domains(constraint: str) -> tuple[str, str]:
    if constraint == "Ip_beta":
        return "normalized", "normalized"
    if constraint == "Ip":
        return "normalized", "physical"
    if constraint == "beta":
        return "physical", "normalized"
    return "physical", "physical"


def _pressure_keys_for_coordinate(coordinate: str) -> tuple[str, str]:
    if coordinate == "rho":
        return "Pn_r", "P_r"
    return "Pn_psin", "P_psi"


def _pick_ref_profile(
    ref: dict[str, np.ndarray | float],
    normalized_key: str,
    physical_key: str,
    normalized: bool,
) -> np.ndarray:
    key = normalized_key if normalized else physical_key
    return ref[key]


def _build_mode_init_kwargs(
    mode: str,
    coordinate: str,
    constraint: str,
    ref: dict[str, np.ndarray | float],
) -> dict[str, np.ndarray]:
    pressure_keys = _pressure_keys_for_coordinate(coordinate)

    if mode == "PF":
        use_normalized = constraint in {"Ip", "beta"}
        driver_keys = ("FFn_r", "FF_r") if coordinate == "rho" else ("FFn_psin", "FF_psi")
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


def _split_benchmark_inputs(init_kwargs: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    heat_key = next(name for name in init_kwargs if name.startswith("P"))
    current_key = next(name for name in init_kwargs if not name.startswith("P"))
    return init_kwargs[heat_key], init_kwargs[current_key]


def _uniform_source_axis(spec: BenchmarkCaseSpec) -> np.ndarray:
    route_spec = validate_route(spec.mode, spec.coordinate, spec.input_kind)
    if route_spec.source_parameterization == "sqrt_psin":
        return _UNIFORM_SOURCE_AXIS_SQRT_PSIN
    return _UNIFORM_SOURCE_AXIS


def _resample_reference_input(
    values: np.ndarray,
    source_axis: np.ndarray | PreparedInterpAxis,
    spec: BenchmarkCaseSpec,
) -> np.ndarray:
    uniform_axis = _uniform_source_axis(spec)
    return _profile_interp(source_axis, values, uniform_axis)


def _sample_reference_input_on_grid(
    values: np.ndarray,
    source_axis: np.ndarray | PreparedInterpAxis,
    grid_axis: np.ndarray,
) -> np.ndarray:
    return _profile_interp(source_axis, values, grid_axis)


def _profile_coeffs_for_case(
    mode: str,
    coordinate: str,
    input_kind: str,
    *,
    constraint: str | None = None,
) -> dict[str, list[float] | None]:
    route_spec = validate_route(mode, coordinate, input_kind)
    if route_spec.source_strategy == "profile_owned_psin":
        coeffs = {name: list(values) for name, values in PSIN_ROBUST_COEFFS.items()}
    else:
        coeffs = {name: list(values) for name, values in BASE_COEFFS.items()}
    if mode in {"PJ2", "PQ"}:
        f_order = 7 if mode == "PQ" and constraint in {"Ip", "Ip_beta"} else 6
        coeffs["F"] = [0.0] * f_order
    return coeffs


def _make_benchmark_case(spec: BenchmarkCaseSpec, reference: ReferenceBundle) -> OperatorCase:
    init_kwargs = _build_mode_init_kwargs(spec.mode, spec.coordinate, spec.constraint, reference.ref_profiles)
    heat_profile, current_profile = _split_benchmark_inputs(init_kwargs)
    if spec.input_kind == "grid":
        if spec.coordinate == "rho":
            grid_axis = _TEST_GRID_RHO_AXIS
            source_axis = reference.rho_interp_axis
        else:
            grid_axis = _profile_interp(reference.rho_interp_axis, reference.psin_axis, _TEST_GRID_RHO_AXIS)
            source_axis = reference.psin_interp_axis
        heat_input = _sample_reference_input_on_grid(heat_profile, source_axis, grid_axis)
        current_input = _sample_reference_input_on_grid(current_profile, source_axis, grid_axis)
        nodes = "grid"
    else:
        source_axis = reference.rho_interp_axis if spec.coordinate == "rho" else reference.psin_interp_axis
        heat_input = _resample_reference_input(heat_profile, source_axis, spec)
        current_input = _resample_reference_input(current_profile, source_axis, spec)
        nodes = "uniform"
    Ip = REFERENCE_IP if spec.constraint in {"Ip", "Ip_beta"} else None
    beta = float(reference.ref_profiles["beta_constraint"]) if spec.constraint in {"beta", "Ip_beta"} else None
    return OperatorCase(
        route=spec.mode,
        profile_coeffs=_profile_coeffs_for_case(
            spec.mode,
            spec.coordinate,
            spec.input_kind,
            constraint=spec.constraint,
        ),
        boundary=BOUNDARY,
        heat_input=heat_input,
        current_input=current_input,
        coordinate=spec.coordinate,
        nodes=nodes,
        Ip=Ip,
        beta=beta,
    )


def _iter_benchmark_specs():
    for mode in BENCHMARK_MODES:
        for coordinate in ("rho", "psin"):
            for input_kind in BENCHMARK_INPUT_KINDS:
                for constraint in BENCHMARK_MODE_CONSTRAINTS[mode]:
                    yield BenchmarkCaseSpec(
                        mode=mode, coordinate=coordinate, constraint=constraint, input_kind=input_kind
                    )


def _solve_with_timing(case: OperatorCase) -> tuple[object, object, np.ndarray, float, float]:
    solver = Solver(operator=Operator(TEST_GRID, case), config=CONFIG)
    solver.solve(
        method=CONFIG.method,
        rtol=CONFIG.rtol,
        atol=CONFIG.atol,
        root_maxiter=CONFIG.root_maxiter,
        root_maxfev=CONFIG.root_maxfev,
        enable_verbose=False,
        enable_history=False,
        enable_warmstart=False,
    )

    elapsed_ms_samples = np.empty(BENCHMARK_REPEAT_COUNT, dtype=np.float64)
    result = None
    for index in range(BENCHMARK_REPEAT_COUNT):
        solver.solve(
            method=CONFIG.method,
            rtol=CONFIG.rtol,
            atol=CONFIG.atol,
            root_maxiter=CONFIG.root_maxiter,
            root_maxfev=CONFIG.root_maxfev,
            enable_verbose=False,
            enable_history=False,
            enable_warmstart=False,
        )
        result = solver.result
        elapsed_ms_samples[index] = float(result.elapsed) / 1000.0

    if result is None:
        raise RuntimeError("benchmark solve produced no result")

    equilibrium = solver.build_equilibrium()
    shape_x = _extract_shape_x(case.profile_coeffs, result.x)
    return result, equilibrium, shape_x, float(np.mean(elapsed_ms_samples)), float(np.std(elapsed_ms_samples))


def _shape_error(reference_x: np.ndarray, current_x: np.ndarray) -> float:
    n = min(reference_x.shape[0], current_x.shape[0])
    if n == 0:
        return 0.0
    return float(np.max(np.abs(current_x[:n] - reference_x[:n])))


def _relative_profile_errors(reference_values: np.ndarray, current_values: np.ndarray) -> tuple[float, float]:
    reference_values = _as_float64_array(reference_values)
    current_values = _as_float64_array(current_values)
    n = min(reference_values.shape[0], current_values.shape[0])
    if n == 0:
        return 0.0, 0.0
    reference_values = reference_values[:n]
    current_values = current_values[:n]
    diff = current_values - reference_values
    scale = max(float(np.max(np.abs(reference_values))), 1.0e-12)
    rel_rms = float(np.sqrt(np.mean(diff * diff)) / scale)
    rel_max = float(np.max(np.abs(diff)) / scale)
    return rel_rms, rel_max


def _window_derivative_sign_changes(
    values: np.ndarray, *, side: str, window: int = DIAGNOSTIC_SIGN_CHANGE_WINDOW
) -> int:
    values = _as_float64_array(values)
    count = min(int(window), values.shape[0])
    if side == "head":
        sample = values[:count]
    elif side == "tail":
        sample = values[-count:]
    else:
        raise ValueError(f"Unsupported side {side!r}")
    delta = np.diff(sample)
    signs = np.sign(delta)
    nonzero = signs[signs != 0.0]
    if nonzero.size < 2:
        return 0
    return int(np.sum(nonzero[1:] * nonzero[:-1] < 0.0))


def _diagnostic_profile_metrics(
    reference_axis: PreparedInterpAxis,
    reference_values: np.ndarray,
    current_axis: np.ndarray,
    current_values: np.ndarray,
) -> tuple[float, float, int, int]:
    current_axis = _as_float64_array(current_axis)
    current_values = _as_float64_array(current_values)
    reference_on_current = _profile_interp(reference_axis, reference_values, current_axis)
    rel_rms, rel_max = _relative_profile_errors(reference_on_current, current_values)
    return (
        rel_rms,
        rel_max,
        _window_derivative_sign_changes(current_values, side="head"),
        _window_derivative_sign_changes(current_values, side="tail"),
    )


def _benchmark_case_result(spec: BenchmarkCaseSpec, reference: ReferenceBundle) -> BenchmarkCaseResult:
    case = _make_benchmark_case(spec, reference)
    result, equilibrium, shape_x, avg_ms, std_ms = _solve_with_timing(case)
    psi_r_rel_rms_error, psi_r_rel_max_error, psi_r_head_sign_changes, psi_r_tail_sign_changes = (
        _diagnostic_profile_metrics(
            reference.rho_interp_axis,
            reference.ref_profiles["psi_r"],
            equilibrium.rho,
            equilibrium.alpha2 * equilibrium.psin_r,
        )
    )
    ff_psi_rel_rms_error, ff_psi_rel_max_error, ff_psi_head_sign_changes, ff_psi_tail_sign_changes = (
        _diagnostic_profile_metrics(
            reference.rho_interp_axis,
            reference.ref_profiles["FF_psi"],
            equilibrium.rho,
            equilibrium.alpha1 * equilibrium.FFn_psin,
        )
    )
    mu0_p_psi_rel_rms_error, mu0_p_psi_rel_max_error, mu0_p_psi_head_sign_changes, mu0_p_psi_tail_sign_changes = (
        _diagnostic_profile_metrics(
            reference.rho_interp_axis,
            (4.0e-7 * np.pi) * reference.ref_profiles["P_psi"],
            equilibrium.rho,
            equilibrium.alpha1 * equilibrium.Pn_psin,
        )
    )
    return BenchmarkCaseResult(
        spec=spec,
        result=result,
        equilibrium=equilibrium,
        avg_ms=avg_ms,
        std_ms=std_ms,
        shape_error=_shape_error(reference.reference_shape_x, shape_x),
        psi_r_rel_rms_error=psi_r_rel_rms_error,
        psi_r_rel_max_error=psi_r_rel_max_error,
        psi_r_head_sign_changes=psi_r_head_sign_changes,
        psi_r_tail_sign_changes=psi_r_tail_sign_changes,
        ff_psi_rel_rms_error=ff_psi_rel_rms_error,
        ff_psi_rel_max_error=ff_psi_rel_max_error,
        ff_psi_head_sign_changes=ff_psi_head_sign_changes,
        ff_psi_tail_sign_changes=ff_psi_tail_sign_changes,
        mu0_p_psi_rel_rms_error=mu0_p_psi_rel_rms_error,
        mu0_p_psi_rel_max_error=mu0_p_psi_rel_max_error,
        mu0_p_psi_head_sign_changes=mu0_p_psi_head_sign_changes,
        mu0_p_psi_tail_sign_changes=mu0_p_psi_tail_sign_changes,
    )


def _write_report(
    reference: ReferenceBundle,
    rows: list[BenchmarkCaseResult],
    plot_failures: list[str] | None = None,
) -> None:
    worst_shape = max(rows, key=lambda row: row.shape_error)
    slowest_case = max(rows, key=lambda row: row.avg_ms)
    largest_nfev_case = max(rows, key=lambda row: int(row.result.nfev))
    worst_psi_r_case = max(rows, key=lambda row: row.psi_r_rel_rms_error)
    worst_ff_psi_case = max(rows, key=lambda row: row.ff_psi_rel_rms_error)
    worst_mu0_p_psi_case = max(rows, key=lambda row: row.mu0_p_psi_rel_rms_error)
    most_oscillatory_psi_r_case = max(rows, key=lambda row: row.psi_r_head_sign_changes + row.psi_r_tail_sign_changes)
    most_oscillatory_ff_psi_case = max(
        rows, key=lambda row: row.ff_psi_head_sign_changes + row.ff_psi_tail_sign_changes
    )
    most_oscillatory_mu0_p_psi_case = max(
        rows, key=lambda row: row.mu0_p_psi_head_sign_changes + row.mu0_p_psi_tail_sign_changes
    )
    failing_rows = [row for row in rows if row.shape_error > SHAPE_MATCH_TOL]
    rows_by_error = _sort_rows_desc(rows, lambda row: row.shape_error)
    rows_by_time = _sort_rows_desc(rows, lambda row: row.avg_ms)
    rows_by_nfev = _sort_rows_desc(rows, lambda row: int(row.result.nfev))
    rows_by_psi_r_rms = _sort_rows_desc(rows, lambda row: row.psi_r_rel_rms_error)
    rows_by_ff_psi_rms = _sort_rows_desc(rows, lambda row: row.ff_psi_rel_rms_error)
    rows_by_mu0_p_psi_rms = _sort_rows_desc(rows, lambda row: row.mu0_p_psi_rel_rms_error)
    rows_by_psi_r_oscillation = _sort_rows_desc(
        rows, lambda row: row.psi_r_head_sign_changes + row.psi_r_tail_sign_changes
    )
    rows_by_ff_psi_oscillation = _sort_rows_desc(
        rows, lambda row: row.ff_psi_head_sign_changes + row.ff_psi_tail_sign_changes
    )
    rows_by_mu0_p_psi_oscillation = _sort_rows_desc(
        rows, lambda row: row.mu0_p_psi_head_sign_changes + row.mu0_p_psi_tail_sign_changes
    )

    lines = [f"PF-rho-Ip reference vs {len(rows)} low-resolution route-specific cases", ""]
    lines.extend(
        _render_pairs(
            [
                ("backend", BACKEND),
                ("reference_case", "PF_RHO + Ip"),
                ("reference_grid", f"{REFERENCE_GRID.Nr}x{REFERENCE_GRID.Nt} ({REFERENCE_GRID.scheme})"),
                ("reference_source_samples", str(REFERENCE_SOURCE_SAMPLE_COUNT)),
                ("test_grid", f"{TEST_GRID.Nr}x{TEST_GRID.Nt} ({TEST_GRID.scheme})"),
                ("test_source_samples", str(TEST_SOURCE_SAMPLE_COUNT)),
                ("repeat_count", str(BENCHMARK_REPEAT_COUNT)),
                ("shape_tol", f"{SHAPE_MATCH_TOL:.3e}"),
                ("failure_count", f"{len(failing_rows)}/{len(rows)}"),
                ("worst_shape_case", f"{worst_shape.case_name} ({worst_shape.shape_error:.6e})"),
                (
                    "worst_psi_r_rel_rms_case",
                    f"{worst_psi_r_case.case_name} ({worst_psi_r_case.psi_r_rel_rms_error:.6e})",
                ),
                (
                    "worst_ff_psi_rel_rms_case",
                    f"{worst_ff_psi_case.case_name} ({worst_ff_psi_case.ff_psi_rel_rms_error:.6e})",
                ),
                (
                    "worst_mu0_p_psi_rel_rms_case",
                    f"{worst_mu0_p_psi_case.case_name} ({worst_mu0_p_psi_case.mu0_p_psi_rel_rms_error:.6e})",
                ),
                (
                    "most_oscillatory_psi_r_case",
                    f"{most_oscillatory_psi_r_case.case_name} "
                    f"(h/t={most_oscillatory_psi_r_case.psi_r_head_sign_changes}/{most_oscillatory_psi_r_case.psi_r_tail_sign_changes})",
                ),
                (
                    "most_oscillatory_ff_psi_case",
                    f"{most_oscillatory_ff_psi_case.case_name} "
                    f"(h/t={most_oscillatory_ff_psi_case.ff_psi_head_sign_changes}/{most_oscillatory_ff_psi_case.ff_psi_tail_sign_changes})",
                ),
                (
                    "most_oscillatory_mu0_p_psi_case",
                    f"{most_oscillatory_mu0_p_psi_case.case_name} "
                    f"(h/t={most_oscillatory_mu0_p_psi_case.mu0_p_psi_head_sign_changes}/{most_oscillatory_mu0_p_psi_case.mu0_p_psi_tail_sign_changes})",
                ),
                ("slowest_case", f"{slowest_case.case_name} ({slowest_case.avg_ms:.3f} ms)"),
                ("largest_nfev_case", f"{largest_nfev_case.case_name} ({int(largest_nfev_case.result.nfev)})"),
            ]
        )
    )

    lines.extend(["", "Case results", ""])
    lines.append(
        "case".ljust(24)
        + " | "
        + "shape_error".rjust(12)
        + " | "
        + "avg_ms".rjust(12)
        + " | "
        + "std_ms".rjust(12)
        + " | "
        + "nfev".rjust(6)
        + " | "
        + "nit".rjust(6)
        + " | "
        + "residual".rjust(12)
        + " | "
        + "ok".rjust(4)
    )
    lines.append("-" * 114)
    for row in rows:
        ok = "yes" if row.shape_error <= SHAPE_MATCH_TOL else "no"
        lines.append(
            f"{row.case_name:<24} | "
            f"{row.shape_error:>12.6e} | "
            f"{row.avg_ms:>12.3f} | "
            f"{row.std_ms:>12.3f} | "
            f"{int(row.result.nfev):>6d} | "
            f"{int(row.result.nit):>6d} | "
            f"{float(row.result.residual_norm_final):>12.6e} | "
            f"{ok:>4}"
        )

    lines.extend(["", "psi_r / FF_psi / mu0P_psi diagnostics", ""])
    lines.append(
        "case".ljust(24)
        + " | "
        + "psi_r_rms".rjust(10)
        + " | "
        + "psi_r_max".rjust(10)
        + " | "
        + "psi_r_h/t".rjust(9)
        + " | "
        + "FF_psi_rms".rjust(10)
        + " | "
        + "FF_psi_max".rjust(10)
        + " | "
        + "FF_psi_h/t".rjust(10)
        + " | "
        + "mu0P_rms".rjust(10)
        + " | "
        + "mu0P_max".rjust(10)
        + " | "
        + "mu0P_h/t".rjust(9)
    )
    lines.append("-" * 132)
    for row in rows:
        lines.append(
            f"{row.case_name:<24} | "
            f"{row.psi_r_rel_rms_error:>10.3e} | "
            f"{row.psi_r_rel_max_error:>10.3e} | "
            f"{f'{row.psi_r_head_sign_changes}/{row.psi_r_tail_sign_changes}':>9} | "
            f"{row.ff_psi_rel_rms_error:>10.3e} | "
            f"{row.ff_psi_rel_max_error:>10.3e} | "
            f"{f'{row.ff_psi_head_sign_changes}/{row.ff_psi_tail_sign_changes}':>10} | "
            f"{row.mu0_p_psi_rel_rms_error:>10.3e} | "
            f"{row.mu0_p_psi_rel_max_error:>10.3e} | "
            f"{f'{row.mu0_p_psi_head_sign_changes}/{row.mu0_p_psi_tail_sign_changes}':>9}"
        )

    lines.extend(
        _render_ranking_section(
            "Largest shape_error ranking",
            rows_by_error,
            columns=[
                ("right", "rank", 4, lambda index, row: index),
                ("left", "case", 24, lambda index, row: row.case_name),
                ("right", "shape_error", 12, lambda index, row: f"{row.shape_error:.6e}"),
                ("right", "avg_ms", 12, lambda index, row: f"{row.avg_ms:.3f}"),
                ("right", "std_ms", 12, lambda index, row: f"{row.std_ms:.3f}"),
                ("right", "nfev", 6, lambda index, row: int(row.result.nfev)),
            ],
        )
    )
    lines.extend(
        _render_ranking_section(
            "Largest psi_r relative RMS error ranking",
            rows_by_psi_r_rms,
            columns=[
                ("right", "rank", 4, lambda index, row: index),
                ("left", "case", 24, lambda index, row: row.case_name),
                ("right", "psi_r_rms", 10, lambda index, row: f"{row.psi_r_rel_rms_error:.3e}"),
                ("right", "psi_r_max", 10, lambda index, row: f"{row.psi_r_rel_max_error:.3e}"),
                (
                    "right",
                    "psi_r_h/t",
                    9,
                    lambda index, row: f"{row.psi_r_head_sign_changes}/{row.psi_r_tail_sign_changes}",
                ),
                ("right", "shape_error", 12, lambda index, row: f"{row.shape_error:.6e}"),
            ],
        )
    )
    lines.extend(
        _render_ranking_section(
            "Largest FF_psi relative RMS error ranking",
            rows_by_ff_psi_rms,
            columns=[
                ("right", "rank", 4, lambda index, row: index),
                ("left", "case", 24, lambda index, row: row.case_name),
                ("right", "FF_psi_rms", 10, lambda index, row: f"{row.ff_psi_rel_rms_error:.3e}"),
                ("right", "FF_psi_max", 10, lambda index, row: f"{row.ff_psi_rel_max_error:.3e}"),
                (
                    "right",
                    "FF_psi_h/t",
                    10,
                    lambda index, row: f"{row.ff_psi_head_sign_changes}/{row.ff_psi_tail_sign_changes}",
                ),
                ("right", "shape_error", 12, lambda index, row: f"{row.shape_error:.6e}"),
            ],
        )
    )
    lines.extend(
        _render_ranking_section(
            "Largest mu0P_psi relative RMS error ranking",
            rows_by_mu0_p_psi_rms,
            columns=[
                ("right", "rank", 4, lambda index, row: index),
                ("left", "case", 24, lambda index, row: row.case_name),
                (
                    "right",
                    "mu0P_h/t",
                    9,
                    lambda index, row: f"{row.mu0_p_psi_head_sign_changes}/{row.mu0_p_psi_tail_sign_changes}",
                ),
                ("right", "mu0P_rms", 10, lambda index, row: f"{row.mu0_p_psi_rel_rms_error:.3e}"),
                ("right", "mu0P_max", 10, lambda index, row: f"{row.mu0_p_psi_rel_max_error:.3e}"),
                ("right", "shape_error", 12, lambda index, row: f"{row.shape_error:.6e}"),
            ],
        )
    )
    lines.extend(
        _render_ranking_section(
            "Most oscillatory psi_r ranking",
            rows_by_psi_r_oscillation,
            columns=[
                ("right", "rank", 4, lambda index, row: index),
                ("left", "case", 24, lambda index, row: row.case_name),
                (
                    "right",
                    "psi_r_h/t",
                    9,
                    lambda index, row: f"{row.psi_r_head_sign_changes}/{row.psi_r_tail_sign_changes}",
                ),
                ("right", "psi_r_rms", 10, lambda index, row: f"{row.psi_r_rel_rms_error:.3e}"),
                ("right", "psi_r_max", 10, lambda index, row: f"{row.psi_r_rel_max_error:.3e}"),
                ("right", "shape_error", 12, lambda index, row: f"{row.shape_error:.6e}"),
            ],
        )
    )
    lines.extend(
        _render_ranking_section(
            "Most oscillatory FF_psi ranking",
            rows_by_ff_psi_oscillation,
            columns=[
                ("right", "rank", 4, lambda index, row: index),
                ("left", "case", 24, lambda index, row: row.case_name),
                (
                    "right",
                    "FF_psi_h/t",
                    10,
                    lambda index, row: f"{row.ff_psi_head_sign_changes}/{row.ff_psi_tail_sign_changes}",
                ),
                ("right", "FF_psi_rms", 10, lambda index, row: f"{row.ff_psi_rel_rms_error:.3e}"),
                ("right", "FF_psi_max", 10, lambda index, row: f"{row.ff_psi_rel_max_error:.3e}"),
                ("right", "shape_error", 12, lambda index, row: f"{row.shape_error:.6e}"),
            ],
        )
    )
    lines.extend(
        _render_ranking_section(
            "Most oscillatory mu0P_psi ranking",
            rows_by_mu0_p_psi_oscillation,
            columns=[
                ("right", "rank", 4, lambda index, row: index),
                ("left", "case", 24, lambda index, row: row.case_name),
                (
                    "right",
                    "mu0P_h/t",
                    9,
                    lambda index, row: f"{row.mu0_p_psi_head_sign_changes}/{row.mu0_p_psi_tail_sign_changes}",
                ),
                ("right", "mu0P_rms", 10, lambda index, row: f"{row.mu0_p_psi_rel_rms_error:.3e}"),
                ("right", "mu0P_max", 10, lambda index, row: f"{row.mu0_p_psi_rel_max_error:.3e}"),
                ("right", "shape_error", 12, lambda index, row: f"{row.shape_error:.6e}"),
            ],
        )
    )
    lines.extend(
        _render_ranking_section(
            "Slowest avg_ms ranking",
            rows_by_time,
            columns=[
                ("right", "rank", 4, lambda index, row: index),
                ("left", "case", 24, lambda index, row: row.case_name),
                ("right", "avg_ms", 12, lambda index, row: f"{row.avg_ms:.3f}"),
                ("right", "std_ms", 12, lambda index, row: f"{row.std_ms:.3f}"),
                ("right", "shape_error", 12, lambda index, row: f"{row.shape_error:.6e}"),
                ("right", "nfev", 6, lambda index, row: int(row.result.nfev)),
            ],
        )
    )
    lines.extend(
        _render_ranking_section(
            "Largest nfev ranking",
            rows_by_nfev,
            columns=[
                ("right", "rank", 4, lambda index, row: index),
                ("left", "case", 24, lambda index, row: row.case_name),
                ("right", "nfev", 6, lambda index, row: int(row.result.nfev)),
                ("right", "avg_ms", 12, lambda index, row: f"{row.avg_ms:.3f}"),
                ("right", "std_ms", 12, lambda index, row: f"{row.std_ms:.3f}"),
                ("right", "shape_error", 12, lambda index, row: f"{row.shape_error:.6e}"),
            ],
        )
    )

    if plot_failures:
        lines.extend(["", "Plot failures", ""])
        lines.extend(plot_failures)

    (_artifact_dir() / "benchmark_compare.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_reference_summary_json(reference: ReferenceBundle) -> None:
    native_eq = reference.equilibrium
    summary_eq = native_eq.resample(REFERENCE_SUMMARY_GRID)

    boundary_R = _as_float64_array(summary_eq.geometry.R[-1])
    boundary_Z = _as_float64_array(summary_eq.geometry.Z[-1])
    boundary_R_closed = np.concatenate([boundary_R, boundary_R[:1]])
    boundary_Z_closed = np.concatenate([boundary_Z, boundary_Z[:1]])
    R_in = float(np.min(boundary_R))
    R_out = float(np.max(boundary_R))
    Z_top = float(np.max(boundary_Z))
    Z_bottom = float(np.min(boundary_Z))
    a_lcfs = 0.5 * (R_out - R_in)
    if a_lcfs <= 1.0e-14:
        raise ValueError("LCFS minor radius is too small to compute delta/elongation")
    R_top = float(boundary_R[int(np.argmax(boundary_Z))])
    R_bottom = float(boundary_R[int(np.argmin(boundary_Z))])
    elongation = 0.5 * (Z_top - Z_bottom) / a_lcfs
    delta_top = (float(native_eq.R0) - R_top) / a_lcfs
    delta_bottom = (float(native_eq.R0) - R_bottom) / a_lcfs
    delta_average = 0.5 * (delta_top + delta_bottom)

    rho = _REFERENCE_SUMMARY_RHO_AXIS
    native_rho_axis = _prepare_interp_axis(native_eq.rho)
    psin = _profile_interp(native_rho_axis, native_eq.psin, rho)
    np.maximum(psin, 0.0, out=psin)
    if psin.size:
        psin[0] = 0.0
        psin[-1] = 1.0

    mu0 = 4.0e-7 * np.pi
    native_P_psi = _as_float64_array(native_eq.alpha1 * native_eq.Pn_psin / mu0)
    P_psi = _profile_interp(native_rho_axis, native_P_psi, rho)
    q = _profile_interp(native_rho_axis, native_eq.q, rho)

    payload = {
        "sampling": {
            "Nr": int(REFERENCE_SUMMARY_GRID.Nr),
            "Nt": int(REFERENCE_SUMMARY_GRID.Nt),
            "scheme": REFERENCE_SUMMARY_GRID.scheme,
        },
        "geometry": {
            "R0": float(native_eq.R0),
            "Z0": float(native_eq.Z0),
            "a": float(native_eq.a),
            "B0": float(native_eq.B0),
            "aspect_ratio": float(native_eq.R0 / native_eq.a),
            "Ip": float(native_eq.Ip),
        },
        "outer_closed_surface": {
            "R": boundary_R_closed.tolist(),
            "Z": boundary_Z_closed.tolist(),
            "R_in": R_in,
            "R_out": R_out,
            "Z_top": Z_top,
            "Z_bottom": Z_bottom,
            "a_from_lcfs": a_lcfs,
            "elongation": float(elongation),
            "delta_top": float(delta_top),
            "delta_bottom": float(delta_bottom),
            "delta_average": float(delta_average),
        },
        "profiles": {
            "rho": rho.tolist(),
            "psin": psin.tolist(),
            "P_psi": P_psi.tolist(),
            "q": q.tolist(),
        },
    }

    _reference_summary_json_path().write_text(json.dumps(payload, indent=2), encoding="utf-8")


def run_full_benchmark(*, show_progress: bool = SHOW_PROGRESS) -> tuple[ReferenceBundle, list[BenchmarkCaseResult]]:
    reference = _solve_reference(show_progress=show_progress)
    rows: list[BenchmarkCaseResult] = []
    plot_failures: list[str] = []
    specs = list(_iter_benchmark_specs())
    plot_dir = _plot_dir() if PLOT else None

    for index, spec in enumerate(specs, start=1):
        row = _benchmark_case_result(spec, reference)
        rows.append(row)
        if show_progress:
            print(
                f"[{BACKEND}] [{index:02d}/{len(specs)}] {row.case_name}: "
                f"time={row.avg_ms:.3f}+/-{row.std_ms:.3f} ms | "
                f"shape={row.shape_error:.3e} | "
                f"psi_r={row.psi_r_rel_rms_error:.2e} ({row.psi_r_head_sign_changes}/{row.psi_r_tail_sign_changes}) | "
                f"FF_psi={row.ff_psi_rel_rms_error:.2e} ({row.ff_psi_head_sign_changes}/{row.ff_psi_tail_sign_changes}) | "
                f"mu0P_psi={row.mu0_p_psi_rel_rms_error:.2e} "
                f"({row.mu0_p_psi_head_sign_changes}/{row.mu0_p_psi_tail_sign_changes})"
            )
        if plot_dir is not None:
            try:
                reference.equilibrium.compare(
                    row.equilibrium,
                    plot_dir / f"{row.case_name}_compare.png",
                    label_ref="PF_RHO_ref",
                    label_other=row.case_name,
                )
                # row.equilibrium.plot(plot_dir / f"{row.case_name}_summary.png")
            except Exception as exc:
                message = f"{row.case_name}: {type(exc).__name__}: {exc}"
                plot_failures.append(message)
                if show_progress:
                    print(f"[{BACKEND}] plot warning: {message}")

    _write_report(reference, rows, plot_failures)
    _write_reference_summary_json(reference)

    if plot_dir is not None:
        try:
            reference.equilibrium.plot(outpath=_artifact_dir() / "reference_summary.png")
        except Exception as exc:
            message = f"reference_summary: {type(exc).__name__}: {exc}"
            plot_failures.append(message)
            if show_progress:
                print(f"[{BACKEND}] plot warning: {message}")
            _write_report(reference, rows, plot_failures)

    return reference, rows


def _run_as_script() -> int:
    run_full_benchmark(show_progress=SHOW_PROGRESS)
    return 0


if __name__ == "__main__":
    raise SystemExit(_run_as_script())

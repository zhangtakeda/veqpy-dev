from __future__ import annotations

import importlib
import importlib.util
import json
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter

import numpy as np

BACKEND = os.environ.get("VEQPY_BACKEND", "numba")
os.environ["VEQPY_BACKEND"] = BACKEND

SOLVE_REPEAT_COUNT = int(os.environ.get("VEQPY_HOTSPOT_SOLVE_REPEAT_COUNT", "10"))
RESIDUAL_REPEAT_COUNT = int(os.environ.get("VEQPY_HOTSPOT_RESIDUAL_REPEAT_COUNT", "200"))
HOTSPOT_SCOPE = os.environ.get("VEQPY_HOTSPOT_SCOPE", "full").strip().lower()
HOTSPOT_TOPK = int(os.environ.get("VEQPY_HOTSPOT_TOPK", "10"))

STAGE_NAMES = ("stage_a", "stage_b", "stage_c", "stage_d")

geometry_module = importlib.import_module("veqpy.model.geometry")
operator_module = importlib.import_module("veqpy.operator.operator")
Solver = importlib.import_module("veqpy.solver").Solver


@dataclass(slots=True)
class TimeStat:
    total_ms: float = 0.0
    calls: int = 0

    def add(self, elapsed_ms: float) -> None:
        self.total_ms += float(elapsed_ms)
        self.calls += 1


@dataclass(frozen=True, slots=True)
class SolveProfile:
    avg_solve_ms: float
    avg_residual_ms: float
    avg_other_ms: float
    residual_share: float
    avg_residual_calls: float


@dataclass(frozen=True, slots=True)
class ResidualStageProfile:
    total_ms: float
    share_of_residual: float
    engine_ms: float
    non_engine_ms: float
    engine_share_of_stage: float
    engine_breakdown_ms: dict[str, float]


@dataclass(frozen=True, slots=True)
class ResidualProfile:
    avg_total_ms: float
    avg_other_ms: float
    other_share: float
    total_engine_ms: float
    total_non_engine_ms: float
    total_non_engine_share: float
    stages: dict[str, ResidualStageProfile]


@dataclass(frozen=True, slots=True)
class HotspotCaseResult:
    case_name: str
    solve: SolveProfile
    residual: ResidualProfile


@dataclass(slots=True)
class _ProfilerSnapshot:
    residual_total_ms: float
    residual_calls: int
    stage_total_ms: dict[str, float]
    stage_calls: dict[str, int]
    engine_detail_ms: dict[str, float]
    engine_detail_calls: dict[str, int]


class OperatorHotspotProfiler:
    def __init__(self, operator):
        self.operator = operator
        self.current_stage: str | None = None
        self.residual_stat = TimeStat()
        self.stage_stats = {name: TimeStat() for name in STAGE_NAMES}
        self.engine_detail_stats = {
            "stage_a.update_profiles_packed_bulk": TimeStat(),
            "stage_b.update_geometry": TimeStat(),
            "stage_b.update_fourier_family_fields": TimeStat(),
            "stage_c.materialize_profile_owned_psin_source": TimeStat(),
            "stage_c.materialize_projected_source_inputs": TimeStat(),
            "stage_c.resolve_source_inputs": TimeStat(),
            "stage_c.source_runner": TimeStat(),
            "stage_c.update_fixed_point_psin_query": TimeStat(),
            "stage_d.residual_stage_runner": TimeStat(),
        }
        self._originals: dict[str, object] = {}

    def __enter__(self) -> "OperatorHotspotProfiler":
        self._patch_operator_methods()
        self._patch_engine_functions()
        self._patch_operator_runners()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self._restore_operator_runners()
        self._restore_engine_functions()
        self._restore_operator_methods()

    def snapshot(self) -> _ProfilerSnapshot:
        return _ProfilerSnapshot(
            residual_total_ms=float(self.residual_stat.total_ms),
            residual_calls=int(self.residual_stat.calls),
            stage_total_ms={name: float(stat.total_ms) for name, stat in self.stage_stats.items()},
            stage_calls={name: int(stat.calls) for name, stat in self.stage_stats.items()},
            engine_detail_ms={name: float(stat.total_ms) for name, stat in self.engine_detail_stats.items()},
            engine_detail_calls={name: int(stat.calls) for name, stat in self.engine_detail_stats.items()},
        )

    def record_engine(self, detail_name: str, elapsed_ms: float) -> None:
        self.engine_detail_stats[detail_name].add(elapsed_ms)

    def _patch_operator_methods(self) -> None:
        operator_cls = operator_module.Operator
        self._originals["residual"] = operator_cls.residual
        self._originals["stage_a_profile"] = operator_cls.stage_a_profile
        self._originals["stage_b_geometry"] = operator_cls.stage_b_geometry
        self._originals["stage_c_source"] = operator_cls.stage_c_source
        self._originals["stage_d_residual"] = operator_cls.stage_d_residual

        profiler = self
        original_residual = self._originals["residual"]

        def build_residual_wrapper(original):
            def wrapper(this, *args, **kwargs):
                if this is not profiler.operator:
                    return original(this, *args, **kwargs)
                started = perf_counter()
                try:
                    return original(this, *args, **kwargs)
                finally:
                    profiler.residual_stat.add((perf_counter() - started) * 1e3)

            return wrapper

        residual_wrapper = build_residual_wrapper(original_residual)

        def build_stage_wrapper(stage_name: str):
            original = self._originals[stage_name]

            def wrapper(this, *args, **kwargs):
                if this is not profiler.operator:
                    return original(this, *args, **kwargs)
                previous_stage = profiler.current_stage
                profiler.current_stage = (
                    stage_name.replace("_profile", "")
                    .replace("_geometry", "")
                    .replace("_source", "")
                    .replace("_residual", "")
                )
                started = perf_counter()
                try:
                    return original(this, *args, **kwargs)
                finally:
                    profiler.stage_stats[profiler.current_stage].add((perf_counter() - started) * 1e3)
                    profiler.current_stage = previous_stage

            return wrapper

        operator_cls.residual = residual_wrapper
        operator_cls.stage_a_profile = build_stage_wrapper("stage_a_profile")
        operator_cls.stage_b_geometry = build_stage_wrapper("stage_b_geometry")
        operator_cls.stage_c_source = build_stage_wrapper("stage_c_source")
        operator_cls.stage_d_residual = build_stage_wrapper("stage_d_residual")

    def _restore_operator_methods(self) -> None:
        operator_cls = operator_module.Operator
        operator_cls.residual = self._originals["residual"]
        operator_cls.stage_a_profile = self._originals["stage_a_profile"]
        operator_cls.stage_b_geometry = self._originals["stage_b_geometry"]
        operator_cls.stage_c_source = self._originals["stage_c_source"]
        operator_cls.stage_d_residual = self._originals["stage_d_residual"]

    def _patch_engine_functions(self) -> None:
        self._originals["update_profiles_packed_bulk"] = operator_module.update_profiles_packed_bulk
        self._originals["update_fourier_family_fields"] = operator_module.update_fourier_family_fields
        self._originals["update_geometry_operator"] = operator_module.update_geometry
        self._originals["materialize_profile_owned_psin_source"] = operator_module.materialize_profile_owned_psin_source
        self._originals["materialize_projected_source_inputs"] = operator_module.materialize_projected_source_inputs
        self._originals["resolve_source_inputs"] = operator_module.resolve_source_inputs
        self._originals["update_fixed_point_psin_query"] = operator_module.update_fixed_point_psin_query
        self._originals["update_geometry"] = geometry_module.update_geometry

        profiler = self

        def wrap_global(name: str, original):
            def wrapper(*args, **kwargs):
                started = perf_counter()
                try:
                    return original(*args, **kwargs)
                finally:
                    profiler.record_engine(name, (perf_counter() - started) * 1e3)

            return wrapper

        operator_module.update_profiles_packed_bulk = wrap_global(
            "stage_a.update_profiles_packed_bulk",
            self._originals["update_profiles_packed_bulk"],
        )
        operator_module.update_fourier_family_fields = wrap_global(
            "stage_b.update_fourier_family_fields",
            self._originals["update_fourier_family_fields"],
        )
        operator_module.update_geometry = wrap_global(
            "stage_b.update_geometry",
            self._originals["update_geometry_operator"],
        )
        operator_module.materialize_profile_owned_psin_source = wrap_global(
            "stage_c.materialize_profile_owned_psin_source",
            self._originals["materialize_profile_owned_psin_source"],
        )
        operator_module.materialize_projected_source_inputs = wrap_global(
            "stage_c.materialize_projected_source_inputs",
            self._originals["materialize_projected_source_inputs"],
        )
        operator_module.resolve_source_inputs = wrap_global(
            "stage_c.resolve_source_inputs",
            self._originals["resolve_source_inputs"],
        )
        operator_module.update_fixed_point_psin_query = wrap_global(
            "stage_c.update_fixed_point_psin_query",
            self._originals["update_fixed_point_psin_query"],
        )
        geometry_module.update_geometry = wrap_global(
            "stage_b.update_geometry",
            self._originals["update_geometry"],
        )

    def _restore_engine_functions(self) -> None:
        operator_module.update_profiles_packed_bulk = self._originals["update_profiles_packed_bulk"]
        operator_module.update_fourier_family_fields = self._originals["update_fourier_family_fields"]
        operator_module.update_geometry = self._originals["update_geometry_operator"]
        operator_module.materialize_profile_owned_psin_source = self._originals["materialize_profile_owned_psin_source"]
        operator_module.materialize_projected_source_inputs = self._originals["materialize_projected_source_inputs"]
        operator_module.resolve_source_inputs = self._originals["resolve_source_inputs"]
        operator_module.update_fixed_point_psin_query = self._originals["update_fixed_point_psin_query"]
        geometry_module.update_geometry = self._originals["update_geometry"]

    def _patch_operator_runners(self) -> None:
        self._originals["source_runner"] = self.operator._source_runner
        self._originals["residual_stage_runner"] = self.operator.residual_stage_runner
        profiler = self
        original_source_runner = self.operator._source_runner
        original_residual_stage_runner = self.operator.residual_stage_runner

        def source_runner_wrapper(*args, **kwargs):
            started = perf_counter()
            try:
                return original_source_runner(*args, **kwargs)
            finally:
                profiler.record_engine("stage_c.source_runner", (perf_counter() - started) * 1e3)

        def residual_stage_runner_wrapper(*args, **kwargs):
            started = perf_counter()
            try:
                return original_residual_stage_runner(*args, **kwargs)
            finally:
                profiler.record_engine("stage_d.residual_stage_runner", (perf_counter() - started) * 1e3)

        self.operator._source_runner = source_runner_wrapper
        self.operator.residual_stage_runner = residual_stage_runner_wrapper

    def _restore_operator_runners(self) -> None:
        self.operator._source_runner = self._originals["source_runner"]
        self.operator.residual_stage_runner = self._originals["residual_stage_runner"]


def _load_benchmark_module():
    benchmark_path = Path(__file__).with_name("benchmark.py")
    spec = importlib.util.spec_from_file_location("veqpy_tests_benchmark_hotspot", benchmark_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load benchmark module from {benchmark_path}")
    benchmark = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = benchmark
    spec.loader.exec_module(benchmark)
    return benchmark


def _artifact_dir() -> Path:
    outdir = Path(__file__).resolve().parent / "hotspot" / f"full46-{BACKEND}"
    outdir.mkdir(parents=True, exist_ok=True)
    return outdir


def _json_path() -> Path:
    return _artifact_dir() / "hotspot_report.json"


def _txt_path() -> Path:
    return _artifact_dir() / "hotspot_report.txt"


def _safe_ratio(part: float, whole: float) -> float:
    if abs(whole) <= 1e-15:
        return 0.0
    return float(part / whole)


def _format_percent(value: float) -> str:
    return f"{100.0 * float(value):.2f}%"


def _format_ms(value: float) -> str:
    return f"{float(value):.3f} ms"


def _default_specs(benchmark) -> tuple[object, ...]:
    return (
        benchmark.BenchmarkCaseSpec("PF", "rho", "Ip", "uniform"),
        benchmark.BenchmarkCaseSpec("PI", "psin", "Ip_beta", "uniform"),
        benchmark.BenchmarkCaseSpec("PJ2", "psin", "Ip_beta", "uniform"),
        benchmark.BenchmarkCaseSpec("PQ", "psin", "Ip_beta", "grid"),
        benchmark.BenchmarkCaseSpec("PQ", "psin", "Ip_beta", "uniform"),
    )


def _iter_specs(benchmark) -> tuple[object, ...]:
    if HOTSPOT_SCOPE == "smoke":
        return _default_specs(benchmark)
    if HOTSPOT_SCOPE == "full":
        return tuple(benchmark._iter_benchmark_specs())
    raise ValueError(f"Unsupported VEQPY_HOTSPOT_SCOPE={HOTSPOT_SCOPE!r}")


def _extract_case_name(spec) -> str:
    return str(spec.case_name)


def _mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return float(np.mean(np.asarray(values, dtype=np.float64)))


def _solve_once(benchmark, solver: Solver) -> None:
    solver.solve(
        method=benchmark.CONFIG.method,
        rtol=benchmark.CONFIG.rtol,
        atol=benchmark.CONFIG.atol,
        root_maxiter=benchmark.CONFIG.root_maxiter,
        root_maxfev=benchmark.CONFIG.root_maxfev,
        enable_verbose=False,
        enable_history=False,
        enable_warmstart=False,
    )


def _profile_case(benchmark, reference, spec) -> HotspotCaseResult:
    case = benchmark._make_benchmark_case(spec, reference)
    operator = operator_module.Operator(benchmark.TEST_GRID, case)
    solver = Solver(operator=operator, config=benchmark.CONFIG)

    _solve_once(benchmark, solver)
    x_final = np.asarray(solver.result.x, dtype=np.float64).copy()

    solve_total_samples: list[float] = []
    solve_residual_samples: list[float] = []
    solve_other_samples: list[float] = []
    solve_residual_call_samples: list[float] = []

    with OperatorHotspotProfiler(operator) as profiler:
        for _ in range(SOLVE_REPEAT_COUNT):
            before = profiler.snapshot()
            started = perf_counter()
            _solve_once(benchmark, solver)
            solve_ms = (perf_counter() - started) * 1e3
            after = profiler.snapshot()

            residual_ms = after.residual_total_ms - before.residual_total_ms
            residual_calls = after.residual_calls - before.residual_calls
            solve_total_samples.append(solve_ms)
            solve_residual_samples.append(residual_ms)
            solve_other_samples.append(max(solve_ms - residual_ms, 0.0))
            solve_residual_call_samples.append(float(residual_calls))

        x_final = np.asarray(solver.result.x, dtype=np.float64).copy()

    solve_total_ms = _mean(solve_total_samples)
    solve_residual_ms = _mean(solve_residual_samples)
    solve_other_ms = _mean(solve_other_samples)
    solve_profile = SolveProfile(
        avg_solve_ms=solve_total_ms,
        avg_residual_ms=solve_residual_ms,
        avg_other_ms=solve_other_ms,
        residual_share=_safe_ratio(solve_residual_ms, solve_total_ms),
        avg_residual_calls=_mean(solve_residual_call_samples),
    )

    with OperatorHotspotProfiler(operator) as profiler:
        operator.residual(x_final)
        before = profiler.snapshot()
        for _ in range(RESIDUAL_REPEAT_COUNT):
            operator.residual(x_final)
        after = profiler.snapshot()

    residual_total_ms = (after.residual_total_ms - before.residual_total_ms) / RESIDUAL_REPEAT_COUNT
    residual_stage_ms = {
        name: (after.stage_total_ms[name] - before.stage_total_ms[name]) / RESIDUAL_REPEAT_COUNT for name in STAGE_NAMES
    }

    engine_detail_ms = {
        name: (after.engine_detail_ms[name] - before.engine_detail_ms[name]) / RESIDUAL_REPEAT_COUNT
        for name in profiler.engine_detail_stats
    }

    stage_engine_detail_map = {
        "stage_a": ["stage_a.update_profiles_packed_bulk"],
        "stage_b": ["stage_b.update_fourier_family_fields", "stage_b.update_geometry"],
        "stage_c": [
            "stage_c.materialize_profile_owned_psin_source",
            "stage_c.materialize_projected_source_inputs",
            "stage_c.resolve_source_inputs",
            "stage_c.source_runner",
            "stage_c.update_fixed_point_psin_query",
        ],
        "stage_d": ["stage_d.residual_stage_runner"],
    }

    stage_profiles: dict[str, ResidualStageProfile] = {}
    stage_total_sum = 0.0
    for stage_name in STAGE_NAMES:
        stage_total = float(residual_stage_ms[stage_name])
        stage_total_sum += stage_total
        detail_names = stage_engine_detail_map[stage_name]
        detail_payload = {
            detail_name.split(".", 1)[1]: float(engine_detail_ms[detail_name]) for detail_name in detail_names
        }
        engine_total = float(sum(detail_payload.values()))
        non_engine_ms = max(stage_total - engine_total, 0.0)
        stage_profiles[stage_name] = ResidualStageProfile(
            total_ms=stage_total,
            share_of_residual=_safe_ratio(stage_total, residual_total_ms),
            engine_ms=engine_total,
            non_engine_ms=non_engine_ms,
            engine_share_of_stage=_safe_ratio(engine_total, stage_total),
            engine_breakdown_ms=detail_payload,
        )

    residual_other_ms = max(residual_total_ms - stage_total_sum, 0.0)
    total_engine_ms = float(sum(stage.engine_ms for stage in stage_profiles.values()))
    total_non_engine_ms = max(residual_total_ms - total_engine_ms, 0.0)
    residual_profile = ResidualProfile(
        avg_total_ms=float(residual_total_ms),
        avg_other_ms=float(residual_other_ms),
        other_share=_safe_ratio(residual_other_ms, residual_total_ms),
        total_engine_ms=total_engine_ms,
        total_non_engine_ms=total_non_engine_ms,
        total_non_engine_share=_safe_ratio(total_non_engine_ms, residual_total_ms),
        stages=stage_profiles,
    )

    return HotspotCaseResult(
        case_name=_extract_case_name(spec),
        solve=solve_profile,
        residual=residual_profile,
    )


def _render_case(result: HotspotCaseResult) -> list[str]:
    lines = [result.case_name, ""]
    lines.append(
        f"solve: total={_format_ms(result.solve.avg_solve_ms)}, "
        f"residual={_format_ms(result.solve.avg_residual_ms)} ({_format_percent(result.solve.residual_share)}), "
        f"other={_format_ms(result.solve.avg_other_ms)} ({_format_percent(1.0 - result.solve.residual_share)}), "
        f"avg_residual_calls={result.solve.avg_residual_calls:.2f}"
    )
    lines.append(
        f"single_residual: total={_format_ms(result.residual.avg_total_ms)}, "
        f"other={_format_ms(result.residual.avg_other_ms)} ({_format_percent(result.residual.other_share)}), "
        f"total_non_engine={_format_ms(result.residual.total_non_engine_ms)} "
        f"({_format_percent(result.residual.total_non_engine_share)})"
    )
    for stage_name in STAGE_NAMES:
        stage = result.residual.stages[stage_name]
        detail_summary = ", ".join(f"{name}={_format_ms(ms)}" for name, ms in stage.engine_breakdown_ms.items())
        lines.append(
            f"{stage_name}: total={_format_ms(stage.total_ms)} ({_format_percent(stage.share_of_residual)}), "
            f"engine={_format_ms(stage.engine_ms)} ({_format_percent(stage.engine_share_of_stage)}), "
            f"non_engine={_format_ms(stage.non_engine_ms)} ({_format_percent(1.0 - stage.engine_share_of_stage)})"
        )
        lines.append(f"{stage_name}_engine_breakdown: {detail_summary}")
    lines.append("")
    return lines


def _select_topk_results(results: list[HotspotCaseResult]) -> list[HotspotCaseResult]:
    topk = max(int(HOTSPOT_TOPK), 0)
    return sorted(
        results,
        key=lambda result: (
            -float(result.residual.total_non_engine_share),
            -float(result.residual.total_non_engine_ms),
            result.case_name,
        ),
    )[:topk]


def _write_reports(results: list[HotspotCaseResult], *, analyzed_case_count: int) -> None:
    payload = {
        "backend": BACKEND,
        "scope": HOTSPOT_SCOPE,
        "solve_repeat_count": SOLVE_REPEAT_COUNT,
        "residual_repeat_count": RESIDUAL_REPEAT_COUNT,
        "analyzed_case_count": int(analyzed_case_count),
        "saved_case_count": int(len(results)),
        "selection_metric": "single_residual_total_non_engine_share",
        "cases": [asdict(result) for result in results],
    }
    _json_path().write_text(json.dumps(payload, indent=2), encoding="utf-8")

    lines = [
        f"Hotspot report ({BACKEND})",
        "",
        f"scope                 : {HOTSPOT_SCOPE}",
        f"solve_repeat_count    : {SOLVE_REPEAT_COUNT}",
        f"residual_repeat_count : {RESIDUAL_REPEAT_COUNT}",
        f"analyzed_case_count   : {analyzed_case_count}",
        f"saved_case_count      : {len(results)}",
        "selection_metric      : single_residual_total_non_engine_share",
        "",
    ]
    for result in results:
        lines.extend(_render_case(result))
    _txt_path().write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_hotspot(*, show_progress: bool = True) -> list[HotspotCaseResult]:
    benchmark = _load_benchmark_module()
    reference = benchmark._solve_reference()
    all_results: list[HotspotCaseResult] = []
    specs = _iter_specs(benchmark)

    for index, spec in enumerate(specs, start=1):
        if show_progress:
            print(f"[{BACKEND}] hotspot [{index}/{len(specs)}] {_extract_case_name(spec)}")
        all_results.append(_profile_case(benchmark, reference, spec))

    results = _select_topk_results(all_results)
    _write_reports(results, analyzed_case_count=len(all_results))
    return results


def _run_as_script() -> int:
    run_hotspot(show_progress=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(_run_as_script())

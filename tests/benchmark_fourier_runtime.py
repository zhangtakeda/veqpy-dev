from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from veqpy.model import Boundary, Grid
from veqpy.operator import Operator, OperatorCase
from veqpy.operator.layout import build_profile_names

BACKEND = os.environ.get("VEQPY_BACKEND", "numba")
REPEATS = int(os.environ.get("VEQPY_FOURIER_BENCH_REPEATS", "400"))
WARMUP_REPEATS = int(os.environ.get("VEQPY_FOURIER_BENCH_WARMUP", "12"))


@dataclass(frozen=True)
class StageTiming:
    stage: str
    mean_ms: float
    std_ms: float
    min_ms: float
    max_ms: float


@dataclass(frozen=True)
class CaseTiming:
    case_name: str
    backend: str
    repeats: int
    grid_nr: int
    grid_nt: int
    k_max: int
    active_profiles: list[str]
    stage_timings: list[StageTiming]


def _artifact_root() -> Path:
    root = Path(__file__).resolve().parent / "benchmark" / f"fourier-runtime-{BACKEND}"
    root.mkdir(parents=True, exist_ok=True)
    return root


def _pf_reference_profiles(rho: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    beta0 = 0.75
    psin = rho**2
    psin_r = 2.0 * rho

    alpha_p, alpha_f = 5.0, 3.32
    exp_ap, exp_af = np.exp(alpha_p), np.exp(alpha_f)
    den_p, den_f = 1.0 + exp_ap * (alpha_p - 1.0), 1.0 + exp_af * (alpha_f - 1.0)

    current_input = (1.0 - beta0) * alpha_f * (np.exp(alpha_f * psin) - exp_af) / den_f * psin_r
    heat_input = beta0 * alpha_p * (np.exp(alpha_p * psin) - exp_ap) / den_p * psin_r
    return current_input, heat_input


def _make_case(grid: Grid, *, activate_high_order: bool) -> OperatorCase:
    profile_coeffs = {name: None for name in build_profile_names(grid.K_max)}
    profile_coeffs.update(
        {
            "psin": [0.0, 1.0],
            "F": [1.0],
            "h": [0.0, 0.0, 0.0],
            "k": [0.0, 0.0, 0.0],
            "s1": [0.0, 0.0, 0.0],
        }
    )
    if "v" in profile_coeffs:
        profile_coeffs["v"] = None
    if activate_high_order:
        profile_coeffs["c3"] = [0.0, 0.0, 0.0]
        profile_coeffs["s4"] = [0.0, 0.0, 0.0]

    current_input, heat_input = _pf_reference_profiles(grid.rho)
    c_offsets = np.zeros(grid.K_max + 1, dtype=np.float64)
    s_offsets = np.zeros(grid.K_max + 1, dtype=np.float64)
    s_offsets[1] = float(np.arcsin(0.5))

    return OperatorCase(
        profile_coeffs=profile_coeffs,
        boundary=Boundary(
            a=1.05 / 1.85,
            R0=1.05,
            Z0=0.0,
            B0=3.0,
            ka=2.2,
            c_offsets=c_offsets,
            s_offsets=s_offsets,
        ),
        heat_input=heat_input,
        current_input=current_input,
        Ip=3.0e6,
    )


def _measure(label: str, repeats: int, func) -> StageTiming:
    samples = np.empty(repeats, dtype=np.float64)
    for i in range(repeats):
        t0 = time.perf_counter_ns()
        func()
        t1 = time.perf_counter_ns()
        samples[i] = (t1 - t0) / 1.0e6
    return StageTiming(
        stage=label,
        mean_ms=float(np.mean(samples)),
        std_ms=float(np.std(samples)),
        min_ms=float(np.min(samples)),
        max_ms=float(np.max(samples)),
    )


def _build_operator_case(case_name: str) -> tuple[Operator, np.ndarray]:
    if case_name == "k2-low-order":
        grid = Grid(Nr=12, Nt=12, scheme="legendre", K_max=2)
        case = _make_case(grid, activate_high_order=False)
    elif case_name == "k4-low-order":
        grid = Grid(Nr=12, Nt=12, scheme="legendre", K_max=4)
        case = _make_case(grid, activate_high_order=False)
    elif case_name == "k4-high-order":
        grid = Grid(Nr=12, Nt=12, scheme="legendre", K_max=4)
        case = _make_case(grid, activate_high_order=True)
    else:
        raise ValueError(f"Unknown benchmark case {case_name!r}")

    operator = Operator(grid=grid, case=case, name="PF", derivative="rho")
    x = operator.encode_initial_state()
    operator.residual(x)
    return operator, x


def _benchmark_operator(case_name: str) -> CaseTiming:
    operator, x = _build_operator_case(case_name)
    active_profiles = [name for name in operator.profile_names if operator.case.profile_coeffs.get(name) is not None]

    for _ in range(WARMUP_REPEATS):
        operator.stage_a_profile(x)
    operator.stage_a_profile(x)
    for _ in range(WARMUP_REPEATS):
        operator.stage_b_geometry()
    operator.stage_b_geometry()
    for _ in range(WARMUP_REPEATS):
        operator.stage_c_source()
    operator.stage_c_source()
    for _ in range(WARMUP_REPEATS):
        operator.stage_d_residual()
    operator.stage_d_residual()

    stage_timings = [
        _measure("stage_a_profile", REPEATS, lambda: operator.stage_a_profile(x)),
        _measure("stage_b_geometry", REPEATS, operator.stage_b_geometry),
        _measure("stage_c_source", REPEATS, operator.stage_c_source),
        _measure("stage_d_residual", REPEATS, operator.stage_d_residual),
        _measure("full_residual", REPEATS, lambda: operator.residual(x)),
    ]
    return CaseTiming(
        case_name=case_name,
        backend=BACKEND,
        repeats=REPEATS,
        grid_nr=operator.grid.Nr,
        grid_nt=operator.grid.Nt,
        k_max=operator.grid.K_max,
        active_profiles=active_profiles,
        stage_timings=stage_timings,
    )


def _render_report(rows: list[CaseTiming]) -> str:
    baseline = {row.case_name: {stage.stage: stage.mean_ms for stage in row.stage_timings} for row in rows}
    base_case = baseline["k2-low-order"]
    lines = [
        f"Fourier runtime microbenchmark ({BACKEND})",
        "",
        f"repeats={REPEATS}",
        f"warmup_repeats={WARMUP_REPEATS}",
        "",
    ]
    for row in rows:
        lines.append(f"[{row.case_name}]")
        lines.append(f"grid={row.grid_nr}x{row.grid_nt}  K_max={row.k_max}")
        lines.append(f"active_profiles={', '.join(row.active_profiles)}")
        for stage in row.stage_timings:
            ratio = stage.mean_ms / base_case[stage.stage]
            lines.append(
                f"  {stage.stage:<18} mean={stage.mean_ms:8.4f} ms  std={stage.std_ms:7.4f}  "
                f"min={stage.min_ms:7.4f}  max={stage.max_ms:7.4f}  ratio_vs_k2={ratio:6.3f}"
            )
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def run_fourier_runtime_benchmark() -> list[CaseTiming]:
    rows = [
        _benchmark_operator("k2-low-order"),
        _benchmark_operator("k4-low-order"),
        _benchmark_operator("k4-high-order"),
    ]
    outdir = _artifact_root()
    report = _render_report(rows)
    (outdir / "report.txt").write_text(report, encoding="utf-8")
    (outdir / "report.json").write_text(
        json.dumps([asdict(row) for row in rows], indent=2),
        encoding="utf-8",
    )
    return rows


def main() -> int:
    rows = run_fourier_runtime_benchmark()
    print(_render_report(rows), end="")
    print(f"artifacts={_artifact_root()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

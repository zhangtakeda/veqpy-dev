from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from veqpy.model import Boundary, Geqdsk, Grid
from veqpy.model.geqdsk import _safe_float_conversion, _sanitize_line
from veqpy.operator import Operator, OperatorCase
from veqpy.operator.layout import build_shape_profile_names
from veqpy.solver import Solver, SolverConfig

MU0 = 4.0e-7 * np.pi


@dataclass(frozen=True)
class GFilePayload:
    fpol: np.ndarray
    pres: np.ndarray
    ffprim: np.ndarray
    pprime: np.ndarray
    psirz: np.ndarray
    qpsi: np.ndarray


@dataclass(frozen=True)
class CompareConfig:
    gfile_path: Path
    plot_path: Path
    summary_path: Path
    profile_coeffs: Mapping[str, list[float]]
    nr: int
    nt: int
    plot_nr: int = 128
    plot_nt: int = 256
    surface_count: int = 10
    levels: tuple[float, ...] | None = None
    solver_method: str = "trf"
    solver_maxfev: int = 2000
    boundary_fit_m: int | None = None
    boundary_maxtol: float = 1.0e-2


@dataclass(frozen=True)
class CompareResult:
    gfile_path: Path
    plot_path: Path
    summary_path: Path
    boundary_cache_path: Path
    fit_rms: float
    solver_residual: float
    solve_elapsed_ms: float
    target_ip: float
    solved_ip: float
    surface_metrics: dict[float, dict[str, float]]


def default_levels_from_count(surface_count: int) -> tuple[float, ...]:
    if surface_count <= 0:
        raise ValueError(f"surface_count must be positive, got {surface_count!r}")
    return tuple(level / surface_count for level in range(1, surface_count + 1))


def boundary_cache_path(gfile_path: Path) -> Path:
    return gfile_path.with_name(f"{gfile_path.name}-boundary.json")


def normalize_levels(levels: tuple[float, ...] | None, surface_count: int) -> list[float]:
    raw_levels = default_levels_from_count(surface_count) if levels is None else tuple(float(v) for v in levels)
    normalized = sorted({float(level) for level in raw_levels})
    if not normalized:
        raise ValueError("At least one normalized psi level is required")
    for level in normalized:
        if not 0.0 < level <= 1.0:
            raise ValueError(f"Normalized psi levels must lie in (0, 1], got {level!r}")
    return normalized


def read_gfile_payload(path: Path | str) -> tuple[Geqdsk, GFilePayload]:
    geqdsk = Geqdsk(path)

    with path.open("r", encoding="utf-8") as file:
        for _ in range(5):
            file.readline()
        payload = _sanitize_line(file.read().replace("\n", " "))

    fields = re.split(r"\s+", payload.strip())
    data = np.array([_safe_float_conversion(value) for value in fields if value], dtype=np.float64)
    nr = geqdsk.nr
    nz = geqdsk.nz
    idx = 0

    fpol = np.nan_to_num(data[idx : idx + nr], nan=0.0, copy=True)
    idx += nr
    pres = np.nan_to_num(data[idx : idx + nr], nan=0.0, copy=True)
    idx += nr
    ffprim = np.nan_to_num(data[idx : idx + nr], nan=0.0, copy=True)
    idx += nr
    pprime = np.nan_to_num(data[idx : idx + nr], nan=0.0, copy=True)
    idx += nr
    psirz = np.nan_to_num(data[idx : idx + nr * nz], nan=0.0, copy=True).reshape(nz, nr)
    idx += nr * nz
    qpsi = np.nan_to_num(data[idx : idx + nr], nan=0.0, copy=True)
    return geqdsk, GFilePayload(fpol=fpol, pres=pres, ffprim=ffprim, pprime=pprime, psirz=psirz, qpsi=qpsi)


def _boundary_signature(
    *,
    gfile_path: Path,
    geqdsk: Geqdsk,
    boundary_m: int | None,
    boundary_maxtol: float,
) -> dict[str, object]:
    stat = gfile_path.stat()
    boundary_n = None if boundary_m is None else int(boundary_m) + 1
    return {
        "gfile": str(gfile_path.resolve()),
        "gfile_size": int(stat.st_size),
        "gfile_mtime_ns": int(stat.st_mtime_ns),
        "boundary_m": None if boundary_m is None else int(boundary_m),
        "boundary_n": None if boundary_n is None else int(boundary_n),
        "boundary_maxtol": float(boundary_maxtol),
        "Bt0": float(geqdsk.Bt0),
        "nr": int(geqdsk.nr),
        "nz": int(geqdsk.nz),
    }


def _boundary_from_fit(geqdsk: Geqdsk, fit: Mapping[str, object]) -> tuple[Boundary, dict[str, float | np.ndarray]]:
    normalized = {
        "M": int(fit["M"]),
        "N": int(fit["N"]),
        "rms": float(fit["rms"]),
        "max_curve_error": float(fit["max_curve_error"]),
        "a": float(fit["a"]),
        "R0": float(fit["R0"]),
        "Z0": float(fit["Z0"]),
        "ka": float(fit["ka"]),
        "c_offsets": np.asarray(fit["c_offsets"], dtype=np.float64),
        "s_offsets": np.asarray(fit["s_offsets"], dtype=np.float64),
    }
    boundary = Boundary(
        a=float(normalized["a"]),
        R0=float(normalized["R0"]),
        Z0=float(normalized["Z0"]),
        B0=float(geqdsk.Bt0),
        ka=float(normalized["ka"]),
        c_offsets=np.asarray(normalized["c_offsets"], dtype=np.float64),
        s_offsets=np.asarray(normalized["s_offsets"], dtype=np.float64),
    )
    return boundary, normalized


def build_boundary(
    gfile_path: Path,
    geqdsk: Geqdsk,
    *,
    boundary_m: int | None,
    boundary_maxtol: float,
) -> tuple[Boundary, dict[str, float | np.ndarray]]:
    cache_path = boundary_cache_path(gfile_path)
    signature = _boundary_signature(
        gfile_path=gfile_path,
        geqdsk=geqdsk,
        boundary_m=boundary_m,
        boundary_maxtol=boundary_maxtol,
    )
    if cache_path.exists():
        try:
            cached = json.loads(cache_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            cached = None
        if isinstance(cached, dict) and cached.get("signature") == signature and isinstance(cached.get("fit"), dict):
            try:
                return _boundary_from_fit(geqdsk, cached["fit"])
            except (KeyError, TypeError, ValueError):
                pass

    boundary_n = None if boundary_m is None else int(boundary_m) + 1
    fit = geqdsk.fit_boundary_params(
        M=boundary_m, N=boundary_n, maxtol=boundary_maxtol, R0=None, Z0=None, a=None, ka=None
    )
    boundary, normalized = _boundary_from_fit(geqdsk, fit)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_payload = {
        "signature": signature,
        "fit": {
            "M": int(normalized["M"]),
            "N": int(normalized["N"]),
            "rms": float(normalized["rms"]),
            "max_curve_error": float(normalized["max_curve_error"]),
            "a": float(normalized["a"]),
            "R0": float(normalized["R0"]),
            "Z0": float(normalized["Z0"]),
            "ka": float(normalized["ka"]),
            "c_offsets": np.asarray(normalized["c_offsets"], dtype=np.float64).tolist(),
            "s_offsets": np.asarray(normalized["s_offsets"], dtype=np.float64).tolist(),
        },
    }
    tmp_path = cache_path.with_suffix(f"{cache_path.suffix}.tmp")
    tmp_path.write_text(json.dumps(cache_payload, indent=2), encoding="utf-8")
    tmp_path.replace(cache_path)
    return boundary, normalized


def build_boundary_curve(boundary: Boundary, *, rho: float = 1.0, n: int = 512) -> np.ndarray:
    theta = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    theta_bar = theta + boundary.c_offsets[0]
    for order in range(1, boundary.c_offsets.size):
        theta_bar += boundary.c_offsets[order] * np.cos(order * theta)
    for order in range(1, boundary.s_offsets.size):
        theta_bar += boundary.s_offsets[order] * np.sin(order * theta)
    return np.column_stack(
        (
            boundary.R0 + boundary.a * rho * np.cos(theta_bar),
            boundary.Z0 - boundary.a * rho * boundary.ka * np.sin(theta),
        )
    )


def close_curve(points: np.ndarray) -> np.ndarray:
    arr = np.asarray(points, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError(f"Expected an (N, 2) curve, got {arr.shape}")
    return np.vstack((arr, arr[:1]))


def curve_distance_metrics(points_a: np.ndarray, points_b: np.ndarray) -> dict[str, float]:
    dist = np.sqrt(np.sum((np.asarray(points_a)[:, None, :] - np.asarray(points_b)[None, :, :]) ** 2, axis=2))
    nearest_a = dist.min(axis=1)
    nearest_b = dist.min(axis=0)
    return {
        "hausdorff": float(max(nearest_a.max(), nearest_b.max())),
        "rms": float(np.sqrt(0.5 * (np.mean(nearest_a**2) + np.mean(nearest_b**2)))),
    }


def extract_gfile_surfaces(geqdsk: Geqdsk, payload: GFilePayload, levels: list[float]) -> dict[float, np.ndarray]:
    psi_span = float(geqdsk.psi_bound - geqdsk.psi_axis)
    if abs(psi_span) < 1.0e-14:
        raise ValueError("GEQDSK psi_axis and psi_bound are too close to normalize")
    psin_grid = (np.asarray(payload.psirz, dtype=np.float64) - float(geqdsk.psi_axis)) / psi_span
    R = np.linspace(geqdsk.Rmin, geqdsk.Rmax, geqdsk.nr, dtype=np.float64)
    Z = np.linspace(geqdsk.Zmin, geqdsk.Zmax, geqdsk.nz, dtype=np.float64)

    surfaces: dict[float, np.ndarray] = {}
    contour_levels = [level for level in levels if level < 1.0 - 1.0e-12]
    if contour_levels:
        fig, ax = plt.subplots()
        contour = ax.contour(R, Z, psin_grid, levels=contour_levels)
        plt.close(fig)
        for idx, level in enumerate(contour_levels):
            candidates = [
                np.asarray(segment, dtype=np.float64) for segment in contour.allsegs[idx] if len(segment) >= 8
            ]
            if candidates:
                surfaces[level] = max(candidates, key=len)
    if any(abs(level - 1.0) <= 1.0e-12 for level in levels):
        surfaces[1.0] = np.asarray(geqdsk.boundary, dtype=np.float64)
    return surfaces


def build_solver_case(boundary: Boundary, payload: GFilePayload, geqdsk: Geqdsk, config: CompareConfig) -> OperatorCase:
    return OperatorCase(
        route="PF",
        coordinate="psin",
        nodes="uniform",
        profile_coeffs={name: list(values) for name, values in config.profile_coeffs.items()},
        boundary=boundary,
        heat_input=MU0 * np.asarray(payload.pprime, dtype=np.float64),
        current_input=np.asarray(payload.ffprim, dtype=np.float64),
        Ip=float(geqdsk.I_total),
    )


def solve_equilibrium(case: OperatorCase, config: CompareConfig) -> tuple[Solver, object, object]:
    solve_grid = Grid(Nr=config.nr, Nt=config.nt, scheme="legendre")
    plot_grid = Grid(
        Nr=max(config.plot_nr, config.nr),
        Nt=max(config.plot_nt, config.nt),
        scheme="uniform",
        L_max=solve_grid.L_max,
        M_max=solve_grid.M_max,
    )
    solver = Solver(
        operator=Operator(solve_grid, case),
        config=SolverConfig(
            method=config.solver_method,
            root_maxfev=config.solver_maxfev,
            enable_warmstart=False,
            enable_fallback=False,
            enable_verbose=False,
            enable_history=False,
        ),
    )
    solver.solve(enable_verbose=False, enable_history=False, enable_warmstart=False, enable_fallback=False)
    equilibrium = solver.build_equilibrium()
    return solver, equilibrium, equilibrium.resample(target_grid=plot_grid)


def build_surface_from_psin(equilibrium, level: float) -> np.ndarray:
    psin = np.asarray(equilibrium.psin, dtype=np.float64)
    rho = np.asarray(equilibrium.rho, dtype=np.float64)
    order = np.argsort(psin)
    psin_unique, unique_idx = np.unique(psin[order], return_index=True)
    rho_level = float(np.interp(level, psin_unique, rho[order][unique_idx]))
    geometry = equilibrium.geometry
    R = np.array(
        [np.interp(rho_level, rho, geometry.R[:, idx]) for idx in range(equilibrium.grid.Nt)], dtype=np.float64
    )
    Z = np.array(
        [np.interp(rho_level, rho, geometry.Z[:, idx]) for idx in range(equilibrium.grid.Nt)], dtype=np.float64
    )
    return np.column_stack((R, Z))


def collect_surface_metrics(
    geqdsk_surfaces: dict[float, np.ndarray], equilibrium, levels: list[float]
) -> tuple[dict[float, np.ndarray], dict[float, dict[str, float]]]:
    veqpy_surfaces: dict[float, np.ndarray] = {}
    metrics: dict[float, dict[str, float]] = {}
    for level in levels:
        if level not in geqdsk_surfaces:
            continue
        veqpy_surface = build_surface_from_psin(equilibrium, level)
        veqpy_surfaces[level] = veqpy_surface
        metrics[level] = curve_distance_metrics(geqdsk_surfaces[level], veqpy_surface)
    return veqpy_surfaces, metrics


def extract_active_coeffs(solver: Solver) -> dict[str, list[float]]:
    coeffs = solver.build_coeffs(include_none=False)
    shape_names = {"psin", *build_shape_profile_names(solver.operator.grid.M_max)}
    active: dict[str, list[float]] = {}
    for name, values in coeffs.items():
        if name in shape_names and values is not None:
            arr = np.asarray(values, dtype=np.float64)
            if arr.size:
                active[name] = arr.astype(float, copy=False).tolist()
    return active


def build_summary_lines(
    *,
    gfile_path: Path,
    fit: Mapping[str, float | np.ndarray],
    solver: Solver,
    equilibrium,
    surface_metrics: dict[float, dict[str, float]],
) -> list[str]:
    result = solver.result
    if result is None:
        raise RuntimeError("solver.result is unavailable after solve")
    lines = [
        f"gfile: {gfile_path.name}",
        f"boundary fit: M={int(fit['M'])}, N={int(fit['N'])}",
        f"boundary rms: {float(fit['rms']):.3e}",
        f"boundary max curve error: {float(fit['max_curve_error']):.3e}",
        f"boundary a: {float(fit['a']):.6e}",
        f"boundary R0: {float(fit['R0']):.6e}",
        f"boundary Z0: {float(fit['Z0']):.6e}",
        f"boundary ka: {float(fit['ka']):.6e}",
        "",
        f"solver method: {solver.config.method}",
        f"solver success: {bool(result.success)}",
        f"solver residual: {float(result.residual_norm_final):.3e}",
        f"solver nfev: {int(result.nfev)}",
        f"solve elapsed: {float(result.elapsed) / 1000.0:.3f} ms",
        f"target Ip [A]: {float(solver.operator.case.Ip):.6e}",
        f"solved Ip [A]: {float(equilibrium.Ip):.6e}",
        f"solved beta_t: {float(equilibrium.beta_t):.6e}",
        f"alpha1: {float(equilibrium.alpha1):.6e}",
        f"alpha2: {float(equilibrium.alpha2):.6e}",
        "",
        "surface mismatch:",
    ]
    if not surface_metrics:
        lines.append("  no closed gfile contours found for the requested psin levels")
    else:
        lines.extend(
            f"  psin={level:.2f}: hausdorff={metrics['hausdorff']:.3e}, rms={metrics['rms']:.3e}"
            for level, metrics in sorted(surface_metrics.items())
        )
    return lines


def render_plot(
    *,
    plot_path: Path,
    payload: GFilePayload,
    boundary: Boundary,
    geqdsk: Geqdsk,
    geqdsk_surfaces: dict[float, np.ndarray],
    veqpy_surfaces: dict[float, np.ndarray],
    summary_lines: list[str],
) -> None:
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(14, 10))
    grid = fig.add_gridspec(2, 2, height_ratios=(1.0, 1.25), hspace=0.24, wspace=0.22)
    ax_boundary = fig.add_subplot(grid[0, 0])
    ax_profiles = fig.add_subplot(grid[0, 1])
    ax_surfaces = fig.add_subplot(grid[1, 0])
    ax_text = fig.add_subplot(grid[1, 1])

    original_boundary = np.asarray(geqdsk.boundary, dtype=np.float64)
    fitted_boundary = build_boundary_curve(boundary, rho=1.0)
    ax_boundary.plot(
        close_curve(original_boundary)[:, 0],
        close_curve(original_boundary)[:, 1],
        color="black",
        linestyle="--",
        linewidth=1.4,
        label="gfile boundary",
    )
    ax_boundary.plot(
        close_curve(fitted_boundary)[:, 0],
        close_curve(fitted_boundary)[:, 1],
        color="#d62728",
        linewidth=1.5,
        label="Boundary.from_geqdsk",
    )
    ax_boundary.set_title("(a) Boundary Fit")
    ax_boundary.set_xlabel("R [m]")
    ax_boundary.set_ylabel("Z [m]")
    ax_boundary.set_aspect("equal")
    ax_boundary.grid(True, linestyle=":", alpha=0.35)
    ax_boundary.legend(loc="best")

    psin_axis = np.linspace(0.0, 1.0, geqdsk.nr, dtype=np.float64)
    ax_profiles.plot(psin_axis, payload.ffprim, color="#1f77b4", linewidth=1.5, label="FFPrime")
    ax_profiles.plot(psin_axis, MU0 * payload.pprime, color="#ff7f0e", linewidth=1.5, label="mu0 * PPrime")
    ax_profiles.set_title("(b) Uniform-psin Source Inputs")
    ax_profiles.set_xlabel("psin")
    ax_profiles.set_ylabel("Input value")
    ax_profiles.grid(True, linestyle=":", alpha=0.35)
    ax_profiles.legend(loc="best")

    sorted_levels = sorted(veqpy_surfaces)
    for idx, level in enumerate(sorted_levels):
        linewidth = 1.0 if idx < len(sorted_levels) - 1 else 1.35
        ax_surfaces.plot(
            close_curve(geqdsk_surfaces[level])[:, 0],
            close_curve(geqdsk_surfaces[level])[:, 1],
            linestyle="--",
            color="black",
            linewidth=linewidth,
            alpha=0.85,
        )
        ax_surfaces.plot(
            close_curve(veqpy_surfaces[level])[:, 0],
            close_curve(veqpy_surfaces[level])[:, 1],
            linestyle="-",
            color="#d62728",
            linewidth=linewidth,
            alpha=0.9,
        )
    ax_surfaces.set_title("(c) Closed Flux Surfaces (dashed=gfile, solid=veqpy)")
    ax_surfaces.set_xlabel("R [m]")
    ax_surfaces.set_ylabel("Z [m]")
    ax_surfaces.set_aspect("equal")
    ax_surfaces.grid(True, linestyle=":", alpha=0.35)

    ax_text.axis("off")
    ax_text.set_title("(d) Summary", loc="left")
    ax_text.text(0.0, 1.0, "\n".join(summary_lines), ha="left", va="top", family="monospace", fontsize=10)
    fig.savefig(plot_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def write_summary_json(
    *,
    summary_path: Path,
    gfile_path: Path,
    fit: Mapping[str, float | np.ndarray],
    solver: Solver,
    equilibrium,
    active_coeffs: dict[str, list[float]],
    surface_metrics: dict[float, dict[str, float]],
    levels: list[float],
) -> None:
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    result = solver.result
    if result is None:
        raise RuntimeError("solver.result is unavailable after solve")
    payload = {
        "gfile": str(gfile_path.resolve()),
        "compare_levels": [float(level) for level in levels],
        "grid": {
            "Nr": int(solver.operator.grid.Nr),
            "Nt": int(solver.operator.grid.Nt),
            "scheme": solver.operator.grid.scheme,
        },
        "boundary_fit": {
            "M": int(fit["M"]),
            "N": int(fit["N"]),
            "rms": float(fit["rms"]),
            "max_curve_error": float(fit["max_curve_error"]),
            "a": float(fit["a"]),
            "R0": float(fit["R0"]),
            "Z0": float(fit["Z0"]),
            "ka": float(fit["ka"]),
            "c_offsets": np.asarray(fit["c_offsets"], dtype=np.float64).tolist(),
            "s_offsets": np.asarray(fit["s_offsets"], dtype=np.float64).tolist(),
        },
        "solver": {
            "method": solver.config.method,
            "success": bool(result.success),
            "message": str(result.message),
            "residual_norm_final": float(result.residual_norm_final),
            "nfev": int(result.nfev),
            "njev": int(result.njev),
            "nit": int(result.nit),
            "elapsed_us": float(result.elapsed),
        },
        "equilibrium": {
            "Ip": float(equilibrium.Ip),
            "beta_t": float(equilibrium.beta_t),
            "alpha1": float(equilibrium.alpha1),
            "alpha2": float(equilibrium.alpha2),
        },
        "active_coeffs": active_coeffs,
        "surface_metrics": {
            f"{level:.2f}": {"hausdorff": float(metrics["hausdorff"]), "rms": float(metrics["rms"])}
            for level, metrics in sorted(surface_metrics.items())
        },
    }
    summary_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def run_compare(config: CompareConfig) -> CompareResult:
    levels = normalize_levels(config.levels, config.surface_count)
    geqdsk, payload = read_gfile_payload(config.gfile_path)
    boundary, fit = build_boundary(
        config.gfile_path,
        geqdsk,
        boundary_m=config.boundary_fit_m,
        boundary_maxtol=config.boundary_maxtol,
    )
    case = build_solver_case(boundary, payload, geqdsk, config)
    solver, equilibrium, plot_equilibrium = solve_equilibrium(case, config)
    geqdsk_surfaces = extract_gfile_surfaces(geqdsk, payload, levels)
    veqpy_surfaces, surface_metrics = collect_surface_metrics(geqdsk_surfaces, plot_equilibrium, levels)
    summary_lines = build_summary_lines(
        gfile_path=config.gfile_path,
        fit=fit,
        solver=solver,
        equilibrium=equilibrium,
        surface_metrics=surface_metrics,
    )
    render_plot(
        plot_path=config.plot_path,
        payload=payload,
        boundary=boundary,
        geqdsk=geqdsk,
        geqdsk_surfaces=geqdsk_surfaces,
        veqpy_surfaces=veqpy_surfaces,
        summary_lines=summary_lines,
    )
    write_summary_json(
        summary_path=config.summary_path,
        gfile_path=config.gfile_path,
        fit=fit,
        solver=solver,
        equilibrium=equilibrium,
        active_coeffs=extract_active_coeffs(solver),
        surface_metrics=surface_metrics,
        levels=levels,
    )

    result = solver.result
    if result is None:
        raise RuntimeError("solver.result is unavailable after solve")
    cache_path = boundary_cache_path(config.gfile_path)
    print(f"gfile={config.gfile_path.resolve()}")
    print(f"plot={config.plot_path.resolve()}")
    print(f"summary={config.summary_path.resolve()}")
    print(f"boundary_cache={cache_path.resolve()}")
    print(f"boundary_fit_rms={float(fit['rms']):.6e}")
    print(f"solver_residual={float(result.residual_norm_final):.6e}")
    print(f"solve_elapsed_ms={float(result.elapsed) / 1000.0:.6f}")
    print(f"target_Ip={float(case.Ip):.6e}")
    print(f"solved_Ip={float(equilibrium.Ip):.6e}")
    for level, metrics in sorted(surface_metrics.items()):
        print(f"psin={level:.2f}: hausdorff={metrics['hausdorff']:.6e}, rms={metrics['rms']:.6e}")
    return CompareResult(
        gfile_path=config.gfile_path.resolve(),
        plot_path=config.plot_path.resolve(),
        summary_path=config.summary_path.resolve(),
        boundary_cache_path=cache_path.resolve(),
        fit_rms=float(fit["rms"]),
        solver_residual=float(result.residual_norm_final),
        solve_elapsed_ms=float(result.elapsed) / 1000.0,
        target_ip=float(case.Ip),
        solved_ip=float(equilibrium.Ip),
        surface_metrics=surface_metrics,
    )

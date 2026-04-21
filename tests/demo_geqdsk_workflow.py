"""No-argument GEQDSK -> veqpy workflow demo.

This script shows the intended user flow:
1. read an EFIT GEQDSK,
2. fit a veqpy boundary from it,
3. solve a veqpy equilibrium,
4. compare magnetic surfaces in one simple figure.

Note: The first run may be slower due to JIT compilation.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from veqpy.model import Boundary, Geqdsk, Grid
from veqpy.operator import Operator, OperatorCase
from veqpy.solver import Solver, SolverConfig

MU0 = 4.0e-7 * np.pi
BOUNDARY_FIT_M = 10
BOUNDARY_FIT_N = 11
DEFAULT_LEVELS = tuple(np.linspace(0.1, 1.0, 10, dtype=np.float64))


def ensure_output_dir() -> Path:
    outdir = Path(__file__).resolve().parent / "demo"
    outdir.mkdir(parents=True, exist_ok=True)
    return outdir


def close_curve(points: np.ndarray) -> np.ndarray:
    arr = np.asarray(points, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError(f"Expected an (N, 2) curve, got {arr.shape}")
    return np.vstack((arr, arr[:1]))


def compute_rz_limits(
    curves: list[np.ndarray], *, pad_fraction: float = 0.06
) -> tuple[tuple[float, float], tuple[float, float]]:
    valid = []
    for curve in curves:
        arr = np.asarray(curve, dtype=np.float64)
        if arr.ndim == 2 and arr.shape[1] == 2 and arr.size:
            valid.append(arr)
    if not valid:
        raise ValueError("At least one non-empty curve is required")
    stacked = np.vstack(valid)
    r_min = float(np.min(stacked[:, 0]))
    r_max = float(np.max(stacked[:, 0]))
    z_min = float(np.min(stacked[:, 1]))
    z_max = float(np.max(stacked[:, 1]))
    r_pad = max((r_max - r_min) * pad_fraction, 1.0e-3)
    z_pad = max((z_max - z_min) * pad_fraction, 1.0e-3)
    return (r_min - r_pad, r_max + r_pad), (z_min - z_pad, z_max + z_pad)


def build_surface_from_psin(equilibrium, level: float) -> np.ndarray:
    psin = np.asarray(equilibrium.psin, dtype=np.float64)
    rho = np.asarray(equilibrium.rho, dtype=np.float64)
    order = np.argsort(psin)
    psin_unique, unique_idx = np.unique(psin[order], return_index=True)
    rho_level = float(np.interp(float(level), psin_unique, rho[order][unique_idx]))
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


def build_geqdsk_surfaces(geqdsk: Geqdsk, *, levels: tuple[float, ...]) -> dict[float, np.ndarray]:
    psi_span = float(geqdsk.psi_bound - geqdsk.psi_axis)
    if abs(psi_span) <= 1.0e-14:
        raise ValueError("GEQDSK psi_axis and psi_bound are too close to normalize")
    psin_grid = (np.asarray(geqdsk.psi, dtype=np.float64).T - float(geqdsk.psi_axis)) / psi_span
    R = np.linspace(geqdsk.Rmin, geqdsk.Rmax, geqdsk.NR, dtype=np.float64)
    Z = np.linspace(geqdsk.Zmin, geqdsk.Zmax, geqdsk.NZ, dtype=np.float64)

    surfaces: dict[float, np.ndarray] = {}
    contour_levels = [float(level) for level in levels if level < 1.0 - 1.0e-12]
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


def build_profile_coeffs() -> dict[str, list[float]]:
    coeffs: dict[str, list[float]] = {
        "psin": [0.0] * 10,
        "h": [0.0] * 10,
        "k": [0.0] * 10,
        "v": [0.0] * 10,
    }
    for order in range(8):
        coeffs[f"c{order}"] = [0.0] * 5
    for order in range(1, 9):
        coeffs[f"s{order}"] = [0.0] * 5
    return coeffs


def style_surface_axis(ax: plt.Axes, *, title: str, rz_limits: tuple[tuple[float, float], tuple[float, float]]) -> None:
    ax.set_title(title)
    ax.set_xlabel("R [m]")
    ax.set_ylabel("Z [m]")
    ax.set_xlim(*rz_limits[0])
    ax.set_ylim(*rz_limits[1])
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, linestyle=":", alpha=0.35)
    ax.tick_params(direction="in", top=True, right=True)


def main() -> None:
    tests_dir = Path(__file__).resolve().parent
    gfile_path = tests_dir / "EFIT.geqdsk"
    outdir = ensure_output_dir()
    figure_path = outdir / "demo_geqdsk_workflow.png"
    equilibrium_path = outdir / "demo_geqdsk_equilibrium.json"

    geqdsk = Geqdsk(path=gfile_path)
    # Use an explicit M/N here so the fitted boundary is deterministic.
    boundary = Boundary.from_geqdsk(
        geqdsk,
        M=BOUNDARY_FIT_M,
        N=BOUNDARY_FIT_N,
        maxtol=1.0e-2,
    )
    case = OperatorCase(
        route="PF",
        coordinate="psin",
        nodes="uniform",
        profile_coeffs=build_profile_coeffs(),
        boundary=boundary,
        heat_input=MU0 * np.asarray(geqdsk.P_psi, dtype=np.float64),
        current_input=np.asarray(geqdsk.FF_psi, dtype=np.float64),
        Ip=MU0 * float(geqdsk.Ip),
    )
    solve_grid = Grid(Nr=32, Nt=32, scheme="legendre")
    plot_grid = Grid(Nr=128, Nt=256, scheme="uniform", L_max=solve_grid.L_max, M_max=solve_grid.M_max)
    solver = Solver(
        operator=Operator(grid=solve_grid, case=case),
        config=SolverConfig(
            method="hybr",
            root_maxfev=2000,
            enable_warmstart=False,
            enable_fallback=False,
            enable_verbose=False,
            enable_history=False,
        ),
    )

    for _ in range(10):
        solver.solve()
        solver.reset()

    solver.solve(enable_verbose=False, enable_history=False, enable_warmstart=False, enable_fallback=False)
    print(solver.result)
    equilibrium = solver.build_equilibrium()
    plot_equilibrium = equilibrium.resample(grid=plot_grid)

    geqdsk_surfaces = build_geqdsk_surfaces(geqdsk, levels=DEFAULT_LEVELS)
    shared_levels = [level for level in DEFAULT_LEVELS if level in geqdsk_surfaces]
    veqpy_surfaces = {float(level): build_surface_from_psin(plot_equilibrium, float(level)) for level in shared_levels}
    rz_limits = compute_rz_limits(list(geqdsk_surfaces.values()) + list(veqpy_surfaces.values()))

    fig, ax = plt.subplots(figsize=(7.6, 6.8), constrained_layout=True)
    for index, level in enumerate(shared_levels):
        linewidth = 1.0 if level < 1.0 - 1.0e-12 else 1.35
        ax.plot(
            close_curve(geqdsk_surfaces[level])[:, 0],
            close_curve(geqdsk_surfaces[level])[:, 1],
            linestyle="--",
            color="black",
            linewidth=linewidth * 1.75,
            label="EFIT" if index == 0 else None,
        )
        ax.plot(
            close_curve(veqpy_surfaces[level])[:, 0],
            close_curve(veqpy_surfaces[level])[:, 1],
            linestyle="-",
            color="#d62728",
            linewidth=linewidth,
            label="veqpy" if index == 0 else None,
        )
    ax.scatter(
        [boundary.R0],
        [boundary.Z0],
        marker="x",
        color="#d62728",
        s=42,
        linewidths=1.4,
        label="Boundary (R0, Z0)",
    )
    style_surface_axis(ax, title="EFIT vs veqpy Flux Surfaces", rz_limits=rz_limits)
    ax.legend(loc="upper right")
    fig.savefig(figure_path, dpi=220)
    plt.close(fig)

    equilibrium.write(str(equilibrium_path))

    result = solver.result
    if result is None:
        raise RuntimeError("solver.result is unavailable after GEQDSK workflow solve")

    print(f"Read GEQDSK       : {gfile_path}")
    print(f"Saved figure      : {figure_path}")
    print(f"Saved equilibrium : {equilibrium_path}")
    print(
        "Boundary fit      : "
        f"M={BOUNDARY_FIT_M}, N={BOUNDARY_FIT_N}, "
        f"R0={boundary.R0:.6f}, Z0={boundary.Z0:.6f}, a={boundary.a:.6f}, ka={boundary.ka:.6f}"
    )
    print(f"Solver success    : {result.success}")
    print(f"Residual norm     : {result.residual_norm_final:.6e}")
    print(f"Ip target [MA]    : {geqdsk.Ip / 1.0e6:.6f}")
    print(f"Ip solved [MA]    : {equilibrium.Ip / 1.0e6:.6f}")


if __name__ == "__main__":
    main()

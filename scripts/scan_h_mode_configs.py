"""
Scan initial_policy × residual_normalization for H-mode (CHEASE) hybr solve.
Round 2: fix normalization names, add balance, try fallback + larger maxfev.
"""
from __future__ import annotations

import sys, itertools, warnings
from pathlib import Path
from time import perf_counter

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from veqpy.model import Boundary, Geqdsk, Grid
from veqpy.model.boundary import _fit_boundary_params
from veqpy.operator import Operator, OperatorCase
from veqpy.solver import Solver, SolverConfig

warnings.filterwarnings("ignore")

MU0 = 4.0e-7 * np.pi
SOLVE_NR, SOLVE_NT = 64, 64

COEFFS = {
    "psin": [0.0] * 10, "h": [0.0] * 10, "k": [0.0] * 10, "v": [0.0] * 10,
    "c0": [0.0] * 5, "c1": [0.0] * 5, "c2": [0.0] * 5, "c3": [0.0] * 5,
    "c4": [0.0] * 5, "c5": [0.0] * 5, "c6": [0.0] * 5, "c7": [0.0] * 5,
    "c8": [0.0] * 5, "c9": [0.0] * 5,
    "s1": [0.0] * 5, "s2": [0.0] * 5, "s3": [0.0] * 5, "s4": [0.0] * 5,
    "s5": [0.0] * 5, "s6": [0.0] * 5, "s7": [0.0] * 5, "s8": [0.0] * 5,
    "s9": [0.0] * 5, "s10": [0.0] * 5,
}


def build_case():
    geqdsk = Geqdsk()
    geqdsk.read_geqdsk("data/CHEASE.geqdsk")
    fit = _fit_boundary_params(geqdsk, M=10, N=10, maxtol=1.0,
                               R0=None, Z0=None, a=None, ka=None)
    boundary = Boundary(
        a=float(fit["a"]), R0=float(fit["R0"]), Z0=float(fit["Z0"]),
        B0=float(geqdsk.Bt0), ka=float(fit["ka"]),
        c_offsets=np.asarray(fit["c_offsets"], dtype=np.float64),
        s_offsets=np.asarray(fit["s_offsets"], dtype=np.float64),
    )
    return OperatorCase(
        route="PF", coordinate="psin", nodes="uniform",
        profile_coeffs={k: list(v) for k, v in COEFFS.items()},
        boundary=boundary,
        heat_input=MU0 * np.asarray(geqdsk.P_psi, dtype=np.float64),
        current_input=np.asarray(geqdsk.FF_psi, dtype=np.float64),
        Ip=MU0 * float(geqdsk.Ip),
    )


def solve_one(*, initial_policy, homothetic_lambda, residual_normalization,
              maxfev=3000, enable_fallback=False):
    case = build_case()
    grid = Grid(Nr=SOLVE_NR, Nt=SOLVE_NT, quadrature_scheme="chebyshev")
    op = Operator(grid, case)
    solver = Solver(
        operator=op,
        config=SolverConfig(
            method="hybr",
            max_evaluations=maxfev,
            enable_warmstart=False,
            enable_fallback=enable_fallback,
            fallback_methods=("lm", "trf") if enable_fallback else (),
            enable_verbose=False,
            enable_history=False,
            initial_policy=initial_policy,
            initial_homothetic_lambda=homothetic_lambda,
            residual_normalization=residual_normalization,
        ),
    )
    t0 = perf_counter()
    try:
        solver.solve(enable_verbose=False, enable_history=False,
                     enable_warmstart=False, enable_fallback=enable_fallback)
        elapsed = (perf_counter() - t0) * 1e3
    except Exception as exc:
        return {"success": False, "nfev": 0, "|res|_final": float("nan"),
                "elapsed_ms": 0, "error": f"{type(exc).__name__}: {exc}"}

    r = solver.result
    return {"success": r.success, "nfev": r.function_evaluations,
            "|res|_final": r.residual_norm_final, "elapsed_ms": elapsed,
            "error": None if r.success else r.message[:80]}


def main():
    results = []
    policies = [
        ("homothetic", [0.25, 0.5, 0.75, 1.0]),
        ("zeros", [1.0]),
    ]
    norms = ["none", "fast", "balance"]
    fallback_opts = [False, True]
    maxfev_opts = [3000, 5000]

    combos = list(itertools.product(policies, norms, fallback_opts, maxfev_opts))
    print(f"Total configs: {len(combos)}")

    for (ip, lambdas), rn, use_fb, mf in combos:
        for lam in lambdas:
            fb_str = "+FB" if use_fb else ""
            label = f"policy={ip}"
            if ip == "homothetic":
                label += f",λ={lam:.2f}"
            label += f",norm={rn}{fb_str},maxfev={mf}"
            print(f"Running: {label} ...", end=" ", flush=True)
            rec = solve_one(initial_policy=ip, homothetic_lambda=lam,
                           residual_normalization=rn, maxfev=mf,
                           enable_fallback=use_fb)
            rec["label"] = label
            results.append(rec)
            status = "✓ CONVERGED" if rec["success"] else "✗ failed"
            print(f"{status}  nfev={rec['nfev']:5d}  |res|={rec['|res|_final']:.3e}  "
                  f"{rec['elapsed_ms']:.0f}ms")

    # Summary
    print("\n" + "=" * 110)
    print(f"{'Configuration':60s} {'Conv':>6s} {'nfev':>6s} {'|res|_final':>14s} {'ms':>8s}")
    print("-" * 110)
    for rec in sorted(results, key=lambda r: (0 if r["success"] else 1, r["|res|_final"])):
        conv = "YES" if rec["success"] else "NO"
        res_str = f"{rec['|res|_final']:.3e}" if np.isfinite(rec["|res|_final"]) else "inf"
        print(f"{rec['label']:60s} {conv:>6s} {rec['nfev']:>6d} {res_str:>14s} {rec['elapsed_ms']:>7.0f}ms")
    print("-" * 110)
    converged = [r for r in results if r["success"]]
    print(f"Converged: {len(converged)}/{len(results)}")
    for r in converged:
        print(f"  {r['label']}")


if __name__ == "__main__":
    main()

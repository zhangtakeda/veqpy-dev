"""
Diagnostic: probe H-mode (CHEASE) hybr solve failure from 07-pareto-analysis.py.

Tracks nfev, residual norm, geometry metrics, and shape parameters at each
function evaluation to locate the blowup.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from veqpy.model import Boundary, Geqdsk, Grid
from veqpy.model.boundary import _fit_boundary_params
from veqpy.operator import Operator, OperatorCase

MU0 = 4.0e-7 * np.pi

# --- replicate 07-pareto-analysis.py H-mode configuration ---
SOLVE_NR = 64
SOLVE_NT = 64
SOLVER_MAXFEV = 2000
BOUNDARY_FIT_M = 10
BOUNDARY_FIT_N = 10
BOUNDARY_MAXTOL = 1.0

CHEASE_PROFILE_COEFFS = {
    "psin": [0.0] * 10, "h": [0.0] * 10, "k": [0.0] * 10, "v": [0.0] * 10,
    "c0": [0.0] * 5, "c1": [0.0] * 5, "c2": [0.0] * 5, "c3": [0.0] * 5,
    "c4": [0.0] * 5, "c5": [0.0] * 5, "c6": [0.0] * 5, "c7": [0.0] * 5,
    "c8": [0.0] * 5, "c9": [0.0] * 5,
    "s1": [0.0] * 5, "s2": [0.0] * 5, "s3": [0.0] * 5, "s4": [0.0] * 5,
    "s5": [0.0] * 5, "s6": [0.0] * 5, "s7": [0.0] * 5, "s8": [0.0] * 5,
    "s9": [0.0] * 5, "s10": [0.0] * 5,
}


def load_case():
    geqdsk = Geqdsk()
    geqdsk.read_geqdsk("data/CHEASE.geqdsk")
    fit = _fit_boundary_params(
        geqdsk, M=BOUNDARY_FIT_M, N=BOUNDARY_FIT_N, maxtol=BOUNDARY_MAXTOL,
        R0=None, Z0=None, a=None, ka=None,
    )
    boundary = Boundary(
        a=float(fit["a"]), R0=float(fit["R0"]), Z0=float(fit["Z0"]),
        B0=float(geqdsk.Bt0), ka=float(fit["ka"]),
        c_offsets=np.asarray(fit["c_offsets"], dtype=np.float64),
        s_offsets=np.asarray(fit["s_offsets"], dtype=np.float64),
    )
    case = OperatorCase(
        route="PF", coordinate="psin", nodes="uniform",
        profile_coeffs={k: list(v) for k, v in CHEASE_PROFILE_COEFFS.items()},
        boundary=boundary,
        heat_input=MU0 * np.asarray(geqdsk.P_psi, dtype=np.float64),
        current_input=np.asarray(geqdsk.FF_psi, dtype=np.float64),
        Ip=MU0 * float(geqdsk.Ip),
    )
    return case, boundary, fit, geqdsk


class DiagnosticOperator:
    """Wraps an Operator, logging geometry metrics after each residual evaluation."""

    def __init__(self, operator: Operator):
        self._op = operator
        self.nfev = 0
        self.blowup_nfev = None
        self.history = []

    def __call__(self, x: np.ndarray, *args, **kwargs) -> np.ndarray:
        self.nfev += 1
        nfev = self.nfev
        out = np.asarray(self._op(x, *args, **kwargs), dtype=np.float64)

        gw = self._op.geometry_workspace
        J = gw.surface_fields[4]
        JdivR = gw.surface_fields[5]
        gttdivJR = gw.surface_fields[7]
        gttdivJR_r = gw.surface_fields[8]

        rec = {
            "nfev": nfev,
            "|J|_max": float(np.max(np.abs(J))),
            "|J|_min": float(np.min(np.abs(J))),
            "|JdivR|_max": float(np.max(np.abs(JdivR))),
            "|gttdivJR|_max": float(np.max(np.abs(gttdivJR))),
            "|gttdivJR_r|_max": float(np.max(np.abs(gttdivJR_r))),
            "|res|_max": float(np.max(np.abs(out))),
            "|res|_rms": float(np.linalg.norm(out) / np.sqrt(out.size)),
        }
        blocks = self._op.build_coeffs(x, include_none=False)
        for name in ("c0", "c1", "s1", "c2", "s2", "c3", "s3"):
            if name in blocks and blocks[name] is not None:
                vals = np.asarray(blocks[name], dtype=np.float64)
                rec[f"max|{name}|"] = float(np.max(np.abs(vals)))

        self.history.append(rec)

        if self.blowup_nfev is None:
            if rec["|gttdivJR|_max"] > 1e10:
                self.blowup_nfev = nfev
                self._print_blowup(rec, kind="GEOMETRY")
            elif rec["|res|_max"] > 1e10:
                self.blowup_nfev = nfev
                self._print_blowup(rec, kind="RESIDUAL")

        if nfev <= 20 or nfev % 50 == 0:
            print(
                f"nfev={nfev:5d}  "
                f"|gttdivJR|_max={rec['|gttdivJR|_max']:12.3e}  "
                f"|J|_min={rec['|J|_min']:12.3e}  "
                f"|res|_max={rec['|res|_max']:12.3e}  "
                f"|res|_rms={rec['|res|_rms']:12.3e}"
            )

        return out

    def _print_blowup(self, rec, kind):
        print(f"\n!!! {kind} BLOWUP at nfev={rec['nfev']} !!!")
        for k in ("|gttdivJR|_max", "|gttdivJR_r|_max", "|J|_min",
                   "|JdivR|_max", "|res|_max"):
            print(f"  {k} = {rec[k]:.3e}")
        for name in ("c0", "c1", "s1", "c2", "s2", "c3", "s3"):
            key = f"max|{name}|"
            if key in rec:
                print(f"  {key} = {rec[key]:.3e}")


def run_diagnostic():
    case, boundary, fit, geqdsk = load_case()
    print(f"boundary fit rms: {float(fit['rms']):.3e}")

    grid = Grid(Nr=SOLVE_NR, Nt=SOLVE_NT, quadrature_scheme="chebyshev")
    operator = Operator(grid, case)
    diag = DiagnosticOperator(operator)

    from scipy.optimize import root

    x0 = operator.encode_initial_state()
    print(f"packed size: {x0.shape[0]}")
    print(f"x0 all zeros: {bool(np.all(x0 == 0.0))}")
    print(f"|x0|_max: {float(np.max(np.abs(x0))):.3e}")
    print()

    try:
        opt = root(
            diag, x0, method="hybr", tol=1e-6,
            options={"maxfev": SOLVER_MAXFEV, "eps": 1e-6},
        )
        print(f"\nFinal: success={opt.success}, nfev={opt.nfev}, msg={opt.message}")
        if hasattr(opt, "fun"):
            print(f"final |fun|_max: {float(np.max(np.abs(opt.fun))):.3e}")
    except Exception as exc:
        print(f"\nException: {type(exc).__name__}: {exc}")

    # --- summary ---
    print(f"\n--- history summary: {len(diag.history)} evaluations ---")
    prev_gtt = 0.0
    for rec in diag.history[:40]:
        gtt = rec["|gttdivJR|_max"]
        marker = ""
        if prev_gtt > 0 and gtt > prev_gtt * 100:
            marker = "  <== SPIKE"
        print(
            f"nfev={rec['nfev']:5d}  "
            f"gttdivJR={gtt:.3e}  "
            f"J_min={rec['|J|_min']:.3e}"
            f"{marker}"
        )
        prev_gtt = gtt

    if len(diag.history) > 40:
        print("... [snip middle] ...")
        for rec in diag.history[-15:]:
            print(
                f"nfev={rec['nfev']:5d}  "
                f"gttdivJR={rec['|gttdivJR|_max']:.3e}  "
                f"J_min={rec['|J|_min']:.3e}"
            )

    if diag.blowup_nfev is not None:
        print(f"\nBlowup confirmed at nfev={diag.blowup_nfev}")
        # Show the pre-blowup state
        idx = next(i for i, r in enumerate(diag.history) if r["nfev"] == diag.blowup_nfev)
        if idx > 0:
            prev = diag.history[idx - 1]
            print(f"Pre-blowup nfev={prev['nfev']}:")
            for k, v in prev.items():
                print(f"  {k} = {v}")
    else:
        print("\nNo blowup detected")

    # Also compare: what does the geometry look like at nfev=1 (initial)?
    print("\n--- shape coefficients at nfev=1, 2, 3 (if available) ---")
    for rec in diag.history[:3]:
        print(f"nfev={rec['nfev']}: ", end="")
        for name in ("c0", "c1", "s1", "c2", "s2", "c3", "s3"):
            key = f"max|{name}|"
            if key in rec:
                print(f"{name}={rec[key]:.4f}  ", end="")
        print()


if __name__ == "__main__":
    run_diagnostic()

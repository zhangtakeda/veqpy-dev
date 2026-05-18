"""
Trace the exact numerical chain from shape params → J → JR → metric → residual
at the worst grid point during the nfev=16 spike.
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

COEFFS = {
    "psin": [0.0] * 10, "h": [0.0] * 10, "k": [0.0] * 10, "v": [0.0] * 10,
    "c0": [0.0] * 5, "c1": [0.0] * 5, "c2": [0.0] * 5, "c3": [0.0] * 5,
    "c4": [0.0] * 5, "c5": [0.0] * 5, "c6": [0.0] * 5, "c7": [0.0] * 5,
    "c8": [0.0] * 5, "c9": [0.0] * 5,
    "s1": [0.0] * 5, "s2": [0.0] * 5, "s3": [0.0] * 5, "s4": [0.0] * 5,
    "s5": [0.0] * 5, "s6": [0.0] * 5, "s7": [0.0] * 5, "s8": [0.0] * 5,
    "s9": [0.0] * 5, "s10": [0.0] * 5,
}


def trace_worst_point():
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
    case = OperatorCase(
        route="PF", coordinate="psin", nodes="uniform",
        profile_coeffs={k: list(v) for k, v in COEFFS.items()},
        boundary=boundary,
        heat_input=MU0 * np.asarray(geqdsk.P_psi, dtype=np.float64),
        current_input=np.asarray(geqdsk.FF_psi, dtype=np.float64),
        Ip=MU0 * float(geqdsk.Ip),
    )
    grid = Grid(Nr=64, Nt=64, quadrature_scheme="chebyshev")
    op = Operator(grid, case)
    x0 = op.encode_initial_state()

    # Evaluate residual at x0
    op(x0)
    gw = op.geometry_workspace

    # Find the worst |gttdivJR_r| point
    gttdivJR_r = gw.surface_fields[8]
    worst_idx = np.unravel_index(np.argmax(np.abs(gttdivJR_r)), gttdivJR_r.shape)
    i_w, j_w = worst_idx

    print("=" * 90)
    print("TRACE: Most extreme metric point at x0=0")
    print("=" * 90)

    rho_i = op.plan.grid_workspace.rho[i_w]
    theta_j = 2.0 * np.pi * j_w / 64

    # Read geometry workspace at this point
    sin_tb = gw.surface_fields[0, i_w, j_w]
    R_ij   = gw.surface_fields[1, i_w, j_w]
    R_t_ij = gw.surface_fields[2, i_w, j_w]
    Z_t_ij = gw.surface_fields[3, i_w, j_w]
    J_ij   = gw.surface_fields[4, i_w, j_w]
    JdivR  = gw.surface_fields[5, i_w, j_w]
    grtdivJR_t = gw.surface_fields[6, i_w, j_w]
    gttdivJR   = gw.surface_fields[7, i_w, j_w]
    gttdivJR_r = gw.surface_fields[8, i_w, j_w]

    print(f"Location: rho[{i_w}]={rho_i:.6f}, theta[{j_w}]={theta_j:.4f} rad = {np.degrees(theta_j):.1f}°")
    print(f"R={R_ij:.6f}, sin_tb={sin_tb:.6f}")
    print(f"R_t={R_t_ij:.6e}, Z_t={Z_t_ij:.6e}")
    print(f"J={J_ij:.6e}, JdivR={JdivR:.6e}")
    print(f"gttdivJR={gttdivJR:.6e}, gttdivJR_r={gttdivJR_r:.6e}")
    print(f"grtdivJR_t={grtdivJR_t:.6e}")

    # Now compute what R_r, Z_r would be (not stored but can be derived)
    # J = R_t * Z_r - R_r * Z_t
    # We know J, R_t, Z_t. Need another relation.
    # From R_r formula: R_r = a*(h_r + cos(tb) - rho*sin(tb)*tb_r)
    # From Z_r formula: Z_r = a*(v_r - (k + rho*k_r)*sin(theta))
    # These require profile derivatives.

    # Read profile fields
    pw = op.profile_workspace
    cf = pw.c_family_fields
    sf = pw.s_family_fields

    # Get profile values and derivatives at this rho
    k_val = op.k_profile.u_fields[0, i_w]
    k_r   = op.k_profile.u_fields[1, i_w]
    v_r   = op.v_profile.u_fields[1, i_w]
    h_r   = op.h_profile.u_fields[1, i_w]
    h_rr  = op.h_profile.u_fields[2, i_w]
    v_rr  = op.v_profile.u_fields[2, i_w]
    k_rr  = op.k_profile.u_fields[2, i_w]

    # c/s at this rho
    c_vals = [cf[m, 0, i_w] for m in range(11)]  # c0..c10
    c_rs   = [cf[m, 1, i_w] for m in range(11)]
    c_rrs  = [cf[m, 2, i_w] for m in range(11)]
    s_vals = [sf[m, 0, i_w] for m in range(11)]  # s0..s10
    s_rs   = [sf[m, 1, i_w] for m in range(11)]
    s_rrs  = [sf[m, 2, i_w] for m in range(11)]

    a = float(fit["a"])
    R0 = float(fit["R0"])
    Z0 = float(fit["Z0"])
    theta_geom = theta_j
    cos_t = np.cos(theta_geom)
    sin_t = np.sin(theta_geom)

    print(f"\n--- Profile values at rho={rho_i:.6f} ---")
    print(f"k={k_val:.6f}, k_r={k_r:.6e}, k_rr={k_rr:.6e}")
    print(f"h_r={h_r:.6e}, v_r={v_r:.6e}")
    print(f"a={a:.6f}, R0={R0:.6f}, Z0={Z0:.6f}")
    print(f"c0={c_vals[0]:.6f}, c1={c_vals[1]:.6f}, c2={c_vals[2]:.6f}")
    print(f"s1={s_vals[1]:.6f}, s2={s_vals[2]:.6f}")
    print(f"c0_r={c_rs[0]:.6e}, c1_r={c_rs[1]:.6e}, s1_r={s_rs[1]:.6e}")

    # --- Recompute geometry step by step ---
    print(f"\n--- Recomputing geometry step-by-step ---")

    # theta_b and derivatives
    tb = theta_geom
    tb_r = 0.0
    tb_t = 1.0
    tb_rr = 0.0
    tb_rt = 0.0
    tb_tt = 0.0

    for m in range(1, 11):
        cos_mt = np.cos(m * theta_geom)
        sin_mt = np.sin(m * theta_geom)
        # c contribution
        tb   += c_vals[m] * cos_mt
        tb_r += c_rs[m] * cos_mt
        tb_t -= c_vals[m] * m * sin_mt
        tb_rr += c_rrs[m] * cos_mt
        tb_rt -= c_rs[m] * m * sin_mt
        tb_tt -= c_vals[m] * m * m * cos_mt
        # s contribution
        tb   += s_vals[m] * sin_mt
        tb_r += s_rs[m] * sin_mt
        tb_t += s_vals[m] * m * cos_mt
        tb_rr += s_rrs[m] * sin_mt
        tb_rt += s_rs[m] * m * cos_mt
        tb_tt -= s_vals[m] * m * m * sin_mt
    # c0 contribution (order 0)
    tb   += c_vals[0]
    tb_r += c_rs[0]
    tb_rr += c_rrs[0]

    print(f"theta_b = {tb:.6f}  (theta={theta_geom:.4f})")
    print(f"tb_r={tb_r:.6e}, tb_t={tb_t:.6e}")
    print(f"tb_rr={tb_rr:.6e}, tb_rt={tb_rt:.6e}, tb_tt={tb_tt:.6e}")

    cos_tb = np.cos(tb)
    sin_tb = np.sin(tb)

    # R and derivatives
    R_calc    = R0 + a * (0.0 + rho_i * cos_tb)
    R_r_calc  = a * (h_r + cos_tb - rho_i * sin_tb * tb_r)
    R_t_calc  = -a * rho_i * sin_tb * tb_t
    R_rr_calc = a * (h_rr - 2.0*sin_tb*tb_r - rho_i*(cos_tb*tb_r*tb_r + sin_tb*tb_rr))
    R_rt_calc = -a * (sin_tb*tb_t + rho_i*(cos_tb*tb_r*tb_t + sin_tb*tb_rt))
    R_tt_calc = -a * rho_i * (cos_tb*tb_t*tb_t + sin_tb*tb_tt)

    # Z and derivatives
    Z_r_calc  = a * (v_r - (k_val + rho_i*k_r) * sin_t)
    Z_t_calc  = -a * rho_i * k_val * cos_t
    Z_rr_calc = a * (v_rr - (2.0*k_r + rho_i*k_rr) * sin_t)
    Z_rt_calc = -a * (k_val + rho_i*k_r) * cos_t
    Z_tt_calc = a * rho_i * k_val * sin_t

    print(f"\nR  = {R_calc:.6f}  (stored: {R_ij:.6f})")
    print(f"R_r= {R_r_calc:.6e}, R_t= {R_t_calc:.6e}  (stored: R_t={R_t_ij:.6e})")
    print(f"Z_r= {Z_r_calc:.6e}, Z_t= {Z_t_calc:.6e}  (stored: Z_t={Z_t_ij:.6e})")

    # J
    J_calc = R_t_calc * Z_r_calc - R_r_calc * Z_t_calc
    print(f"\nJ = R_t*Z_r - R_r*Z_t = {R_t_calc:.6e}*{Z_r_calc:.6e} - {R_r_calc:.6e}*{Z_t_calc:.6e}")
    print(f"  = {R_t_calc*Z_r_calc:.6e} - {R_r_calc*Z_t_calc:.6e}")
    print(f"  = {J_calc:.6e}  (stored: {J_ij:.6e})")

    # JR and metric
    JR_calc = J_calc * R_calc
    inv_JR = 1.0 / JR_calc
    print(f"\nJR = J*R = {J_calc:.6e} * {R_calc:.4f} = {JR_calc:.6e}")
    print(f"1/JR = {inv_JR:.6e}")

    # metric components
    grt = R_r_calc * R_t_calc  # actually this should be grt_ij = R_r*R_t + Z_r*Z_t
    # Actually: grt_ij = R_r_ij * R_t_ij + Z_r_ij * Z_t_ij
    grt_calc = R_r_calc * R_t_calc + Z_r_calc * Z_t_calc
    gtt_calc = R_t_calc * R_t_calc + Z_t_calc * Z_t_calc

    grt_t = R_rt_calc * R_t_calc + R_r_calc * R_tt_calc + Z_rt_calc * Z_t_calc + Z_r_calc * Z_tt_calc
    gtt_r = 2.0 * (R_t_calc * R_rt_calc + Z_t_calc * Z_rt_calc)

    print(f"\ngrt = {grt_calc:.6e}, gtt = {gtt_calc:.6e}")
    print(f"grt_t = {grt_t:.6e}, gtt_r = {gtt_r:.6e}")

    JR_r = J_calc * R_r_calc  # Wait, this is wrong. JR_r = J_r*R + J*R_r
    # J_r = -(R_rr*Z_t - R_rt*Z_r + R_r*Z_rt - R_t*Z_rr)
    J_r_calc = -(R_rr_calc * Z_t_calc - R_rt_calc * Z_r_calc + R_r_calc * Z_rt_calc - R_t_calc * Z_rr_calc)
    JR_r_calc = J_r_calc * R_calc + J_calc * R_r_calc

    print(f"J_r = {J_r_calc:.6e}")
    print(f"JR_r = J_r*R + J*R_r = {J_r_calc:.6e}*{R_calc:.4f} + {J_calc:.6e}*{R_r_calc:.6e} = {JR_r_calc:.6e}")

    # THE EXPLOSIVE TERMS
    grtdivJR_t_calc = (grt_t - grt_calc * JR_r_calc * inv_JR) * inv_JR  # wait, this uses JR_t, not JR_r
    # Let me recompute with JR_t
    J_t_calc = -(R_rt_calc * Z_t_calc - R_tt_calc * Z_r_calc + R_r_calc * Z_tt_calc - R_t_calc * Z_rt_calc)
    JR_t_calc = J_t_calc * R_calc + J_calc * R_t_calc
    grtdivJR_t_calc = (grt_t - grt_calc * JR_t_calc * inv_JR) * inv_JR

    gttdivJR_calc = gtt_calc * inv_JR
    gttdivJR_r_calc = gtt_r * inv_JR - gtt_calc * JR_r_calc * inv_JR * inv_JR

    print(f"\n=== METRIC TERMS (the explosive ones) ===")
    print(f"gttdivJR   = gtt/JR = {gtt_calc:.4e} * {inv_JR:.4e} = {gttdivJR_calc:.4e}")
    print(f"  stored: {gttdivJR:.6e}")
    print(f"gttdivJR_r = gtt_r/JR - gtt*JR_r/JR²")
    print(f"           = {gtt_r:.4e}/{JR_calc:.4e} - {gtt_calc:.4e}*{JR_r_calc:.4e}/{JR_calc:.4e}²")
    term1 = gtt_r * inv_JR
    term2 = gtt_calc * JR_r_calc * inv_JR * inv_JR
    print(f"           = {term1:.4e} - {term2:.4e}")
    print(f"           = {gttdivJR_r_calc:.4e}")
    print(f"  stored: {gttdivJR_r:.6e}")

    # Summary: which term dominates the explosion?
    print(f"\n=== EXPLOSION ANALYSIS ===")
    print(f"JR = {JR_calc:.6e}")
    print(f"If JR were clipped at 1e-10: inv_JR = 1e10, inv_JR² = 1e20")
    print(f"Current JR = {JR_calc:.6e}, 1/JR = {inv_JR:.6e}, 1/JR² = {inv_JR**2:.6e}")
    print(f"gtt term contributes: {gtt_calc:.4e} (should be O(1))")
    print(f"gtt_r term contributes: {gtt_r:.4e} (should be O(1))")
    print(f"JR_r term contributes: {JR_r_calc:.4e} (should be O(1))")
    print(f"The explosion is from 1/JR² = {inv_JR**2:.6e} multiplying gtt*JR_r = {gtt_calc*JR_r_calc:.6e}")
    print(f"Result: {gtt_calc*JR_r_calc*inv_JR**2:.6e}")

    # Find the minimum JR across the grid
    J_all = gw.surface_fields[4]
    R_all = gw.surface_fields[1]
    JR_all = J_all * R_all
    min_jr_idx = np.unravel_index(np.argmin(np.abs(JR_all)), JR_all.shape)
    print(f"\n=== GLOBAL JR STATS ===")
    print(f"JR min abs: {np.min(np.abs(JR_all)):.6e} at {min_jr_idx}")
    print(f"  R={R_all[min_jr_idx]:.6f}, J={J_all[min_jr_idx]:.6e}")
    print(f"If JR_min were clipped at 1e-10: max inv_JR = 1e10")
    print(f"If JR_min were clipped at 1e-8: max inv_JR = 1e8")


if __name__ == "__main__":
    trace_worst_point()

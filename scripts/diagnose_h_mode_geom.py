"""
Follow-up: find WHERE in the grid J clips to 1e-6 at x0=0 for H-mode.
Also check: what are the boundary offsets for CHEASE?
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

CHEASE_PROFILE_COEFFS = {
    "psin": [0.0] * 10, "h": [0.0] * 10, "k": [0.0] * 10, "v": [0.0] * 10,
    "c0": [0.0] * 5, "c1": [0.0] * 5, "c2": [0.0] * 5, "c3": [0.0] * 5,
    "c4": [0.0] * 5, "c5": [0.0] * 5, "c6": [0.0] * 5, "c7": [0.0] * 5,
    "c8": [0.0] * 5, "c9": [0.0] * 5,
    "s1": [0.0] * 5, "s2": [0.0] * 5, "s3": [0.0] * 5, "s4": [0.0] * 5,
    "s5": [0.0] * 5, "s6": [0.0] * 5, "s7": [0.0] * 5, "s8": [0.0] * 5,
    "s9": [0.0] * 5, "s10": [0.0] * 5,
}

geqdsk = Geqdsk()
geqdsk.read_geqdsk("data/CHEASE.geqdsk")
print("=== CHEASE GEQDSK ===")
print(f"NR={geqdsk.NR}, NZ={geqdsk.NZ}")
print(f"R0={geqdsk.R0}, Raxis={geqdsk.Raxis}, Zaxis={geqdsk.Zaxis}")
print(f"Bt0={geqdsk.Bt0}, Ip={geqdsk.Ip}")
print(f"Rmin={geqdsk.Rmin}, Rmax={geqdsk.Rmax}")
print(f"Zmin={geqdsk.Zmin}, Zmax={geqdsk.Zmax}")
print(f"psi_bound={geqdsk.psi_bound}, psi_axis={geqdsk.psi_axis}")
print(f"boundary points: {geqdsk.boundary.shape}")
print(f"boundary min/max R: {geqdsk.boundary[:,0].min():.4f}, {geqdsk.boundary[:,0].max():.4f}")
print(f"boundary min/max Z: {geqdsk.boundary[:,1].min():.4f}, {geqdsk.boundary[:,1].max():.4f}")

fit = _fit_boundary_params(
    geqdsk, M=10, N=10, maxtol=1.0, R0=None, Z0=None, a=None, ka=None,
)
print(f"\n=== Boundary Fit ===")
print(f"rms={float(fit['rms']):.3e}")
print(f"a={float(fit['a']):.4f}, R0={float(fit['R0']):.4f}, Z0={float(fit['Z0']):.4f}, ka={float(fit['ka']):.4f}")
c_offsets = np.asarray(fit["c_offsets"], dtype=np.float64)
s_offsets = np.asarray(fit["s_offsets"], dtype=np.float64)
print(f"c_offsets (len={len(c_offsets)}): {c_offsets[:8]}")
print(f"s_offsets (len={len(s_offsets)}): {s_offsets[:8]}")
print(f"max|c_offset|={float(np.max(np.abs(c_offsets))):.4f}")
print(f"max|s_offset|={float(np.max(np.abs(s_offsets))):.4f}")

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

grid = Grid(Nr=64, Nt=64, quadrature_scheme="chebyshev")
operator = Operator(grid, case)

# --- Evaluate at x0=0 ---
x0 = operator.encode_initial_state()
print(f"\nx0 size={x0.shape[0]}, all zeros={bool(np.all(x0==0))}")

# Run the residual and sample geometry
res = operator(x0)
gw = operator.geometry_workspace

J = gw.surface_fields[4]  # (9, Nr, Nt) -> shape (Nr, Nt)
rho = operator.plan.grid_workspace.rho  # (Nr,)

print(f"\nJ shape: {J.shape}, rho shape: {rho.shape}")
print(f"J min/max: {J.min():.6e} / {J.max():.6e}")

# Find where J clips
clip_mask = J <= 1.01e-6
n_clip = int(np.sum(clip_mask))
print(f"J_clipped cells: {n_clip} / {J.size} ({100*n_clip/J.size:.1f}%)")

# By rho surface
print("\nJ stats by rho surface:")
for i in range(J.shape[0]):
    Ji = J[i]
    n_c = int(np.sum(Ji <= 1.01e-6))
    if n_c > 0 or i < 5 or i >= J.shape[0] - 5:
        print(f"  rho[{i:2d}]={rho[i]:.6f}  J_min={Ji.min():.6e}  J_max={Ji.max():.4f}  "
              f"clip_count={n_c}/{Ji.size}")

# Sample the surface geometry at a problem location
# Find the first (i,j) where J clips
clip_ij = np.argwhere(clip_mask)
if len(clip_ij) > 0:
    i0, j0 = clip_ij[0]
    print(f"\nFirst clip at (i={i0}, j={j0}), rho={rho[i0]:.6f}, theta_idx={j0}")
    print(f"  R = {gw.surface_fields[1, i0, j0]:.4f}")
    print(f"  Z_theta = {gw.surface_fields[3, i0, j0]:.4f}")
    print(f"  J = {J[i0, j0]:.6e}")
    print(f"  R_t = {gw.surface_fields[2, i0, j0]:.6e}")
    # R_t = -a * rho * sin(tb) * tb_t
    # Z_t = -a * rho * k * cos(t)
    # These depend on the profile values...

# --- Also check: what do the profile values look like at the boundary? ---
print("\n\n=== Profile evaluation at x0=0 ===")
# Run stage A manually
operator.stage_a_profile(x0)
operator.stage_b_geometry()

# Check c_fields at the last rho surface
# c_fields shape: (M+1, 3, Nr) where 3 = (value, deriv, 2nd deriv)
print("c_fields shape:", gw.surface_fields.shape)
# Actually the c/s fields are in profile_workspace
pw = operator.profile_workspace
print("c_family_fields shape:", pw.c_family_fields.shape if hasattr(pw, 'c_family_fields') else "N/A")

# Look at the actual R, Z at boundary
R_surf = gw.surface_fields[1]  # (Nr, Nt)
Z_t_surf = gw.surface_fields[3]  # Z_t
print(f"\nR at boundary (rho=1): min={R_surf[-1].min():.4f}, max={R_surf[-1].max():.4f}")
print(f"Expected boundary R: min={geqdsk.boundary[:,0].min():.4f}, max={geqdsk.boundary[:,0].max():.4f}")

# What does the boundary shape look like at x0=0?
# sin_tb = surface_fields[0]
sin_tb = gw.surface_fields[0]
print(f"\nsin_tb at boundary: min={sin_tb[-1].min():.4f}, max={sin_tb[-1].max():.4f}")

# Calculate Z at boundary from geometry:
# Z = Z0 + a * (v(rho) - rho * k(rho) * sin(theta))
# With v=0, k=0: Z = Z0 = 0 for all theta!
# But H-mode has significant elongation...
print(f"\nZ0={float(fit['Z0']):.4f}, ka={float(fit['ka']):.4f}")
print("With k=0, v=0 at x0: Z(rho,theta) = Z0 = 0 for all rho,theta")
print("But H-mode boundary has Z up to ~ka*a, which is large!")
print(f"ka * a = {float(fit['ka'])*float(fit['a']):.4f}")

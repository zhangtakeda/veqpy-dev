from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from veqpy.model import Boundary
from veqpy.model.boundary import _fit_boundary_params
from veqpy.model.geqdsk import Geqdsk

ROOT = Path(__file__).resolve().parent / "fitting"
GEQDSK_PATH = ROOT / "geqdsk.txt"
OUTPUT_PATH = ROOT / "geqdsk-boundary.png"
NR = 6


def build_boundary_curve(boundary: Boundary, *, rho: float = 1.0, n: int = 144) -> np.ndarray:
    rho = float(rho)
    if not 0.0 <= rho <= 1.0:
        raise ValueError(f"rho must lie in [0, 1], got {rho!r}")

    theta = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    theta_bar = theta + boundary.c_offsets[0]

    for order in range(1, len(boundary.c_offsets)):
        theta_bar += boundary.c_offsets[order] * np.cos(order * theta)
    for order in range(1, len(boundary.s_offsets)):
        theta_bar += boundary.s_offsets[order] * np.sin(order * theta)

    # When all optimization-controlled shape profiles are zero, the boundary
    # parameters alone define a rho-scaled family of nested curves.
    R = boundary.R0 + boundary.a * rho * np.cos(theta_bar)
    Z = boundary.Z0 - boundary.a * rho * boundary.ka * np.sin(theta)
    return np.column_stack((R, Z))


def close_path(points: np.ndarray) -> np.ndarray:
    return np.vstack((points, points[0]))


def format_boundary_summary(boundary: Boundary, *, nr: int) -> str:
    c_offsets = np.array2string(boundary.c_offsets, precision=3, separator=", ")
    s_offsets = np.array2string(boundary.s_offsets, precision=3, separator=", ")
    return "\n".join(
        [
            "Boundary",
            f"a: {boundary.a:.3f} [m]",
            f"R0: {boundary.R0:.3f} [m]",
            f"Z0: {boundary.Z0:.3f} [m]",
            f"B0: {boundary.B0:.3f} [T]",
            f"ka: {boundary.ka:.3f}",
            f"c_offsets: {c_offsets}",
            f"s_offsets: {s_offsets}",
            "",
            f"surface lines: {nr}",
            "shape control: boundary only",
            "optimized profiles: all zero",
        ]
    )


def main() -> None:
    if NR < 1:
        raise ValueError(f"NR must be positive, got {NR!r}")

    geqdsk = Geqdsk(str(GEQDSK_PATH))
    fit = _fit_boundary_params(geqdsk, M=None, N=None, maxtol=1.0e-2, R0=None, Z0=None, a=None, ka=None)
    boundary = Boundary.from_geqdsk(str(GEQDSK_PATH))
    surface_rhos = np.linspace(1.0 / NR, 1.0, NR)
    fitted_surfaces = [close_path(build_boundary_curve(boundary, rho=rho)) for rho in surface_rhos]

    original_closed = close_path(np.asarray(geqdsk.boundary, dtype=np.float64))

    fig = plt.figure(figsize=(6, 3))
    ax = fig.add_subplot(121)
    ax_text = fig.add_subplot(122)

    colors = plt.cm.Blues(np.linspace(0.45, 0.90, NR))
    for idx, fitted_closed in enumerate(fitted_surfaces):
        rho = surface_rhos[idx]
        ax.plot(
            fitted_closed[:, 0],
            fitted_closed[:, 1],
            color=colors[idx],
            linewidth=1.0 + 0.8 * rho,
        )
    # ax.plot(original_closed[:, 0], original_closed[:, 1], "--", color="#d62728", linewidth=1.2, alpha=0.75)
    ax.set_aspect("equal")
    ax.set_xlabel("R (m)")
    ax.set_ylabel("Z (m)")
    ax.set_title(f"Boundary-Driven Surfaces (Nr={NR})")
    # ax.grid(alpha=0.2)
    # ax.legend(loc="best")

    ax_text.axis("off")
    ax_text.text(
        0.0,
        1.0,
        format_boundary_summary(boundary, nr=NR),
        va="top",
        ha="left",
        family="monospace",
        fontsize=11,
    )

    fig.tight_layout()
    fig.savefig(OUTPUT_PATH, dpi=180, bbox_inches="tight")
    plt.close(fig)

    print(f"source={GEQDSK_PATH}")
    print(f"output={OUTPUT_PATH}")
    print(f"Nr={NR}")
    print(f"a={boundary.a}")
    print(f"R0={boundary.R0}")
    print(f"Z0={boundary.Z0}")
    print(f"B0={boundary.B0}")
    print(f"ka={boundary.ka}")
    print(f"rms={fit['rms']}")
    print(f"max_curve_error={fit['max_curve_error']}")
    print(f"c_offsets={boundary.c_offsets}")
    print(f"s_offsets={boundary.s_offsets}")


if __name__ == "__main__":
    main()

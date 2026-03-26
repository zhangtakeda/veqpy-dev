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


def build_boundary_curve(boundary: Boundary, *, n: int = 144) -> np.ndarray:
    theta = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    theta_bar = theta + boundary.c_offsets[0]

    for order in range(1, len(boundary.c_offsets)):
        theta_bar += boundary.c_offsets[order] * np.cos(order * theta)
    for order in range(1, len(boundary.s_offsets)):
        theta_bar += boundary.s_offsets[order] * np.sin(order * theta)

    R = boundary.R0 + boundary.a * np.cos(theta_bar)
    Z = boundary.Z0 - boundary.a * boundary.ka * np.sin(theta)
    return np.column_stack((R, Z))


def close_path(points: np.ndarray) -> np.ndarray:
    return np.vstack((points, points[0]))


def main() -> None:
    geqdsk = Geqdsk(str(GEQDSK_PATH))
    fit = _fit_boundary_params(geqdsk, M=None, N=None, maxtol=1.0e-2, R0=None, Z0=None, a=None, ka=None)
    boundary = Boundary.from_geqdsk(str(GEQDSK_PATH))
    fitted = build_boundary_curve(boundary)

    original_closed = close_path(np.asarray(geqdsk.boundary, dtype=np.float64))
    fitted_closed = close_path(fitted)

    fig = plt.figure(figsize=(10.5, 5.5))
    ax = fig.add_subplot(121)
    ax_text = fig.add_subplot(122)

    ax.plot(original_closed[:, 0], original_closed[:, 1], color="#1f77b4", linewidth=2.2, label="Original boundary")
    ax.plot(
        fitted_closed[:, 0], fitted_closed[:, 1], "--", color="#d62728", linewidth=1.8, label="Reconstructed boundary"
    )
    ax.set_aspect("equal")
    ax.set_xlabel("R (m)")
    ax.set_ylabel("Z (m)")
    ax.set_title("Boundary Reconstruction")
    ax.grid(alpha=0.2)
    ax.legend(loc="best")

    ax_text.axis("off")
    ax_text.text(
        0.0,
        1.0,
        "\n".join([(str(boundary))]),
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

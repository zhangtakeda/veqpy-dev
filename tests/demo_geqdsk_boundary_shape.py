import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from veqpy.model import Boundary
from veqpy.model.boundary import _fit_boundary_params
from veqpy.model.geqdsk import Geqdsk

TRUTH = {
    "R0": 1.72,
    "Z0": -0.08,
    "a": 0.43,
    "ka": 1.68,
    "c_offsets": np.array([0.01, 0.08]),
    "s_offsets": np.array([0.0, -0.01, 0.01]),
}


def build_boundary(*, R0, Z0, a, ka, c_offsets, s_offsets, n=721):
    theta = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    theta_bar = theta + c_offsets[0]
    for order in range(1, len(c_offsets)):
        theta_bar += c_offsets[order] * np.cos(order * theta)
    for order in range(1, len(s_offsets)):
        theta_bar += s_offsets[order] * np.sin(order * theta)
    R = R0 + a * np.cos(theta_bar)
    Z = Z0 - a * ka * np.sin(theta)
    return np.column_stack((R, Z))


def close_path(points):
    return np.vstack((points, points[0]))


def max_bidirectional_distance(points_a, points_b):
    diff = points_a[:, None, :] - points_b[None, :, :]
    distances = np.sqrt(np.sum(diff * diff, axis=2))
    return max(distances.min(axis=1).max(), distances.min(axis=0).max())


def build_info_lines(title, params, boundary_error, extra_lines=None, truth=None):
    lines = [
        title,
        "",
        f"rms residual     : {params['rms']:.6e}",
        f"max curve error  : {boundary_error:.6e}",
        "",
    ]
    if extra_lines:
        lines.extend(extra_lines)
        lines.append("")
    if truth is not None:
        lines.extend(
            [
                "Truth vs fit",
                f"R0   : {truth['R0']:+.6f} -> {params['R0']:+.6f}",
                f"Z0   : {truth['Z0']:+.6f} -> {params['Z0']:+.6f}",
                f"a    : {truth['a']:+.6f} -> {params['a']:+.6f}",
                f"ka   : {truth['ka']:+.6f} -> {params['ka']:+.6f}",
                f"c0   : {truth['c_offsets'][0]:+.6f} -> {params['c_offsets'][0]:+.6f}",
                f"c1   : {truth['c_offsets'][1]:+.6f} -> {params['c_offsets'][1]:+.6f}",
                f"s1   : {truth['s_offsets'][1]:+.6f} -> {params['s_offsets'][1]:+.6f}",
                f"s2   : {truth['s_offsets'][2]:+.6f} -> {params['s_offsets'][2]:+.6f}",
            ]
        )
    else:
        lines.extend(
            [
                "Fitted parameters",
                f"R0   : {params['R0']:+.6f}",
                f"Z0   : {params['Z0']:+.6f}",
                f"a    : {params['a']:+.6f}",
                f"ka   : {params['ka']:+.6f}",
                f"M/N  : {params['M']}/{params['N']}",
                f"c0   : {params['c_offsets'][0]:+.6f}",
                f"c1   : {params['c_offsets'][1]:+.6f}" if len(params["c_offsets"]) > 1 else "c1   : n/a",
                f"s1   : {params['s_offsets'][1]:+.6f}" if len(params["s_offsets"]) > 1 else "s1   : n/a",
                f"s2   : {params['s_offsets'][2]:+.6f}" if len(params["s_offsets"]) > 2 else "s2   : n/a",
            ]
        )
    return lines


def render_plot(boundary, fitted_boundary, params, output_path, *, title, info_lines):
    boundary_closed = close_path(boundary)
    fitted_boundary_closed = close_path(fitted_boundary)

    fig = plt.figure(figsize=(11, 5.5))
    ax = fig.add_subplot(121)
    ax_text = fig.add_subplot(122)

    ax.plot(
        boundary_closed[:, 0],
        boundary_closed[:, 1],
        color="#1f77b4",
        linewidth=2.2,
        label="Original boundary",
    )
    ax.plot(
        fitted_boundary_closed[:, 0],
        fitted_boundary_closed[:, 1],
        "--",
        color="#d62728",
        linewidth=1.8,
        label="Fitted boundary",
    )
    ax.scatter(
        [boundary[0, 0], fitted_boundary[0, 0]],
        [boundary[0, 1], fitted_boundary[0, 1]],
        color=["#1f77b4", "#d62728"],
        s=18,
        zorder=5,
    )
    ax.set_aspect("equal")
    ax.set_xlabel("R (m)")
    ax.set_ylabel("Z (m)")
    ax.set_title(title)
    ax.grid(alpha=0.2)
    ax.legend(loc="best")

    ax_text.axis("off")
    ax_text.text(
        0.0,
        1.0,
        "\n".join(info_lines),
        va="top",
        ha="left",
        family="monospace",
        fontsize=10.5,
    )

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(output_path)


def run_synthetic(output_path: Path):
    geqdsk = Geqdsk()
    geqdsk.boundary = build_boundary(**TRUTH)
    geqdsk.R0 = TRUTH["R0"]
    geqdsk.Z0 = TRUTH["Z0"]

    params = _fit_boundary_params(geqdsk, M=1, N=2, R0=None, Z0=None, a=None, ka=None)
    boundary = Boundary.from_geqdsk(geqdsk, M=1, N=2)
    fitted_boundary = build_boundary(
        R0=boundary.R0,
        Z0=boundary.Z0,
        a=boundary.a,
        ka=boundary.ka,
        c_offsets=boundary.c_offsets,
        s_offsets=boundary.s_offsets,
    )
    boundary_error = max_bidirectional_distance(geqdsk.boundary, fitted_boundary)
    info_lines = build_info_lines("Synthetic boundary fit", params, boundary_error, truth=TRUTH)
    render_plot(
        geqdsk.boundary,
        fitted_boundary,
        params,
        output_path,
        title="Boundary fit overlay",
        info_lines=info_lines,
    )


def run_geqdsk(input_path: Path, output_path: Path):
    geqdsk = Geqdsk(str(input_path))
    params = _fit_boundary_params(geqdsk, M=None, N=None, R0=None, Z0=None, a=None, ka=None)
    boundary = Boundary.from_geqdsk(geqdsk)
    fitted_boundary = build_boundary(
        R0=boundary.R0,
        Z0=boundary.Z0,
        a=boundary.a,
        ka=boundary.ka,
        c_offsets=boundary.c_offsets,
        s_offsets=boundary.s_offsets,
        n=len(geqdsk.boundary),
    )
    boundary_error = max_bidirectional_distance(geqdsk.boundary, fitted_boundary)
    extra_lines = [
        f"Source: {input_path.name}",
        f"Raxis : {geqdsk.Raxis:+.6f}",
        f"Zaxis : {geqdsk.Zaxis:+.6f}",
        f"Bt0   : {geqdsk.Bt0:+.6f}",
        f"npts  : {len(geqdsk.boundary)}",
    ]
    info_lines = build_info_lines("GEQDSK boundary fit", params, boundary_error, extra_lines=extra_lines)
    render_plot(
        geqdsk.boundary,
        fitted_boundary,
        params,
        output_path,
        title=f"Boundary fit overlay: {input_path.name}",
        info_lines=info_lines,
    )


if __name__ == "__main__":
    if len(sys.argv) == 1:
        run_synthetic(Path("tests/geqdsk-boundary-shape-fit.png"))
    elif len(sys.argv) == 2:
        input_path = Path(sys.argv[1])
        if input_path.exists():
            run_geqdsk(input_path, Path(f"tests/{input_path.name}-boundary-fit.png"))
        else:
            run_synthetic(Path(sys.argv[1]))
    else:
        run_geqdsk(Path(sys.argv[1]), Path(sys.argv[2]))

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from veqpy.model import Boundary, Grid

ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = ROOT / "generated" / "input-diagrams"

BLUE = "#2F5D7E"
LIGHT_BLUE = "#D7E7F3"
ORANGE = "#D97941"
LIGHT_ORANGE = "#F4D1BC"
GREEN = "#5E8B5A"
GRAY = "#7A828A"
LIGHT_GRAY = "#E9EDF1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate schematic input diagrams for the veqpy architecture figure.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where the generated figures will be written.",
    )
    parser.add_argument(
        "--node-count",
        type=int,
        default=12,
        help="Number of radial sample points used in the node comparison figure.",
    )
    return parser.parse_args()


def pf_reference_profiles(rho: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    rho = np.asarray(rho, dtype=np.float64)
    beta0 = 0.72
    psin = rho**2
    psin_r = 2.0 * rho

    alpha_p = 5.0
    alpha_f = 3.32
    exp_ap = np.exp(alpha_p)
    exp_af = np.exp(alpha_f)
    den_p = 1.0 + exp_ap * (alpha_p - 1.0)
    den_f = 1.0 + exp_af * (alpha_f - 1.0)

    current_input = (1.0 - beta0) * alpha_f * (np.exp(alpha_f * psin) - exp_af) / den_f * psin_r
    heat_input = beta0 * alpha_p * (np.exp(alpha_p * psin) - exp_ap) / den_p * psin_r

    current_input = current_input.copy()
    heat_input = heat_input.copy()
    current_input[0] = 0.0
    current_input[-1] = 0.0
    heat_input[0] = 0.0
    heat_input[-1] = 0.0
    return current_input, heat_input


def normalize_pair(first: np.ndarray, second: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    scale = max(float(np.max(np.abs(first))), float(np.max(np.abs(second))), 1.0e-12)
    return first / scale, second / scale


def build_boundary_curve(boundary: Boundary, *, n: int = 360) -> np.ndarray:
    theta = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    theta_bar = theta + boundary.c_offsets[0]

    for order in range(1, len(boundary.c_offsets)):
        theta_bar += boundary.c_offsets[order] * np.cos(order * theta)
    for order in range(1, len(boundary.s_offsets)):
        theta_bar += boundary.s_offsets[order] * np.sin(order * theta)

    R = boundary.R0 + boundary.a * np.cos(theta_bar)
    Z = boundary.Z0 - boundary.a * boundary.ka * np.sin(theta)
    return np.column_stack((R, Z))


def close_curve(points: np.ndarray) -> np.ndarray:
    return np.vstack((points, points[0]))


def build_boundary_examples() -> tuple[Boundary, Boundary]:
    base_kwargs = {
        "a": 0.55,
        "R0": 1.05,
        "Z0": 0.00,
        "B0": 3.00,
        "ka": 1.85,
    }

    low_precision = Boundary(
        **base_kwargs,
        c_offsets=np.array([0.00, 0.04], dtype=np.float64),
        s_offsets=np.array([0.0, 0.18], dtype=np.float64),
    )
    high_precision = Boundary(
        **base_kwargs,
        c_offsets=np.array([0.02, 0.10, -0.05, 0.025, -0.012], dtype=np.float64),
        s_offsets=np.array([0.0, 0.30, -0.09, 0.045, -0.020], dtype=np.float64),
    )
    return low_precision, high_precision


def configure_matplotlib() -> None:
    plt.style.use("seaborn-v0_8-paper")
    plt.rcParams.update(
        {
            "font.size": 12,
            "axes.titlesize": 14,
            "axes.labelsize": 12,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "legend.fontsize": 10,
            "figure.titlesize": 16,
            "axes.facecolor": "white",
        }
    )


def save_figure(fig: plt.Figure, outdir: Path, stem: str) -> tuple[Path, Path]:
    png_path = outdir / f"{stem}.png"
    svg_path = outdir / f"{stem}.svg"
    fig.savefig(png_path, dpi=220, bbox_inches="tight", facecolor="white")
    fig.savefig(svg_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return png_path, svg_path


def plot_source_node_diagram(outdir: Path, *, node_count: int) -> tuple[Path, Path]:
    gauss_grid = Grid(Nr=node_count, Nt=16, scheme="legendre")
    uniform_grid = Grid(Nr=node_count, Nt=16, scheme="uniform")

    rho_dense = np.linspace(0.0, 1.0, 512)
    current_dense, heat_dense = pf_reference_profiles(rho_dense)
    current_dense = np.abs(current_dense)
    heat_dense = np.abs(heat_dense)
    current_dense, heat_dense = normalize_pair(current_dense, heat_dense)

    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.6), sharex=True, sharey=True)

    node_specs = (
        ("Gauss-like nodes (Legendre grid)", gauss_grid.rho, LIGHT_BLUE),
        ("Uniform nodes", uniform_grid.rho, LIGHT_ORANGE),
    )
    for ax, (title, nodes, panel_fill) in zip(axes, node_specs, strict=True):
        current_nodes, heat_nodes = pf_reference_profiles(nodes)
        current_nodes = np.abs(current_nodes)
        heat_nodes = np.abs(heat_nodes)
        current_nodes, heat_nodes = normalize_pair(current_nodes, heat_nodes)

        ax.set_facecolor(panel_fill)
        for x_coord in nodes:
            ax.axvline(float(x_coord), color=GRAY, linewidth=0.8, alpha=0.20, zorder=0)

        ax.plot(rho_dense, heat_dense, color=BLUE, linewidth=2.2, label="heat input")
        ax.plot(rho_dense, current_dense, color=ORANGE, linewidth=2.2, label="current input")
        ax.scatter(nodes, heat_nodes, s=44, color=BLUE, edgecolors="white", linewidths=0.8, zorder=4)
        ax.scatter(nodes, current_nodes, s=44, marker="s", color=ORANGE, edgecolors="white", linewidths=0.8, zorder=4)
        ax.scatter(nodes, np.full_like(nodes, -0.07), s=18, color=GREEN, zorder=3)

        ax.set_title(title, color="#243746", fontweight="bold")
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(-0.12, 1.08)
        ax.set_xlabel("normalized radius")
        ax.grid(alpha=0.15, linewidth=0.8)

    axes[0].set_ylabel("normalized source magnitude")
    axes[0].legend(loc="upper center", ncol=2, frameon=True)

    fig.suptitle("Input Sampling Diversity", fontweight="bold")
    fig.text(0.5, 0.02, "Green dots mark the actual source-node placement used for interpolation or direct sampling.", ha="center")
    fig.tight_layout(rect=(0.0, 0.05, 1.0, 0.94))
    return save_figure(fig, outdir, "input-nodes-comparison")


def plot_boundary_diversity_diagram(outdir: Path) -> tuple[Path, Path]:
    low_precision, high_precision = build_boundary_examples()
    low_curve = close_curve(build_boundary_curve(low_precision))
    high_curve = close_curve(build_boundary_curve(high_precision))

    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.8), sharex=True, sharey=True)

    panel_specs = (
        ("Lower-precision boundary", low_curve, BLUE, LIGHT_BLUE, "fewer Fourier offsets"),
        ("Higher-precision boundary", high_curve, ORANGE, LIGHT_ORANGE, "richer Fourier offsets"),
    )

    x_min = min(float(np.min(low_curve[:, 0])), float(np.min(high_curve[:, 0]))) - 0.08
    x_max = max(float(np.max(low_curve[:, 0])), float(np.max(high_curve[:, 0]))) + 0.08
    y_min = min(float(np.min(low_curve[:, 1])), float(np.min(high_curve[:, 1]))) - 0.08
    y_max = max(float(np.max(low_curve[:, 1])), float(np.max(high_curve[:, 1]))) + 0.08

    for ax, (title, curve, edge_color, fill_color, subtitle) in zip(axes, panel_specs, strict=True):
        ax.set_facecolor("white")
        ax.fill(curve[:, 0], curve[:, 1], color=fill_color, alpha=0.85, zorder=1)
        ax.plot(curve[:, 0], curve[:, 1], color=edge_color, linewidth=2.6, zorder=2)
        ax.axhline(0.0, color=LIGHT_GRAY, linewidth=1.0, zorder=0)
        ax.axvline(1.05, color=LIGHT_GRAY, linewidth=1.0, zorder=0)
        ax.scatter([1.05], [0.0], s=26, color="#243746", zorder=3)

        ax.set_title(title, color="#243746", fontweight="bold")
        ax.text(0.5, 0.03, subtitle, transform=ax.transAxes, ha="center", va="bottom", color=GRAY)
        ax.set_aspect("equal")
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

    fig.suptitle("Boundary Input Diversity", fontweight="bold")
    fig.text(0.5, 0.02, "Both examples share the same major radius and minor radius, but differ in harmonic detail.", ha="center")
    fig.tight_layout(rect=(0.0, 0.05, 1.0, 0.93))
    return save_figure(fig, outdir, "boundary-diversity")


def main() -> None:
    args = parse_args()
    configure_matplotlib()

    outdir = args.output_dir.resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    nodes_png, nodes_svg = plot_source_node_diagram(outdir, node_count=args.node_count)
    boundary_png, boundary_svg = plot_boundary_diversity_diagram(outdir)

    print(f"output_dir={outdir}")
    print(f"nodes_png={nodes_png}")
    print(f"nodes_svg={nodes_svg}")
    print(f"boundary_png={boundary_png}")
    print(f"boundary_svg={boundary_svg}")


if __name__ == "__main__":
    main()

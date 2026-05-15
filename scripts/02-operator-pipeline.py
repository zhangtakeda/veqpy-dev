import argparse
import os
from pathlib import Path

import matplotlib
import sys

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.config import (
    DOUBLE_COLUMN_WIDTH,
    SAVE_DPI,
    SAVE_TRANSPARENT,
    TITLE_FONT_SIZE,
    apply_plot_style,
    scaled_font_size,
)

FIGURE_WIDTH = DOUBLE_COLUMN_WIDTH
FIGURE_HEIGHT = 4.15
FIGURE_SIZE = (FIGURE_WIDTH, FIGURE_HEIGHT)

# Horizontal two-panel layout.  PANEL_A_WIDTH_RATIO controls how much of the
# usable width is assigned to panel (a); the remaining width is assigned to
# panel (b).  Both source images are fit into their allocated column while
# preserving their original aspect ratios.
PANEL_A_WIDTH_RATIO = 0.60
PANEL_LEFT_MARGIN = 0.15
PANEL_RIGHT_MARGIN = 0.10
PANEL_TOP_MARGIN = 0.10
PANEL_BOTTOM_MARGIN = 0.05
PANEL_GAP = 0.20
PANEL_TITLE_HEIGHT = 0.34
PANEL_TITLE_GAP = 0.08

INPUT_PANEL_PATHS = {
    "a": "figures/02a.png",
    "b": "figures/02b.png",
}
OUTPUT_PNG_PATH = "figures/02.png"
OUTPUT_PDF_PATH = "figures/02.pdf"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compose Figure 02 from two PNG panels.")
    parser.add_argument("--input-a", default=INPUT_PANEL_PATHS["a"])
    parser.add_argument("--input-b", default=INPUT_PANEL_PATHS["b"])
    parser.add_argument(
        "--a-width-ratio",
        type=float,
        default=PANEL_A_WIDTH_RATIO,
        help="Fraction of usable horizontal space assigned to panel (a).",
    )
    parser.add_argument("--output-png", default=OUTPUT_PNG_PATH)
    parser.add_argument("--output-pdf", default=OUTPUT_PDF_PATH)
    return parser.parse_args()


def resolve_input_path(path: str | os.PathLike[str]) -> Path:
    input_path = Path(path)
    if input_path.suffix.lower() == ".png":
        if input_path.is_file():
            return input_path
        raise FileNotFoundError(f"Missing input PNG: {input_path}")
    raise ValueError(f"Figure 02 inputs must be PNG files exported from PPT, got {input_path}")


def load_panel(input_path: Path) -> np.ndarray:
    with Image.open(input_path) as image:
        return np.asarray(image.convert("RGBA"))


def panel_height_over_width(image: np.ndarray) -> float:
    height, width = image.shape[:2]
    return float(height) / max(float(width), 1.0)


def add_sized_panel(
    fig: plt.Figure,
    image: np.ndarray,
    label: str,
    *,
    left: float,
    bottom: float,
    max_width: float,
    max_height: float,
) -> None:
    fig_width, fig_height = fig.get_size_inches()
    image_aspect = panel_height_over_width(image)
    width = max_width
    height = width * image_aspect
    if height > max_height:
        height = max_height
        width = height / image_aspect

    panel_left = left + 0.5 * (max_width - width)
    panel_bottom = bottom + max_height - height
    ax = fig.add_axes(
        [
            panel_left / fig_width,
            panel_bottom / fig_height,
            width / fig_width,
            height / fig_height,
        ]
    )
    ax.imshow(image, aspect="equal")
    ax.set_axis_off()
    fig.text(
        left / fig_width,
        (bottom + max_height + PANEL_TITLE_HEIGHT - PANEL_TITLE_GAP) / fig_height,
        rf"$\bf{{({label})}}$",
        ha="left",
        va="top",
        fontsize=scaled_font_size(TITLE_FONT_SIZE),
    )


def validate_a_width_ratio(a_width_ratio: float) -> float:
    if not 0.1 <= a_width_ratio <= 0.9:
        raise ValueError("--a-width-ratio must be between 0.1 and 0.9")
    return a_width_ratio


def build_figure(
    images: dict[str, np.ndarray], *, a_width_ratio: float = PANEL_A_WIDTH_RATIO
) -> plt.Figure:
    fig = plt.figure(figsize=FIGURE_SIZE)
    a_width_ratio = validate_a_width_ratio(a_width_ratio)
    usable_width = FIGURE_WIDTH - PANEL_LEFT_MARGIN - PANEL_RIGHT_MARGIN - PANEL_GAP
    panel_a_width = usable_width * a_width_ratio
    panel_b_width = usable_width - panel_a_width
    panel_height = FIGURE_HEIGHT - PANEL_TOP_MARGIN - PANEL_BOTTOM_MARGIN - PANEL_TITLE_HEIGHT
    panel_bottom = PANEL_BOTTOM_MARGIN
    add_sized_panel(
        fig,
        images["a"],
        "a",
        left=PANEL_LEFT_MARGIN,
        bottom=panel_bottom,
        max_width=panel_a_width,
        max_height=panel_height,
    )
    add_sized_panel(
        fig,
        images["b"],
        "b",
        left=PANEL_LEFT_MARGIN + panel_a_width + PANEL_GAP,
        bottom=panel_bottom,
        max_width=panel_b_width,
        max_height=panel_height,
    )
    return fig


def main() -> None:
    args = parse_args()
    apply_plot_style()
    input_paths = {
        "a": resolve_input_path(args.input_a),
        "b": resolve_input_path(args.input_b),
    }
    images = {key: load_panel(path) for key, path in input_paths.items()}

    fig = build_figure(images, a_width_ratio=args.a_width_ratio)
    output_png = Path(args.output_png)
    output_pdf = Path(args.output_pdf)
    output_png.parent.mkdir(parents=True, exist_ok=True)
    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=SAVE_DPI, transparent=SAVE_TRANSPARENT)
    fig.savefig(output_pdf, dpi=SAVE_DPI, transparent=SAVE_TRANSPARENT)
    plt.close(fig)
    print(f"saved: {output_png}")
    print(f"saved: {output_pdf}")


if __name__ == "__main__":
    main()

import ast
from pathlib import Path

import numpy as np

from veqpy.engine.numba_source import (
    _regularize_axis_linear_psin_r,
)


def test_psin_coordinate_update_uses_base_accumulator():
    """Route kernels should integrate psin_r with the base grid integration matrix."""

    source = Path("veqpy/engine/numba_source.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    call_args: list[str] = []

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if not isinstance(node.func, ast.Name):
            continue
        if node.func.id != "_update_psin_coordinate":
            continue
        assert len(node.args) == 3
        call_args.append(ast.unparse(node.args[2]))

    assert call_args
    assert set(call_args) == {"accumulator"}


def test_axis_psin_r_regularizer_dynamic_anchors():
    rho = np.array(
        [0.002, 0.012, 0.035, 0.062, 0.095, 0.14, 0.20],
        dtype=np.float64,
    )
    polluted_slopes = np.array([0.0, 12.0, 5.0, 2.38, 2.35, 2.352, 2.353], dtype=np.float64)
    psin_r = rho * polluted_slopes

    _regularize_axis_linear_psin_r(psin_r, rho)

    # First rho >= 0.05 is rho[3] = 0.062, so anchors are indices 3, 4
    anchor0, anchor1 = 3, 4
    x0 = rho[anchor0] * rho[anchor0]
    x1 = rho[anchor1] * rho[anchor1]
    slope_gradient = (polluted_slopes[anchor1] - polluted_slopes[anchor0]) / (x1 - x0)
    expected_head_slopes = polluted_slopes[anchor0] + slope_gradient * (
        rho[:anchor0] * rho[:anchor0] - x0
    )

    assert np.allclose(psin_r[:anchor0] / rho[:anchor0], expected_head_slopes)
    assert np.allclose(psin_r[anchor0:] / rho[anchor0:], polluted_slopes[anchor0:])
    assert np.ptp(psin_r[:anchor0] / rho[:anchor0]) > 1e-4

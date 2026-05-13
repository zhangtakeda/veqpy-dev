import ast
from pathlib import Path

import numpy as np

from veqpy.engine.numba_source import (
    _regularize_psin_r,
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


def test_axis_psin_r_regularizer_precomputed_n_fix():
    rho = np.array(
        [0.002, 0.012, 0.035, 0.062, 0.095, 0.14, 0.20],
        dtype=np.float64,
    )
    polluted_slopes = np.array([0.0, 12.0, 5.0, 2.38, 2.35, 2.352, 2.353], dtype=np.float64)
    psin_r = rho * polluted_slopes

    # n_fix = count of rho < 0.05 = 3 (indices 0, 1, 2)
    n_fix = 3
    _regularize_psin_r(psin_r, rho, n_fix)

    anchor0, anchor1 = n_fix, n_fix + 1
    x0 = rho[anchor0] * rho[anchor0]
    x1 = rho[anchor1] * rho[anchor1]
    slope_gradient = (polluted_slopes[anchor1] - polluted_slopes[anchor0]) / (x1 - x0)
    expected_head_slopes = polluted_slopes[anchor0] + slope_gradient * (
        rho[:anchor0] * rho[:anchor0] - x0
    )

    assert np.allclose(psin_r[:anchor0] / rho[:anchor0], expected_head_slopes)
    assert np.allclose(psin_r[anchor0:] / rho[anchor0:], polluted_slopes[anchor0:])
    assert np.ptp(psin_r[:anchor0] / rho[:anchor0]) > 1e-4

    # n_fix=0 should be a no-op (all points already in clean region)
    psin_r2 = rho * polluted_slopes
    _regularize_psin_r(psin_r2, rho, 0)
    assert np.allclose(psin_r2, rho * polluted_slopes)


def test_axis_psin_r_regularizer_applies_global_floor():
    rho = np.array([0.002, 0.012, 0.035, 0.062], dtype=np.float64)
    psin_r = np.array([0.0, -1.0e-12, 2.0e-10, 4.0e-10], dtype=np.float64)

    _regularize_psin_r(psin_r, rho, 0)

    assert np.all(psin_r >= 1.0e-10)
    assert psin_r[2] == 2.0e-10


def test_numba_source_uses_regularized_psin_r_directly():
    source = Path("veqpy/engine/numba_source.py").read_text(encoding="utf-8")

    assert "psin_r_safe" not in source
    assert "_SLOT_PSIN_R_SAFE" not in source
    assert "maximum_floor_into(psin_r" not in source
    assert "_regularize_psin_r(Itor_r" not in source

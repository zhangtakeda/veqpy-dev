import ast
from pathlib import Path

import numpy as np

from veqpy.engine.numba_source import (
    _regularize_axis_linear_psin_r,
    _regularize_true_axis_ratio_profile,
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


def test_axis_psin_r_regularizer_extrapolates_from_stable_interior():
    rho = np.array(
        [0.00060227, 0.00541175, 0.01498437, 0.02922797, 0.04800535, 0.07113569],
        dtype=np.float64,
    )
    polluted_slopes = np.array([0.0, 7.7, 2.1, 2.38, 2.348, 2.352], dtype=np.float64)
    psin_r = rho * polluted_slopes

    _regularize_axis_linear_psin_r(psin_r, rho)

    x0 = rho[4] * rho[4]
    x1 = rho[5] * rho[5]
    slope_gradient = (polluted_slopes[5] - polluted_slopes[4]) / (x1 - x0)
    expected_head_slopes = polluted_slopes[4] + slope_gradient * (rho[:4] * rho[:4] - x0)

    assert np.allclose(psin_r[:4] / rho[:4], expected_head_slopes)
    assert np.allclose(psin_r[4:] / rho[4:], polluted_slopes[4:])
    assert np.ptp(psin_r[:4] / rho[:4]) > 1e-4


def test_true_axis_ratio_regularizer_skips_off_axis_nodes():
    rho = np.array([0.0006, 0.0054, 0.015, 0.029], dtype=np.float64)
    profile = np.array([100.0, 2.0, 3.0, 4.0], dtype=np.float64)
    original = profile.copy()

    _regularize_true_axis_ratio_profile(profile, rho)

    assert np.array_equal(profile, original)


def test_true_axis_ratio_regularizer_replaces_only_axis_sample():
    rho = np.array([0.0, 0.1, 0.2, 0.3], dtype=np.float64)
    profile = np.array([0.0, 2.0, 3.0, 4.0], dtype=np.float64)

    _regularize_true_axis_ratio_profile(profile, rho)

    assert profile[0] == np.float64(1.0)
    assert np.array_equal(profile[1:], np.array([2.0, 3.0, 4.0], dtype=np.float64))

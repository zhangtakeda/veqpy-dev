import sys
import types
from pathlib import Path

import numpy as np
import pytest

units_module = types.ModuleType("units")
units_module.base = types.SimpleNamespace(get_mesh=lambda mesh: mesh)
sys.modules.setdefault("units", units_module)

from veqpy.model.geqdsk import Geqdsk


def _build_boundary(*, R0, Z0, a, ka, c0a, c1a, s1a, s2a, n=721):
    theta = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    theta_bar = theta + c0a + c1a * np.cos(theta) + s1a * np.sin(theta) + s2a * np.sin(2.0 * theta)
    R = R0 + a * np.cos(theta_bar)
    Z = Z0 - a * ka * np.sin(theta)
    return np.column_stack((R, Z))


def _max_bidirectional_distance(points_a, points_b):
    diff = points_a[:, None, :] - points_b[None, :, :]
    distances = np.sqrt(np.sum(diff * diff, axis=2))
    return max(distances.min(axis=1).max(), distances.min(axis=0).max())


def test_boundary_shape_params_recovers_synthetic_boundary():
    truth = {
        "R0": 1.72,
        "Z0": -0.08,
        "a": 0.43,
        "ka": 1.68,
        "c0a": 0.01,
        "c1a": 0.08,
        "s1a": -0.01,
        "s2a": 0.01,
    }

    geqdsk = Geqdsk()
    geqdsk.boundary = _build_boundary(**truth)

    params = geqdsk.boundary_shape_params(R0=truth["R0"], Z0=truth["Z0"], a=truth["a"])
    fitted_boundary = _build_boundary(
        R0=params["R0"],
        Z0=params["Z0"],
        a=params["a"],
        ka=params["ka"],
        c0a=params["c0a"],
        c1a=params["c1a"],
        s1a=params["s1a"],
        s2a=params["s2a"],
    )

    assert params["rms"] < 1.0e-2
    assert params["R0"] == pytest.approx(truth["R0"])
    assert params["Z0"] == pytest.approx(truth["Z0"])
    assert params["a"] == pytest.approx(truth["a"])
    assert params["ka"] == pytest.approx(truth["ka"], abs=1.0e-2)
    assert np.sign(params["c1a"]) == np.sign(truth["c1a"])
    assert np.sign(params["s1a"]) == np.sign(truth["s1a"])
    assert np.sign(params["s2a"]) == np.sign(truth["s2a"])
    assert _max_bidirectional_distance(geqdsk.boundary, fitted_boundary) < 1.5e-2


def test_boundary_shape_params_jointly_optimizes_r0_z0_a():
    truth = {
        "R0": 1.72,
        "Z0": -0.08,
        "a": 0.43,
        "ka": 1.68,
        "c0a": 0.01,
        "c1a": 0.08,
        "s1a": -0.01,
        "s2a": 0.01,
    }

    geqdsk = Geqdsk()
    geqdsk.boundary = _build_boundary(**truth)
    geqdsk.R0 = truth["R0"] + 0.12
    geqdsk.Z0 = truth["Z0"] - 0.15

    params = geqdsk.boundary_shape_params()
    fitted_boundary = _build_boundary(
        R0=params["R0"],
        Z0=params["Z0"],
        a=params["a"],
        ka=params["ka"],
        c0a=params["c0a"],
        c1a=params["c1a"],
        s1a=params["s1a"],
        s2a=params["s2a"],
    )

    assert params["rms"] < 1.0e-3
    assert params["R0"] == pytest.approx(truth["R0"], abs=1.0e-3)
    assert params["Z0"] == pytest.approx(truth["Z0"], abs=1.0e-3)
    assert params["a"] == pytest.approx(truth["a"], abs=1.0e-3)
    assert params["ka"] == pytest.approx(truth["ka"], abs=1.0e-2)
    assert _max_bidirectional_distance(geqdsk.boundary, fitted_boundary) < 1.5e-2


def test_boundary_shape_params_raises_for_empty_boundary():
    geqdsk = Geqdsk()

    with pytest.raises(ValueError, match="Boundary is empty"):
        geqdsk.boundary_shape_params()


def test_boundary_shape_params_fits_real_gfile():
    geqdsk = Geqdsk(str(Path("tests") / "gfile"))
    params = geqdsk.boundary_shape_params()

    assert params["rms"] < 6.0e-2
    assert params["a"] > 0.0
    assert params["ka"] > 1.0
    assert -np.pi <= params["c0a"] <= np.pi
